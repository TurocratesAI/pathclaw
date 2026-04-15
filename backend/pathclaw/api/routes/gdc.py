"""GDC/TCGA API routes — search and async download."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

PATHCLAW_DATA_DIR = Path(os.environ.get("PATHCLAW_DATA_DIR", "~/.pathclaw")).expanduser()
DOWNLOADS_DIR = PATHCLAW_DATA_DIR / "downloads"
JOBS_DIR = PATHCLAW_DATA_DIR / "jobs"
DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
JOBS_DIR.mkdir(parents=True, exist_ok=True)

GDC_FILES_URL = "https://api.gdc.cancer.gov/files"
GDC_DATA_URL = "https://api.gdc.cancer.gov/data"

router = APIRouter()


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class GDCSearchRequest(BaseModel):
    project: str = ""                   # e.g. "TCGA-UCEC"
    data_type: str = "Slide Image"      # "Slide Image" | "Masked Somatic Mutation" | "Clinical Supplement"
    experimental_strategy: str = ""     # "Diagnostic Slide" | "Tissue Slide"
    workflow_type: str = ""             # e.g. "MuTect2 Variant Aggregation and Masking", "STAR - Counts"
    primary_diagnosis: list[str] = Field(default_factory=list)
    access: str = "open"                # "open" | "controlled"
    limit: int = 500


class GDCDownloadRequest(BaseModel):
    file_ids: list[str] = Field(default_factory=list)
    output_dir: str = ""           # default: ~/.pathclaw/downloads/{project}
    project: str = ""              # used for default output_dir naming
    gdc_token: str = ""            # required for controlled-access files
    max_concurrent: int = 4        # parallel download streams


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_filters(req: GDCSearchRequest) -> dict:
    content = []

    if req.project:
        content.append({
            "op": "in",
            "content": {"field": "cases.project.project_id", "value": [req.project]},
        })

    if req.data_type:
        content.append({
            "op": "in",
            "content": {"field": "data_type", "value": [req.data_type]},
        })

    if req.experimental_strategy:
        content.append({
            "op": "in",
            "content": {"field": "experimental_strategy", "value": [req.experimental_strategy]},
        })

    if req.workflow_type:
        content.append({
            "op": "in",
            "content": {"field": "analysis.workflow_type", "value": [req.workflow_type]},
        })

    if req.primary_diagnosis:
        content.append({
            "op": "in",
            "content": {"field": "cases.diagnoses.primary_diagnosis", "value": req.primary_diagnosis},
        })

    if req.access:
        content.append({
            "op": "in",
            "content": {"field": "access", "value": [req.access]},
        })

    if not content:
        return {}
    return {"op": "and", "content": content}


def _job_path(job_id: str) -> Path:
    return JOBS_DIR / f"dl_{job_id}.json"


def _update_job(job_id: str, **kwargs):
    p = _job_path(job_id)
    data = json.loads(p.read_text()) if p.exists() else {}
    data.update(kwargs)
    p.write_text(json.dumps(data, indent=2))


# ---------------------------------------------------------------------------
# Standalone downloader script (runs as detached subprocess — survives restarts)
# ---------------------------------------------------------------------------

_DOWNLOADER_SCRIPT = '''
import asyncio, json, sys, os
from pathlib import Path
from datetime import datetime, timezone

async def run(job_id, file_ids, out_dir, token, max_concurrent, jobs_dir):
    import httpx
    out_dir = Path(out_dir)
    jobs_dir = Path(jobs_dir)
    job_file = jobs_dir / f"dl_{job_id}.json"
    out_dir.mkdir(parents=True, exist_ok=True)

    def update(**kw):
        data = json.loads(job_file.read_text()) if job_file.exists() else {}
        data.update(kw)
        job_file.write_text(json.dumps(data, indent=2))

    GDC_DATA_URL = "https://api.gdc.cancer.gov/data"
    total = len(file_ids)
    done = 0
    bytes_done = 0
    failed = []
    errors = {}
    headers = {"X-Auth-Token": token} if token else {}
    sem = asyncio.Semaphore(max_concurrent)

    async def fetch_one(fid):
        nonlocal done, bytes_done
        async with sem:
            # Check if already downloaded (by UUID filename or SVS)
            existing = list(out_dir.glob(f"*{fid[:8]}*"))
            if existing:
                done += 1
                fsize = existing[0].stat().st_size
                bytes_done += fsize
                update(done=done, total=total, bytes_done=bytes_done, status="running",
                       message=f"Downloading... {done}/{total} ({bytes_done/1e9:.2f} GB)")
                return

            tmp = out_dir / f".{fid}.part"
            last_err = ""
            for attempt in range(3):
                try:
                    async with httpx.AsyncClient(timeout=None, follow_redirects=True) as client:
                        async with client.stream("GET", f"{GDC_DATA_URL}/{fid}", headers=headers) as resp:
                            if resp.status_code == 401:
                                errors[fid] = "401 Unauthorized — GDC token required for controlled-access files"
                                failed.append(fid)
                                return
                            if resp.status_code != 200:
                                last_err = f"HTTP {resp.status_code}"
                                continue
                            cd = resp.headers.get("content-disposition", "")
                            fname = fid
                            if "filename=" in cd:
                                part = cd.split("filename=")[-1].strip().strip(chr(34))
                                if part:
                                    fname = part
                            real_dest = out_dir / fname
                            file_bytes = 0
                            with open(tmp, "wb") as f:
                                async for chunk in resp.aiter_bytes(1024 * 1024):
                                    f.write(chunk)
                                    file_bytes += len(chunk)
                            tmp.rename(real_dest)
                            done += 1
                            bytes_done += file_bytes
                            update(done=done, total=total, bytes_done=bytes_done, status="running",
                                   message=f"Downloading... {done}/{total} ({bytes_done/1e9:.2f} GB)")
                            return
                except Exception as e:
                    last_err = f"{type(e).__name__}: {e}"
                    if tmp.exists():
                        tmp.unlink(missing_ok=True)
                    if attempt < 2:
                        await asyncio.sleep(5 * (attempt + 1))

            errors[fid] = last_err
            failed.append(fid)
            update(errors=errors)

    await asyncio.gather(*[fetch_one(fid) for fid in file_ids])

    status = "completed" if not failed else ("partial" if done > 0 else "failed")
    update(
        status=status,
        done=done,
        total=total,
        bytes_done=bytes_done,
        failed=failed,
        errors=errors,
        output_dir=str(out_dir),
        finished_at=datetime.now(timezone.utc).isoformat(),
        message=f"Done: {done}/{total} files" + (f" | {len(failed)} failed: {list(errors.values())[:2]}" if failed else ""),
    )

args = json.loads(sys.argv[1])
asyncio.run(run(**args))
'''


def _launch_download_subprocess(job_id: str, file_ids: list[str], out_dir: Path,
                                 token: str, max_concurrent: int) -> None:
    """Spawn a detached Python subprocess to download files.
    The subprocess writes progress to the job file and survives server restarts."""
    args = json.dumps({
        "job_id": job_id,
        "file_ids": file_ids,
        "out_dir": str(out_dir),
        "token": token,
        "max_concurrent": max_concurrent,
        "jobs_dir": str(JOBS_DIR),
    })
    # Write downloader script to a temp file
    script_path = JOBS_DIR / f"_dl_{job_id}.py"
    script_path.write_text(_DOWNLOADER_SCRIPT)

    # Spawn detached (POSIX: new session so it won't be killed when server dies)
    log_path = JOBS_DIR / f"dl_{job_id}.log"
    with open(log_path, "w") as log_f:
        proc = subprocess.Popen(
            [sys.executable, str(script_path), args],
            stdout=log_f,
            stderr=log_f,
            start_new_session=True,  # detach from server's process group
        )

    _update_job(job_id, pid=proc.pid)


# ---------------------------------------------------------------------------
# Routes — search
# ---------------------------------------------------------------------------

@router.post("/search")
async def search_gdc(req: GDCSearchRequest):
    """Search GDC for files matching project/data_type/diagnosis criteria."""
    filters = _build_filters(req)
    payload: dict = {
        "size": req.limit,
        "fields": "file_id,file_name,file_size,data_type,cases.project.project_id",
    }
    if filters:
        payload["filters"] = filters   # pass as object, not stringified

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(GDC_FILES_URL, json=payload)
            resp.raise_for_status()
            data = resp.json()

        hits = data.get("data", {}).get("hits", [])
        total = data.get("data", {}).get("pagination", {}).get("total", 0)

        files = []
        for h in hits:
            files.append({
                "file_id": h.get("file_id"),
                "file_name": h.get("file_name"),
                "file_size": h.get("file_size", 0),
                "data_type": h.get("data_type"),
            })

        return {"total": total, "returned": len(files), "files": files}

    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"GDC API error: {e}")


@router.post("/search/all-ids")
async def search_all_ids(req: GDCSearchRequest):
    """Retrieve ALL file IDs matching criteria (paginates automatically)."""
    filters = _build_filters(req)
    all_files: list[dict] = []
    from_idx = 0
    page_size = 500
    total = None

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            while True:
                payload: dict = {
                    "size": page_size,
                    "from": from_idx,
                    "fields": "file_id,file_name,file_size",
                }
                if filters:
                    payload["filters"] = filters

                resp = await client.post(GDC_FILES_URL, json=payload)
                resp.raise_for_status()
                data = resp.json()

                hits = data["data"]["hits"]
                total = data["data"]["pagination"]["total"]
                all_files.extend(hits)
                from_idx += len(hits)

                if from_idx >= total or not hits:
                    break

        return {
            "total": total,
            "file_ids": [f["file_id"] for f in all_files],
            "files": [{"file_id": f["file_id"], "file_name": f.get("file_name"), "file_size": f.get("file_size", 0)} for f in all_files],
        }

    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"GDC API error: {e}")


# ---------------------------------------------------------------------------
# Routes — download
# ---------------------------------------------------------------------------

@router.post("/download")
async def download_from_gdc(req: GDCDownloadRequest):
    """Start a detached background download of GDC files.
    The download runs as a separate process — survives server restarts."""
    if not req.file_ids:
        raise HTTPException(status_code=400, detail="file_ids list is required")

    out_dir = Path(req.output_dir).expanduser() if req.output_dir else DOWNLOADS_DIR / (req.project or "gdc")
    job_id = str(uuid.uuid4())[:8]

    # Write initial job state to disk
    _update_job(
        job_id,
        status="running",
        total=len(req.file_ids),
        done=0,
        bytes_done=0,
        failed=[],
        errors={},
        output_dir=str(out_dir),
        started_at=datetime.now(timezone.utc).isoformat(),
        message=f"Starting download of {len(req.file_ids)} files…",
    )

    # Resolve GDC token: request > config file
    token = req.gdc_token
    if not token:
        cfg_path = PATHCLAW_DATA_DIR / "config.json"
        if cfg_path.exists():
            try:
                token = json.loads(cfg_path.read_text()).get("gdc_token", "")
            except Exception:
                pass

    # Spawn detached subprocess (survives server restart)
    _launch_download_subprocess(job_id, req.file_ids, out_dir, token, req.max_concurrent)

    return {
        "job_id": job_id,
        "status": "running",
        "total_files": len(req.file_ids),
        "output_dir": str(out_dir),
        "message": f"Download started. Poll /api/gdc/jobs/{job_id} for progress.",
    }


@router.get("/jobs/{job_id}")
async def get_download_job(job_id: str):
    """Get status of a GDC download job."""
    p = _job_path(job_id)
    if not p.exists():
        # Also check without dl_ prefix
        p2 = JOBS_DIR / f"{job_id}.json"
        if p2.exists():
            p = p2
        else:
            return {"job_id": job_id, "status": "unknown", "progress": 0}
    data = json.loads(p.read_text())
    done = data.get("done", 0)
    total = max(data.get("total", 1), 1)
    data["progress"] = done / total
    return data


@router.get("/jobs")
async def list_download_jobs():
    """List all GDC download jobs."""
    jobs = []
    for p in sorted(JOBS_DIR.glob("dl_*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            d = json.loads(p.read_text())
            jobs.append(d)
        except Exception:
            pass
    return {"jobs": jobs}


@router.get("/cohorts")
async def list_cohorts():
    """Scan ~/.pathclaw/downloads/ and return per-cohort file counts for the UI."""
    cohorts = []
    if not DOWNLOADS_DIR.exists():
        return {"cohorts": []}
    for cohort_dir in sorted(DOWNLOADS_DIR.iterdir()):
        if not cohort_dir.is_dir():
            continue
        entry = {"name": cohort_dir.name, "path": str(cohort_dir), "subdirs": []}
        total_size = 0
        for sub in ("slides", "maf", "clinical", "expression", "cnv"):
            sub_path = cohort_dir / sub
            if not sub_path.exists():
                continue
            files = [f for f in sub_path.rglob("*") if f.is_file()]
            if not files:
                continue
            size = sum(f.stat().st_size for f in files)
            total_size += size
            sample = sorted(f.name for f in files[:5])
            entry["subdirs"].append({
                "name": sub,
                "count": len(files),
                "size_mb": round(size / (1024 * 1024), 1),
                "sample": sample,
            })
        labels = [f.name for f in cohort_dir.glob("*_labels*.csv")]
        if labels:
            entry["labels"] = labels
        entry["total_size_mb"] = round(total_size / (1024 * 1024), 1)
        if entry["subdirs"] or labels:
            cohorts.append(entry)
    return {"cohorts": cohorts}
