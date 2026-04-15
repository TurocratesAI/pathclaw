"""One-button uploads — classify each file by extension and route to the right place.

Routing (per session, isolated PhD-student workspaces):

    .svs .ndpi .tif .tiff .mrxs .vsi .scn .bif .qptiff   → uploads/slides/
    .pdf                                                 → uploads/papers/
    .csv .tsv                                            → uploads/tables/
    .maf .vcf .vcf.gz                                    → uploads/genomics/
    .json .yaml .yml .toml                               → workspace/user_code/
    .py .ipynb .md .txt .rst                             → workspace/user_code/
    .png .jpg .jpeg .gif .bmp .webp                      → workspace/user_code/
    anything else                                        → uploads/misc/

All paths land under ~/.pathclaw/sessions/<sid>/ when session_id is supplied,
so session 1 cannot see session 2's files (the "parallel PhD students" model).
Empty session_id falls back to the legacy ~/.pathclaw/uploads/.
"""
from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from pathclaw.api.routes.workspace_fs import session_workspace_root


PATHCLAW_DATA_DIR = Path(os.environ.get("PATHCLAW_DATA_DIR", "~/.pathclaw")).expanduser()
_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")

# 4 GB hard ceiling per file (whole-slide images are big — 1–3 GB common).
MAX_FILE_BYTES = 4 * 1024 * 1024 * 1024

SLIDE_EXTS = {".svs", ".ndpi", ".tif", ".tiff", ".mrxs", ".vsi", ".scn", ".bif", ".qptiff"}
PDF_EXTS = {".pdf"}
TABLE_EXTS = {".csv", ".tsv"}
GENOMIC_EXTS = {".maf", ".vcf", ".gz", ".bed", ".gff", ".gtf"}
CODE_EXTS = {".py", ".ipynb", ".md", ".txt", ".rst", ".json", ".yaml", ".yml", ".toml"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}


router = APIRouter()


def _safe_filename(name: str) -> str:
    base = os.path.basename(name or "upload")
    base = re.sub(r"[^A-Za-z0-9._\-]+", "_", base).strip("._-") or "upload"
    return base[:200]


def _session_uploads_root(session_id: str | None) -> Path:
    if session_id:
        if not _SESSION_ID_RE.fullmatch(session_id):
            raise HTTPException(400, detail=f"Invalid session_id: {session_id!r}")
        return PATHCLAW_DATA_DIR / "sessions" / session_id / "uploads"
    return PATHCLAW_DATA_DIR / "uploads"


def _classify(filename: str) -> tuple[str, str]:
    """Return (category, subdir). Special-cases vcf.gz which has a compound suffix."""
    lower = filename.lower()
    if lower.endswith(".vcf.gz") or lower.endswith(".maf.gz"):
        return "genomic", "genomics"
    suffix = Path(lower).suffix
    if suffix in SLIDE_EXTS:
        return "slide", "slides"
    if suffix in PDF_EXTS:
        return "pdf", "papers"
    if suffix in TABLE_EXTS:
        return "table", "tables"
    if suffix in GENOMIC_EXTS:
        return "genomic", "genomics"
    if suffix in CODE_EXTS:
        return "code", "user_code"  # routed into workspace, not uploads
    if suffix in IMAGE_EXTS:
        return "image", "user_code"
    return "misc", "misc"


@router.post("")
async def upload_files(
    session_id: str = Form(""),
    files: list[UploadFile] = File(...),
):
    """Accept N files, route each by extension, return a per-file report.

    The frontend's "Upload" button drops files here without asking the user
    where they go — the classifier does the routing.
    """
    if not files:
        raise HTTPException(400, detail="No files provided.")

    uploads_root = _session_uploads_root(session_id)
    workspace_root = session_workspace_root(session_id)

    results = []
    for f in files:
        name = _safe_filename(f.filename or "upload")
        category, subdir = _classify(name)

        # Workspace-bound (code/images) goes under the editor tree, not uploads/
        if category in {"code", "image"}:
            dest_dir = workspace_root / subdir
        else:
            dest_dir = uploads_root / subdir
        dest_dir.mkdir(parents=True, exist_ok=True)

        dest = dest_dir / name
        # Don't silently clobber — append _N if the name is taken
        if dest.exists():
            stem, ext = dest.stem, dest.suffix
            n = 1
            while (dest_dir / f"{stem}_{n}{ext}").exists():
                n += 1
            dest = dest_dir / f"{stem}_{n}{ext}"

        # Stream-write to disk (avoid loading 3 GB slides into RAM)
        size = 0
        with dest.open("wb") as out:
            while True:
                chunk = await f.read(8 * 1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_FILE_BYTES:
                    out.close()
                    dest.unlink(missing_ok=True)
                    raise HTTPException(413, detail=f"{name}: exceeds 4 GB limit.")
                out.write(chunk)

        rel = str(dest.relative_to(PATHCLAW_DATA_DIR))
        results.append({
            "filename": name,
            "category": category,
            "stored_at": rel,
            "size_bytes": size,
        })

    # Hint: if any slide files landed, the user typically wants a dataset.
    slide_count = sum(1 for r in results if r["category"] == "slide")
    pdf_count = sum(1 for r in results if r["category"] == "pdf")
    next_steps = []
    if slide_count:
        next_steps.append(f"{slide_count} slide(s) uploaded — register as a dataset to preprocess + extract features.")
    if pdf_count:
        next_steps.append(f"{pdf_count} PDF(s) uploaded — attach a folder to this chat session to read them with the agent.")

    return {
        "status": "ok",
        "session_id": session_id or "",
        "uploaded": results,
        "summary": {
            "total": len(results),
            "by_category": {
                cat: sum(1 for r in results if r["category"] == cat)
                for cat in {r["category"] for r in results}
            },
        },
        "next_steps": next_steps,
    }
