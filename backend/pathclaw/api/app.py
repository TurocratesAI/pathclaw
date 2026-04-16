"""FastAPI application for the PathClaw backend."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from pathclaw.api.routes import datasets, preprocess, training, evaluation, artifacts, gdc, chat, features, config_space, tileserver, folders, telegram, queue as queue_route, workspace_fs, plugins as plugins_route, tasks as tasks_route, ihc as ihc_route

# ---------------------------------------------------------------------------
# App configuration
# ---------------------------------------------------------------------------

PATHCLAW_DATA_DIR = Path(os.environ.get("PATHCLAW_DATA_DIR", "~/.pathclaw")).expanduser()
PATHCLAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

PORT_FILE = PATHCLAW_DATA_DIR / "server.port"

app = FastAPI(
    title="PathClaw Backend",
    description="Computational pathology platform — MIL/MAMMOTH training backend",
    version="0.3.0",
)


@app.on_event("startup")
async def _write_port():
    """Write the server port to a file so chat.py can self-reference correctly."""
    import sys
    # 1. Explicit env var (most reliable — set by start scripts)
    port = os.environ.get("PATHCLAW_PORT") or os.environ.get("PORT")
    # 2. Parse --port N from uvicorn CLI args
    if not port:
        argv = sys.argv
        for i, arg in enumerate(argv):
            if arg in ("--port", "-p") and i + 1 < len(argv):
                port = argv[i + 1]
                break
            if arg.startswith("--port="):
                port = arg.split("=", 1)[1]
                break
    # 3. Safe default
    if not port:
        port = "8101"
    PORT_FILE.write_text(str(port))

# CORS — allow the web UI to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Register route modules
# ---------------------------------------------------------------------------

app.include_router(datasets.router, prefix="/api/datasets", tags=["datasets"])
app.include_router(preprocess.router, prefix="/api/preprocess", tags=["preprocess"])
app.include_router(training.router, prefix="/api/training", tags=["training"])
app.include_router(evaluation.router, prefix="/api/eval", tags=["evaluation"])
app.include_router(artifacts.router, prefix="/api/artifacts", tags=["artifacts"])
app.include_router(gdc.router, prefix="/api/gdc", tags=["gdc"])
app.include_router(chat.router, prefix="/api/chat", tags=["chat"])
app.include_router(features.router, prefix="/api/features", tags=["features"])
app.include_router(config_space.router, prefix="/api/config-space", tags=["config-space"])
app.include_router(tileserver.router, prefix="/api/tiles", tags=["tiles"])
app.include_router(folders.router, prefix="/api/folders", tags=["folders"])
app.include_router(telegram.router, prefix="/api/telegram", tags=["telegram"])
app.include_router(queue_route.router, prefix="/api/queue", tags=["queue"])
app.include_router(workspace_fs.router, prefix="/api/workspace", tags=["workspace"])
app.include_router(plugins_route.router, prefix="/api/plugins", tags=["plugins"])
app.include_router(tasks_route.router, prefix="/api/task-plan", tags=["tasks"])
app.include_router(ihc_route.router, prefix="/api/ihc", tags=["ihc"])
from pathclaw.api.routes import upload as upload_route  # noqa: E402
app.include_router(upload_route.router, prefix="/api/upload", tags=["upload"])


@app.on_event("startup")
async def _start_queue_worker():
    queue_route.start_worker()

# Serve the web UI
STATIC_DIR = Path(__file__).parent.parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/")
    async def serve_ui():
        """Serve the web UI."""
        from fastapi.responses import FileResponse
        return FileResponse(
            STATIC_DIR / "index.html",
            headers={"Cache-Control": "no-store"},
        )


# ---------------------------------------------------------------------------
# Health & status
# ---------------------------------------------------------------------------

@app.get("/api/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "version": "0.3.0"}


@app.get("/api/jobs/all")
async def all_jobs(session_id: str = ""):
    """Aggregate view of all active/recent jobs across types for the UI jobs panel.

    When `session_id` is set, only return jobs owned by that session. Each session
    is treated as an isolated workspace ("parallel PhD students") — sidebar must
    not bleed jobs across sessions.
    """
    import json as _json
    from pathclaw.api.routes import training as _tr, evaluation as _ev, preprocess as _pp

    jobs = []

    # Training jobs (in-memory)
    for jid, j in list(_tr._training_jobs.items()):
        jobs.append({
            "job_id": jid,
            "type": "training",
            "status": j.get("status", "unknown"),
            "progress": j.get("progress", 0),
            "config": j.get("config", {}),
            "metrics": j.get("metrics", {}),
            "session_id": j.get("session_id") or j.get("config", {}).get("session_id", ""),
        })

    # Eval jobs (in-memory)
    for jid, j in list(_ev._eval_jobs.items()):
        jobs.append({
            "job_id": jid,
            "type": "eval",
            "status": j.get("status", "unknown"),
            "progress": j.get("progress", 0),
            "metrics": j.get("metrics", {}),
            "session_id": j.get("session_id", ""),
        })

    # Preprocess jobs (in-memory)
    for jid, j in list(_pp._jobs.items()):
        jobs.append({
            "job_id": jid,
            "type": "preprocess",
            "status": j.get("status", "unknown"),
            "progress": j.get("progress", 0),
            "dataset_id": j.get("dataset_id", ""),
            "session_id": j.get("session_id", ""),
        })

    # Feature jobs (on disk in jobs/feat-*/status.json)
    for p in sorted(PATHCLAW_DATA_DIR.glob("jobs/feat-*/status.json"), key=lambda x: -x.stat().st_mtime)[:10]:
        try:
            d = _json.loads(p.read_text())
            jobs.append({
                "job_id": d.get("job_id", p.parent.name),
                "type": "features",
                "status": d.get("status", "unknown"),
                "progress": d.get("progress", 0),
                "slides_completed": d.get("slides_completed", 0),
                "slides_total": d.get("slides_total", 0),
                "backbone": d.get("backbone", ""),
                "session_id": d.get("session_id", ""),
            })
        except Exception:
            pass

    # GDC download jobs (from disk)
    for p in sorted(PATHCLAW_DATA_DIR.glob("jobs/dl_*.json"), key=lambda x: -x.stat().st_mtime)[:10]:
        try:
            d = _json.loads(p.read_text())
            jobs.append({
                "job_id": d.get("job_id", p.stem),
                "type": "download",
                "status": d.get("status", "unknown"),
                "progress": d.get("done", 0) / max(d.get("total", 1), 1),
                "done": d.get("done", 0),
                "total": d.get("total", 0),
                "bytes_done": d.get("bytes_done", 0),
                "output_dir": d.get("output_dir", ""),
                "message": d.get("message", ""),
                "session_id": d.get("session_id", ""),
            })
        except Exception:
            pass

    # Per-session segregation: a job with no session_id is considered orphan
    # (typically a job started before sessionization) — show only when no
    # session filter is applied.
    if session_id:
        jobs = [j for j in jobs if j.get("session_id") == session_id]

    # Sort: running first, then by job type
    status_order = {"running": 0, "queued": 1, "completed": 2, "failed": 3, "unknown": 4}
    jobs.sort(key=lambda j: (status_order.get(j["status"], 4), j["type"]))

    return {"jobs": jobs}


@app.get("/api/status/ollama")
async def ollama_status():
    """Proxy Ollama status so the UI never needs to know Ollama's address."""
    import httpx
    ollama_base = os.environ.get("OLLAMA_BASE", "http://localhost:11434")
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{ollama_base}/api/tags")
            data = resp.json()
            models = [m["name"] for m in data.get("models", [])]
            return {"online": True, "models": models}
    except Exception:
        return {"online": False, "models": []}


@app.get("/api/status")
async def status():
    """System status — GPU, storage, jobs."""
    import shutil

    gpu_available = False
    gpu_name = None
    gpu_count = 0
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if gpu_available else None
        gpu_count = torch.cuda.device_count() if gpu_available else 0
    except ImportError:
        # torch not installed in this env — fall back to nvidia-smi
        import subprocess
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                timeout=5, stderr=subprocess.DEVNULL,
            ).decode().strip()
            gpus = [l.strip() for l in out.splitlines() if l.strip()]
            if gpus:
                gpu_available = True
                gpu_count = len(gpus)
                gpu_name = gpus[0].split(",")[0].strip()
        except Exception:
            pass

    disk = shutil.disk_usage(PATHCLAW_DATA_DIR)

    return {
        "gpu": {
            "available": gpu_available,
            "name": gpu_name,
            "count": gpu_count,
        },
        "storage": {
            "total_gb": round(disk.total / (1024**3), 1),
            "used_gb": round(disk.used / (1024**3), 1),
            "free_gb": round(disk.free / (1024**3), 1),
        },
        "data_dir": str(PATHCLAW_DATA_DIR),
    }


# ---------------------------------------------------------------------------
# Config (onboarding)
# ---------------------------------------------------------------------------

CONFIG_PATH = PATHCLAW_DATA_DIR / "config.json"

def _load_config() -> dict:
    import json
    if CONFIG_PATH.exists():
        return json.loads(CONFIG_PATH.read_text())
    return {}

def _save_config(config: dict):
    import json
    CONFIG_PATH.write_text(json.dumps(config, indent=2))


@app.get("/api/config")
async def get_config():
    """Get config (redacted secrets)."""
    cfg = _load_config()
    return {
        "huggingface_token_set": bool(cfg.get("huggingface_token")),
        "gdc_token_set": bool(cfg.get("gdc_token")),
        "data_dir": cfg.get("data_dir", str(PATHCLAW_DATA_DIR)),
        "onboarding_complete": cfg.get("onboarding_complete", False),
        "disclaimer_acknowledged": bool(cfg.get("disclaimer_acknowledged", False)),
        "disclaimer_version": cfg.get("disclaimer_version", 0),
        "disclaimer_at": cfg.get("disclaimer_at", ""),
    }


from pydantic import BaseModel as _BaseModel

class ConfigUpdate(_BaseModel):
    huggingface_token: str = ""
    gdc_token: str = ""
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    google_api_key: str = ""
    semantic_scholar_api_key: str = ""
    llm_provider: str = ""   # ollama | anthropic | openai | google
    llm_model: str = ""
    ollama_base: str = ""    # override for remote Ollama server
    openai_base: str = ""    # OpenAI-compatible endpoint (OpenRouter, LM Studio, vLLM, Together, Groq)
    disclaimer_acknowledged: bool | None = None
    disclaimer_version: int | None = None
    disclaimer_at: str = ""

@app.post("/api/config")
async def update_config(update: ConfigUpdate):
    """Update config (set tokens and LLM provider)."""
    cfg = _load_config()
    if update.huggingface_token:
        cfg["huggingface_token"] = update.huggingface_token
        os.environ["HUGGINGFACE_TOKEN"] = update.huggingface_token
    if update.gdc_token:
        cfg["gdc_token"] = update.gdc_token
    if update.anthropic_api_key:
        cfg["anthropic_api_key"] = update.anthropic_api_key
    if update.openai_api_key:
        cfg["openai_api_key"] = update.openai_api_key
    if update.google_api_key:
        cfg["google_api_key"] = update.google_api_key
    if update.semantic_scholar_api_key:
        cfg["semantic_scholar_api_key"] = update.semantic_scholar_api_key
    if update.llm_provider:
        cfg["llm_provider"] = update.llm_provider
    if update.llm_model:
        cfg["llm_model"] = update.llm_model
    if update.ollama_base:
        cfg["ollama_base"] = update.ollama_base
        os.environ["OLLAMA_BASE"] = update.ollama_base
    if update.openai_base:
        cfg["openai_base"] = update.openai_base
    # Disclaimer acknowledgement is independent of credential onboarding
    # (a user can ack the disclaimer without supplying a HF token yet).
    disclaimer_only = update.disclaimer_acknowledged is not None and not any([
        update.huggingface_token, update.gdc_token, update.anthropic_api_key,
        update.openai_api_key, update.google_api_key, update.llm_provider,
        update.llm_model, update.ollama_base, update.openai_base,
    ])
    if update.disclaimer_acknowledged is not None:
        cfg["disclaimer_acknowledged"] = bool(update.disclaimer_acknowledged)
    if update.disclaimer_version is not None:
        cfg["disclaimer_version"] = int(update.disclaimer_version)
    if update.disclaimer_at:
        cfg["disclaimer_at"] = update.disclaimer_at
    if not disclaimer_only:
        cfg["onboarding_complete"] = True
    _save_config(cfg)
    return {"status": "ok", "huggingface_token_set": bool(cfg.get("huggingface_token"))}


@app.get("/api/status/llm")
async def llm_status():
    """Current LLM provider, model, and connectivity."""
    from pathclaw.api import llm_providers
    provider, model = llm_providers.get_active_provider()
    cfg = _load_config()
    keys_set = {
        "anthropic": bool(cfg.get("anthropic_api_key")),
        "openai": bool(cfg.get("openai_api_key")),
        "google": bool(cfg.get("google_api_key")),
    }
    # For Ollama, check connectivity
    online = False
    models: list[str] = []
    if provider == "ollama":
        import httpx as _httpx
        ollama_base = os.environ.get("OLLAMA_BASE", "http://localhost:11434")
        try:
            async with _httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get(f"{ollama_base}/api/tags")
                data = resp.json()
                models = [m["name"] for m in data.get("models", [])]
                online = True
        except Exception:
            pass
    else:
        online = keys_set.get(provider, False)
        models = llm_providers.list_provider_models(provider)
    return {
        "provider": provider,
        "model": model,
        "online": online,
        "models": models,
        "keys_set": keys_set,
    }