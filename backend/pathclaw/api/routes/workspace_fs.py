"""Workspace filesystem API — user code, cloned repos, plugin drafts.

Sessions are isolated like parallel PhD students:
    ~/.pathclaw/workspace/                 — shared/legacy fallback (no session)
    ~/.pathclaw/sessions/<sid>/workspace/  — per-session workspace
        user_code/   — user-authored scripts (editable via Monaco)
        repos/       — cloned git repos
        plugins/     — registered plugin source (see pathclaw.plugins)
        methods/     — agent-implemented paper methodologies (drafts)

All paths are validated against the session's workspace root to reject
traversal attempts.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel


PATHCLAW_DATA_DIR = Path(os.environ.get("PATHCLAW_DATA_DIR", "~/.pathclaw")).expanduser()
WORKSPACE_ROOT = PATHCLAW_DATA_DIR / "workspace"  # legacy/shared root, kept for back-compat
_SUBDIRS = ("user_code", "repos", "plugins", "methods")
_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")


def session_workspace_root(session_id: str | None) -> Path:
    """Resolve the workspace root for a session.

    Empty/None session_id falls back to the legacy shared root so older
    clients and the agent's pre-session bootstrap keep working. Anything
    else MUST match `_SESSION_ID_RE` to prevent traversal via the session_id.
    """
    if not session_id:
        return WORKSPACE_ROOT
    if not _SESSION_ID_RE.fullmatch(session_id):
        raise HTTPException(status_code=400, detail=f"Invalid session_id: {session_id!r}")
    return PATHCLAW_DATA_DIR / "sessions" / session_id / "workspace"

# Cap to keep editor responsive and avoid accidentally loading training artifacts.
MAX_TEXT_BYTES = 2 * 1024 * 1024  # 2 MB
_BINARY_HINT_EXTS = {
    ".pt", ".pth", ".bin", ".safetensors", ".npz", ".npy", ".zip", ".tar", ".gz",
    ".png", ".jpg", ".jpeg", ".gif", ".pdf", ".svs", ".tif", ".tiff", ".ndpi", ".mrxs",
}


def _ensure_workspace(session_id: str | None = None):
    root = session_workspace_root(session_id)
    for sub in _SUBDIRS:
        (root / sub).mkdir(parents=True, exist_ok=True)


def safe_workspace_path(rel: str, session_id: str | None = None) -> Path:
    """Resolve a relative path inside the session's workspace root.

    Callable from chat.py's workspace tools — shared path-whitelist logic.
    A leading ``workspace/`` (or ``./workspace/``) is stripped: tool outputs
    render paths as ``workspace/<rel>``, and models often echo that prefix
    back on the next call. Accept both forms.
    """
    rel = (rel or "").strip()
    if rel.startswith("./"):
        rel = rel[2:]
    if rel.startswith("workspace/"):
        rel = rel[len("workspace/"):]
    if not rel or rel.startswith("/") or ".." in Path(rel).parts:
        raise HTTPException(status_code=400, detail=f"Invalid workspace path: {rel!r}")
    _ensure_workspace(session_id)
    root = session_workspace_root(session_id).resolve()
    p = (root / rel).resolve()
    if root != p and root not in p.parents:
        raise HTTPException(status_code=400, detail="Path escapes workspace root.")
    return p


def _is_probably_text(p: Path) -> bool:
    if p.suffix.lower() in _BINARY_HINT_EXTS:
        return False
    try:
        sample = p.read_bytes()[:4096]
    except Exception:
        return False
    if b"\x00" in sample:
        return False
    return True


router = APIRouter()


# ---------------------------------------------------------------------------
# Read-side: tree + file contents
# ---------------------------------------------------------------------------

@router.get("/tree")
def tree(session_id: str = ""):
    """Recursive listing of the workspace root with sizes."""
    _ensure_workspace(session_id)
    root = session_workspace_root(session_id).resolve()

    def walk(dir_: Path) -> dict:
        try:
            entries = sorted(dir_.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        except OSError:
            entries = []
        children = []
        for p in entries:
            # Skip .git internals and caches to keep the tree responsive
            if p.name in {".git", "__pycache__", ".venv", ".pytest_cache", "node_modules"}:
                continue
            if p.is_dir():
                children.append({
                    "type": "dir",
                    "name": p.name,
                    "path": str(p.relative_to(root)),
                    "children": walk(p)["children"],
                })
            else:
                try:
                    size = p.stat().st_size
                except OSError:
                    size = 0
                children.append({
                    "type": "file",
                    "name": p.name,
                    "path": str(p.relative_to(root)),
                    "size": size,
                })
        return {"children": children}

    return {"root": str(root), "children": walk(root)["children"]}


_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}


@router.get("/raw")
def read_raw(path: str, session_id: str = ""):
    """Serve a workspace file as a raw binary response — used for image previews.

    Unlike /file this doesn't reject binary content or wrap it in JSON. Only
    files under 10 MB are served to keep the UI responsive.
    """
    from fastapi.responses import FileResponse
    p = safe_workspace_path(path, session_id)
    if not p.exists() or not p.is_file():
        raise HTTPException(status_code=404, detail=f"File not found: {path}")
    size = p.stat().st_size
    if size > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File too large ({size} bytes > 10 MB).")
    ext = p.suffix.lower()
    media = {
        ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".gif": "image/gif", ".bmp": "image/bmp", ".webp": "image/webp",
    }.get(ext, "application/octet-stream")
    return FileResponse(p, media_type=media, headers={"Cache-Control": "no-store"})


@router.get("/file")
def read_file(path: str, session_id: str = ""):
    """Return text contents of a workspace file. Rejects >2 MB and binary files."""
    p = safe_workspace_path(path, session_id)
    if not p.exists() or not p.is_file():
        raise HTTPException(status_code=404, detail=f"File not found: {path}")
    size = p.stat().st_size
    if size > MAX_TEXT_BYTES:
        raise HTTPException(status_code=413, detail=f"File too large ({size} bytes > {MAX_TEXT_BYTES}).")
    if not _is_probably_text(p):
        raise HTTPException(status_code=415, detail="Refusing to load binary file in the editor.")
    try:
        content = p.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not decode file: {e}")
    return {"path": str(p.relative_to(session_workspace_root(session_id).resolve())), "size": size, "content": content}


# ---------------------------------------------------------------------------
# Write-side
# ---------------------------------------------------------------------------

class WriteFile(BaseModel):
    path: str
    content: str
    session_id: str = ""


@router.put("/file")
def write_file(body: WriteFile):
    p = safe_workspace_path(body.path, body.session_id)
    if len(body.content.encode("utf-8")) > MAX_TEXT_BYTES:
        raise HTTPException(status_code=413, detail="Content exceeds 2 MB text cap.")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body.content, encoding="utf-8")
    return {
        "status": "ok",
        "path": str(p.relative_to(session_workspace_root(body.session_id).resolve())),
        "size": p.stat().st_size,
    }


class MkdirBody(BaseModel):
    path: str
    session_id: str = ""


@router.post("/mkdir")
def mkdir(body: MkdirBody):
    p = safe_workspace_path(body.path, body.session_id)
    p.mkdir(parents=True, exist_ok=True)
    return {"status": "ok", "path": str(p.relative_to(session_workspace_root(body.session_id).resolve()))}


@router.delete("/file")
def delete_file(path: str, session_id: str = ""):
    p = safe_workspace_path(path, session_id)
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"Not found: {path}")
    if p.is_dir():
        # Reject recursive dir deletes through this endpoint — too dangerous to
        # expose to the agent without confirmation UI.
        raise HTTPException(status_code=400, detail="Refusing to delete directory. Delete files individually.")
    p.unlink()
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Git clone
# ---------------------------------------------------------------------------

_CLONE_HOST_RE = re.compile(r"^https?://(github\.com|gitlab\.com|huggingface\.co|codeberg\.org|bitbucket\.org)/")


class CloneBody(BaseModel):
    url: str
    name: str | None = None
    session_id: str = ""


@router.post("/clone")
def clone_repo(body: CloneBody):
    url = body.url.strip()
    if not _CLONE_HOST_RE.match(url):
        raise HTTPException(
            status_code=400,
            detail="Only https:// URLs from github.com, gitlab.com, huggingface.co, codeberg.org, or bitbucket.org are allowed.",
        )
    name = body.name or re.sub(r"\.git$", "", url.rsplit("/", 1)[-1])
    if not re.fullmatch(r"[A-Za-z0-9._\-]+", name):
        raise HTTPException(status_code=400, detail=f"Invalid repo name: {name!r}")
    _ensure_workspace(body.session_id)
    root = session_workspace_root(body.session_id)
    dest = root / "repos" / name
    if dest.exists():
        raise HTTPException(status_code=409, detail=f"Repo already exists in this session: repos/{name}")
    try:
        result = subprocess.run(
            ["git", "clone", "--depth", "1", url, str(dest)],
            capture_output=True, text=True, timeout=180,
        )
    except subprocess.TimeoutExpired:
        # Clean up partial clone if any
        shutil.rmtree(dest, ignore_errors=True)
        raise HTTPException(status_code=504, detail="git clone timed out after 180s.")
    if result.returncode != 0:
        shutil.rmtree(dest, ignore_errors=True)
        raise HTTPException(status_code=400, detail=f"git clone failed: {result.stderr.strip()[:800]}")
    # Count files for a useful summary
    file_count = sum(1 for p in dest.rglob("*") if p.is_file() and ".git" not in p.parts)
    return {
        "status": "ok",
        "name": name,
        "path": str(dest.relative_to(root.resolve())),
        "files": file_count,
        "stdout": result.stdout[-400:],
    }
