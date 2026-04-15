"""Plugin registry HTTP API — list, toggle-install, register, delete.

Plugins are the extension point for custom patch-embed modules, MIL heads,
losses, augmentations, and paper-implemented methods. See
`pathclaw.plugins.__init__` for the registry contract.
"""
from __future__ import annotations

import subprocess
import sys
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from pathclaw import plugins as plugin_registry

router = APIRouter()


@router.get("")
def list_plugins() -> dict[str, Any]:
    """All plugins from the merged registry, with installation status."""
    reg = plugin_registry.load_registry()
    out = []
    for pid, manifest in reg.items():
        out.append({
            "id": pid,
            "name": manifest.get("name", pid),
            "kind": manifest.get("kind", "unknown"),
            "description": manifest.get("description", ""),
            "applies_to": manifest.get("applies_to", []),
            "source": manifest.get("source", ""),
            "default_config": manifest.get("default_config", {}),
            "enabled_default": bool(manifest.get("enabled_default", False)),
            "builtin": bool(manifest.get("builtin", False)),
            "installed": plugin_registry.is_installed(manifest),
        })
    out.sort(key=lambda p: (not p["builtin"], p["id"]))
    return {"plugins": out}


class RegisterManifest(BaseModel):
    id: str
    name: str
    kind: str  # patch_embed | mil | loss | augment | method
    import_path: str  # module.path:attr
    description: str = ""
    applies_to: list[str] = []
    source: str = ""
    default_config: dict = {}


@router.post("/register")
def register_plugin(body: RegisterManifest) -> dict[str, Any]:
    if body.kind not in plugin_registry.VALID_KINDS:
        raise HTTPException(400, detail=f"Invalid kind {body.kind!r}. Valid: {sorted(plugin_registry.VALID_KINDS)}")
    if ":" not in body.import_path:
        raise HTTPException(400, detail="import_path must be of the form 'module.path:attr'.")
    manifest = {
        "name": body.name,
        "kind": body.kind,
        "import_path": body.import_path,
        "description": body.description,
        "applies_to": body.applies_to,
        "source": body.source,
        "default_config": body.default_config,
        "enabled_default": False,
        "builtin": False,
    }
    try:
        plugin_registry.save_user_entry(body.id, manifest)
    except ValueError as e:
        raise HTTPException(400, detail=str(e))
    return {"status": "ok", "id": body.id, "installed": plugin_registry.is_installed(manifest)}


class UpdateConfigBody(BaseModel):
    default_config: dict


@router.put("/{plugin_id}/config")
def update_plugin_config(plugin_id: str, body: UpdateConfigBody) -> dict[str, Any]:
    """Update a plugin's default_config — gemma & the UI both call this.

    Built-ins are shadowed via the user registry, so updates are reversible
    by deleting the user entry.
    """
    try:
        manifest = plugin_registry.update_default_config(plugin_id, body.default_config)
    except KeyError as e:
        raise HTTPException(404, detail=str(e))
    except ValueError as e:
        raise HTTPException(400, detail=str(e))
    return {"status": "ok", "id": plugin_id, "default_config": manifest.get("default_config", {})}


class DeleteBody(BaseModel):
    id: str


@router.delete("/{plugin_id}")
def delete_plugin(plugin_id: str) -> dict[str, Any]:
    if not plugin_registry.delete_user_entry(plugin_id):
        raise HTTPException(404, detail=f"Plugin {plugin_id!r} is not user-registered (built-ins cannot be deleted).")
    return {"status": "ok", "id": plugin_id}


class InstallBody(BaseModel):
    id: str


@router.post("/install")
def install_plugin(body: InstallBody) -> dict[str, Any]:
    """pip-install the source declared by a plugin manifest.

    Refuses anything other than `pip:<package>` sources for safety.
    """
    reg = plugin_registry.load_registry()
    manifest = reg.get(body.id)
    if not manifest:
        raise HTTPException(404, detail=f"Plugin {body.id!r} not in registry.")
    source = manifest.get("source", "")
    if not source.startswith("pip:"):
        raise HTTPException(400, detail=f"Plugin source {source!r} is not pip-installable. Install manually.")
    pkg = source.split(":", 1)[1].strip()
    if not pkg:
        raise HTTPException(400, detail="Empty pip source.")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", pkg],
        capture_output=True, text=True, timeout=300,
    )
    if result.returncode != 0:
        raise HTTPException(400, detail=f"pip install failed: {result.stderr.strip()[-800:]}")
    return {
        "status": "ok",
        "id": body.id,
        "installed": plugin_registry.is_installed(manifest),
        "stdout_tail": result.stdout[-400:],
    }
