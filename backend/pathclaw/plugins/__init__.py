"""Plugin registry — optional modules the user/agent can toggle on trainings.

A plugin declares a build callable that returns an nn.Module of one of several
"kinds". The trainer looks up plugins in the registry by id and applies them at
the matching hook point (e.g. `patch_embed` plugins replace the input Linear in
MIL models).

Registry layout:
    registry.json — built-in plugins bundled with PathClaw
    ~/.pathclaw/plugins/user_registry.json — user/agent-registered plugins

User entries override built-ins with the same id.
"""
from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable

_BUILTIN_REG = Path(__file__).parent / "registry.json"
_PATHCLAW_DATA_DIR = Path(os.environ.get("PATHCLAW_DATA_DIR", "~/.pathclaw")).expanduser()
_USER_REG = _PATHCLAW_DATA_DIR / "plugins" / "user_registry.json"

# Make the workspace dir importable so user-authored plugins/methods under
# ~/.pathclaw/workspace/plugins/foo.py are reachable as `workspace.plugins.foo`.
_WORKSPACE_PARENT = str(_PATHCLAW_DATA_DIR.resolve())
if _WORKSPACE_PARENT not in sys.path:
    sys.path.insert(0, _WORKSPACE_PARENT)

VALID_KINDS = {"patch_embed", "mil", "loss", "augment", "method"}


def _read_json(p: Path) -> dict:
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def load_registry() -> dict[str, dict]:
    """Merged registry (built-in ← user overrides)."""
    reg = dict(_read_json(_BUILTIN_REG))
    for k, v in _read_json(_USER_REG).items():
        reg[k] = v
    return reg


def save_user_entry(plugin_id: str, manifest: dict) -> None:
    """Persist a user/agent-authored plugin manifest."""
    if manifest.get("kind") not in VALID_KINDS:
        raise ValueError(f"Invalid plugin kind: {manifest.get('kind')!r}")
    _USER_REG.parent.mkdir(parents=True, exist_ok=True)
    data = _read_json(_USER_REG)
    data[plugin_id] = manifest
    _USER_REG.write_text(json.dumps(data, indent=2))


def update_default_config(plugin_id: str, new_default_config: dict) -> dict:
    """Persist a new `default_config` for `plugin_id`.

    Built-in plugins are not edited in place: we write a user-registry entry
    that shadows the built-in (the merged loader picks user > built-in). This
    lets gemma/the user retune cellpose, mammoth, etc. without touching the
    bundled JSON.
    """
    reg = dict(_read_json(_BUILTIN_REG))
    user_data = _read_json(_USER_REG)
    reg.update(user_data)
    if plugin_id not in reg:
        raise KeyError(f"Plugin {plugin_id!r} not in registry.")
    base = dict(reg[plugin_id])
    base["default_config"] = dict(new_default_config)
    _USER_REG.parent.mkdir(parents=True, exist_ok=True)
    user_data[plugin_id] = base
    _USER_REG.write_text(json.dumps(user_data, indent=2))
    return base


def delete_user_entry(plugin_id: str) -> bool:
    data = _read_json(_USER_REG)
    if plugin_id not in data:
        return False
    del data[plugin_id]
    _USER_REG.write_text(json.dumps(data, indent=2))
    return True


def resolve_builder(import_path: str) -> Callable[..., Any]:
    """Resolve 'module.path:attr' to a callable."""
    if ":" not in import_path:
        raise ValueError(f"import_path must be 'module:attr', got {import_path!r}")
    mod_path, attr = import_path.split(":", 1)
    mod = importlib.import_module(mod_path)
    return getattr(mod, attr)


def is_installed(manifest: dict) -> bool:
    """Can we import the plugin's build callable right now?"""
    ip = manifest.get("import_path", "")
    if not ip or ":" not in ip:
        return False
    mod_path = ip.split(":", 1)[0]
    try:
        importlib.import_module(mod_path)
        return True
    except Exception:
        return False


def build_patch_embed(
    in_dim: int,
    embed_dim: int,
    plugins: list[dict] | None,
    legacy_mammoth: dict | None = None,
):
    """Dispatch patch-embed plugins to a single nn.Module.

    plugins: list of {"id": str, "config": dict} entries from training config.
    legacy_mammoth: old-shape `config.mammoth` dict — auto-translated if `.enabled`.

    Returns the last patch-embed plugin's module, or nn.Linear baseline.
    """
    import torch.nn as nn

    # Legacy shim: old configs have `mammoth: {enabled: true, ...}` at top level.
    effective = list(plugins or [])
    if legacy_mammoth and legacy_mammoth.get("enabled") and not any(
        p.get("id") == "mammoth" for p in effective
    ):
        effective.append({"id": "mammoth", "config": {k: v for k, v in legacy_mammoth.items() if k != "enabled"}})

    reg = load_registry()
    active_patch = None
    for entry in effective:
        pid = entry.get("id")
        manifest = reg.get(pid)
        if not manifest or manifest.get("kind") != "patch_embed":
            continue
        cfg = {**manifest.get("default_config", {}), **entry.get("config", {})}
        builder = resolve_builder(manifest["import_path"])
        active_patch = builder(in_dim, embed_dim, cfg)

    if active_patch is not None:
        return active_patch
    return nn.Linear(in_dim, embed_dim)
