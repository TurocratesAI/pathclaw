"""Referential validation for agent tool inputs.

Schema validation (handled by the LLM-provider's JSON-schema enforcement) catches
shape errors. This module catches *referential* errors — IDs that look right but
don't point to anything real. Without this layer, smaller LLMs invent job_ids,
dataset_ids, etc. and the tool either silently fails or burns a turn polling
nothing.

Pattern follows the OpenAI Agents SDK guardrail shape: each resolver either
returns the resolved entity (or just returns silently if it exists) or raises
ToolInputError with a model-readable message that the dispatcher feeds back
into the conversation as the tool result. The model sees the error on the next
round and self-corrects.

Resolvers are intentionally cheap (filesystem stat / single-file read). They
must not block the event loop; if you add anything that would, run it in a
threadpool.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

PATHCLAW_DATA_DIR = Path(os.environ.get("PATHCLAW_DATA_DIR", "~/.pathclaw")).expanduser()
DATASETS_DIR = PATHCLAW_DATA_DIR / "datasets"
EXPERIMENTS_DIR = PATHCLAW_DATA_DIR / "experiments"
JOBS_DIR = PATHCLAW_DATA_DIR / "jobs"
QUEUE_PATH = PATHCLAW_DATA_DIR / "queue.json"
PLUGINS_USER = PATHCLAW_DATA_DIR / "plugins" / "user_registry.json"
PLUGINS_BUILTIN = Path(__file__).resolve().parent.parent / "plugins" / "registry.json"


class ToolInputError(Exception):
    """Raised by a resolver when a referenced entity does not exist.
    The .message is fed back to the LLM as the tool result so it can self-correct."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


# ---------------------------------------------------------------------------
# Cheap listing helpers (cap output so error messages stay under the model's
# attention budget)
# ---------------------------------------------------------------------------

def _list_dir(p: Path, limit: int = 12) -> list[str]:
    if not p.exists():
        return []
    names = sorted(x.name for x in p.iterdir() if not x.name.startswith("."))
    if len(names) > limit:
        return names[:limit] + [f"... ({len(names) - limit} more)"]
    return names


# ---------------------------------------------------------------------------
# Resolvers — each raises ToolInputError on miss with a self-correcting hint.
# Naming is uniform so the table-driven dispatch in chat.py can map them.
# ---------------------------------------------------------------------------

def resolve_dataset_id(did: str) -> str:
    """Verify a dataset is registered. Returns the canonical id."""
    if not did or did in ("?", "unknown", "null", "None", ""):
        raise ToolInputError(
            "ERROR: dataset_id is empty or a placeholder. Call list_datasets to see "
            "what's registered, or register_dataset first to create one."
        )
    if not (DATASETS_DIR / did / "meta.json").exists():
        available = _list_dir(DATASETS_DIR)
        raise ToolInputError(
            f"ERROR: dataset_id '{did}' does not exist. "
            f"Available datasets: {available or '[]'}. "
            f"Call list_datasets to confirm, or register_dataset to create a new one. "
            f"Do NOT invent dataset ids."
        )
    return did


def resolve_experiment_id(eid: str) -> str:
    """Verify an experiment / training run exists."""
    if not eid or eid in ("?", "unknown", "null", "None", ""):
        raise ToolInputError(
            "ERROR: experiment_id (training run id) is empty or placeholder. "
            "Call list_artifacts to see existing runs, or start_training first."
        )
    if not (EXPERIMENTS_DIR / eid).exists():
        available = _list_dir(EXPERIMENTS_DIR)
        raise ToolInputError(
            f"ERROR: experiment_id '{eid}' does not exist. "
            f"Recent runs: {available or '[]'}. "
            f"Call list_artifacts to confirm, or start_training to create a run. "
            f"Do NOT invent experiment ids."
        )
    return eid


def resolve_job_id(jid: str, job_type: str = "training") -> str:
    """Verify a background job exists. Job persistence varies by type:
    - training/eval/lora: ~/.pathclaw/experiments/<jid>/
    - features/preprocess: ~/.pathclaw/jobs/<jid>.json
    - gdc: ~/.pathclaw/jobs/dl_<suffix>.json (the jid is dl_<suffix>)
    Queue entries (q-XXXX) live in queue.json."""
    if not jid or jid in ("?", "unknown", "null", "None", ""):
        raise ToolInputError(
            f"ERROR: job_id is empty or placeholder. Call the corresponding start_* "
            f"tool first ({_start_tool_for(job_type)}) and use the job_id it returns. "
            f"Do NOT invent job_ids."
        )

    # Quick paths
    if (EXPERIMENTS_DIR / jid).exists():
        return jid
    if (JOBS_DIR / f"{jid}.json").exists():
        return jid
    # Queue task ids
    if jid.startswith("q-") and QUEUE_PATH.exists():
        try:
            tasks = json.loads(QUEUE_PATH.read_text())
            if any(t.get("task_id") == jid for t in tasks):
                return jid
        except Exception:
            pass

    # Build hint: list the most recent jobs of the requested type
    hint = _recent_jobs_hint(job_type)
    raise ToolInputError(
        f"ERROR: {job_type} job '{jid}' does not exist. "
        f"{hint} "
        f"Call list_queue or get_job_status to find real job ids, or call "
        f"{_start_tool_for(job_type)} to create one. Do NOT invent job_ids."
    )


def resolve_plugin_id(pid: str) -> str:
    """Verify a plugin is registered (built-in or user)."""
    if not pid:
        raise ToolInputError(
            "ERROR: plugin_id is empty. Call list_plugins to see what's registered."
        )
    known = _all_plugin_ids()
    if pid not in known:
        raise ToolInputError(
            f"ERROR: plugin_id '{pid}' is not registered. "
            f"Available plugins: {sorted(known) or '[]'}. "
            f"Call list_plugins to confirm, or register_plugin to add a new one."
        )
    return pid


def resolve_session_path(rel: str, session_id: str, kind: str = "workspace") -> Path:
    """Resolve a workspace/uploads path under a session, blocking traversal."""
    if not session_id:
        raise ToolInputError("ERROR: session context unavailable.")
    base = PATHCLAW_DATA_DIR / "sessions" / session_id / kind
    base.mkdir(parents=True, exist_ok=True)
    p = (base / (rel or "")).resolve()
    if not str(p).startswith(str(base.resolve())):
        raise ToolInputError(
            f"ERROR: path '{rel}' resolves outside the session {kind} root. "
            f"Use a relative path under {kind}/."
        )
    return p


def resolve_slide_stem(dataset_id: str, slide_stem: str) -> str:
    """Verify a slide exists in the dataset (by stem, no extension)."""
    resolve_dataset_id(dataset_id)  # raises if dataset itself is bad
    if not slide_stem:
        raise ToolInputError(
            f"ERROR: slide_stem is empty. Call list_dataset_slides('{dataset_id}') "
            f"to see available slides."
        )
    meta_p = DATASETS_DIR / dataset_id / "meta.json"
    try:
        meta = json.loads(meta_p.read_text())
    except Exception:
        return slide_stem  # can't validate without metadata; defer to the tool
    slides = meta.get("slides") or []
    stems = {Path(s.get("filename", "")).stem for s in slides if isinstance(s, dict)}
    if stems and slide_stem not in stems:
        sample = sorted(stems)[:5]
        raise ToolInputError(
            f"ERROR: slide '{slide_stem}' not in dataset '{dataset_id}'. "
            f"Sample slides: {sample}. Call list_dataset_slides for the full list."
        )
    return slide_stem


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _start_tool_for(job_type: str) -> str:
    return {
        "training": "start_training",
        "features": "start_feature_extraction",
        "preprocess": "start_preprocessing",
        "eval": "start_evaluation",
        "lora": "start_lora_finetuning",
        "gdc": "download_gdc",
    }.get(job_type, "the appropriate start_* tool")


def _recent_jobs_hint(job_type: str, limit: int = 5) -> str:
    """Return a short string of the N most recent jobs of the given type."""
    candidates: list[tuple[float, str]] = []

    if job_type in ("training", "eval", "lora") and EXPERIMENTS_DIR.exists():
        for d in EXPERIMENTS_DIR.iterdir():
            if d.is_dir():
                try:
                    candidates.append((d.stat().st_mtime, d.name))
                except OSError:
                    pass

    if JOBS_DIR.exists():
        prefix = {"features": "feat-", "preprocess": "pre-", "gdc": "dl_"}.get(job_type)
        for f in JOBS_DIR.iterdir():
            if f.suffix == ".json" and (prefix is None or f.stem.startswith(prefix)):
                try:
                    candidates.append((f.stat().st_mtime, f.stem))
                except OSError:
                    pass

    if not candidates:
        return f"No recent {job_type} jobs found."
    recent = sorted(candidates, reverse=True)[:limit]
    return f"Recent {job_type} jobs: {[name for _, name in recent]}."


def _all_plugin_ids() -> set[str]:
    ids: set[str] = set()
    for p in (PLUGINS_BUILTIN, PLUGINS_USER):
        if p.exists():
            try:
                data = json.loads(p.read_text())
                for plugin in data.get("plugins", []):
                    if pid := plugin.get("id"):
                        ids.add(pid)
            except Exception:
                pass
    return ids


# ---------------------------------------------------------------------------
# Per-tool validation table
# Maps tool name → list of (arg_name, resolver_callable, optional kwargs).
# Resolvers receive arg_value as positional; kwargs come from the tool_args
# dict at runtime via `aux_keys` (e.g. job_type for resolve_job_id).
# ---------------------------------------------------------------------------

# Each entry: (arg_name, resolver, aux_keys)
#   aux_keys: list of additional arg_names to pull from tool_args and pass as
#             positional args after the primary value.
TOOL_VALIDATORS: dict[str, list[tuple[str, callable, list[str]]]] = {
    # Datasets
    "list_dataset_slides": [("dataset_id", resolve_dataset_id, [])],
    "get_dataset_profile": [("dataset_id", resolve_dataset_id, [])],
    "start_preprocessing": [("dataset_id", resolve_dataset_id, [])],
    "start_feature_extraction": [("dataset_id", resolve_dataset_id, [])],
    "start_training": [("dataset_id", resolve_dataset_id, [])],
    "start_evaluation": [("dataset_id", resolve_dataset_id, [])],
    "start_lora_finetuning": [("dataset_id", resolve_dataset_id, [])],

    # Experiments
    "get_eval_metrics": [("experiment_id", resolve_experiment_id, [])],
    "get_eval_plots": [("experiment_id", resolve_experiment_id, [])],
    "get_training_logs": [("experiment_id", resolve_experiment_id, [])],
    "list_artifacts": [],  # listing is fine without an id
    "compare_experiments": [
        ("experiment_a", resolve_experiment_id, []),
        ("experiment_b", resolve_experiment_id, []),
    ],
    "generate_heatmap": [
        ("experiment_id", resolve_experiment_id, []),
    ],

    # Jobs
    "wait_for_job": [("job_id", resolve_job_id, ["job_type"])],
    "get_job_status": [("job_id", resolve_job_id, [])],
    "gdc_job_status": [("job_id", lambda j: resolve_job_id(j, "gdc"), [])],

    # Plugins
    "update_plugin_config": [("plugin_id", resolve_plugin_id, [])],
    "smoke_test_plugin": [("plugin_id", resolve_plugin_id, [])],
}


def validate_tool_args(name: str, args: dict) -> str | None:
    """Run the registered validators for `name`. Returns an error message
    string on the first failure (so the caller can short-circuit and feed it
    back to the model), or None if everything checks out (or the tool has no
    validators registered)."""
    rules = TOOL_VALIDATORS.get(name)
    if not rules:
        return None
    for primary_key, resolver, aux_keys in rules:
        primary_val = args.get(primary_key, "")
        try:
            extras = [args.get(k, "") for k in aux_keys]
            resolver(primary_val, *extras)
        except ToolInputError as e:
            return e.message
        except TypeError:
            # Resolver signature mismatch — surface as a generic input error so
            # the caller still short-circuits instead of crashing.
            return f"ERROR: invalid input for {name}: {primary_key}={primary_val!r}"
    return None
