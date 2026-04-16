"""Ollama-powered chat endpoint for PathClaw.

Connects the web UI to a local LLM via Ollama with:
- 14 tool definitions covering the full backend API
- Keyword-based SKILL.md loading for per-turn expert context
- SSE streaming endpoint for low-latency token delivery
- Smart conversation trimming to stay within the context window
"""

from __future__ import annotations

import asyncio as _asyncio
import json
import logging
import subprocess
import uuid
from pathlib import Path
from typing import Any, AsyncGenerator, Optional

import httpx
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from pathclaw.api import llm_providers
from pathclaw.api.validators import validate_tool_args

logger = logging.getLogger(__name__)

router = APIRouter()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

import os as _os
import time as _time_mod

OLLAMA_BASE = _os.environ.get("OLLAMA_BASE", "http://localhost:11434")
DEFAULT_MODEL = _os.environ.get("PATHCLAW_MODEL", "qwen3:8b")
NUM_CTX = 16384          # qwen3:8b supports 32K; 16K gives room for system prompt + history
MAX_HISTORY_MSGS = 40    # system + up to 40 conversation messages before trimming

WORKSPACE_DIR = Path(__file__).parent.parent.parent.parent / "workspace"
DATA_DIR = Path(_os.environ.get("PATHCLAW_DATA_DIR", "~/.pathclaw")).expanduser()
CHATS_DIR = DATA_DIR / "chats"
MEMORY_FILE = DATA_DIR / "memory.json"
CHATS_DIR.mkdir(parents=True, exist_ok=True)

_BACKEND_BASE_DEFAULT = _os.environ.get("PATHCLAW_BACKEND_BASE", "")

def _get_backend_base() -> str:
    """Read the server port from the port file (written by app.py startup) so we
    always call our own backend on the correct port, even after server restarts."""
    if _BACKEND_BASE_DEFAULT:
        return _BACKEND_BASE_DEFAULT
    port_file = DATA_DIR / "server.port"
    if port_file.exists():
        try:
            port = port_file.read_text().strip()
            if port.isdigit():
                return f"http://localhost:{port}"
        except Exception:
            pass
    return "http://localhost:8101"

# In-memory conversation store  session_id → messages[]
_conversations: dict[str, list[dict]] = {}


# ---------------------------------------------------------------------------
# GDC zero-result recovery hints
#
# GDC's /files endpoint returns count=0 when filters are *valid JSON* but
# *semantically wrong* — e.g. `experimental_strategy="Diagnostic Slide"` on a
# Masked Somatic Mutation search (that strategy applies to slides, not MAFs).
# Small LLMs often fail to diagnose this; they just stop. The tool itself now
# returns a structured retry recipe so the model sees exactly which filter is
# wrong and can re-call search_gdc in the next round.
# ---------------------------------------------------------------------------

# Last search_gdc result per session, used by download_gdc to resolve
# filter_pattern + max_count without forcing the model to copy UUIDs out of
# a long text blob. Values are the raw file records: [{file_id, file_name,
# file_size}, ...].
_GDC_SEARCH_CACHE: dict[str, list[dict]] = {}


_GDC_DATA_TYPE_FILTERS: dict[str, dict] = {
    "Slide Image": {
        "valid_strategies": ["Diagnostic Slide", "Tissue Slide"],
        "note": "Use experimental_strategy='Diagnostic Slide' for DX FFPE slides (gold standard for MIL), or 'Tissue Slide' for frozen sections.",
    },
    "Masked Somatic Mutation": {
        "valid_strategies": ["WXS"],
        "note": "MAFs come from whole-exome sequencing. Use experimental_strategy='WXS' OR omit the field entirely. NEVER pass 'Diagnostic Slide' — that's for slides.",
    },
    "Clinical Supplement": {
        "valid_strategies": [],
        "note": "Clinical supplements accept ONLY project + data_type. Drop experimental_strategy, primary_diagnosis, file_name, and access — they will zero the result set.",
    },
    "Gene Expression Quantification": {
        "valid_strategies": ["RNA-Seq"],
        "note": "Set experimental_strategy='RNA-Seq' OR omit.",
    },
    "Copy Number Segment": {
        "valid_strategies": ["Genotyping Array"],
        "note": "Set experimental_strategy='Genotyping Array' OR omit.",
    },
    "Methylation Beta Value": {
        "valid_strategies": ["Methylation Array"],
        "note": "Set experimental_strategy='Methylation Array' OR omit.",
    },
}


def _gdc_zero_result_hint(args: dict) -> str:
    """Return a model-readable retry recipe when search_gdc returned 0 files."""
    data_type = args.get("data_type") or "Slide Image"
    project = args.get("project") or "?"
    strategy = args.get("experimental_strategy")
    primary_diagnosis = args.get("primary_diagnosis")
    file_name = args.get("file_name")
    access = args.get("access")

    lines = [
        f"Found 0 {data_type} files in {project} for the given filters.",
        "",
        "This almost always means a filter is INCOMPATIBLE with the data_type, not that no such files exist.",
        "Read the recovery recipe below and call search_gdc AGAIN with corrected arguments. Do NOT ask the user — diagnose and retry.",
        "",
        f"## Expected filters for data_type='{data_type}'",
    ]

    spec = _GDC_DATA_TYPE_FILTERS.get(data_type)
    if spec:
        lines.append(spec["note"])
        if spec["valid_strategies"]:
            lines.append(f"Valid experimental_strategy values: {spec['valid_strategies']} (or omit).")
        else:
            lines.append("experimental_strategy is NOT APPLICABLE for this data_type — remove it.")
    else:
        lines.append("(No specific recipe for this data_type — try dropping filters one at a time.)")

    # Identify the most likely offending filter
    bad: list[str] = []
    if spec:
        if strategy and strategy not in spec["valid_strategies"] and spec["valid_strategies"]:
            bad.append(f"experimental_strategy='{strategy}' (expected one of {spec['valid_strategies']} or omit)")
        if strategy and not spec["valid_strategies"]:
            bad.append(f"experimental_strategy='{strategy}' (not valid for {data_type} — remove)")
    if primary_diagnosis and data_type != "Slide Image":
        bad.append(f"primary_diagnosis={primary_diagnosis!r} (only applies to Slide Image searches — remove for {data_type})")
    if file_name:
        bad.append(f"file_name={file_name!r} (remove; too restrictive for a first pass)")

    if bad:
        lines.append("")
        lines.append("## Likely offending filter(s) in your call")
        for b in bad:
            lines.append(f"- {b}")

    # Concrete retry
    retry = {"project": project, "data_type": data_type}
    if spec and spec["valid_strategies"]:
        retry["experimental_strategy"] = spec["valid_strategies"][0]
    if data_type == "Slide Image":
        if not strategy:
            retry["experimental_strategy"] = "Diagnostic Slide"
        if primary_diagnosis:
            retry["primary_diagnosis"] = primary_diagnosis

    lines += [
        "",
        "## RETRY NOW with these arguments",
        f"search_gdc({json.dumps(retry)})",
        "",
        "If that still returns 0, drop filters one at a time in this order: experimental_strategy → primary_diagnosis → file_name → access. Keep data_type and project; they are required.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Memory helpers
# ---------------------------------------------------------------------------

def _memory_path(session_id: str) -> Path:
    """Per-session memory file. Memory is scoped to the session so facts from a
    UCEC project don't leak into a new BRCA project in a different chat."""
    return CHATS_DIR / f"{session_id}.memory.json"


def _load_memory(session_id: str = "") -> dict:
    if not session_id:
        return {}
    p = _memory_path(session_id)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return {}
    return {}


def _save_memory(session_id: str, key: str, value: str) -> None:
    if not session_id:
        return
    mem = _load_memory(session_id)
    mem[key] = value
    _memory_path(session_id).write_text(json.dumps(mem, indent=2))


def _memory_block(session_id: str = "") -> str:
    mem = _load_memory(session_id)
    if not mem:
        return ""
    lines = ["## Session Memory\nFacts you've remembered in THIS session:\n"]
    for k, v in mem.items():
        lines.append(f"- **{k}**: {v}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Per-session notes — each session is like a PhD student's running notebook.
# The agent appends findings/decisions/progress under topic headings; the whole
# file is injected into every round's system prompt, so context survives
# message-trimming and auto-compaction.
# ---------------------------------------------------------------------------

def _notes_path(session_id: str) -> Path:
    return CHATS_DIR / f"{session_id}.notes.md"


def _read_session_notes(session_id: str) -> str:
    p = _notes_path(session_id)
    if p.exists():
        try:
            return p.read_text()
        except Exception:
            return ""
    return ""


def _append_session_note(session_id: str, topic: str, content: str) -> str:
    from datetime import datetime, timezone
    p = _notes_path(session_id)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    topic_clean = topic.strip() or "note"
    block = f"\n### {topic_clean}  _{ts}_\n{content.strip()}\n"
    if not p.exists():
        header = "# Session Notebook\n\nRunning log of decisions, findings, and context for this session.\n"
        p.write_text(header + block)
    else:
        with p.open("a") as f:
            f.write(block)
    return str(p)


def _session_notes_block(session_id: str) -> str:
    text = _read_session_notes(session_id)
    if not text.strip():
        return ""
    # Cap to keep the system prompt bounded; keep tail (most recent notes).
    if len(text) > 6000:
        text = "# Session Notebook (truncated — showing most recent entries)\n\n..." + text[-6000:]
    return f"## Session Notebook (per-session memory — read this before planning)\n\n{text}"


# ---------------------------------------------------------------------------
# Per-session manuscript workspace — each session has a LaTeX project folder
# so the agent can draft a paper alongside the research (PhD-student thesis
# model: notes → figures → manuscript in one place).
# ---------------------------------------------------------------------------

_ALLOWED_MS_EXT = {".tex", ".bib", ".cls", ".sty", ".bst", ".md"}


def _manuscript_dir(session_id: str) -> Path:
    d = CHATS_DIR / f"{session_id}_manuscript"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _safe_manuscript_path(session_id: str, filename: str) -> Path:
    """Resolve a filename inside the manuscript dir, refusing traversal."""
    if not filename or "/" in filename or ".." in filename or filename.startswith("."):
        raise ValueError("Invalid filename (no path separators, leading dot, or '..').")
    root = _manuscript_dir(session_id)
    p = (root / filename).resolve()
    if root.resolve() not in p.parents and p != root.resolve():
        raise ValueError("Path escapes manuscript dir.")
    return p


def _list_manuscript_files(session_id: str) -> list[dict]:
    root = _manuscript_dir(session_id)
    files = []
    for p in sorted(root.iterdir()):
        if p.is_file():
            files.append({
                "name": p.name,
                "size": p.stat().st_size,
                "modified": p.stat().st_mtime,
            })
    return files


_DEFAULT_LATEX_TEMPLATE = r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{graphicx}
\graphicspath{{figures/}}
\usepackage{hyperref}
\usepackage{booktabs}

\title{Working Title}
\author{Your Name}
\date{\today}

\begin{document}
\maketitle

\begin{abstract}
Short abstract here.
\end{abstract}

\section{Introduction}
Write the motivation and prior work.

\section{Methods}
Datasets, preprocessing, model, training.

\section{Results}
Metrics, figures, tables.

\section{Discussion}
Interpretation and limitations.

\bibliographystyle{plain}
\bibliography{refs}

\end{document}
"""


def _compile_latex(session_id: str, main_file: str = "main.tex") -> dict:
    """Compile a LaTeX project with tectonic → pdflatex fallback.

    If main_file doesn't exist, auto-scaffold a default template so the user
    always gets a compilable starting point on first Compile click.
    Returns {"status", "pdf", "log", "compiler", "scaffolded"}.
    """
    import shutil
    root = _manuscript_dir(session_id)
    src = root / main_file
    scaffolded = False
    if not src.exists():
        if main_file == "main.tex":
            src.write_text(_DEFAULT_LATEX_TEMPLATE)
            scaffolded = True
        else:
            return {"status": "error", "log": f"{main_file} not found in manuscript dir."}

    tectonic = shutil.which("tectonic")
    pdflatex = shutil.which("pdflatex")
    compiler = None
    log_parts: list[str] = []

    if tectonic:
        compiler = "tectonic"
        try:
            proc = subprocess.run(
                [tectonic, "--keep-logs", "--outdir", str(root), str(src)],
                capture_output=True, text=True, timeout=120, cwd=str(root),
            )
            log_parts.append(proc.stdout)
            log_parts.append(proc.stderr)
            if proc.returncode != 0:
                return {"status": "error", "compiler": compiler, "log": "\n".join(log_parts)[-4000:]}
        except Exception as e:
            return {"status": "error", "compiler": compiler, "log": f"tectonic failed: {e}"}
    elif pdflatex:
        compiler = "pdflatex"
        try:
            for _ in range(2):  # two passes for refs
                proc = subprocess.run(
                    [pdflatex, "-interaction=nonstopmode", "-halt-on-error",
                     "-output-directory", str(root), str(src)],
                    capture_output=True, text=True, timeout=90, cwd=str(root),
                )
                log_parts.append(proc.stdout)
            if proc.returncode != 0:
                return {"status": "error", "compiler": compiler, "log": "\n".join(log_parts)[-4000:]}
        except Exception as e:
            return {"status": "error", "compiler": compiler, "log": f"pdflatex failed: {e}"}
    else:
        return {
            "status": "error",
            "log": "No LaTeX compiler found. Install `tectonic` (recommended: single binary, auto-downloads packages) or `texlive-latex-base`.",
        }

    pdf = src.with_suffix(".pdf")
    if not pdf.exists():
        return {"status": "error", "compiler": compiler, "log": "Compiler returned success but no PDF produced."}
    return {"status": "ok", "compiler": compiler, "pdf": str(pdf), "log": "\n".join(log_parts)[-2000:], "scaffolded": scaffolded}


# ---------------------------------------------------------------------------
# Chat persistence helpers
# ---------------------------------------------------------------------------

def _chat_path(session_id: str) -> Path:
    return CHATS_DIR / f"{session_id}.json"


def _slugify(s: str) -> str:
    """Normalize to a short kebab-case slug. Returns '' if input cleans to empty."""
    import re as _re
    s = (s or "").strip().lower()
    s = _re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s[:40]


def _resolve_session(id_or_slug: str) -> str:
    """Map a slug OR session_id (or unique id prefix) to an actual session_id.

    Returns '' if nothing matches. Session_id match wins over slug match."""
    if not id_or_slug:
        return ""
    key = id_or_slug.strip()
    # Exact id hit
    if _chat_path(key).exists():
        return key
    key_slug = _slugify(key)
    id_prefix_matches: list[str] = []
    for p in CHATS_DIR.glob("*.json"):
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        sid = d.get("session_id", p.stem)
        if d.get("slug") == key_slug and key_slug:
            return sid
        if sid.startswith(key) and len(key) >= 4:
            id_prefix_matches.append(sid)
    if len(id_prefix_matches) == 1:
        return id_prefix_matches[0]
    return ""


def _save_chat(session_id: str) -> None:
    msgs = _conversations.get(session_id, [])
    visible = [m for m in msgs if m.get("role") in ("user", "assistant") and m.get("content")]
    auto_title = next((m["content"][:80] for m in visible if m["role"] == "user"), "New Session")
    now = _time_mod.time()

    # Preserve manually-set title, slug, and created_at if file already exists
    existing_title = None
    existing_slug = None
    created_at = now
    p = _chat_path(session_id)
    if p.exists():
        try:
            existing = json.loads(p.read_text())
            existing_title = existing.get("title")
            existing_slug = existing.get("slug")
            created_at = existing.get("created_at", now)
        except Exception:
            pass

    # Only auto-title if still the default
    title = auto_title if (not existing_title or existing_title == "New Session") else existing_title

    payload = {
        "session_id": session_id,
        "slug": existing_slug,
        "title": title,
        "created_at": created_at,
        "updated_at": now,
        "messages": visible,                          # user/assistant text only (for UI display)
        "full_messages": [m for m in msgs[1:]],       # everything except system prompt (for LLM hydration)
    }
    p.write_text(json.dumps(payload, indent=2))


def _list_chats() -> list[dict]:
    chats = []
    for p in sorted(CHATS_DIR.glob("*.json"), key=lambda x: -x.stat().st_mtime):
        try:
            d = json.loads(p.read_text())
            chats.append({
                "session_id": d["session_id"],
                "slug": d.get("slug"),
                "title": d.get("title", "New Session"),
                "created_at": d.get("created_at", d.get("timestamp", 0)),
                "updated_at": d.get("updated_at", d.get("timestamp", 0)),
                "message_count": len(d.get("messages", [])),
            })
        except Exception:
            continue
    return chats

# ---------------------------------------------------------------------------
# Skill keywords — maps skill name to trigger terms
# ---------------------------------------------------------------------------

SKILL_TRIGGERS: dict[str, list[str]] = {
    "dataset-intake": [
        "upload", "register", "ingest", "add dataset", "folder", "scan", "slide files",
        "dataset path", "validate", "svs", "ndpi", "mrxs", "tiff",
    ],
    "gdc-tcga": [
        "gdc", "tcga", "download", "brca", "luad", "lusc", "kirc", "kirp", "ucec", "stad",
        "blca", "skcm", "prad", "coad", "read", "crc", "endometrial", "uterine",
        "cancer genome atlas", "genomic data commons", "manifest", "dbgap", "controlled access",
        "diagnostic slide", "dx slide", "maf file", "clinical data", "msi status",
    ],
    "data-profiling": [
        "profil", "quality", "class balance", "class distribution", "cohort", "statistics",
        "label", "missing", "corrupt", "imbalance", "eda", "analyze dataset",
    ],
    "data-cleaning": [
        "clean", "harmonize", "standardize", "rename label", "duplicate", "outlier",
        "metadata", "map label", "recode",
    ],
    "data-lifecycle": [
        "storage", "disk space", "delete", "retention", "cleanup", "temp files",
        "free up", "archive", "purge",
    ],
    "wsi-preprocess": [
        "preprocess", "otsu", "patch", "tissue", "magnification",
        "extract patch", "tiling", "preview", "qc preview", "patch size", "stride",
        "feature extract", "extract feature", "uni", "conch", "virchow", "gigapath",
        "ctranspath", "backbone", "foundation model",
    ],
    "segmentation": [
        "segment", "segmentation", "nuclei", "cell", "mask", "iou", "dice",
        "hovernet", "unet", "cellpose", "instance", "semantic", "tissue mask",
        "nuclear", "cytoplasm", "panoptic", "boundary",
    ],
    "lora-finetune": [
        "lora", "fine-tune", "finetune", "fine tune", "adapter", "low-rank",
        "finetuning", "domain adapt", "adapt backbone", "peft",
    ],
    "train-config": [
        "config", "configure", "setup training", "training config", "mil method",
        "mammoth", "abmil", "transmil", "clam", "dsmil", "rrtmil", "wikg",
        "num experts", "num_experts", "attention", "embed", "hyperparameter",
    ],
    "train-exec": [
        "train", "start training", "launch", "run model", "epoch", "loss", "converge",
        "job", "monitor training", "training job", "training log", "checkpoint",
    ],
    "evaluation": [
        "evaluat", "metrics", "auroc", "auc", "accuracy", "f1", "confusion matrix",
        "roc curve", "precision", "recall", "balanced accuracy", "inference",
    ],
    "results": [
        "result", "interpret", "explain result", "compare run", "compare experiment",
        "recommend", "next step", "improve", "why", "what does this mean",
        "baseline", "published", "benchmark",
    ],
    "genomic-analysis": [
        "maf", "vcf", "mutation", "genomic", "variant", "tmb", "somatic", "snv", "indel",
        "oncoplot", "mutated gene", "variant classification", "hugo symbol", "parse maf",
        "parse vcf", "tumor mutational burden", "nonsynonymous", "missense", "nonsense",
        "frameshift", "splice", "maf file", "vcf file",
    ],
    "label-engineering": [
        "extract label", "msi label", "mutation label", "cbioportal", "combine data",
        "harmonize label", "slide label", "patient barcode", "tcga barcode",
        "labels.csv", "label extraction", "clinical field", "msi status", "tmb class",
        "mutation status", "wildtype", "mutant",
    ],
    "survival-biomarker": [
        "survival", "kaplan", "meier", "cox", "hazard", "prognosis", "overall survival",
        "biomarker", "differential expression", "enrichment", "log-rank",
        "progression free", "disease free", "survival curve",
    ],
}


def _match_skills(message: str) -> list[str]:
    """Return skill names whose keywords match the user message (up to 2)."""
    msg_lower = message.lower()
    scores: dict[str, int] = {}
    for skill, kws in SKILL_TRIGGERS.items():
        score = sum(1 for kw in kws if kw in msg_lower)
        if score > 0:
            scores[skill] = score
    # Return up to 2 best-matching skills
    return [s for s, _ in sorted(scores.items(), key=lambda x: -x[1])[:2]]


def _load_skill(skill_name: str) -> str:
    skill_path = WORKSPACE_DIR / "skills" / skill_name / "SKILL.md"
    if skill_path.exists():
        return f"\n\n---\n## Active Skill: {skill_name}\n\n{skill_path.read_text()}"
    return ""


def _load_skills_summary() -> str:
    p = WORKSPACE_DIR / "SKILLS_SUMMARY.md"
    return p.read_text() if p.exists() else ""


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_TOOL_GROUPS = [
    ("GDC / TCGA", {"search_gdc", "download_gdc", "gdc_job_status"}),
    ("Datasets", {"register_dataset", "list_datasets", "get_dataset_info", "delete_dataset", "attach_folder"}),
    ("Preprocessing", {"start_preprocessing", "get_preprocess_status", "preview_tissue_segmentation"}),
    ("Features", {"start_feature_extraction", "list_features"}),
    ("Training", {"start_training", "list_experiments", "get_experiment_info", "resume_training"}),
    ("Evaluation", {"start_evaluation", "get_eval_metrics", "get_eval_predictions", "make_plot", "list_artifacts", "attach_figure_to_manuscript"}),
    ("Jobs", {"get_job_status", "wait_for_job", "list_jobs", "cancel_job", "get_job_logs"}),
    ("Genomics", {"parse_genomic_file", "query_mutations", "compute_tmb", "extract_labels_from_genomic", "query_cbioportal", "run_survival_analysis", "build_multi_omic_labels", "generate_oncoplot", "parse_gene_expression", "biomarker_discovery"}),
    ("Python / notes", {"run_python", "write_note", "read_notes", "remember_fact", "recall_facts"}),
    ("Literature", {"deep_literature_review", "search_literature", "pubmed_search", "fetch_url", "get_paper_citations"}),
    ("Manuscript", {"write_manuscript", "compile_manuscript", "list_manuscript_files", "read_manuscript_file"}),
    ("Plugins", {"list_plugins", "install_plugin", "update_plugin_config", "smoke_test_plugin", "run_cellpose_segmentation", "clone_repo"}),
    ("Telegram", {"telegram_status"}),
    ("System", {"get_system_status"}),
]


def _tools_catalog() -> str:
    """Build an inline tool catalog for the system prompt.

    Small models (gemma4:e4b) sometimes hallucinate tool names when relying only
    on the function-calling schema surface. Injecting the canonical names into
    the system prompt makes them lexically present every round so the model
    copies rather than invents."""
    by_name = {t.get("function", {}).get("name", ""): t.get("function", {}).get("description", "") for t in TOOLS}
    lines = ["## Tool Catalog (call by EXACT name — do not invent new ones)"]
    seen: set[str] = set()
    for group_name, names in _TOOL_GROUPS:
        rows = []
        for n in sorted(names):
            if n in by_name:
                desc = (by_name[n] or "").split(".")[0].strip()[:120]
                rows.append(f"- `{n}` — {desc}")
                seen.add(n)
        if rows:
            lines.append(f"\n**{group_name}**")
            lines.extend(rows)
    other = sorted(set(by_name) - seen)
    if other:
        lines.append("\n**Other**")
        for n in other:
            desc = (by_name[n] or "").split(".")[0].strip()[:120]
            lines.append(f"- `{n}` — {desc}")
    lines.append("\nIf a name you want isn't in this list, IT DOES NOT EXIST. Pick the closest one above; do not prefix with namespaces like `google:` or `ds_api/`.")
    return "\n".join(lines)


def _build_system_prompt(extra_skill: str = "", session_id: str = "") -> str:
    """Load core workspace files + optional active skill into the system prompt."""
    parts = []
    for fname in ["AGENTS.md", "SOUL.md"]:
        fpath = WORKSPACE_DIR / fname
        if fpath.exists():
            parts.append(fpath.read_text())
    parts.append(_tools_catalog())

    summary = _load_skills_summary()
    if summary:
        parts.append(summary)

    mem = _memory_block(session_id)
    if mem:
        parts.append(mem)

    if session_id:
        notes = _session_notes_block(session_id)
        if notes:
            parts.append(notes)
        try:
            from pathclaw.api.routes.tasks import render_plan_for_prompt
            plan_block = render_plan_for_prompt(session_id)
            if plan_block:
                parts.append(plan_block)
        except Exception:
            pass

    parts.append("""## Tool Usage Instructions

**Pipeline execution (key behaviors):**
- When given a high-level goal (e.g. "train an MSI classifier on TCGA-UCEC"), execute the full pipeline autonomously:
  1. search_gdc → **show a summary table first** (data type, file count, est. GB) → confirm → download_gdc
     → After search, ALWAYS present a table like: | Data type | Files | Est. size | before starting downloads.
     → download_gdc returns immediately — downloads run as background subprocesses that survive restarts.
     → Tell the user the download is running and they can continue chatting/planning while it waits.
     → The Active Jobs panel (left sidebar) shows live file count and GB progress.
     → Only call wait_for_job(gdc) right before you need the files (e.g., before register_dataset).
  2. run_python to extract labels from MAF/CSV files
  3. start_preprocessing → wait_for_job(preprocess) → start_feature_extraction → wait_for_job(features)
  4. **Pause here — confirm** backbone, MIL method, hyperparams, **and the primary evaluation metric** with the user.
     Ask which metric matters for their task before training: `auroc` (default for binary tumor/normal, mutation calling),
     `accuracy` / `balanced_accuracy` (balanced cohorts), `macro_f1` / `weighted_f1` (class imbalance), `qwk`
     (ordinal targets — e.g., grade I–IV, Gleason), `sensitivity` / `specificity` (clinical screening, fixed threshold).
     Pass the user's choice in `evaluation.metrics` on start_training — the trainer will report all of them and
     early-stop on the first one listed.
  5. start_training → wait_for_job(training) → start_evaluation → wait_for_job(eval) → get_eval_metrics
- Use **wait_for_job** instead of manually polling get_job_status. It blocks until the job completes and emits live status updates.
- Use **parse_genomic_file** to parse MAF, VCF, or clinical XML files — it returns structured summaries, gene-specific variants, TMB stats, and clinical fields. Much more reliable than writing ad-hoc pandas code.
- Use **query_mutations** to query mutation data across a cohort (e.g. "which patients have EGFR mutations?").
- Use **compute_tmb** to compute Tumor Mutational Burden from MAF files.
- Use **extract_labels_from_genomic** to automatically extract slide-level labels from genomic data (MSI status, mutation status, TMB class, clinical fields) with TCGA barcode resolution and patient deduplication.
- Use **query_cbioportal** to get mutation data, clinical attributes, MSI scores, and CNA from cBioPortal without downloading raw files. Accepts TCGA project names (e.g. TCGA-UCEC) directly.
- Use **run_survival_analysis** to run Kaplan-Meier survival analysis on clinical data, stratified by slide labels.
- Use **build_multi_omic_labels** to combine data from multiple sources (MAF mutations, clinical, model predictions) into a unified matrix.
- Use **generate_oncoplot** to create mutation landscape plots from MAF files — shows top mutated genes across samples.
- Use **parse_gene_expression** to parse STAR/HTSeq expression files and query specific genes.
- Use **biomarker_discovery** to find differentially mutated genes between groups or correlate MIL attention with mutations.
- Use **run_python** for any data wrangling without a dedicated tool. pandas, numpy, pathlib are pre-imported. Use print() for output.
- For any **literature review / state-of-the-art / "what's recent on X"** request, **use `deep_literature_review`** as your primary tool. It fans out 6 sub-queries in parallel across Semantic Scholar + OpenAlex + PubMed + Europe PMC, dedupes, ranks by citations+recency, and returns 20 papers with clickable links. Use `search_literature` or `pubmed_search` only for narrow follow-ups on a single specific question.
- After you have the ranked list from `deep_literature_review`, use **`get_paper_citations`** on the 2–3 most relevant papers to follow the citation graph (forward citations + references) before synthesizing.
- **Before starting a literature review, ask the user clarifying questions first** — do NOT jump straight to search tools. Specifically ask:
  1. **Objective** — what's the goal? Mapping SOTA, finding method baselines, clinical validation evidence, mechanistic understanding, or something else?
  2. **Paper types** — primary research / systematic reviews / methods papers / clinical trials / benchmarks?
  3. **Time range** — last 2 years (strict SOTA), last 5 years, or all-time landmark papers?
  4. **Scope** — specific subtypes, models, datasets, or evaluation settings to prioritize?
  5. **Depth** — quick scan (5–10 papers) vs. deep dive (30+ papers with synthesis)?
  Skip this ONLY if the user explicitly says "just search" or has already answered these in-session.
- **When citing papers in your synthesis, always preserve the clickable links from the tool output.** Every citation must include at least one URL (PubMed, DOI, Semantic Scholar, or PDF) — never cite by PMID/year alone. Format: `Author et al. (Year) [PubMed](url)` or similar.
- Do not invent citations. If you don't have a tool-returned result for a claim, either drop the claim or mark it as "unverified".
- Only ask for user confirmation before: GPU-intensive jobs (training, feature extraction), large downloads (>50 GB), data deletion.
- After evaluation completes, analyze the metrics and suggest next experiments.
- Use remember_fact for facts about THIS session (chosen dataset_id, label column, backbone, training config). Memory is session-scoped — it won't leak into other sessions and it won't see other sessions' memory.
- Use **write_note** liberally for this session's running log: dataset decisions, job IDs, label files, experiment plans, errors encountered, interim results. Each session is a PhD student's notebook — future-you (after context trimming) relies on it. Write a note at the start of every major phase (data registration, preprocessing, training kick-off) and after every result.
- Use **write_manuscript** when the user wants to draft/revise a paper. Each session has its own LaTeX project (main.tex + sections + refs.bib). Go from ideation → results → manuscript in the same session: write sections as results come in, cite papers you pulled via search_literature/pubmed_search (add them to refs.bib), then compile_manuscript to get a PDF. Assume a standard article class unless told otherwise.
- Format numbers and metrics clearly. If a tool call fails, explain and suggest alternatives.

**Cellpose segmentation (built-in plugin):**
- Cellpose is registered as a `method` plugin (id=`cellpose`). It does nuclei / cell instance segmentation on patches of a registered dataset. Pretrained, no training needed.
- Run it via `run_cellpose_segmentation(dataset_id, max_slides?, max_patches_per_slide?, config_overrides?)`. Output goes under `~/.pathclaw/sessions/<sid>/cellpose/<dataset_id>/<slide_stem>/` as instance-mask PNGs.
- Editable knobs (read defaults via `list_plugins`, persist new defaults via `update_plugin_config`, or pass per-call via `config_overrides`):
  - `model_type` ∈ {`cpsam` (default — SAM-based generalist), `nuclei` (H&E-biased), `cyto3` (brightfield cytoplasm)}
  - `diameter` (px, null = auto-estimate)
  - `flow_threshold` (0–3, default 0.4 — lower is stricter)
  - `cellprob_threshold` (-6 to 6, default 0.0 — raise to reject dim detections)
  - `niter` (0 = auto; raise to ~2000 for crowded objects)
  - `channels` ([cyto, nucleus]; 0=grayscale, 1=R, 2=G, 3=B; default [0,0])
  - `min_size` (px, default 15)
  - `tile_norm` (bool, default true — recommended for WSI)
  - `gpu` (bool, default true)
- When the user says e.g. *"lower cellpose's flow_threshold to 0.2 and re-run on the first 3 slides"*: call `update_plugin_config(id="cellpose", default_config={...})` with the merged config, then `run_cellpose_segmentation(dataset_id, max_slides=3)`.
- When the user wants to A/B compare without persisting changes: pass `config_overrides={"flow_threshold": 0.2}` directly to `run_cellpose_segmentation`.
- Cellpose requires preprocessed patches in the dataset (run `start_preprocessing` first). It does NOT require feature extraction.""")

    if extra_skill:
        parts.append(extra_skill)

    return "\n\n---\n\n".join(parts)


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

TOOLS = [
    # --- Session identity ---
    {
        "type": "function",
        "function": {
            "name": "rename_session",
            "description": (
                "Set a short kebab-case slug (and optionally a human title) on THIS session so it can be referenced by name. "
                "Useful for Telegram (`/session <slug>`), the sidebar, and your own cross-reference. "
                "Slug must be a-z/0-9/- up to 40 chars and unique across sessions. "
                "Call this ONCE near the start of a new project, e.g. rename_session(slug='chol-idh', title='TCGA-CHOL IDH1 MIL')."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "slug": {"type": "string", "description": "Kebab-case identifier (e.g. 'chol-idh', 'ucec-msi-v2')"},
                    "title": {"type": "string", "description": "Optional human-readable title for the sidebar/history list"},
                },
                "required": ["slug"],
            },
        },
    },
    # --- Memory ---
    {
        "type": "function",
        "function": {
            "name": "remember_fact",
            "description": "Save a fact to THIS session's memory. Memory is session-scoped — a UCEC session's facts do NOT leak into a BRCA session. Use for this-session dataset paths, label columns, chosen backbone, training decisions. Cross-session info (HF tokens, conventions) lives in config, not here.",
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {"type": "string", "description": "Short descriptive key (e.g. 'dataset_id', 'label_column', 'backbone')"},
                    "value": {"type": "string", "description": "The fact to remember"},
                },
                "required": ["key", "value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recall_facts",
            "description": "List facts stored in THIS session's memory. Does not see other sessions' memory.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    # --- Per-session notebook (PhD-student model) ---
    {
        "type": "function",
        "function": {
            "name": "write_note",
            "description": "Append an entry to THIS session's running notebook under a topic heading. Use liberally to record: dataset choices, label extraction outputs, job IDs, hyperparameter decisions, error diagnoses, interim results, links to papers read. The notebook is always visible in your system prompt, so it survives message trimming and auto-compaction. Think of each session as a PhD student's working log.",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "Short heading (e.g. 'Dataset', 'Labels', 'Experiment 01 — UNI + ABMIL', 'Bug — Virchow OOM')"},
                    "content": {"type": "string", "description": "The note content, markdown allowed. Be concise but complete — facts, numbers, paths, job IDs."},
                },
                "required": ["topic", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_notes",
            "description": "Read the full session notebook. Rarely needed since notes are auto-injected into your system prompt, but useful if you want to review before planning a complex multi-step operation.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    # --- Task plan (mandatory for ≥3-step requests; prevents drift on small LLMs) ---
    {
        "type": "function",
        "function": {
            "name": "create_task_plan",
            "description": (
                "Commit to a step-by-step plan BEFORE doing multi-step work. "
                "Call this ONCE at the start of any request that needs 3+ tool calls "
                "(paper generation, pipeline runs, dataset prep, IHC scoring, etc.). "
                "Each task is one discrete chunk of work (\"download slides\", \"extract features\", "
                "\"write manuscript methods section\"). Use `pause_after=true` on a task only when "
                "you need the user's input before continuing (e.g., after showing intermediate "
                "results). Default is auto-advance: finish a task, mark completed, immediately "
                "start the next one in the same turn. The current plan is always visible at the "
                "top of your system prompt so you cannot lose track."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "tasks": {
                        "type": "array",
                        "description": "Ordered list of tasks. Each is {title, description, pause_after?}.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "title": {"type": "string", "description": "Short imperative title (<60 chars)"},
                                "description": {"type": "string", "description": "What tool calls / outcome this task covers"},
                                "pause_after": {"type": "boolean", "description": "Set true to stop and wait for user input after this task"},
                            },
                            "required": ["title"],
                        },
                    },
                },
                "required": ["tasks"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_task_status",
            "description": (
                "Flip a task's status. Call this TWICE per task: "
                "first with status=\"in_progress\" BEFORE you start the task's tool calls, "
                "then with status=\"completed\" once done. Use \"skipped\" if a task is no longer "
                "needed given what you've learned. The frontend checklist updates live."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {"type": "integer", "description": "ID from create_task_plan response / plan view"},
                    "status": {"type": "string", "enum": ["pending", "in_progress", "completed", "skipped"]},
                },
                "required": ["task_id", "status"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_task_plan",
            "description": "Read the current plan. Usually not needed — the plan is always injected into your system prompt — but available if you want to re-read.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    # --- IHC scoring (rule-based + learned) ---
    {
        "type": "function",
        "function": {
            "name": "list_ihc_rules",
            "description": (
                "List built-in IHC scoring presets (Ki-67 PI, ER/PR Allred, HER2 0/1+/2+/3+, "
                "PD-L1 TPS) with their compartments, DAB thresholds, and aggregation rules. "
                "Call this before score_ihc when the user mentions a marker but you're not "
                "sure which preset fits."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "score_ihc",
            "description": (
                "Run rule-based IHC scoring across every slide in a registered dataset. "
                "No training required — uses color deconvolution (H/E/DAB) + cellpose (or "
                "a morphology fallback) to compute a clinical score per slide (e.g. Ki-67 "
                "proliferation index, HER2 0/1+/2+/3+, ER Allred 0-8, PD-L1 TPS). "
                "`rule_override` lets you tweak thresholds/patches_per_slide on the fly — "
                "great when the user says \"use a stricter DAB cutoff\" or \"sample 500 "
                "patches per slide\". Writes a CSV under the dataset dir and returns summary "
                "stats (mean/median/min/max score, n scored, n failed)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset_id": {"type": "string", "description": "Dataset to score (e.g. tcga-brca-her2)"},
                    "rule": {
                        "type": "string",
                        "description": "Preset name.",
                        "enum": ["ki67_pi", "er_allred", "pr_allred", "her2_membrane", "pdl1_tps"],
                    },
                    "rule_override": {
                        "type": "object",
                        "description": "Patch any Rule field on the fly. Common keys: dab_threshold (float|list), patches_per_slide (int), patch_size (int), target_mpp (float).",
                    },
                    "max_slides": {"type": "integer", "description": "Cap for quick dry-runs. Omit to score all."},
                    "use_cellpose": {"type": "boolean", "description": "Use cellpose for nuclear segmentation (slower, more accurate). Default true."},
                },
                "required": ["dataset_id", "rule"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "build_ihc_patch_labels",
            "description": (
                "Bootstrap a per-patch training set from the same rule engine. For each slide, "
                "samples N tissue patches, computes the rule's continuous score per patch, and "
                "writes two CSVs: patch-level (slide_id, patch_idx, label) and slide-level "
                "(mean over patches). Use these as supervision for a learned MIL/regressor on "
                "top of UNI/GigaPath features — same rule, but now GPU-efficient at inference."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset_id": {"type": "string"},
                    "rule": {
                        "type": "string",
                        "enum": ["ki67_pi", "er_allred", "pr_allred", "her2_membrane", "pdl1_tps"],
                    },
                    "rule_override": {"type": "object"},
                    "patches_per_slide": {"type": "integer", "description": "Overrides the rule's default (e.g. 400 for denser supervision)."},
                    "out_name": {"type": "string", "description": "Base name for the output CSVs (default: patchlabels_<rule>)."},
                },
                "required": ["dataset_id", "rule"],
            },
        },
    },
    # --- Manuscript (LaTeX) — draft, edit, and compile the session's paper ---
    {
        "type": "function",
        "function": {
            "name": "write_manuscript",
            "description": (
                "Create or overwrite a file in THIS session's LaTeX manuscript project "
                "(~/.pathclaw/chats/{session}_manuscript/). Use for .tex source (main.tex, sections), "
                ".bib bibliography, .cls/.sty style files. Start a paper by writing main.tex with \\documentclass, "
                "then add \\section content over subsequent calls. Each session is one manuscript — like a PhD thesis repo."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "description": "File name (e.g. 'main.tex', 'refs.bib', 'sections/methods.tex'). No path escapes."},
                    "content": {"type": "string", "description": "Full file content — this REPLACES any existing file of the same name."},
                    "mode": {"type": "string", "enum": ["write", "append"], "default": "write", "description": "'write' replaces file; 'append' adds to end (useful for growing sections)."},
                },
                "required": ["filename", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_manuscript",
            "description": "Read a file from the session's manuscript project (e.g. to revise a section). Omit filename to list all files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "description": "File name inside the manuscript dir. If omitted, returns a listing of all files."},
                    "max_chars": {"type": "integer", "description": "Truncate content to at most this many characters (default 8000).", "default": 8000},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compile_manuscript",
            "description": (
                "Compile the LaTeX project to PDF (tries tectonic, falls back to pdflatex). "
                "Returns the PDF path on success or the compiler log tail on error. "
                "Call after finishing edits; fix any errors reported and recompile."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "main_file": {"type": "string", "description": "Top-level .tex file (default 'main.tex').", "default": "main.tex"},
                },
                "required": [],
            },
        },
    },
    # --- System & Config ---
    {
        "type": "function",
        "function": {
            "name": "system_status",
            "description": "Get system status: GPU availability, storage, and data directory path.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_config",
            "description": "Get PathClaw config: whether HuggingFace/GDC tokens are set, data directory, onboarding status.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    # --- Datasets ---
    {
        "type": "function",
        "function": {
            "name": "list_datasets",
            "description": "List all registered datasets with slide counts and sizes.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "register_dataset",
            "description": "Register a new dataset from a local path. Accepts either a directory containing WSI files, or a single WSI file path (.svs, .tiff, .ndpi, .mrxs). Scans recursively for slide files and records metadata.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Display name for the dataset"},
                    "path": {"type": "string", "description": "Absolute path to a directory of slides, or a single WSI file (e.g. /data/slides/my_slide.svs)"},
                    "description": {"type": "string", "description": "Brief description of the dataset"},
                },
                "required": ["name", "path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dataset_slides",
            "description": (
                "List concrete slide file paths for a dataset so you can iterate over them from "
                "run_python. Returns absolute paths ready for openslide.OpenSlide()."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset_id": {"type": "string", "description": "Dataset ID from list_datasets"},
                    "limit": {"type": "integer", "description": "Maximum number of slides to return", "default": 10},
                    "offset": {"type": "integer", "description": "Skip first N slides", "default": 0},
                },
                "required": ["dataset_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_dataset_profile",
            "description": "Get quality profile for a dataset: class distribution, slide sizes, label coverage, missing data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset_id": {"type": "string", "description": "Dataset ID from list_datasets"},
                },
                "required": ["dataset_id"],
            },
        },
    },
    # --- GDC / TCGA ---
    {
        "type": "function",
        "function": {
            "name": "search_gdc",
            "description": (
                "Search GDC/TCGA for files. Supports rich filtering by project, histological type, "
                "slide type (diagnostic vs tissue), data type, and workflow type. "
                "Returns file_ids needed for download_gdc. "
                "For TCGA slides use experimental_strategy='Diagnostic Slide' for FFPE diagnostic slides. "
                "primary_diagnosis values are lowercase, e.g. 'endometrioid adenocarcinoma, nos'. "
                "For MSI labels use data_type='Masked Somatic Mutation'. "
                "For gene expression use data_type='Gene Expression Quantification', workflow_type='STAR - Counts'. "
                "For copy number use data_type='Copy Number Segment'. "
                "For methylation use data_type='Methylation Beta Value'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "project": {"type": "string", "description": "GDC project ID, e.g. TCGA-UCEC, TCGA-LUAD"},
                    "data_type": {
                        "type": "string",
                        "description": "Data type: 'Slide Image', 'Masked Somatic Mutation', 'Clinical Supplement', 'Gene Expression Quantification', 'Copy Number Segment', 'Methylation Beta Value', 'miRNA Expression Quantification', 'Biospecimen Supplement'",
                        "default": "Slide Image",
                    },
                    "experimental_strategy": {
                        "type": "string",
                        "description": "Slide type: 'Diagnostic Slide' (FFPE diagnostic) or 'Tissue Slide'. Leave empty for non-slide data types.",
                        "default": "",
                    },
                    "workflow_type": {
                        "type": "string",
                        "description": "Analysis workflow filter. E.g. 'MuTect2 Variant Aggregation and Masking', 'STAR - Counts', 'HTSeq - Counts', 'BWA with Mark Duplicates and Cocleaning'. Leave empty if not needed.",
                        "default": "",
                    },
                    "primary_diagnosis": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Filter by histological type (lowercase). E.g. ['endometrioid adenocarcinoma, nos', 'endometrioid adenocarcinoma, secretory variant']",
                        "default": [],
                    },
                    "access": {"type": "string", "description": "'open' or 'controlled'", "default": "open"},
                    "limit": {"type": "integer", "description": "Max files to retrieve. Default 1000 fetches all files in most cohorts. The handler auto-paginates if total exceeds this.", "default": 1000},
                },
                "required": ["project"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "download_gdc",
            "description": (
                "Download GDC/TCGA files. Runs as a background job — returns job_id immediately. "
                "Two ways to pick files: (A) pass explicit file_ids (exact 36-char UUIDs from a prior search_gdc), OR "
                "(B) pass filter_pattern and/or max_count and it will resolve against the LAST search_gdc result in this session. "
                "Option B is preferred for small models — you never have to copy UUIDs. "
                "Poll gdc_job_status for progress. Downloaded files land in output_dir and can be registered as a dataset."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_ids": {"type": "array", "items": {"type": "string"}, "description": "List of GDC file UUIDs (36-char hex+dashes). Omit if using filter_pattern."},
                    "filter_pattern": {"type": "string", "description": "Glob/substring applied to file_name on the LAST search_gdc result (e.g. '*-01Z-*DX*' for diagnostic FFPE slides). Bare substrings are auto-wrapped in '*'."},
                    "max_count": {"type": "integer", "description": "Cap the number of files picked (e.g. 10 for a quick pilot)."},
                    "output_dir": {"type": "string", "description": "Local directory to save files. Default: ~/.pathclaw/downloads/{project}"},
                    "project": {"type": "string", "description": "Project name used for default output_dir (e.g. 'tcga-ucec')"},
                    "max_concurrent": {"type": "integer", "description": "Parallel download streams (default: 4)", "default": 4},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "gdc_job_status",
            "description": "Poll the status of a running GDC download job. Returns progress (done/total), output_dir, and any failed file_ids.",
            "parameters": {
                "type": "object",
                "properties": {
                    "job_id": {"type": "string", "description": "Job ID returned by download_gdc"},
                },
                "required": ["job_id"],
            },
        },
    },
    # --- Preprocessing ---
    {
        "type": "function",
        "function": {
            "name": "start_preprocessing",
            "description": (
                "Launch WSI preprocessing: Otsu tissue segmentation + patch extraction. "
                "Returns a job_id immediately. Run directly with preview_only=false for known cohorts; only set preview_only=true when the user explicitly asks for a preview or when tissue-segmentation params are unvalidated for a brand-new dataset."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset_id": {"type": "string", "description": "Dataset ID to preprocess"},
                    "patch_size": {"type": "integer", "description": "Patch size in pixels (default 256)", "default": 256},
                    "stride": {"type": "integer", "description": "Stride between patches (default 256 = non-overlapping)", "default": 256},
                    "magnification": {"type": "number", "description": "Target magnification (default 20.0)", "default": 20.0},
                    "otsu_level": {"type": "integer", "description": "OpenSlide pyramid level for Otsu mask (default 1)", "default": 1},
                    "min_tissue_pct": {"type": "number", "description": "Min tissue fraction per patch 0-1 (default 0.5)", "default": 0.5},
                    "preview_only": {"type": "boolean", "description": "Process only 3 slides for QC preview", "default": False},
                },
                "required": ["dataset_id"],
            },
        },
    },
    # --- Training ---
    {
        "type": "function",
        "function": {
            "name": "start_training",
            "description": (
                "Launch a MIL training job (task_type='mil') or segmentation training (task_type='segmentation'). "
                "Returns a job_id. For MIL, requires pre-extracted features (.pt files). "
                "For segmentation, requires mask PNG files in datasets/{id}/masks/. "
                "ALWAYS get user confirmation before calling this — it launches GPU computation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {"type": "string", "description": "Task description, e.g. 'BRCA subtype classification' or 'nuclei segmentation'"},
                    "dataset_id": {"type": "string", "description": "Dataset ID"},
                    "task_type": {"type": "string", "description": "Task type: mil (default) or segmentation", "default": "mil"},
                    "label_column": {"type": "string", "description": "Label column for MIL (not needed for segmentation)"},
                    "mil_method": {
                        "type": "string",
                        "description": "MIL method: abmil (default), meanpool, transmil, clam, dsmil, rrtmil, wikg",
                        "default": "abmil",
                    },
                    "seg_model": {
                        "type": "string",
                        "description": "Segmentation model: seg_unet (default), hovernet, cellpose",
                        "default": "seg_unet",
                    },
                    "feature_backbone": {
                        "type": "string",
                        "description": "Feature backbone (for MIL): uni (1024), conch (512), ctranspath (768), virchow (1280), virchow2 (2560), gigapath (1536)",
                        "default": "uni",
                    },
                    "feature_dim": {"type": "integer", "description": "Feature dimension (must match backbone)", "default": 1024},
                    "num_classes": {"type": "integer", "description": "Number of MIL output classes", "default": 2},
                    "num_seg_classes": {"type": "integer", "description": "Number of segmentation classes (including background)", "default": 2},
                    "mammoth_enabled": {"type": "boolean", "description": "Enable MAMMOTH MoE patch embedding (MIL only, default true)", "default": True},
                    "epochs": {"type": "integer", "description": "Training epochs", "default": 100},
                    "lr": {"type": "number", "description": "Learning rate (default 1e-4)", "default": 1e-4},
                    "optimizer": {"type": "string", "description": "Optimizer: adam (default), adamw, sgd, radam", "default": "adam"},
                    "eval_strategy": {
                        "type": "string",
                        "description": "Evaluation strategy for MIL: holdout (default) or 5-fold-cv, 3-fold-cv",
                        "default": "holdout",
                    },
                },
                "required": ["task", "dataset_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_training_logs",
            "description": "Get recent training logs for a running or completed training job.",
            "parameters": {
                "type": "object",
                "properties": {
                    "job_id": {"type": "string", "description": "Training job ID"},
                },
                "required": ["job_id"],
            },
        },
    },
    # --- Task queue (serializes GPU-heavy work) ---
    {
        "type": "function",
        "function": {
            "name": "queue_task",
            "description": (
                "Queue a GPU-heavy task (training / features / preprocess / evaluation) for serialized execution. "
                "Use this INSTEAD OF start_training/extract_features/start_preprocessing when there are already GPU jobs running "
                "or when chaining multiple jobs (e.g. run feature extraction THEN training). "
                "The queue ensures only one GPU-exclusive task runs at a time, avoiding OOM and race conditions. "
                "Returns a task_id you can poll via list_queue."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task_type": {
                        "type": "string",
                        "description": "One of: training, features, preprocess, evaluation",
                    },
                    "payload": {
                        "type": "object",
                        "description": (
                            "Same fields you'd pass to the corresponding tool (start_training / extract_features / etc). "
                            "For training: use flat keys like task, dataset_id, mil_method, feature_backbone, feature_dim, "
                            "num_classes, label_column, mammoth_enabled, epochs, lr, optimizer, eval_strategy — "
                            "these are normalized to the nested TrainingConfig automatically."
                        ),
                    },
                    "note": {"type": "string", "description": "Short human-readable description of why this task is queued"},
                },
                "required": ["task_type", "payload"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_queue",
            "description": "List queued/running/recent tasks. Use after queue_task to check dispatch status.",
            "parameters": {"type": "object", "properties": {"status": {"type": "string"}}},
        },
    },
    # --- Evaluation ---
    {
        "type": "function",
        "function": {
            "name": "start_evaluation",
            "description": "Launch model evaluation. Computes AUROC, balanced accuracy, F1, confusion matrix, ROC curves.",
            "parameters": {
                "type": "object",
                "properties": {
                    "model_path": {"type": "string", "description": "Path to model.pth (from list_artifacts)"},
                    "dataset_id": {"type": "string", "description": "Dataset ID to evaluate on"},
                    "split": {"type": "string", "description": "Data split to evaluate: val (default) or all", "default": "val"},
                },
                "required": ["model_path", "dataset_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_eval_metrics",
            "description": "Get evaluation metrics (accuracy, AUROC, F1, confusion matrix) for a completed eval job.",
            "parameters": {
                "type": "object",
                "properties": {
                    "job_id": {"type": "string", "description": "Evaluation job ID"},
                },
                "required": ["job_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_eval_plots",
            "description": "List plot files (ROC curve, confusion matrix) generated by an evaluation job.",
            "parameters": {
                "type": "object",
                "properties": {
                    "job_id": {"type": "string", "description": "Evaluation job ID"},
                },
                "required": ["job_id"],
            },
        },
    },
    # --- Job Status (unified) ---
    {
        "type": "function",
        "function": {
            "name": "get_job_status",
            "description": (
                "Poll the status of any background job (preprocessing, training, evaluation, feature extraction). "
                "Returns status, progress, and current metrics."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "job_id": {"type": "string", "description": "Job ID returned by a start_* tool"},
                    "job_type": {
                        "type": "string",
                        "description": "Job type: preprocess, training, eval, features, lora",
                        "default": "training",
                    },
                },
                "required": ["job_id"],
            },
        },
    },
    # --- Feature Extraction ---
    {
        "type": "function",
        "function": {
            "name": "start_feature_extraction",
            "description": (
                "Launch foundation model feature extraction on preprocessed WSI patches. "
                "Must run preprocessing first. Saves .pt feature files used for training. "
                "ALWAYS get user confirmation before calling — it loads a large model onto GPU."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset_id": {"type": "string", "description": "Dataset ID with preprocessed patches"},
                    "backbone": {
                        "type": "string",
                        "description": "Foundation model: uni (1024d), conch (512d), ctranspath (768d), virchow (1280d), virchow2 (2560d), gigapath (1536d)",
                        "default": "uni",
                    },
                    "batch_size": {"type": "integer", "description": "Patches per batch (default 256)", "default": 256},
                },
                "required": ["dataset_id"],
            },
        },
    },
    # --- Artifacts ---
    {
        "type": "function",
        "function": {
            "name": "list_artifacts",
            "description": "List all experiment artifacts: trained models, config files, metrics, plots.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    # --- LoRA fine-tuning ---
    {
        "type": "function",
        "function": {
            "name": "start_lora_finetuning",
            "description": (
                "Fine-tune a foundation model backbone (Virchow2, UNI, etc.) with LoRA adapters. "
                "Only adapter weights are saved (~10-50MB). After fine-tuning, re-run feature "
                "extraction with the LoRA-adapted model for improved MIL training. "
                "ALWAYS get user confirmation first — loads a large model onto GPU."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "backbone": {
                        "type": "string",
                        "description": "Backbone to fine-tune: uni, conch, virchow, virchow2, gigapath, ctranspath",
                        "default": "uni",
                    },
                    "dataset_id": {"type": "string", "description": "Dataset ID with preprocessed patches"},
                    "num_classes": {"type": "integer", "description": "Number of output classes", "default": 2},
                    "lora_rank": {"type": "integer", "description": "LoRA rank r (default 8, higher = more params)", "default": 8},
                    "lora_alpha": {"type": "integer", "description": "LoRA scaling alpha (default 16)", "default": 16},
                    "epochs": {"type": "integer", "description": "Fine-tuning epochs (default 20)", "default": 20},
                    "lr": {"type": "number", "description": "Learning rate (default 5e-5)", "default": 5e-5},
                    "merge_adapter": {"type": "boolean", "description": "Merge LoRA weights into base model after training", "default": False},
                },
                "required": ["backbone", "dataset_id"],
            },
        },
    },
    # --- Heatmap ---
    {
        "type": "function",
        "function": {
            "name": "generate_heatmap",
            "description": (
                "Generate an attention heatmap overlay for a WSI slide using a trained MIL model. "
                "Requires completed training and feature extraction. "
                "Result can be viewed in the Viewer tab as a semi-transparent overlay."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "experiment_id": {"type": "string", "description": "Experiment ID of the trained model"},
                    "dataset_id": {"type": "string", "description": "Dataset ID the slide belongs to"},
                    "slide_stem": {"type": "string", "description": "Slide filename without extension"},
                },
                "required": ["experiment_id", "dataset_id", "slide_stem"],
            },
        },
    },
    # --- Code Execution ---
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": (
                "Execute a Python script for data wrangling, analysis, or any computation "
                "that lacks a dedicated tool. Common uses: parse MAF files to extract MSI/mutation "
                "labels, read clinical CSVs and build label mappings, compute cohort statistics, "
                "custom filtering or transformation of data files. "
                "pandas, numpy, pathlib, json, csv are pre-imported. DATA_DIR = Path('~/.pathclaw'). "
                "Use print() for output — that's what gets returned. 60-second timeout."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute. Use print() for output.",
                    },
                    "description": {
                        "type": "string",
                        "description": "Brief one-line description of what this script does.",
                    },
                },
                "required": ["code"],
            },
        },
    },
    # --- Experiment Comparison ---
    {
        "type": "function",
        "function": {
            "name": "compare_experiments",
            "description": (
                "Compare metrics across multiple experiments side-by-side. "
                "Returns a formatted table of AUROC, balanced accuracy, F1, and config (backbone, method, MAMMOTH) "
                "for each experiment. Use after running multiple training runs to identify the best configuration."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "experiment_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of experiment IDs to compare",
                    },
                },
                "required": ["experiment_ids"],
            },
        },
    },
    # --- Job Waiting ---
    {
        "type": "function",
        "function": {
            "name": "wait_for_job",
            "description": (
                "Wait for a background job to complete before continuing. "
                "Polls every 10 seconds and returns when status is completed or failed. "
                "Use this instead of manually calling get_job_status in a loop. "
                "Returns final status, progress, and any available metrics. "
                "Timeout: 30 minutes. "
                "CRITICAL: job_id MUST be a real id returned by a prior start_*/download_* tool "
                "call in this session. Do NOT invent, guess, or carry over job_ids from older "
                "sessions — the call will fail fast with an explicit error if the job does not exist."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "job_id": {"type": "string", "description": "Job ID returned by a previous start_*/download_* tool call. Must exist; do not fabricate."},
                    "job_type": {
                        "type": "string",
                        "description": "Job type: preprocess | training | eval | features | lora | gdc",
                        "default": "training",
                    },
                },
                "required": ["job_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_literature",
            "description": (
                "Search scientific literature for pathology and oncology papers using Semantic Scholar. "
                "Returns titles, authors, year, citation count, and abstracts. "
                "Use this before designing experiments, to find prior work on a cancer type or method, "
                "or when the user asks 'what papers exist on X' or 'how has Y been done before'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query, e.g. 'MSI microsatellite instability endometrial deep learning MIL'",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Number of papers to return (1–10)",
                        "default": 5,
                    },
                    "year_from": {
                        "type": "integer",
                        "description": "Only return papers published from this year onward (e.g. 2020)",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "deep_literature_review",
            "description": (
                "Run a multi-query, multi-source literature search and return a ranked, de-duplicated list of papers "
                "with clickable PubMed/DOI/PDF links. Use this for any 'literature review', 'state of the art', "
                "'what's the latest on X' request — it fans out across Semantic Scholar, PubMed, OpenAlex, and Europe PMC "
                "in parallel, dedupes by DOI/PMID, ranks by citations + recency, and returns ~20 papers. "
                "Much more thorough than a single search_literature call."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "Core research topic, e.g. 'deep learning breast cancer histopathology subtyping'",
                    },
                    "year_from": {
                        "type": "integer",
                        "description": "Only return papers published from this year onward (e.g. 2023)",
                        "default": 2022,
                    },
                    "max_papers": {
                        "type": "integer",
                        "description": "Max papers to return after dedup/rank (10–40, default 20)",
                        "default": 20,
                    },
                    "angles": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Optional sub-angles to search in parallel. If omitted, 6 defaults are generated: "
                            "methods / benchmarks / recent reviews / foundation models / multimodal / limitations."
                        ),
                    },
                },
                "required": ["topic"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_paper_citations",
            "description": (
                "Given a paper (by DOI or PubMed ID), fetch its forward citations and references via Semantic Scholar. "
                "Use to follow the citation graph: 'what cites this paper', 'what does this paper build on'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "paper_id": {
                        "type": "string",
                        "description": "DOI (e.g. 10.1038/...) or PubMed ID (e.g. PMID:12345678 or just 12345678)",
                    },
                    "direction": {
                        "type": "string",
                        "enum": ["citations", "references", "both"],
                        "description": "'citations' = papers citing this · 'references' = papers this cites · 'both'",
                        "default": "both",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max per direction (1–20)",
                        "default": 10,
                    },
                },
                "required": ["paper_id"],
            },
        },
    },
    # --- Knowledge: folders (user-uploaded PDFs) ---
    {
        "type": "function",
        "function": {
            "name": "list_folders",
            "description": (
                "List all user folders and their uploaded PDFs. "
                "Use at the start of a session (or when the user says 'check my folder', "
                "'read the paper I uploaded', 'what's in my references') to discover available reading material."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_pdf",
            "description": (
                "Read the full extracted text of a PDF uploaded to a folder. "
                "Use when the user references a paper in their folder, or after list_folders when a PDF is relevant. "
                "Text can be long — summarise or quote selectively rather than dumping the whole content."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "folder_id": {"type": "string", "description": "Folder id returned by list_folders (e.g. fld-ab12cd34)"},
                    "filename": {"type": "string", "description": "PDF filename inside that folder"},
                    "max_chars": {"type": "integer", "description": "Truncate to at most this many characters (default 20000)", "default": 20000},
                },
                "required": ["folder_id", "filename"],
            },
        },
    },
    # --- Web research ---
    {
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": (
                "Fetch any HTTP(S) URL and return readable text content (HTML is stripped, scripts/styles removed). "
                "Use to read documentation pages, supplementary data, gene database entries, blog posts, protocol wikis, etc. "
                "Only follow URLs the user provides or URLs returned by other tools — do not invent URLs."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Full http(s) URL"},
                    "max_chars": {"type": "integer", "description": "Truncate to this many characters (default 8000)", "default": 8000},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "pubmed_search",
            "description": (
                "Search PubMed (NCBI E-utilities) for biomedical literature. Returns titles, authors, abstracts, "
                "journal, year, and PubMed ID. More biomedically focused than search_literature. Use for clinical "
                "/ pathology / oncology questions where MEDLINE coverage matters."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query — supports PubMed syntax, e.g. 'MSI[Title] AND endometrial[Title]'"},
                    "limit": {"type": "integer", "description": "Number of papers to return (1–20)", "default": 10},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "download_paper_pdf",
            "description": (
                "Download a PDF from a URL and save it into a folder (creating the folder if needed). "
                "Returns the saved filename and the first ~5k characters of extracted text. "
                "Use when the user wants to ingest a paper found via search_literature / pubmed_search."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Direct PDF URL (must end in .pdf or return Content-Type: application/pdf)"},
                    "folder_id": {"type": "string", "description": "Existing folder id (e.g. fld-ab12cd34). If omitted, a new folder is created."},
                    "folder_name": {"type": "string", "description": "Name to use if a new folder is created (default 'References')"},
                    "filename": {"type": "string", "description": "Optional override filename (defaults to URL basename)"},
                },
                "required": ["url"],
            },
        },
    },
    # --- Genomics ---
    {
        "type": "function",
        "function": {
            "name": "parse_genomic_file",
            "description": (
                "Parse a genomic data file (MAF, VCF, clinical XML, or TSV/CSV) and return a structured summary. "
                "For MAF: variant counts, top mutated genes, variant classifications, per-sample TMB. "
                "For VCF: variant types, FILTER stats, chromosome distribution. "
                "For clinical XML: all clinical fields (msi_status, histological_type, vital_status, etc.). "
                "Use query parameter to filter: 'summary' for overview, gene name (e.g. 'TP53') for gene-specific variants, "
                "'variants' for a raw variant table."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Absolute path to the genomic file (.maf, .maf.gz, .vcf, .vcf.gz, .xml, .tsv, .csv)"},
                    "file_type": {"type": "string", "description": "File type: maf | vcf | clinical_xml | tsv | auto (auto-detect from extension)", "default": "auto"},
                    "query": {"type": "string", "description": "Query mode: 'summary' (overview), gene name (e.g. 'EGFR'), field name, or 'variants' (raw table)", "default": "summary"},
                    "sample_id": {"type": "string", "description": "Optional: filter to a specific sample/case barcode"},
                    "limit": {"type": "integer", "description": "Max rows to return in detail/variants mode", "default": 50},
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_mutations",
            "description": (
                "Query mutation data across a cohort of MAF files. "
                "Answer questions like 'which patients have EGFR mutations?', "
                "'what are the top mutated genes?', 'show TP53 frameshift variants'. "
                "Scans all .maf/.maf.gz files in the specified directory."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "genomic_dir": {"type": "string", "description": "Directory containing MAF files"},
                    "gene": {"type": "string", "description": "Gene symbol (e.g. 'EGFR', 'TP53'). Use '*' for all genes.", "default": "*"},
                    "variant_class": {"type": "string", "description": "Filter by variant classification (e.g. Missense_Mutation, Frame_Shift_Del). Empty = all", "default": ""},
                    "min_frequency": {"type": "number", "description": "Min mutation frequency across samples (0-1). E.g. 0.05 = ≥5% of samples", "default": 0.0},
                    "output_format": {"type": "string", "description": "Output: summary | table | gene_list", "default": "summary"},
                },
                "required": ["genomic_dir"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compute_tmb",
            "description": (
                "Compute Tumor Mutational Burden (TMB) from MAF files for a cohort. "
                "Returns per-sample TMB values, distribution stats, and high/medium/low classification. "
                "TMB = nonsynonymous variants / exome size (default 30 Mb). "
                "TMB is an FDA-approved biomarker for immunotherapy response."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "maf_dir": {"type": "string", "description": "Directory containing MAF files"},
                    "exome_size_mb": {"type": "number", "description": "Capture region size in Mb for normalization (default: 30 for WXS)", "default": 30.0},
                    "thresholds": {
                        "type": "object",
                        "description": "TMB classification thresholds: {high: 10, medium: 6, low: 0}",
                        "default": {"low": 0, "medium": 6, "high": 10},
                    },
                },
                "required": ["maf_dir"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "extract_labels_from_genomic",
            "description": (
                "Extract slide-level labels from genomic/clinical files for MIL training. "
                "Handles TCGA barcode resolution, patient deduplication (DX1 preferred), "
                "and produces a labels.csv mapping slide filenames to integer labels. "
                "Supported label types: msi_status (from clinical XML or TMB), "
                "mutation_status (mutant vs wildtype for a gene), "
                "tmb_class (high vs low TMB), clinical_field (any clinical attribute)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "genomic_dir": {"type": "string", "description": "Directory containing MAF/VCF/clinical files"},
                    "dataset_id": {"type": "string", "description": "Dataset ID to match slides against"},
                    "label_type": {"type": "string", "description": "Label type: msi_status | mutation_status | tmb_class | clinical_field"},
                    "label_spec": {
                        "type": "object",
                        "description": "Type-specific params. mutation_status: {gene: 'EGFR'}. tmb_class: {threshold: 10}. clinical_field: {field: 'histological_type', mapping: {'serous': 0, 'endometrioid': 1}}",
                        "default": {},
                    },
                    "output_path": {"type": "string", "description": "Output CSV path. Default: ~/.pathclaw/datasets/{dataset_id}/labels.csv"},
                },
                "required": ["genomic_dir", "dataset_id", "label_type"],
            },
        },
    },
    # --- cBioPortal ---
    {
        "type": "function",
        "function": {
            "name": "query_cbioportal",
            "description": (
                "Query cBioPortal for cancer genomics data. Supports mutations, clinical data, "
                "copy number alterations, and MSI scores. Works with TCGA project names (e.g. TCGA-UCEC) "
                "or cBioPortal study IDs (e.g. ucec_tcga_pan_can_atlas_2018). "
                "Great for getting MSI status, subtype labels, mutation data, and clinical attributes "
                "without downloading raw files."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "study_id": {"type": "string", "description": "cBioPortal study ID or TCGA project name (e.g. 'TCGA-UCEC', 'ucec_tcga_pan_can_atlas_2018')"},
                    "data_type": {"type": "string", "description": "Data type: mutations | clinical | cna | msi_scores", "default": "clinical"},
                    "gene_list": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Genes to query (for mutations/cna). E.g. ['EGFR', 'TP53', 'KRAS']",
                    },
                    "clinical_attributes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Clinical attributes to fetch (for clinical type). E.g. ['MSI_STATUS', 'SUBTYPE', 'OS_STATUS']",
                    },
                },
                "required": ["study_id"],
            },
        },
    },
    # --- Survival ---
    {
        "type": "function",
        "function": {
            "name": "run_survival_analysis",
            "description": (
                "Run Kaplan-Meier survival analysis on clinical data, optionally stratified by slide labels. "
                "Extracts survival fields (days_to_death, vital_status) from clinical XML/TSV, "
                "merges with labels.csv, computes KM curves, log-rank test, and median survival. "
                "Saves a KM plot and results JSON. Requires lifelines for full analysis (falls back to summary without it)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "clinical_dir": {"type": "string", "description": "Directory with clinical XML/TSV files"},
                    "dataset_id": {"type": "string", "description": "Dataset ID (to auto-find labels.csv for stratification)"},
                    "labels_path": {"type": "string", "description": "Path to labels.csv (alternative to dataset_id)"},
                    "group_column": {"type": "string", "description": "Column in labels.csv to stratify by", "default": "label_name"},
                    "output_dir": {"type": "string", "description": "Where to save KM plot and results"},
                },
                "required": ["clinical_dir"],
            },
        },
    },
    # --- Multi-omic ---
    {
        "type": "function",
        "function": {
            "name": "build_multi_omic_labels",
            "description": (
                "Build a combined multi-omic feature matrix from multiple data sources. "
                "Merges mutation data (MAF), clinical attributes (TSV), MIL model predictions, "
                "and existing labels into a unified CSV. Useful for multi-label training, "
                "survival analysis, or biomarker discovery."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset_id": {"type": "string", "description": "Dataset ID for default output path"},
                    "sources": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "type": {"type": "string", "description": "Source type: maf | clinical | model_predictions | labels"},
                                "path": {"type": "string", "description": "File or directory path"},
                                "features": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Feature names to extract (genes for MAF, fields for clinical)",
                                },
                            },
                            "required": ["type", "path"],
                        },
                        "description": "List of data sources to merge",
                    },
                    "output_path": {"type": "string", "description": "Output CSV path. Default: datasets/{id}/multi_omic_labels.csv"},
                },
                "required": ["dataset_id", "sources"],
            },
        },
    },
    # --- Visualization ---
    {
        "type": "function",
        "function": {
            "name": "generate_oncoplot",
            "description": (
                "Generate an oncoplot (mutation landscape) from MAF files. "
                "Shows top-N mutated genes across all samples with variant classification "
                "color-coding. Saves a PNG plot and returns text summary."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "maf_dir": {"type": "string", "description": "Directory containing MAF files"},
                    "top_n": {"type": "integer", "description": "Number of top mutated genes to show", "default": 20},
                    "output_path": {"type": "string", "description": "Where to save the plot (default: auto)"},
                    "title": {"type": "string", "description": "Plot title", "default": "Mutation Landscape"},
                    "min_frequency": {"type": "number", "description": "Min mutation frequency to include a gene (0-1)", "default": 0.0},
                },
                "required": ["maf_dir"],
            },
        },
    },
    # --- Gene Expression ---
    {
        "type": "function",
        "function": {
            "name": "parse_gene_expression",
            "description": (
                "Parse gene expression files (STAR counts, HTSeq counts, FPKM). "
                "Returns summary statistics, top expressed genes, or query specific genes. "
                "Supports .tsv, .tsv.gz, .counts formats from GDC/TCGA."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to expression file"},
                    "query": {"type": "string", "description": "'summary' for overview, gene name for specific gene, 'top' for highest expressed", "default": "summary"},
                    "gene_list": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of genes to query (e.g. ['EGFR', 'TP53'])",
                    },
                    "limit": {"type": "integer", "description": "Max genes to return", "default": 50},
                },
                "required": ["file_path"],
            },
        },
    },
    # --- Biomarker Discovery ---
    {
        "type": "function",
        "function": {
            "name": "biomarker_discovery",
            "description": (
                "Run biomarker discovery analysis: find genes differentially mutated between groups "
                "(mutation enrichment) or correlate MIL attention scores with mutations "
                "(attention-gene correlation). Helps identify molecular features that distinguish "
                "morphological subtypes."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "maf_dir": {"type": "string", "description": "Directory with MAF files"},
                    "labels_path": {"type": "string", "description": "Path to labels.csv with group assignments"},
                    "dataset_id": {"type": "string", "description": "Dataset ID (for attention correlation)"},
                    "experiment_id": {"type": "string", "description": "Experiment ID (for attention correlation)"},
                    "analysis_type": {
                        "type": "string",
                        "description": "Analysis type: mutation_enrichment | attention_correlation",
                        "default": "mutation_enrichment",
                    },
                    "gene_list": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Specific genes to analyze",
                    },
                    "group_column": {"type": "string", "description": "Stratification column in labels.csv", "default": "label_name"},
                },
                "required": ["maf_dir", "labels_path"],
            },
        },
    },
    # --- Workspace (user code + cloned repos) ---
    {
        "type": "function",
        "function": {
            "name": "list_workspace_files",
            "description": (
                "List files in the user's workspace (~/.pathclaw/workspace/). "
                "Contains user-authored scripts (user_code/), cloned repos (repos/), "
                "plugin source (plugins/), and paper-implemented methods (methods/). "
                "Call this before trying to import or run a user script."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "subdir": {"type": "string", "description": "Optional sub-path (e.g. 'user_code' or 'repos/CLAM'). Empty = whole workspace.", "default": ""},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_workspace_file",
            "description": "Read a text file from the workspace. Use to inspect user code before applying it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Workspace-relative path, e.g. 'user_code/my_preprocess.py'"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_workspace_file",
            "description": (
                "Create or overwrite a text file in the workspace. Use this to draft new plugins "
                "(plugins/<id>.py), paper methodologies (methods/<name>.py), or fix bugs in user code. "
                "Must stay under 2 MB. Directories are created automatically."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Workspace-relative path"},
                    "content": {"type": "string", "description": "Full file contents"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_workspace_file",
            "description": "Delete a single workspace file (directories are rejected for safety).",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "clone_repo",
            "description": (
                "Clone a public git repo into ~/.pathclaw/workspace/repos/<name>/. "
                "Only github.com, gitlab.com, huggingface.co, codeberg.org, bitbucket.org are allowed. "
                "Does a shallow clone (--depth 1). After cloning, use list_workspace_files + "
                "read_workspace_file to inspect README.md and top-level Python files."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "https://github.com/<org>/<repo>.git or similar"},
                    "name": {"type": "string", "description": "Optional override; defaults to the repo slug"},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_plugins",
            "description": (
                "List all PathClaw plugins (patch-embed, MIL head, loss, augment, method). "
                "Each entry shows id, kind, installed status, and default_config. "
                "Use this before proposing to toggle or install one."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "register_hf_backbone",
            "description": (
                "Register a custom HuggingFace foundation model as a feature-extraction backbone. "
                "Typically used after reading the model card to infer dim and patch size. "
                "Once registered, use the id with start_feature_extraction."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Short id, e.g. 'virchow2_1b'"},
                    "hf_model_id": {"type": "string", "description": "HuggingFace repo id, e.g. 'paige-ai/Virchow2'"},
                    "timm_model": {"type": "string", "description": "timm model name (often same as hf id); blank to default"},
                    "dim": {"type": "integer", "description": "Feature dimension from the model card"},
                    "patch_px": {"type": "integer", "description": "Expected input patch size in pixels (default 224)"},
                    "magnification": {"type": "integer", "description": "Expected magnification (default 20)"},
                    "gated": {"type": "boolean", "description": "True if the HF repo requires a token"},
                },
                "required": ["id", "hf_model_id", "dim"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_backbones",
            "description": "List all available feature-extraction backbones (built-in + user-registered).",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "register_plugin",
            "description": (
                "Register a new plugin in the user registry. The `import_path` must be of the "
                "form 'module.path:attr' and must be importable (e.g. a file under "
                "~/.pathclaw/workspace/plugins/ that has been written via write_workspace_file "
                "and is on the Python path, or a module from a cloned repo). Kind must be one of "
                "patch_embed|mil|loss|augment|method."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Unique slug, e.g. 'clam_attn'"},
                    "name": {"type": "string", "description": "Human-readable name"},
                    "kind": {"type": "string", "enum": ["patch_embed", "mil", "loss", "augment", "method"]},
                    "import_path": {"type": "string", "description": "module.path:attr"},
                    "description": {"type": "string"},
                    "applies_to": {"type": "array", "items": {"type": "string"}, "description": "e.g. ['mil']"},
                    "default_config": {"type": "object", "description": "Default knobs; merged with user overrides"},
                    "source": {"type": "string", "description": "Optional provenance, e.g. 'pip:mammoth-moe' or 'local:plugins/foo.py'"},
                },
                "required": ["id", "name", "kind", "import_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_repo",
            "description": (
                "Produce a structured summary of a cloned repo under workspace/repos/<name>/. "
                "Returns: language mix, top-level file tree (2 levels), README excerpt, any setup.py/"
                "pyproject.toml/requirements.txt snippets, and a list of candidate Python modules "
                "likely to contain model code. Use this before proposing a plugin integration."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Repo directory under workspace/repos/"},
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ask_user",
            "description": (
                "Surface a specific, blocking question to the user when an ambiguity prevents "
                "completing the current task. Use this INSTEAD of silently guessing when the paper, "
                "config, or repo is genuinely ambiguous. The question should be concrete and "
                "answerable in one line (e.g. 'The paper's attention uses ReLU OR GELU — which?')."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "One concrete question"},
                    "context": {"type": "string", "description": "Short context so the user understands why you're asking"},
                },
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_plugin_config",
            "description": (
                "Update a plugin's `default_config` (the knobs it ships with). Use this when the "
                "user says things like 'lower cellpose's flow_threshold to 0.2' or 'set mammoth's "
                "num_experts to 16'. The merged config is stored as a user override and survives "
                "restarts. Pass the FULL new config object — partial updates require reading the "
                "current config first via list_plugins. Built-ins are shadowed, never overwritten."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Plugin id, e.g. 'cellpose', 'mammoth'"},
                    "default_config": {"type": "object", "description": "Complete new default_config dict"},
                },
                "required": ["id", "default_config"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_cellpose_segmentation",
            "description": (
                "Run cellpose nuclei/cell instance segmentation on a dataset's slides. Reads the "
                "cellpose plugin's current `default_config` (override per-call via `config_overrides`). "
                "Writes per-patch instance-mask PNGs and a counts CSV under "
                "~/.pathclaw/sessions/<sid>/cellpose/<dataset_id>/<slide_stem>/. "
                "Knobs you can change: model_type (cpsam|nuclei|cyto3), diameter, flow_threshold, "
                "cellprob_threshold, niter, channels, min_size, tile_norm, gpu. "
                "Use update_plugin_config to persist new defaults across runs."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset_id": {"type": "string"},
                    "max_slides": {"type": "integer", "description": "Cap on slides to process (default 5)"},
                    "max_patches_per_slide": {"type": "integer", "description": "Cap on patches per slide (default 64)"},
                    "config_overrides": {"type": "object", "description": "Per-call overrides to the plugin's default_config"},
                },
                "required": ["dataset_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "smoke_test_plugin",
            "description": (
                "Smoke-test a plugin file under workspace/plugins/ or workspace/methods/. "
                "Imports the module, calls `build(in_dim, embed_dim, config)` with dummy args, "
                "runs a forward pass on a random (B=1, N=64, in_dim) tensor, and verifies the "
                "output shape is (B, N, embed_dim) for patch_embed kind. Returns traceback on "
                "failure so the agent can self-correct. Use this BEFORE calling register_plugin."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "import_path": {"type": "string", "description": "module.path:attr (e.g. 'workspace.methods.gamil:build')"},
                    "kind": {"type": "string", "enum": ["patch_embed", "mil", "loss", "augment", "method"]},
                    "in_dim": {"type": "integer", "description": "Input feature dim (default 1024)"},
                    "embed_dim": {"type": "integer", "description": "Output embed dim (default 512)"},
                    "config": {"type": "object", "description": "Plugin config dict"},
                },
                "required": ["import_path", "kind"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "implement_from_paper",
            "description": (
                "Orchestration helper for implementing a paper's methodology as a plugin. "
                "This tool does NOT write code; the agent does. The tool returns a workflow "
                "prompt reminding the agent of the plugin contract and the exact sequence of "
                "tool calls required: (1) read_pdf the paper, (2) write_workspace_file to "
                "plugins/<method_name>.py implementing `build(in_dim, embed_dim, config)`, "
                "(3) smoke_test_plugin, (4) register_plugin on success. If smoke test fails, "
                "the agent must self-correct (≤2 retries) or call ask_user."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "method_name": {"type": "string", "description": "Plugin id, e.g. 'gamil'"},
                    "paper_path": {"type": "string", "description": "Workspace-relative or absolute path to the PDF"},
                    "target_kind": {"type": "string", "enum": ["patch_embed", "mil", "loss", "method"]},
                },
                "required": ["method_name", "target_kind"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "make_plot",
            "description": (
                "Generate a new plot for a completed training experiment from its predictions.csv "
                "+ history.json. Built-in kinds: roc_curve, pr_curve, per_class_auroc, calibration, "
                "confusion_matrix, prediction_histogram. For kind='custom', supply `spec` as a "
                "matplotlib snippet; it runs with variables `history`, `metrics`, `predictions` "
                "(pandas DataFrame), `num_classes` pre-loaded and must assign `fig`. "
                "The PNG is saved under the experiment's plots/ directory."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "experiment_id": {"type": "string", "description": "Training job id (e.g. 'train-abc123')"},
                    "kind": {
                        "type": "string",
                        "enum": ["roc_curve", "pr_curve", "per_class_auroc", "calibration",
                                "confusion_matrix", "prediction_histogram", "custom"],
                    },
                    "spec": {"type": "string", "description": "Matplotlib snippet when kind='custom'"},
                    "title": {"type": "string", "description": "Optional plot title"},
                },
                "required": ["experiment_id", "kind"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Tool executor
# ---------------------------------------------------------------------------

def _fmt_citation_row(p: dict, idx: int) -> str:
    title = p.get("title") or "(no title)"
    year = p.get("year") or "?"
    cites = p.get("citationCount") or 0
    authors = ", ".join(a.get("name", "") for a in (p.get("authors") or [])[:3])
    if len(p.get("authors") or []) > 3:
        authors += " et al."
    ext = p.get("externalIds") or {}
    links = []
    if ext.get("PubMed"):
        links.append(f"[PubMed](https://pubmed.ncbi.nlm.nih.gov/{ext['PubMed']}/)")
    if ext.get("DOI"):
        links.append(f"[DOI](https://doi.org/{ext['DOI']})")
    if p.get("url"):
        links.append(f"[SemanticScholar]({p['url']})")
    link_str = " · ".join(links) if links else ""
    return f"  {idx}. **{title}** ({year}, {cites} cites) — {authors}\n     {link_str}"


async def _execute_tool(name: str, arguments: dict[str, Any], session_id: str = "") -> str:
    # Referential guardrail — catches fabricated ids before the tool body runs.
    # Returns a model-readable error so the LLM self-corrects on the next turn
    # instead of polling a non-existent job or dispatching against a phantom dataset.
    err = validate_tool_args(name, arguments)
    if err is not None:
        return err

    base = _get_backend_base()
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            # --- Session rename ---
            if name == "rename_session":
                if not session_id:
                    return "Error: session context unavailable."
                raw_slug = (arguments.get("slug") or "").strip()
                slug = _slugify(raw_slug)
                if raw_slug and not slug:
                    return f"Error: slug '{raw_slug}' has no letters or digits after normalization."
                p = _chat_path(session_id)
                if not p.exists():
                    # Create a stub so rename on a brand-new session works
                    now = _time_mod.time()
                    p.write_text(json.dumps({
                        "session_id": session_id, "title": "New Session",
                        "created_at": now, "updated_at": now, "messages": [],
                    }, indent=2))
                data = json.loads(p.read_text())
                if slug:
                    for other in CHATS_DIR.glob("*.json"):
                        if other == p:
                            continue
                        try:
                            od = json.loads(other.read_text())
                        except Exception:
                            continue
                        if od.get("slug") == slug:
                            return f"Error: slug '{slug}' already used by session {od.get('session_id')}."
                    data["slug"] = slug
                title = (arguments.get("title") or "").strip()
                if title:
                    data["title"] = title[:200]
                data["updated_at"] = _time_mod.time()
                p.write_text(json.dumps(data, indent=2))
                return f"Session renamed. slug={data.get('slug')!r} title={data.get('title')!r}"

            # --- Memory (per-session) ---
            elif name == "remember_fact":
                if not session_id:
                    return "Error: session context unavailable for memory."
                key = arguments.get("key", "").strip()
                value = arguments.get("value", "").strip()
                if not key or not value:
                    return "Error: key and value are required."
                _save_memory(session_id, key, value)
                return f"Remembered: {key} = {value}"

            elif name == "recall_facts":
                if not session_id:
                    return "Error: session context unavailable."
                mem = _load_memory(session_id)
                if not mem:
                    return "No facts in this session's memory yet."
                return "\n".join(f"- {k}: {v}" for k, v in mem.items())

            elif name == "write_note":
                if not session_id:
                    return "Error: session context unavailable for note."
                topic = arguments.get("topic", "").strip()
                content = arguments.get("content", "").strip()
                if not topic or not content:
                    return "Error: topic and content are required."
                path = _append_session_note(session_id, topic, content)
                return f"Note saved under '{topic}' → {path}"

            elif name == "read_notes":
                if not session_id:
                    return "Error: session context unavailable."
                text = _read_session_notes(session_id)
                return text or "(notebook is empty — use write_note to add entries)"

            # --- Task plan ---
            elif name == "create_task_plan":
                if not session_id:
                    return "Error: session context unavailable for task plan."
                from pathclaw.api.routes import tasks as _tasks
                tasks_in = arguments.get("tasks") or []
                if not tasks_in:
                    return "Error: tasks list is empty."
                now = _tasks._now_iso()
                new_tasks = []
                for i, t in enumerate(tasks_in, start=1):
                    title = (t.get("title") or "").strip()
                    if not title:
                        continue
                    new_tasks.append({
                        "id": i,
                        "title": title[:120],
                        "description": (t.get("description") or "")[:400],
                        "status": "pending",
                        "pause_after": bool(t.get("pause_after", False)),
                        "created_at": now,
                        "updated_at": now,
                    })
                if not new_tasks:
                    return "Error: all tasks were empty."
                plan = {
                    "session_id": session_id,
                    "created_at": now,
                    "updated_at": now,
                    "tasks": new_tasks,
                }
                _tasks.save_plan(plan)
                lines = [f"Plan created ({len(new_tasks)} tasks):"]
                for t in new_tasks:
                    mark = " (pause_after)" if t["pause_after"] else ""
                    lines.append(f"  [ ] {t['id']}. {t['title']}{mark}")
                lines.append("Start with task 1: call update_task_status(task_id=1, status=\"in_progress\") then do the work.")
                return "\n".join(lines)

            elif name == "update_task_status":
                if not session_id:
                    return "Error: session context unavailable for task plan."
                from pathclaw.api.routes import tasks as _tasks
                task_id = arguments.get("task_id")
                status_val = (arguments.get("status") or "").strip()
                valid = {"pending", "in_progress", "completed", "skipped"}
                if status_val not in valid:
                    return f"Error: status must be one of {sorted(valid)}"
                plan = _tasks.load_plan(session_id)
                if not plan.get("tasks"):
                    return "Error: no plan exists. Call create_task_plan first."
                now = _tasks._now_iso()
                hit = None
                for t in plan["tasks"]:
                    if t["id"] == task_id:
                        t["status"] = status_val
                        t["updated_at"] = now
                        hit = t
                        break
                if not hit:
                    return f"Error: task {task_id} not in plan. Existing ids: {[t['id'] for t in plan['tasks']]}"
                plan["updated_at"] = now
                _tasks.save_plan(plan)
                remaining = [t for t in plan["tasks"] if t["status"] == "pending"]
                if status_val == "completed":
                    if hit.get("pause_after"):
                        return f"Task {task_id} '{hit['title']}' → completed. pause_after=True — stop and wait for user input before continuing."
                    if remaining:
                        nxt = remaining[0]
                        return f"Task {task_id} '{hit['title']}' → completed. Next: task {nxt['id']} '{nxt['title']}' — call update_task_status(task_id={nxt['id']}, status=\"in_progress\") and start."
                    return f"Task {task_id} '{hit['title']}' → completed. All tasks done — summarize for the user."
                return f"Task {task_id} '{hit['title']}' → {status_val}."

            elif name == "get_task_plan":
                if not session_id:
                    return "Error: session context unavailable for task plan."
                from pathclaw.api.routes import tasks as _tasks
                plan = _tasks.load_plan(session_id)
                if not plan.get("tasks"):
                    return "No active plan. Use create_task_plan to start one."
                lines = [f"Plan ({len(plan['tasks'])} tasks):"]
                for t in plan["tasks"]:
                    mark = {"completed": "[x]", "in_progress": "[~]", "pending": "[ ]", "skipped": "[-]"}.get(
                        t.get("status", "pending"), "[ ]"
                    )
                    pause = " (pause_after)" if t.get("pause_after") else ""
                    lines.append(f"  {mark} {t['id']}. {t['title']}{pause}")
                return "\n".join(lines)

            # --- Manuscript (LaTeX) ---
            elif name == "write_manuscript":
                if not session_id:
                    return "Error: session context unavailable for manuscript."
                fname = (arguments.get("filename") or "").strip()
                content = arguments.get("content") or ""
                mode = (arguments.get("mode") or "write").strip().lower()
                if not fname or not content:
                    return "Error: filename and content are required."
                try:
                    path = _safe_manuscript_path(session_id, fname)
                except ValueError as e:
                    return f"Error: {e}"
                path.parent.mkdir(parents=True, exist_ok=True)
                if mode == "append" and path.exists():
                    with path.open("a") as f:
                        f.write(content)
                else:
                    path.write_text(content)
                return f"Manuscript file '{fname}' saved ({path.stat().st_size} bytes) → {path}"

            elif name == "read_manuscript":
                if not session_id:
                    return "Error: session context unavailable for manuscript."
                fname = (arguments.get("filename") or "").strip()
                if not fname:
                    files = _list_manuscript_files(session_id)
                    if not files:
                        return "(manuscript project is empty — use write_manuscript to create main.tex)"
                    lines = [f"Manuscript files ({len(files)}):"]
                    for f in files:
                        lines.append(f"- {f['name']} ({f['size']} bytes)")
                    return "\n".join(lines)
                try:
                    path = _safe_manuscript_path(session_id, fname)
                except ValueError as e:
                    return f"Error: {e}"
                if not path.exists():
                    return f"Error: '{fname}' not found in manuscript project."
                max_chars = int(arguments.get("max_chars", 8000))
                text = path.read_text(errors="replace")
                if len(text) > max_chars:
                    return f"{fname} (truncated, {len(text):,} chars total, first {max_chars}):\n\n{text[:max_chars]}"
                return f"{fname} ({len(text)} chars):\n\n{text}"

            elif name == "compile_manuscript":
                if not session_id:
                    return "Error: session context unavailable for manuscript."
                main = (arguments.get("main_file") or "main.tex").strip()
                result = _compile_latex(session_id, main)
                if result["status"] == "ok":
                    return f"Compiled with {result['compiler']} → {result['pdf']}\n\n(open in the Manuscript tab to view)"
                log = result.get("log", "")
                comp = result.get("compiler", "(none)")
                return f"Compilation FAILED with {comp}.\nLog tail:\n{log}"

            # --- System & Config ---
            elif name == "system_status":
                resp = await client.get(f"{base}/api/status")
                d = resp.json()
                gpu = d["gpu"]
                st = d["storage"]
                gpu_str = f"{gpu['name']} ×{gpu['count']}" if gpu["available"] else "No GPU"
                return (
                    f"GPU: {gpu_str}\n"
                    f"Storage: {st['free_gb']} GB free / {st['total_gb']} GB total\n"
                    f"Data dir: {d['data_dir']}"
                )

            elif name == "get_config":
                resp = await client.get(f"{base}/api/config")
                d = resp.json()
                return (
                    f"HuggingFace token: {'✓ Set' if d.get('huggingface_token_set') else '✗ Not set'}\n"
                    f"GDC token: {'✓ Set' if d.get('gdc_token_set') else '✗ Not set'}\n"
                    f"Data directory: {d.get('data_dir', 'unknown')}\n"
                    f"Onboarding: {'complete' if d.get('onboarding_complete') else 'pending'}"
                )

            # --- Datasets ---
            elif name == "list_datasets":
                params = {"session_id": session_id} if session_id else {}
                resp = await client.get(f"{base}/api/datasets", params=params)
                d = resp.json()
                if not d["datasets"]:
                    return "No datasets registered yet. Use register_dataset to add one."
                lines = [f"• {x['name']} (ID: {x['id']}) — {x['slide_count']} slides, {x['total_size_mb']} MB"
                         for x in d["datasets"]]
                return "\n".join(lines)

            elif name == "register_dataset":
                if session_id:
                    arguments = {**arguments, "session_id": session_id}
                resp = await client.post(f"{base}/api/datasets", json=arguments)
                d = resp.json()
                return (
                    f"Dataset registered: {d.get('name')}\n"
                    f"ID: {d.get('id')}\n"
                    f"Slides found: {d.get('slide_count', 0)}\n"
                    f"Total size: {d.get('total_size_mb', 0)} MB"
                )

            elif name == "list_dataset_slides":
                ds_id = arguments["dataset_id"]
                limit = int(arguments.get("limit", 10))
                offset = int(arguments.get("offset", 0))
                resp = await client.get(f"{base}/api/datasets/{ds_id}/slides")
                if resp.status_code != 200:
                    return f"Failed: {resp.status_code} {resp.text[:200]}"
                d = resp.json()
                slides = d.get("slides", [])[offset: offset + limit]
                if not slides:
                    return f"No slides found in dataset {ds_id} (offset={offset})."
                lines = [f"Dataset {ds_id}: showing {len(slides)} of {len(d.get('slides', []))} slides"]
                for s in slides:
                    # Each slide entry has 'path' (absolute) and 'filename'
                    p = s.get("path") or s.get("filename", "")
                    size = s.get("size_mb")
                    lines.append(f"  {p}" + (f"  ({size} MB)" if size else ""))
                return "\n".join(lines)

            elif name == "get_dataset_profile":
                ds_id = arguments["dataset_id"]
                resp = await client.get(f"{base}/api/datasets/{ds_id}/profile")
                d = resp.json()
                lines = [f"Dataset: {d.get('name', ds_id)}"]
                if "class_distribution" in d:
                    lines.append("Class distribution:")
                    for cls, cnt in d["class_distribution"].items():
                        lines.append(f"  {cls}: {cnt} slides")
                if "label_coverage" in d:
                    lines.append(f"Label coverage: {d['label_coverage']:.1%}")
                return "\n".join(lines)

            # --- GDC ---
            elif name == "search_gdc":
                resp = await client.post(f"{base}/api/gdc/search", json=arguments, timeout=60.0)
                d = resp.json()
                total = d.get("total", 0)
                files = d.get("files", [])
                # Cache so download_gdc can resolve filter_pattern without the
                # model having to copy UUIDs out of a long text blob.
                _GDC_SEARCH_CACHE[session_id or "__global__"] = list(files)

                # Zero-result recovery: GDC queries fail silently when the data_type
                # is incompatible with a filter (e.g. `experimental_strategy=Diagnostic
                # Slide` on a MAF search). Return a structured retry hint so the LLM
                # can self-correct on the next round instead of giving up.
                if total == 0:
                    return _gdc_zero_result_hint(arguments)

                # Auto-paginate: if more files exist than retrieved, fetch all IDs
                all_ids: list[str] = [fi["file_id"] for fi in files]
                if total > len(files) and total <= 2000:
                    paginate_args = {**arguments, "limit": total}
                    try:
                        r2 = await client.post(f"{base}/api/gdc/search", json=paginate_args, timeout=120.0)
                        files2 = r2.json().get("files", [])
                        all_ids = [fi["file_id"] for fi in files2]
                        files = files2
                    except Exception:
                        pass  # fall back to partial set

                # Compute sizes (API returns bytes)
                sizes_bytes = [fi.get("file_size") or 0 for fi in files]
                total_gb = sum(sizes_bytes) / 1e9
                sizes_known = any(s > 0 for s in sizes_bytes)

                project = arguments.get("project", "?")
                data_type = arguments.get("data_type", "Slide Image")
                lines = [
                    f"Found {total} {data_type} files in {project}.",
                    f"Retrieved: {len(all_ids)} file IDs"
                    + (f" | Est. total size: ~{total_gb:.1f} GB" if sizes_known else " | (file sizes not available from this endpoint — typical slide ~1.2 GB each)"),
                ]

                # Sample listing (first 5)
                for fi in files[:5]:
                    size_mb = (fi.get("file_size") or 0) / 1e6
                    size_str = f"{size_mb:.0f} MB" if size_mb > 0 else "size TBD"
                    lines.append(f"  • {fi.get('file_name', fi.get('file_id'))} [{size_str}]")
                if len(files) > 5:
                    lines.append(f"  … and {len(files) - 5} more")

                lines.append(f"\nfile_ids ({len(all_ids)} total): {json.dumps(all_ids)}")
                return "\n".join(lines)

            elif name == "download_gdc":
                # Resolve file_ids from the last search_gdc cache if the model
                # passed a filter_pattern / max_count instead of explicit ids.
                # This lets gemma4 avoid copying UUIDs out of a text blob.
                file_ids = arguments.get("file_ids") or []
                pattern = arguments.get("filter_pattern")
                max_count = arguments.get("max_count")
                if (not file_ids or pattern) and (pattern or max_count):
                    import fnmatch
                    cached = _GDC_SEARCH_CACHE.get(session_id or "__global__", [])
                    if not cached:
                        return (
                            "ERROR: download_gdc needs a prior search_gdc in this session "
                            "before filter_pattern/max_count can resolve. Call search_gdc first."
                        )
                    matched = cached
                    if pattern:
                        pat = pattern if any(c in pattern for c in "*?[]") else f"*{pattern}*"
                        matched = [fi for fi in cached if fnmatch.fnmatch(fi.get("file_name", ""), pat)]
                    if max_count:
                        matched = matched[: int(max_count)]
                    if not matched:
                        return f"ERROR: filter_pattern={pattern!r} matched 0 of {len(cached)} cached files."
                    file_ids = [fi["file_id"] for fi in matched]
                    arguments = {**arguments, "file_ids": file_ids}
                    arguments.pop("filter_pattern", None)
                    arguments.pop("max_count", None)
                resp = await client.post(f"{base}/api/gdc/download", json=arguments, timeout=30.0)
                d = resp.json()
                return (
                    f"Download job started.\n"
                    f"Job ID: {d.get('job_id')}\n"
                    f"Files: {d.get('total_files')}\n"
                    f"Output dir: {d.get('output_dir')}\n"
                    f"Use gdc_job_status with job_id to track progress."
                )

            elif name == "gdc_job_status":
                job_id = arguments["job_id"]
                if not job_id.startswith("dl_"):
                    job_id = f"dl_{job_id}"
                resp = await client.get(f"{base}/api/gdc/jobs/{job_id}", timeout=10.0)
                d = resp.json()
                status = d.get("status", "unknown")
                done = d.get("done", 0)
                total = d.get("total", 0)
                pct = f"{100*done//total}%" if total else "0%"
                lines = [
                    f"Status: {status}",
                    f"Progress: {done}/{total} ({pct})",
                    f"Output: {d.get('output_dir', '?')}",
                ]
                if d.get("failed"):
                    lines.append(f"Failed: {len(d['failed'])} files")
                if d.get("message"):
                    lines.append(d["message"])
                return "\n".join(lines)

            # --- Preprocessing ---
            elif name == "start_preprocessing":
                resp = await client.post(f"{base}/api/preprocess/start", json=arguments)
                d = resp.json()
                return (
                    f"Preprocessing job launched.\n"
                    f"Job ID: {d['job_id']}\n"
                    f"Status: {d['status']}\n"
                    f"Use get_job_status(job_id='{d['job_id']}', job_type='preprocess') to monitor."
                )

            # --- Training ---
            elif name == "start_training":
                # Map flat tool args to nested TrainingConfig structure
                task_type = arguments.pop("task_type", "mil")
                mammoth_enabled = arguments.pop("mammoth_enabled", True)
                eval_strategy = arguments.pop("eval_strategy", "holdout")
                epochs = arguments.pop("epochs", 100 if task_type == "mil" else 50)
                lr = arguments.pop("lr", 1e-4)
                optimizer_name = arguments.pop("optimizer", "adam")
                payload = {
                    **arguments,
                    "task_type": task_type,
                    "mammoth": {"enabled": mammoth_enabled},
                    "training": {"epochs": epochs, "lr": lr, "optimizer": optimizer_name},
                    "evaluation": {"strategy": eval_strategy},
                    "session_id": session_id,
                }
                resp = await client.post(f"{base}/api/training/start", json=payload)
                d = resp.json()
                if "job_id" not in d:
                    return (
                        f"Training failed to start (HTTP {resp.status_code}): "
                        f"{d.get('detail', d)}"
                    )
                if task_type == "segmentation":
                    seg_model = arguments.get("seg_model", "seg_unet")
                    return (
                        f"Segmentation training launched.\n"
                        f"Job ID: {d['job_id']}\n"
                        f"Model: {seg_model}, Epochs: {epochs}\n"
                        f"Use get_job_status(job_id='{d['job_id']}', job_type='training') to monitor."
                    )
                return (
                    f"Training job launched.\n"
                    f"Job ID: {d['job_id']}\n"
                    f"Method: {arguments.get('mil_method', 'abmil')}, "
                    f"MAMMOTH: {'on' if mammoth_enabled else 'off'}, "
                    f"Epochs: {epochs}\n"
                    f"Use get_job_status(job_id='{d['job_id']}', job_type='training') to monitor."
                )

            elif name == "get_training_logs":
                jid = arguments["job_id"]
                resp = await client.get(f"{base}/api/training/{jid}/logs")
                d = resp.json()
                logs = d.get("logs", "No logs yet.")
                # Return last 1000 chars to keep context manageable
                return logs[-1000:] if len(logs) > 1000 else logs

            # --- Task queue ---
            elif name == "queue_task":
                task_type = arguments.get("task_type")
                payload = dict(arguments.get("payload", {}))
                if task_type == "training":
                    # Normalize flat agent-friendly keys → nested TrainingConfig shape
                    if "mammoth_enabled" in payload:
                        payload.setdefault("mammoth", {})["enabled"] = payload.pop("mammoth_enabled")
                    flat_train = {}
                    for k in ("epochs", "lr", "optimizer", "weight_decay", "scheduler", "early_stopping_patience"):
                        if k in payload:
                            flat_train[k] = payload.pop(k)
                    if flat_train:
                        payload.setdefault("training", {}).update(flat_train)
                    if "eval_strategy" in payload:
                        payload.setdefault("evaluation", {})["strategy"] = payload.pop("eval_strategy")
                    payload.setdefault("session_id", session_id)
                body = {
                    "task_type": task_type,
                    "payload": payload,
                    "session_id": session_id,
                    "note": arguments.get("note", ""),
                }
                resp = await client.post(f"{base}/api/queue/submit", json=body)
                d = resp.json()
                if "task_id" not in d:
                    return f"Queue submit failed (HTTP {resp.status_code}): {d.get('detail', d)}"
                return (
                    f"Task queued.\n"
                    f"Task ID: {d['task_id']}\n"
                    f"Type: {d['task_type']}  ·  Status: {d['status']}\n"
                    f"The queue worker will dispatch it when GPU resources are free. "
                    f"Use list_queue to check progress."
                )

            elif name == "list_queue":
                params = {}
                if arguments.get("status"):
                    params["status"] = arguments["status"]
                resp = await client.get(f"{base}/api/queue", params=params)
                d = resp.json()
                tasks = d.get("tasks", [])
                if not tasks:
                    return "Queue is empty."
                lines = [f"Queue: {len(tasks)} task(s)"]
                for t in tasks[-10:]:  # last 10
                    lines.append(
                        f"- {t['task_id']} [{t['task_type']}] {t['status']} "
                        f"dispatched={t.get('dispatched_job_id') or '—'}  "
                        f"note={t.get('note', '')[:40]}"
                    )
                return "\n".join(lines)

            # --- Evaluation ---
            elif name == "start_evaluation":
                resp = await client.post(f"{base}/api/eval/start", json=arguments)
                d = resp.json()
                return (
                    f"Evaluation job launched.\n"
                    f"Job ID: {d['job_id']}\n"
                    f"Use get_job_status(job_id='{d['job_id']}', job_type='eval') to monitor, "
                    f"then get_eval_metrics to retrieve results."
                )

            elif name == "get_eval_metrics":
                jid = arguments["job_id"]
                resp = await client.get(f"{base}/api/eval/{jid}/metrics")
                d = resp.json()
                metrics = d.get("metrics", {})
                lines = []
                for k, v in metrics.items():
                    if isinstance(v, float):
                        lines.append(f"  {k}: {v:.4f}")
                    else:
                        lines.append(f"  {k}: {v}")
                return "Evaluation metrics:\n" + ("\n".join(lines) if lines else "No metrics available yet.")

            elif name == "get_eval_plots":
                jid = arguments["job_id"]
                resp = await client.get(f"{base}/api/eval/{jid}/plots")
                d = resp.json()
                plots = d.get("plots", [])
                if not plots:
                    return "No plots generated yet."
                return "Generated plots:\n" + "\n".join(f"  • {p}" for p in plots)

            # --- IHC scoring ---
            elif name == "list_ihc_rules":
                resp = await client.get(f"{base}/api/ihc/rules")
                if resp.status_code != 200:
                    return f"Error listing IHC rules: HTTP {resp.status_code} {resp.text[:300]}"
                rules = resp.json().get("rules", [])
                lines = []
                for r in rules:
                    lines.append(
                        f"• {r['name']} ({r['marker']}) — compartment={r['compartment']}, "
                        f"agg={r['aggregation']}, DAB thr={r['dab_threshold']}. {r.get('notes','')}"
                    )
                return "Available IHC presets:\n" + "\n".join(lines)

            elif name == "score_ihc":
                payload = dict(arguments)
                if session_id:
                    payload.setdefault("session_id", session_id)
                resp = await client.post(f"{base}/api/ihc/score", json=payload, timeout=None)
                if resp.status_code != 200:
                    return f"IHC score failed: HTTP {resp.status_code} — {resp.text[:500]}"
                d = resp.json()
                lines = [
                    f"IHC scoring complete. Rule: {d.get('rule')}. Dataset: {d.get('dataset_id')}.",
                    f"  slides scored: {d.get('n_scored')}/{d.get('n_slides')} (failed {d.get('n_failed',0)})",
                ]
                for k in ("mean_score", "median_score", "min_score", "max_score"):
                    if k in d:
                        lines.append(f"  {k}: {d[k]:.3f}")
                if d.get("csv_path"):
                    lines.append(f"  CSV: {d['csv_path']}")
                return "\n".join(lines)

            elif name == "build_ihc_patch_labels":
                payload = dict(arguments)
                if session_id:
                    payload.setdefault("session_id", session_id)
                resp = await client.post(f"{base}/api/ihc/patch-labels", json=payload, timeout=None)
                if resp.status_code != 200:
                    return f"build_ihc_patch_labels failed: HTTP {resp.status_code} — {resp.text[:500]}"
                d = resp.json()
                lines = [
                    f"Patch-level labels generated. Rule: {d.get('rule')}. Dataset: {d.get('dataset_id')}.",
                    f"  slides: {d.get('n_slides')}, patches labeled: {d.get('n_patch_labels')}",
                    f"  patch CSV: {d.get('patch_csv')}",
                    f"  slide CSV: {d.get('slide_csv')}",
                    f"  note: {d.get('note','')}",
                ]
                return "\n".join(lines)

            # --- Job Status (unified) ---
            elif name == "get_job_status":
                jid = arguments["job_id"]
                jtype = arguments.get("job_type", "training")
                if jtype == "gdc" and not jid.startswith("dl_"):
                    jid = f"dl_{jid}"
                route_map = {
                    "preprocess": f"/api/preprocess/{jid}",
                    "training": f"/api/training/{jid}",
                    "eval": f"/api/eval/{jid}",
                    "features": f"/api/features/{jid}",
                    "lora": f"/api/training/lora/{jid}",
                    "gdc": f"/api/gdc/jobs/{jid}",
                }
                route = route_map.get(jtype, f"/api/training/{jid}")
                resp = await client.get(f"{base}{route}")
                d = resp.json()
                status = d.get("status", "unknown")
                progress = d.get("progress", 0)
                metrics = d.get("metrics", {})
                result = f"Job {jid}: {status} ({progress:.0%})"
                if metrics:
                    for k, v in metrics.items():
                        if isinstance(v, float):
                            result += f"\n  {k}: {v:.4f}"
                        else:
                            result += f"\n  {k}: {v}"
                if d.get("errors"):
                    result += "\nErrors: " + "; ".join(d["errors"])
                return result

            # --- Feature Extraction ---
            elif name == "start_feature_extraction":
                resp = await client.post(f"{base}/api/features/start", json=arguments)
                d = resp.json()
                return (
                    f"Feature extraction launched.\n"
                    f"Job ID: {d['job_id']}\n"
                    f"Backbone: {d['backbone']}\n"
                    f"Use get_job_status(job_id='{d['job_id']}', job_type='features') to monitor."
                )

            # --- LoRA fine-tuning ---
            elif name == "start_lora_finetuning":
                resp = await client.post(f"{base}/api/training/lora/start", json=arguments)
                d = resp.json()
                return (
                    f"LoRA fine-tuning launched.\n"
                    f"Job ID: {d['job_id']}\n"
                    f"Backbone: {arguments.get('backbone', 'uni')}, "
                    f"Rank: {arguments.get('lora_rank', 8)}\n"
                    f"Use get_job_status(job_id='{d['job_id']}', job_type='training') to monitor.\n"
                    f"After completion, re-run feature extraction to use the fine-tuned backbone."
                )

            # --- Heatmap ---
            elif name == "generate_heatmap":
                resp = await client.post(
                    f"{base}/api/eval/{arguments['experiment_id']}/heatmap",
                    json={"dataset_id": arguments["dataset_id"], "slide_stem": arguments["slide_stem"]},
                )
                d = resp.json()
                return (
                    f"Heatmap generation started.\n"
                    f"Job ID: {d['job_id']}\n"
                    f"Slide: {arguments['slide_stem']}\n"
                    f"When complete, open the Viewer tab, select this slide, and enable the heatmap overlay."
                )

            # --- Artifacts ---
            elif name == "list_artifacts":
                resp = await client.get(f"{base}/api/artifacts")
                d = resp.json()
                if not d.get("artifacts"):
                    return "No experiment artifacts yet. Run a training job first."
                lines = []
                for a in d["artifacts"]:
                    status_tag = "✓ model" if a.get("has_model") else "pending"
                    lines.append(f"• {a['experiment_id']} [{status_tag}] — {', '.join(a.get('files', []))}")
                return "\n".join(lines)

            # --- Experiment Comparison ---
            elif name == "compare_experiments":
                exp_ids = arguments.get("experiment_ids", [])
                if not exp_ids:
                    return "No experiment IDs provided."
                rows = []
                for eid in exp_ids:
                    exp_dir = DATA_DIR / "experiments" / eid
                    cfg_path = exp_dir / "config.json"
                    met_path = exp_dir / "metrics.json"
                    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
                    met = json.loads(met_path.read_text()) if met_path.exists() else {}
                    if not met:
                        # Try status.json
                        st_path = exp_dir / "status.json"
                        if st_path.exists():
                            st = json.loads(st_path.read_text())
                            met = st.get("result", {}).get("metrics", {}) or st.get("metrics", {})
                    rows.append({
                        "id": eid,
                        "backbone": cfg.get("feature_backbone", "?"),
                        "method": cfg.get("mil_method", "?"),
                        "mammoth": "✓" if cfg.get("mammoth", {}).get("enabled") else "✗",
                        "auroc": met.get("auroc", met.get("test_auroc", "?")),
                        "bal_acc": met.get("balanced_accuracy", met.get("test_balanced_accuracy", "?")),
                        "f1": met.get("f1", met.get("test_f1", "?")),
                    })
                if not rows:
                    return "No experiment data found."
                header = f"{'ID':<16} {'Backbone':<12} {'Method':<10} {'MAMMOTH':<8} {'AUROC':<8} {'Bal Acc':<9} {'F1'}"
                sep = "-" * len(header)
                lines = [header, sep]
                for r in rows:
                    auroc = f"{r['auroc']:.4f}" if isinstance(r['auroc'], float) else str(r['auroc'])
                    bal = f"{r['bal_acc']:.4f}" if isinstance(r['bal_acc'], float) else str(r['bal_acc'])
                    f1 = f"{r['f1']:.4f}" if isinstance(r['f1'], float) else str(r['f1'])
                    lines.append(f"{r['id']:<16} {r['backbone']:<12} {r['method']:<10} {r['mammoth']:<8} {auroc:<8} {bal:<9} {f1}")
                return "\n".join(lines)

            # --- Code Execution ---
            elif name == "run_python":
                code = arguments.get("code", "").strip()
                if not code:
                    return "Error: no code provided."
                sandbox = DATA_DIR / "sandbox"
                sandbox.mkdir(parents=True, exist_ok=True)
                preamble = (
                    "import pandas as pd\n"
                    "import numpy as np\n"
                    "from pathlib import Path\n"
                    "import json, csv, os, sys, re\n"
                    f"DATA_DIR = Path(r'{DATA_DIR}')\n\n"
                )
                script = sandbox / f"_run_{uuid.uuid4().hex[:8]}.py"
                script.write_text(preamble + code)
                try:
                    proc = await _asyncio.create_subprocess_exec(
                        "python3", str(script),
                        stdout=_asyncio.subprocess.PIPE,
                        stderr=_asyncio.subprocess.PIPE,
                        cwd=str(sandbox),
                    )
                    try:
                        stdout, stderr = await _asyncio.wait_for(proc.communicate(), timeout=60.0)
                    except _asyncio.TimeoutError:
                        proc.kill()
                        return "Error: script timed out after 60 seconds."
                    out = stdout.decode("utf-8", errors="replace").strip()
                    err = stderr.decode("utf-8", errors="replace").strip()
                    parts = []
                    if out:
                        parts.append(out[:4000])
                    if err:
                        parts.append(f"STDERR:\n{err[:1000]}")
                    if proc.returncode != 0:
                        parts.append(f"Exit code: {proc.returncode}")
                    return "\n".join(parts) if parts else "(no output)"
                finally:
                    script.unlink(missing_ok=True)

            # --- Literature Search ---
            elif name == "search_literature":
                query = arguments.get("query", "").strip()
                if not query:
                    return "Error: query is required."
                limit = min(int(arguments.get("limit", 5)), 10)
                year_from = arguments.get("year_from")
                fields = "title,abstract,authors,year,citationCount,externalIds,openAccessPdf,url"
                params = {"query": query, "limit": limit, "fields": fields}
                if year_from:
                    params["year"] = f"{year_from}-"
                ss_url = "https://api.semanticscholar.org/graph/v1/paper/search"
                # Use configured API key if available (free at semanticscholar.org/product/api)
                cfg_data = {}
                cfg_path = DATA_DIR / "config.json"
                if cfg_path.exists():
                    try:
                        cfg_data = json.loads(cfg_path.read_text())
                    except Exception:
                        pass
                ss_key = cfg_data.get("semantic_scholar_api_key") or _os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")
                ss_headers = {
                    "User-Agent": "PathClaw/0.3 (pathology research platform)",
                    **({"x-api-key": ss_key} if ss_key else {}),
                }
                resp = None
                for attempt, backoff in enumerate((0, 2, 5), start=1):
                    if backoff:
                        await _asyncio.sleep(backoff)
                    try:
                        resp = await client.get(ss_url, params=params, headers=ss_headers, timeout=15.0)
                    except Exception as e:
                        if attempt == 3:
                            return f"Literature search failed: {e}"
                        continue
                    if resp.status_code != 429:
                        break
                if resp is not None and resp.status_code == 429:
                    # Fall back to PubMed so the agent still gets usable references.
                    esearch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
                    esum_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
                    try:
                        r1 = await client.get(
                            esearch_url,
                            params={"db": "pubmed", "term": query, "retmax": limit, "retmode": "json",
                                    **({"mindate": f"{year_from}/01/01", "datetype": "pdat"} if year_from else {})},
                            timeout=15.0,
                        )
                        ids = r1.json().get("esearchresult", {}).get("idlist", []) if r1.status_code == 200 else []
                    except Exception:
                        ids = []
                    if not ids:
                        return (
                            "Semantic Scholar rate limit hit and PubMed fallback returned nothing. "
                            "Try broader keywords or register a free API key at "
                            "semanticscholar.org/product/api (Settings → semantic_scholar_api_key)."
                        )
                    try:
                        r2 = await client.get(esum_url,
                            params={"db": "pubmed", "id": ",".join(ids), "retmode": "json"}, timeout=15.0)
                        sums = r2.json().get("result", {}) if r2.status_code == 200 else {}
                    except Exception:
                        sums = {}
                    lines = [f"(Semantic Scholar rate-limited — falling back to PubMed, {len(ids)} results)\n"]
                    for i, pmid in enumerate(ids, 1):
                        s = sums.get(pmid, {})
                        title = s.get("title", "(no title)")
                        year = (s.get("pubdate") or "")[:4]
                        journal = s.get("source", "")
                        authors = ", ".join(a.get("name", "") for a in (s.get("authors") or [])[:3])
                        if len(s.get("authors") or []) > 3:
                            authors += " et al."
                        doi = ""
                        for aid in s.get("articleids", []) or []:
                            if aid.get("idtype") == "doi":
                                doi = aid.get("value", "")
                                break
                        links = [f"[PubMed](https://pubmed.ncbi.nlm.nih.gov/{pmid}/)"]
                        if doi:
                            links.append(f"[DOI](https://doi.org/{doi})")
                        lines.append(
                            f"**{i}. {title}** ({year}, {journal}) · {' · '.join(links)}\n_{authors}_"
                        )
                    return "\n\n".join(lines)
                if resp is None or resp.status_code != 200:
                    code = resp.status_code if resp is not None else "no response"
                    return f"Semantic Scholar returned {code}. Try a different query."
                try:
                    data = resp.json()
                    papers = data.get("data", [])
                    if not papers:
                        return "No papers found for that query. Try broader keywords."
                    lines = [f"Found {data.get('total', len(papers)):,} matching papers (showing top {len(papers)}):\n"]
                    for i, p in enumerate(papers, 1):
                        authors = ", ".join(a.get("name", "") for a in (p.get("authors") or [])[:3])
                        if len(p.get("authors") or []) > 3:
                            authors += " et al."
                        year = p.get("year", "?")
                        cites = p.get("citationCount", 0)
                        title = p.get("title", "No title")
                        abstract = (p.get("abstract") or "")[:350]
                        if abstract and len(p.get("abstract") or "") > 350:
                            abstract += "…"
                        ext = p.get("externalIds") or {}
                        doi = ext.get("DOI", "")
                        pmid = ext.get("PubMed", "")
                        pdf = (p.get("openAccessPdf") or {}).get("url", "")
                        ss_url = p.get("url", "")  # Semantic Scholar paper page
                        links = []
                        if pmid:
                            links.append(f"[PubMed](https://pubmed.ncbi.nlm.nih.gov/{pmid}/)")
                        if doi:
                            links.append(f"[DOI](https://doi.org/{doi})")
                        if ss_url:
                            links.append(f"[SemanticScholar]({ss_url})")
                        if pdf:
                            links.append(f"[PDF]({pdf})")
                        link_str = (" · " + " · ".join(links)) if links else ""
                        lines.append(
                            f"**{i}. {title}** ({year}) — {cites:,} citations{link_str}\n"
                            f"_{authors}_\n"
                            f"{abstract}"
                        )
                    return "\n\n".join(lines)
                except Exception as e:
                    return f"Literature search failed: {e}"

            # --- Deep multi-source literature review ---
            elif name == "deep_literature_review":
                topic = (arguments.get("topic") or "").strip()
                if not topic:
                    return "Error: topic is required."
                year_from = int(arguments.get("year_from") or 2022)
                max_papers = max(10, min(int(arguments.get("max_papers") or 20), 40))
                angles = arguments.get("angles") or [
                    f"{topic} deep learning methods",
                    f"{topic} benchmark evaluation dataset",
                    f"{topic} review recent advances",
                    f"{topic} foundation model vision transformer",
                    f"{topic} multimodal integration",
                    f"{topic} limitations challenges clinical",
                ]
                angles = angles[:8]

                cfg_data = {}
                cfg_path = DATA_DIR / "config.json"
                if cfg_path.exists():
                    try:
                        cfg_data = json.loads(cfg_path.read_text())
                    except Exception:
                        pass
                ss_key = cfg_data.get("semantic_scholar_api_key") or _os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")
                ss_headers = {"User-Agent": "PathClaw/0.3"}
                if ss_key:
                    ss_headers["x-api-key"] = ss_key

                per_source_limit = 8

                async def _q_ss(q):
                    try:
                        r = await client.get(
                            "https://api.semanticscholar.org/graph/v1/paper/search",
                            params={"query": q, "limit": per_source_limit, "year": f"{year_from}-",
                                    "fields": "title,abstract,authors,year,citationCount,externalIds,openAccessPdf,url,venue"},
                            headers=ss_headers, timeout=15.0)
                        if r.status_code != 200:
                            return []
                        out = []
                        for p in (r.json().get("data") or []):
                            ext = p.get("externalIds") or {}
                            out.append({
                                "source": "ss",
                                "title": p.get("title") or "",
                                "abstract": p.get("abstract") or "",
                                "authors": [a.get("name", "") for a in (p.get("authors") or [])][:5],
                                "year": p.get("year"),
                                "venue": p.get("venue") or "",
                                "citations": p.get("citationCount") or 0,
                                "doi": (ext.get("DOI") or "").lower(),
                                "pmid": ext.get("PubMed") or "",
                                "pdf": (p.get("openAccessPdf") or {}).get("url") or "",
                                "ss_url": p.get("url") or "",
                            })
                        return out
                    except Exception:
                        return []

                async def _q_openalex(q):
                    try:
                        r = await client.get("https://api.openalex.org/works",
                            params={"search": q, "per_page": per_source_limit,
                                    "filter": f"from_publication_date:{year_from}-01-01"},
                            headers={"User-Agent": "PathClaw/0.3 (mailto:devansh@turocrates.ai)"}, timeout=15.0)
                        if r.status_code != 200:
                            return []
                        out = []
                        for w in (r.json().get("results") or []):
                            doi_full = (w.get("doi") or "").lower().replace("https://doi.org/", "")
                            ids = w.get("ids") or {}
                            pmid = ""
                            if ids.get("pmid"):
                                pmid = ids["pmid"].rstrip("/").split("/")[-1]
                            # Reconstruct abstract from inverted index
                            inv = w.get("abstract_inverted_index") or {}
                            abstract = ""
                            if inv:
                                max_pos = max((pos for poses in inv.values() for pos in poses), default=-1)
                                words = [""] * (max_pos + 1)
                                for word, poses in inv.items():
                                    for pos in poses:
                                        if 0 <= pos < len(words):
                                            words[pos] = word
                                abstract = " ".join(words)
                            pdf = ""
                            oa = w.get("best_oa_location") or {}
                            if oa:
                                pdf = oa.get("pdf_url") or ""
                            out.append({
                                "source": "oa",
                                "title": w.get("title") or "",
                                "abstract": abstract,
                                "authors": [a.get("author", {}).get("display_name", "")
                                            for a in (w.get("authorships") or [])][:5],
                                "year": w.get("publication_year"),
                                "venue": ((w.get("primary_location") or {}).get("source") or {}).get("display_name") or "",
                                "citations": w.get("cited_by_count") or 0,
                                "doi": doi_full,
                                "pmid": pmid,
                                "pdf": pdf,
                                "ss_url": "",
                            })
                        return out
                    except Exception:
                        return []

                async def _q_pubmed(q):
                    try:
                        r1 = await client.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
                            params={"db": "pubmed", "term": q, "retmax": per_source_limit, "retmode": "json",
                                    "mindate": f"{year_from}/01/01", "datetype": "pdat"}, timeout=15.0)
                        ids = r1.json().get("esearchresult", {}).get("idlist", []) if r1.status_code == 200 else []
                        if not ids:
                            return []
                        r2 = await client.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
                            params={"db": "pubmed", "id": ",".join(ids), "retmode": "json"}, timeout=15.0)
                        sums = r2.json().get("result", {}) if r2.status_code == 200 else {}
                        out = []
                        for pmid in ids:
                            s = sums.get(pmid, {})
                            if not s:
                                continue
                            doi = ""
                            for aid in s.get("articleids", []) or []:
                                if aid.get("idtype") == "doi":
                                    doi = (aid.get("value") or "").lower()
                                    break
                            out.append({
                                "source": "pm",
                                "title": s.get("title") or "",
                                "abstract": "",
                                "authors": [a.get("name", "") for a in (s.get("authors") or [])][:5],
                                "year": (s.get("pubdate") or "")[:4],
                                "venue": s.get("source") or "",
                                "citations": 0,
                                "doi": doi,
                                "pmid": pmid,
                                "pdf": "",
                                "ss_url": "",
                            })
                        return out
                    except Exception:
                        return []

                async def _q_epmc(q):
                    try:
                        r = await client.get("https://www.ebi.ac.uk/europepmc/webservices/rest/search",
                            params={"query": f"{q} AND PUB_YEAR:[{year_from} TO 3000]",
                                    "format": "json", "resultType": "lite", "pageSize": per_source_limit}, timeout=15.0)
                        if r.status_code != 200:
                            return []
                        out = []
                        for p in (r.json().get("resultList", {}).get("result") or []):
                            out.append({
                                "source": "epmc",
                                "title": p.get("title") or "",
                                "abstract": "",
                                "authors": [x.strip() for x in (p.get("authorString") or "").split(",")][:5],
                                "year": p.get("pubYear"),
                                "venue": p.get("journalTitle") or "",
                                "citations": p.get("citedByCount") or 0,
                                "doi": (p.get("doi") or "").lower(),
                                "pmid": p.get("pmid") or "",
                                "pdf": "",
                                "ss_url": "",
                            })
                        return out
                    except Exception:
                        return []

                jobs = []
                for ang in angles:
                    jobs.extend([_q_ss(ang), _q_openalex(ang), _q_pubmed(ang), _q_epmc(ang)])
                results = await _asyncio.gather(*jobs, return_exceptions=True)

                merged = {}
                source_counts = {"ss": 0, "oa": 0, "pm": 0, "epmc": 0}
                for res in results:
                    if isinstance(res, Exception) or not res:
                        continue
                    for p in res:
                        source_counts[p["source"]] = source_counts.get(p["source"], 0) + 1
                        key = p["doi"] or (f"pmid:{p['pmid']}" if p["pmid"] else f"title:{(p['title'] or '').lower()[:80]}")
                        if not key or key == "title:":
                            continue
                        if key in merged:
                            ex = merged[key]
                            for fld in ("abstract", "pdf", "pmid", "ss_url", "doi", "venue"):
                                if not ex.get(fld) and p.get(fld):
                                    ex[fld] = p[fld]
                            ex["citations"] = max(ex.get("citations") or 0, p.get("citations") or 0)
                        else:
                            merged[key] = dict(p)

                def _score(p):
                    try:
                        y = int(p.get("year") or 0)
                    except Exception:
                        y = 0
                    recency = max(0, y - (year_from - 1))
                    return (p.get("citations") or 0) + recency * 3

                ranked = sorted(merged.values(), key=_score, reverse=True)[:max_papers]
                if not ranked:
                    return (f"Deep review on '{topic}' returned no papers from any source. "
                            f"Tried {len(angles)} angles across Semantic Scholar / OpenAlex / PubMed / Europe PMC.")
                lines = [
                    f"**Deep literature review: {topic}**",
                    f"_Angles searched ({len(angles)}): {'; '.join(angles)}_",
                    f"_Raw hits: SS={source_counts['ss']} · OpenAlex={source_counts['oa']} · "
                    f"PubMed={source_counts['pm']} · EuropePMC={source_counts['epmc']} → "
                    f"{len(merged)} unique → top {len(ranked)} after citation+recency rank._\n",
                ]
                for i, p in enumerate(ranked, 1):
                    authors = ", ".join(p["authors"][:3]) + (" et al." if len(p["authors"]) > 3 else "")
                    links = []
                    if p.get("pmid"):
                        links.append(f"[PubMed](https://pubmed.ncbi.nlm.nih.gov/{p['pmid']}/)")
                    if p.get("doi"):
                        links.append(f"[DOI](https://doi.org/{p['doi']})")
                    if p.get("ss_url"):
                        links.append(f"[SemanticScholar]({p['ss_url']})")
                    if p.get("pdf"):
                        links.append(f"[PDF]({p['pdf']})")
                    link_str = " · ".join(links) if links else "(no links available)"
                    venue = f" · _{p['venue']}_" if p.get("venue") else ""
                    abs_txt = (p.get("abstract") or "").strip()
                    if abs_txt:
                        abs_txt = abs_txt[:300] + ("…" if len(abs_txt) > 300 else "")
                        abs_block = f"\n   {abs_txt}"
                    else:
                        abs_block = ""
                    lines.append(
                        f"**{i}. {p['title']}** ({p.get('year') or '?'}, {p.get('citations') or 0} cites){venue}\n"
                        f"   {authors}\n"
                        f"   {link_str}{abs_block}"
                    )
                return "\n\n".join(lines)

            elif name == "get_paper_citations":
                paper_id = (arguments.get("paper_id") or "").strip()
                if not paper_id:
                    return "Error: paper_id is required."
                direction = arguments.get("direction") or "both"
                limit = max(1, min(int(arguments.get("limit") or 10), 20))
                if paper_id.lower().startswith("pmid:"):
                    ss_id = f"PMID:{paper_id.split(':',1)[1].strip()}"
                elif paper_id.isdigit():
                    ss_id = f"PMID:{paper_id}"
                elif paper_id.lower().startswith("10."):
                    ss_id = f"DOI:{paper_id}"
                else:
                    ss_id = paper_id
                fields = "title,year,authors,citationCount,externalIds,url"

                async def _fetch(edge):
                    try:
                        r = await client.get(
                            f"https://api.semanticscholar.org/graph/v1/paper/{ss_id}/{edge}",
                            params={"fields": fields, "limit": limit}, timeout=15.0)
                        if r.status_code != 200:
                            return []
                        return r.json().get("data") or []
                    except Exception:
                        return []

                blocks = []
                if direction in ("citations", "both"):
                    cits = await _fetch("citations")
                    if cits:
                        blocks.append(f"**Papers citing {paper_id} ({len(cits)}):**\n" + "\n".join(
                            _fmt_citation_row(item.get("citingPaper", {}), j + 1)
                            for j, item in enumerate(cits) if item.get("citingPaper")))
                if direction in ("references", "both"):
                    refs = await _fetch("references")
                    if refs:
                        blocks.append(f"\n**Papers cited BY {paper_id} ({len(refs)}):**\n" + "\n".join(
                            _fmt_citation_row(item.get("citedPaper", {}), j + 1)
                            for j, item in enumerate(refs) if item.get("citedPaper")))
                return "\n\n".join(blocks) if blocks else f"No citation data found for {paper_id}."

            # --- Folders (user-uploaded PDFs) ---
            elif name == "list_folders":
                from pathclaw import folders as _folders
                folders = _folders.list_folders()
                if not folders:
                    return "No folders yet. The user can create one in the Folders tab and upload PDFs there."
                lines = ["Available folders:"]
                for f in folders:
                    files = _folders.list_files(f["id"])
                    file_list = ", ".join(x["name"] for x in files) if files else "(empty)"
                    lines.append(f"- **{f['name']}** (id={f['id']}, {f['file_count']} files): {file_list}")
                return "\n".join(lines)

            elif name == "read_pdf":
                from pathclaw import folders as _folders
                folder_id = arguments["folder_id"]
                filename = arguments["filename"]
                max_chars = int(arguments.get("max_chars", 20000))
                try:
                    text = _folders.read_pdf_text(folder_id, filename)
                except FileNotFoundError as e:
                    return f"Error: {e}"
                except Exception as e:
                    return f"Failed to parse PDF: {e}"
                if not text.strip():
                    return "PDF parsed but no extractable text (likely a scanned/image PDF). Consider OCR."
                truncated = text[:max_chars]
                suffix = f"\n\n… (truncated — showing {max_chars} of {len(text)} chars)" if len(text) > max_chars else ""
                return f"**{filename}** ({len(text):,} chars):\n\n{truncated}{suffix}"

            # --- Web research ---
            elif name == "fetch_url":
                url = (arguments.get("url") or "").strip()
                if not url.startswith(("http://", "https://")):
                    return "Error: url must start with http:// or https://"
                max_chars = int(arguments.get("max_chars", 8000))

                # Smart redirects for JS-rendered biomedical pages
                import re as _re
                pm_match = _re.match(r"https?://pubmed\.ncbi\.nlm\.nih\.gov/(\d+)/?", url)
                pmc_match = _re.match(r"https?://(?:www\.)?ncbi\.nlm\.nih\.gov/pmc/articles/(PMC\d+)/?", url)
                effective_url = url
                note = ""
                if pm_match:
                    effective_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id={pm_match.group(1)}&rettype=abstract&retmode=text"
                    note = f"(auto-routed PubMed {pm_match.group(1)} through E-utilities for clean text)\n"
                elif pmc_match:
                    effective_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id={pmc_match.group(1)[3:]}&rettype=xml"
                    note = f"(auto-routed PMC {pmc_match.group(1)} through E-utilities for full-text XML)\n"

                try:
                    resp = await client.get(
                        effective_url,
                        timeout=20.0,
                        follow_redirects=True,
                        headers={"User-Agent": "PathClaw/0.3 (+research)"},
                    )
                except Exception as e:
                    return f"Fetch failed: {e}"
                if resp.status_code != 200:
                    return f"Fetched {effective_url} → HTTP {resp.status_code}"
                ct = resp.headers.get("content-type", "").lower()
                body = resp.text
                if "html" in ct or "<html" in body[:400].lower():
                    body = _re.sub(r"<(script|style)[^>]*>.*?</\1>", "", body, flags=_re.I | _re.S)
                    body = _re.sub(r"<[^>]+>", " ", body)
                    body = _re.sub(r"&nbsp;|&amp;|&#\d+;|&\w+;", " ", body)
                    body = _re.sub(r"[ \t]+", " ", body)
                    body = _re.sub(r"\n\s*\n+", "\n\n", body).strip()
                    # Detect JS-only SPA skeletons — NCBI and many others
                    if len(body) < 600 and ("needs JavaScript" in body or "JavaScript to work" in body):
                        return f"{url} requires JavaScript and returned only a skeleton. For PubMed/PMC articles, pass the direct PMID/PMCID to pubmed_search or efetch. For journals, find the article PDF URL and use download_paper_pdf instead."
                truncated = body[:max_chars]
                suffix = f"\n\n… (truncated — {len(body):,} chars total)" if len(body) > max_chars else ""
                return f"{note}**{url}** (status {resp.status_code}, {ct or 'text'}):\n\n{truncated}{suffix}"

            elif name == "pubmed_search":
                query = (arguments.get("query") or "").strip()
                if not query:
                    return "Error: query is required."
                limit = max(1, min(int(arguments.get("limit", 10)), 20))
                esearch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
                esum_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
                efetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
                try:
                    r1 = await client.get(
                        esearch_url,
                        params={"db": "pubmed", "term": query, "retmax": limit, "retmode": "json"},
                        timeout=15.0,
                    )
                    if r1.status_code != 200:
                        return f"PubMed esearch returned {r1.status_code}"
                    ids = r1.json().get("esearchresult", {}).get("idlist", [])
                    if not ids:
                        return "No PubMed results for that query."
                    id_str = ",".join(ids)
                    r2 = await client.get(
                        esum_url,
                        params={"db": "pubmed", "id": id_str, "retmode": "json"},
                        timeout=15.0,
                    )
                    sums = r2.json().get("result", {}) if r2.status_code == 200 else {}
                    # Fetch abstracts in one call
                    r3 = await client.get(
                        efetch_url,
                        params={"db": "pubmed", "id": id_str, "rettype": "abstract", "retmode": "text"},
                        timeout=20.0,
                    )
                    abs_text = r3.text if r3.status_code == 200 else ""
                except Exception as e:
                    return f"PubMed search failed: {e}"

                lines = [f"Found {len(ids)} PubMed results for: _{query}_\n"]
                for pmid in ids:
                    s = sums.get(pmid, {})
                    title = s.get("title", "(no title)")
                    year = (s.get("pubdate") or "")[:4]
                    journal = s.get("source", "")
                    authors = ", ".join(a.get("name", "") for a in (s.get("authors") or [])[:3])
                    if len(s.get("authors") or []) > 3:
                        authors += " et al."
                    lines.append(f"**PMID {pmid}** — {title} ({year}, {journal})\n_{authors}_\nhttps://pubmed.ncbi.nlm.nih.gov/{pmid}/")
                if abs_text:
                    lines.append(f"\n---\nFull abstracts:\n{abs_text[:6000]}")
                return "\n\n".join(lines)

            elif name == "download_paper_pdf":
                from pathclaw import folders as _folders
                url = (arguments.get("url") or "").strip()
                if not url.startswith(("http://", "https://")):
                    return "Error: url must start with http:// or https://"
                folder_id = arguments.get("folder_id", "").strip()
                if not folder_id:
                    fname = arguments.get("folder_name", "References")
                    new_folder = _folders.create_folder(fname)
                    folder_id = new_folder["id"]
                try:
                    resp = await client.get(
                        url,
                        timeout=60.0,
                        follow_redirects=True,
                        headers={"User-Agent": "PathClaw/0.3 (+research)"},
                    )
                except Exception as e:
                    return f"Download failed: {e}"
                if resp.status_code != 200:
                    return f"Download failed with HTTP {resp.status_code}"
                ct = resp.headers.get("content-type", "").lower()
                if "pdf" not in ct and not url.lower().endswith(".pdf"):
                    return f"URL did not return a PDF (content-type: {ct}). Aborted."
                default_name = arguments.get("filename") or url.rstrip("/").split("/")[-1] or "paper.pdf"
                try:
                    info = _folders.save_pdf(folder_id, default_name, resp.content)
                    text = _folders.read_pdf_text(folder_id, info["name"])
                except Exception as e:
                    return f"Saved but failed to parse: {e}"
                preview = text[:5000]
                suffix = f"\n\n… ({len(text):,} chars total — use read_pdf to fetch more)" if len(text) > 5000 else ""
                return (
                    f"Saved **{info['name']}** to folder `{folder_id}` ({info['size_bytes']:,} bytes).\n\n"
                    f"Preview:\n{preview}{suffix}"
                )

            # wait_for_job is handled inline in _stream_generator (needs SSE yields)
            elif name == "wait_for_job":
                # Fallback for non-streaming (blocking) endpoint
                jid = arguments["job_id"]
                jtype = arguments.get("job_type", "training")
                # download_gdc returns the job id without the 'dl_' prefix the
                # queue stores internally, so normalise both forms here.
                prefix_map = {"gdc": "dl_"}
                prefix = prefix_map.get(jtype, "")
                if prefix and not jid.startswith(prefix):
                    jid = f"{prefix}{jid}"
                route_map = {
                    "preprocess": f"/api/preprocess/{jid}",
                    "training": f"/api/training/{jid}",
                    "eval": f"/api/eval/{jid}",
                    "features": f"/api/features/{jid}",
                    "lora": f"/api/training/lora/{jid}",
                    "gdc": f"/api/gdc/jobs/{jid}",
                }
                route = route_map.get(jtype, f"/api/training/{jid}")
                url = f"{base}{route}"
                for _ in range(180):
                    try:
                        async with httpx.AsyncClient(timeout=10.0) as pc:
                            d = (await pc.get(url)).json()
                        status = d.get("status", "unknown")
                        if status in ("completed", "failed", "error"):
                            result = f"Job {jid}: {status}"
                            metrics = d.get("metrics", {})
                            for k, v in metrics.items():
                                result += f"\n  {k}: {v:.4f}" if isinstance(v, float) else f"\n  {k}: {v}"
                            return result
                    except Exception:
                        pass
                    await _asyncio.sleep(10)
                return f"Job {jid}: still running after 30 min."

            # --- Genomics ---
            elif name == "parse_genomic_file":
                from pathclaw.genomics.parsers import parse_genomic_file as _parse_gf
                return _parse_gf(
                    file_path=arguments["file_path"],
                    file_type=arguments.get("file_type", "auto"),
                    query=arguments.get("query", "summary"),
                    sample_id=arguments.get("sample_id"),
                    limit=int(arguments.get("limit", 50)),
                )

            elif name == "query_mutations":
                from pathclaw.genomics.parsers import query_mutations_impl
                return query_mutations_impl(
                    genomic_dir=arguments["genomic_dir"],
                    gene=arguments.get("gene", "*"),
                    variant_class=arguments.get("variant_class", ""),
                    min_frequency=float(arguments.get("min_frequency", 0.0)),
                    output_format=arguments.get("output_format", "summary"),
                )

            elif name == "compute_tmb":
                from pathclaw.genomics.parsers import compute_tmb_impl
                return compute_tmb_impl(
                    maf_dir=arguments["maf_dir"],
                    exome_size_mb=float(arguments.get("exome_size_mb", 30.0)),
                    thresholds=arguments.get("thresholds"),
                )

            elif name == "extract_labels_from_genomic":
                from pathclaw.genomics.label_extraction import extract_labels
                return extract_labels(
                    genomic_dir=arguments["genomic_dir"],
                    dataset_id=arguments["dataset_id"],
                    label_type=arguments["label_type"],
                    label_spec=arguments.get("label_spec", {}),
                    output_path=arguments.get("output_path"),
                )

            elif name == "query_cbioportal":
                from pathclaw.genomics.cbioportal import query_cbioportal as _qcbio
                return await _qcbio(
                    study_id=arguments["study_id"],
                    data_type=arguments.get("data_type", "clinical"),
                    gene_list=arguments.get("gene_list"),
                    clinical_attributes=arguments.get("clinical_attributes"),
                )

            elif name == "run_survival_analysis":
                from pathclaw.genomics.survival import run_survival_analysis as _rsa
                return _rsa(
                    clinical_dir=arguments["clinical_dir"],
                    dataset_id=arguments.get("dataset_id"),
                    labels_path=arguments.get("labels_path"),
                    group_column=arguments.get("group_column", "label_name"),
                    output_dir=arguments.get("output_dir"),
                )

            elif name == "build_multi_omic_labels":
                from pathclaw.genomics.multi_omic import build_multi_omic_labels as _bmol
                return _bmol(
                    dataset_id=arguments["dataset_id"],
                    sources=arguments["sources"],
                    output_path=arguments.get("output_path"),
                )

            elif name == "generate_oncoplot":
                from pathclaw.genomics.visualization import generate_oncoplot as _go
                return _go(
                    maf_dir=arguments["maf_dir"],
                    top_n=int(arguments.get("top_n", 20)),
                    output_path=arguments.get("output_path"),
                    title=arguments.get("title", "Mutation Landscape"),
                    min_frequency=float(arguments.get("min_frequency", 0.0)),
                )

            elif name == "parse_gene_expression":
                from pathclaw.genomics.expression import parse_gene_expression as _pge
                return _pge(
                    file_path=arguments["file_path"],
                    query=arguments.get("query", "summary"),
                    gene_list=arguments.get("gene_list"),
                    limit=int(arguments.get("limit", 50)),
                )

            elif name == "biomarker_discovery":
                from pathclaw.genomics.biomarker import biomarker_discovery as _bd
                return _bd(
                    maf_dir=arguments["maf_dir"],
                    labels_path=arguments["labels_path"],
                    dataset_id=arguments.get("dataset_id"),
                    experiment_id=arguments.get("experiment_id"),
                    analysis_type=arguments.get("analysis_type", "mutation_enrichment"),
                    gene_list=arguments.get("gene_list"),
                    group_column=arguments.get("group_column", "label_name"),
                )

            # --- Workspace (user code + cloned repos) — ALL session-scoped ---
            elif name == "list_workspace_files":
                from pathclaw.api.routes.workspace_fs import (
                    safe_workspace_path, _ensure_workspace, session_workspace_root,
                )
                _ensure_workspace(session_id)
                sub = (arguments.get("subdir") or "").strip().lstrip("/")
                root = session_workspace_root(session_id)
                base_path = safe_workspace_path(sub, session_id) if sub else root
                if not base_path.exists():
                    return f"Workspace path not found: {sub or '/'}"
                lines = []
                for p in sorted(base_path.rglob("*")):
                    if any(part in {".git", "__pycache__", ".venv"} for part in p.parts):
                        continue
                    rel = p.relative_to(root)
                    if p.is_dir():
                        lines.append(f"📁 {rel}/")
                    else:
                        try:
                            size_kb = p.stat().st_size / 1024
                            lines.append(f"📄 {rel}  ({size_kb:.1f} KB)")
                        except OSError:
                            lines.append(f"📄 {rel}")
                if not lines:
                    return f"Workspace is empty at '{sub or '/'}'. Create files with write_workspace_file."
                return "\n".join(lines[:500])

            elif name == "read_workspace_file":
                from pathclaw.api.routes.workspace_fs import safe_workspace_path, MAX_TEXT_BYTES, _is_probably_text
                path_arg = (arguments.get("path") or "").strip()
                if not path_arg:
                    return "Error: path is required."
                p = safe_workspace_path(path_arg, session_id)
                if not p.exists() or not p.is_file():
                    return f"Not a file: {path_arg}"
                if p.stat().st_size > MAX_TEXT_BYTES:
                    return f"File too large: {p.stat().st_size} bytes"
                if not _is_probably_text(p):
                    return f"Refusing to read binary file: {path_arg}"
                return p.read_text(encoding="utf-8", errors="replace")

            elif name == "write_workspace_file":
                from pathclaw.api.routes.workspace_fs import safe_workspace_path, MAX_TEXT_BYTES
                path_arg = (arguments.get("path") or "").strip()
                content = arguments.get("content") or ""
                if not path_arg:
                    return "Error: path is required."
                if len(content.encode("utf-8")) > MAX_TEXT_BYTES:
                    return "Error: content exceeds 2 MB workspace text cap."
                p = safe_workspace_path(path_arg, session_id)
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(content, encoding="utf-8")
                return f"Wrote {p.stat().st_size} bytes to workspace/{path_arg}"

            elif name == "delete_workspace_file":
                from pathclaw.api.routes.workspace_fs import safe_workspace_path
                path_arg = (arguments.get("path") or "").strip()
                if not path_arg:
                    return "Error: path is required."
                p = safe_workspace_path(path_arg, session_id)
                if not p.exists():
                    return f"Not found: {path_arg}"
                if p.is_dir():
                    return "Refusing to delete directory through this tool."
                p.unlink()
                return f"Deleted workspace/{path_arg}"

            elif name == "clone_repo":
                r = await client.post(f"{base}/api/workspace/clone", json={
                    "url": arguments.get("url", ""),
                    "name": arguments.get("name"),
                    "session_id": session_id,
                }, timeout=200.0)
                if r.status_code == 200:
                    d = r.json()
                    return f"Cloned {d['name']} ({d['files']} files) → workspace/{d['path']}"
                try:
                    return f"Clone failed: {r.json().get('detail', r.text)}"
                except Exception:
                    return f"Clone failed: HTTP {r.status_code}: {r.text[:400]}"

            elif name == "register_hf_backbone":
                payload = {
                    "id": arguments.get("id", ""),
                    "hf_model_id": arguments.get("hf_model_id", ""),
                    "timm_model": arguments.get("timm_model", ""),
                    "dim": int(arguments.get("dim", 0) or 0),
                    "patch_px": int(arguments.get("patch_px", 224) or 224),
                    "magnification": int(arguments.get("magnification", 20) or 20),
                    "gated": bool(arguments.get("gated", False)),
                }
                r = await client.post(f"{base}/api/config-space/backbones/register", json=payload, timeout=15.0)
                if r.status_code == 200:
                    d = r.json()
                    m = d["manifest"]
                    return (f"Registered backbone '{d['id']}' → {m['hf_model_id']} "
                            f"(dim={m['dim']}, patch={m['patch_px']}px, "
                            f"{'gated' if m['gated'] else 'open'}).")
                try:
                    return f"register_hf_backbone failed: {r.json().get('detail', r.text)}"
                except Exception:
                    return f"register_hf_backbone failed: HTTP {r.status_code}: {r.text[:400]}"

            elif name == "list_backbones":
                r = await client.get(f"{base}/api/config-space/backbones", timeout=10.0)
                if r.status_code != 200:
                    return f"list_backbones failed: HTTP {r.status_code}"
                bbs = r.json().get("backbones", [])
                lines = [f"{len(bbs)} backbone(s):"]
                for b in bbs:
                    tag = " [custom]" if b.get("custom") else ""
                    gated = " (gated)" if b.get("gated") else ""
                    lines.append(f"- {b['id']}{tag}: dim={b.get('dim')}, {b.get('hf_id','?')}{gated}")
                return "\n".join(lines)

            elif name == "list_plugins":
                r = await client.get(f"{base}/api/plugins", timeout=10.0)
                if r.status_code != 200:
                    return f"list_plugins failed: HTTP {r.status_code}: {r.text[:300]}"
                plist = r.json().get("plugins", [])
                if not plist:
                    return "No plugins registered."
                lines = [f"{len(plist)} plugin(s):"]
                for p in plist:
                    inst = "✓ installed" if p["installed"] else "✗ not installed"
                    lines.append(f"- {p['id']} [{p['kind']}] — {p['name']} ({inst}); applies_to={p['applies_to']}")
                return "\n".join(lines)

            elif name == "register_plugin":
                payload = {
                    "id": arguments.get("id", ""),
                    "name": arguments.get("name", ""),
                    "kind": arguments.get("kind", ""),
                    "import_path": arguments.get("import_path", ""),
                    "description": arguments.get("description", ""),
                    "applies_to": arguments.get("applies_to", []),
                    "default_config": arguments.get("default_config", {}),
                    "source": arguments.get("source", ""),
                }
                r = await client.post(f"{base}/api/plugins/register", json=payload, timeout=10.0)
                if r.status_code == 200:
                    d = r.json()
                    status = "importable" if d.get("installed") else "NOT importable yet — check import_path"
                    return f"Registered plugin '{d['id']}' ({status})."
                try:
                    return f"register_plugin failed: {r.json().get('detail', r.text)}"
                except Exception:
                    return f"register_plugin failed: HTTP {r.status_code}: {r.text[:400]}"

            elif name == "update_plugin_config":
                pid = (arguments.get("id") or "").strip()
                new_cfg = arguments.get("default_config") or {}
                if not pid or not isinstance(new_cfg, dict):
                    return "update_plugin_config: 'id' and dict 'default_config' required."
                r = await client.put(f"{base}/api/plugins/{pid}/config", json={"default_config": new_cfg}, timeout=10.0)
                if r.status_code == 200:
                    d = r.json()
                    return f"Updated default_config for {pid}: {json.dumps(d.get('default_config', {}), indent=2)}"
                try:
                    return f"update_plugin_config failed: {r.json().get('detail', r.text)}"
                except Exception:
                    return f"update_plugin_config failed: HTTP {r.status_code}: {r.text[:300]}"

            elif name == "run_cellpose_segmentation":
                ds_id = (arguments.get("dataset_id") or "").strip()
                if not ds_id:
                    return "run_cellpose_segmentation: 'dataset_id' required."
                max_slides = int(arguments.get("max_slides") or 5)
                max_patches = int(arguments.get("max_patches_per_slide") or 64)
                overrides = arguments.get("config_overrides") or {}
                # Resolve current cellpose config
                pr = await client.get(f"{base}/api/plugins", timeout=10.0)
                if pr.status_code != 200:
                    return f"run_cellpose_segmentation: could not load plugin registry ({pr.status_code})"
                cfg = {}
                for p in pr.json().get("plugins", []):
                    if p["id"] == "cellpose":
                        cfg = dict(p.get("default_config") or {})
                        break
                if not cfg:
                    return "run_cellpose_segmentation: cellpose plugin not in registry."
                cfg.update(overrides)
                # Resolve dataset path
                dr = await client.get(f"{base}/api/datasets/{ds_id}", timeout=10.0)
                if dr.status_code != 200:
                    return f"run_cellpose_segmentation: dataset {ds_id!r} not found."
                ds = dr.json()
                slides = (ds.get("slides") or [])[:max_slides]
                if not slides:
                    return f"run_cellpose_segmentation: no slides in dataset {ds_id!r}."
                # Run inline (no job — keep it observable in chat)
                try:
                    from pathclaw.plugins.cellpose import run_on_images
                    from pathclaw.preprocessing.feature_extraction import _open_slide_safe  # type: ignore
                    import numpy as _np
                    from PIL import Image as _Im
                    out_root = (DATA_DIR / "sessions" / (session_id or "default") / "cellpose" / ds_id)
                    out_root.mkdir(parents=True, exist_ok=True)
                    rows = []
                    total_objs = 0
                    for s in slides:
                        stem = s["filename"].rsplit(".", 1)[0]
                        sdir = out_root / stem
                        sdir.mkdir(exist_ok=True)
                        # Pull patches from existing preprocessed dataset if available
                        patches_dir = DATA_DIR / "datasets" / ds_id / "patches" / stem
                        patch_files = sorted(patches_dir.glob("*.png"))[:max_patches] if patches_dir.exists() else []
                        if not patch_files:
                            rows.append(f"  {stem}: no preprocessed patches — run preprocessing first")
                            continue
                        imgs = [_np.array(_Im.open(pf).convert("RGB")) for pf in patch_files]
                        result = run_on_images(imgs, cfg)
                        for pf, mask, n in zip(patch_files, result["masks"], result["num_objects"]):
                            mask_path = sdir / f"{pf.stem}_mask.png"
                            _Im.fromarray((mask % 256).astype("uint8")).save(mask_path)
                            total_objs += n
                        rows.append(f"  {stem}: {len(patch_files)} patches, {sum(result['num_objects'])} objects")
                    summary = (
                        f"Cellpose run complete. Config: {json.dumps(cfg)}\n"
                        f"Output: {out_root}\n"
                        f"Slides processed: {len([r for r in rows if 'no preprocessed' not in r])}\n"
                        f"Total objects detected: {total_objs}\n"
                        + "\n".join(rows)
                    )
                    return summary
                except ImportError as e:
                    return f"run_cellpose_segmentation: cellpose not installed ({e}). Run pip install cellpose."
                except Exception as e:
                    import traceback as _tb
                    return f"run_cellpose_segmentation failed: {e}\n{_tb.format_exc()[-1200:]}"

            elif name == "analyze_repo":
                from pathclaw.api.routes.workspace_fs import session_workspace_root
                repo_name = (arguments.get("name") or "").strip().strip("/")
                if not repo_name or ".." in repo_name.split("/"):
                    return "analyze_repo: invalid 'name'"
                ws = session_workspace_root(session_id)
                repo_dir = (ws / "repos" / repo_name).resolve()
                root = ws.resolve()
                if not str(repo_dir).startswith(str(root)) or not repo_dir.is_dir():
                    return f"analyze_repo: repo not found at workspace/repos/{repo_name}/"
                # Language mix (top 8 extensions)
                from collections import Counter as _C
                ext_ct: _C[str] = _C()
                py_modules: list[str] = []
                for p in repo_dir.rglob("*"):
                    if not p.is_file():
                        continue
                    if "/.git/" in str(p) or str(p).endswith("/.git"):
                        continue
                    ext = p.suffix.lower()
                    if ext:
                        ext_ct[ext] += 1
                    if ext == ".py":
                        rel = str(p.relative_to(repo_dir))
                        if rel.count("/") <= 2 and any(k in rel.lower() for k in ("model", "net", "arch", "layer", "attn", "mil", "transformer")):
                            py_modules.append(rel)
                tree_lines: list[str] = []
                for entry in sorted(repo_dir.iterdir()):
                    if entry.name.startswith(".git"):
                        continue
                    tree_lines.append(f"- {entry.name}{'/' if entry.is_dir() else ''}")
                    if entry.is_dir() and len(tree_lines) < 80:
                        for sub in sorted(entry.iterdir())[:15]:
                            if sub.name.startswith("."):
                                continue
                            tree_lines.append(f"    - {sub.name}{'/' if sub.is_dir() else ''}")
                readme = ""
                for rn in ("README.md", "README.rst", "README.txt", "readme.md"):
                    rp = repo_dir / rn
                    if rp.exists():
                        try:
                            readme = rp.read_text(errors="replace")[:2000]
                        except Exception:
                            pass
                        break
                setup = ""
                for sn in ("pyproject.toml", "setup.py", "requirements.txt"):
                    sp = repo_dir / sn
                    if sp.exists():
                        try:
                            setup += f"\n--- {sn} ---\n" + sp.read_text(errors="replace")[:1500]
                        except Exception:
                            pass
                lang_str = ", ".join(f"{e}:{n}" for e, n in ext_ct.most_common(8))
                out = [f"# Repo analysis: {repo_name}",
                       f"Files by ext: {lang_str or '(none)'}",
                       "",
                       "## Tree (2 levels)", *tree_lines[:80],
                       "", "## Candidate model modules", *(py_modules[:15] or ["(none found by name heuristic)"])]
                if readme:
                    out += ["", "## README (first 2 KB)", readme]
                if setup:
                    out += ["", "## Install metadata", setup]
                return "\n".join(out)

            elif name == "ask_user":
                # This tool is best-effort: the chat loop will display the question to the user
                # in the next agent message. We just return a structured payload the LLM can
                # reference when composing its reply.
                q = (arguments.get("question") or "").strip()
                ctx = (arguments.get("context") or "").strip()
                if not q:
                    return "ask_user: 'question' is required"
                return f"[Pending user question — wait for their reply]\nQuestion: {q}" + (f"\nContext: {ctx}" if ctx else "")

            elif name == "smoke_test_plugin":
                import_path = arguments.get("import_path", "").strip()
                kind = arguments.get("kind", "patch_embed")
                in_dim = int(arguments.get("in_dim", 1024) or 1024)
                embed_dim = int(arguments.get("embed_dim", 512) or 512)
                cfg = arguments.get("config", {}) or {}
                try:
                    from pathclaw.plugins import resolve_builder
                    import torch as _torch
                    import sys as _sys
                    # Invalidate cached workspace modules so edits take effect
                    for mod_name in list(_sys.modules):
                        if mod_name.startswith("workspace."):
                            del _sys.modules[mod_name]
                    builder = resolve_builder(import_path)
                    module = builder(in_dim=in_dim, embed_dim=embed_dim, config=cfg) if kind == "patch_embed" \
                        else builder(**cfg) if callable(builder) else None
                    if module is None:
                        return f"smoke_test: builder at {import_path} returned None"
                    if kind == "patch_embed":
                        x = _torch.randn(1, 64, in_dim)
                        try:
                            module.eval()
                        except Exception:
                            pass
                        y = module(x)
                        shape = tuple(y.shape) if hasattr(y, "shape") else None
                        if shape is None or len(shape) != 3 or shape[-1] != embed_dim:
                            return (f"smoke_test FAILED: forward shape {shape} ≠ expected "
                                    f"(1, 64, {embed_dim}). Fix the module and retry.")
                        n_params = sum(p.numel() for p in module.parameters())
                        return f"smoke_test PASSED: output shape {shape}, params {n_params:,}"
                    return f"smoke_test: kind={kind!r} OK (instantiation only)"
                except Exception as _e:
                    import traceback as _tb
                    return f"smoke_test FAILED ({type(_e).__name__}): {_e}\n{_tb.format_exc()[:1500]}"

            elif name == "implement_from_paper":
                method_name = (arguments.get("method_name") or "").strip()
                target_kind = arguments.get("target_kind", "patch_embed")
                paper_path = arguments.get("paper_path", "") or "(use read_pdf on the uploaded PDF)"
                if not method_name or not method_name.replace("_", "").isalnum():
                    return "implement_from_paper: 'method_name' must be alphanumeric/underscore"
                contract = {
                    "patch_embed": (
                        "def build(in_dim: int, embed_dim: int, config: dict) -> nn.Module:\n"
                        "    # forward: (B, N, in_dim) → (B, N, embed_dim)"
                    ),
                    "mil": (
                        "def build(in_dim: int, num_classes: int, config: dict) -> nn.Module:\n"
                        "    # forward: (B, N, in_dim) → (B, num_classes) plus optional attention weights"
                    ),
                    "loss": (
                        "def build(config: dict) -> nn.Module:\n"
                        "    # forward(logits, targets) -> scalar loss"
                    ),
                    "method": "def build(config: dict) -> Any:  # end-to-end method factory",
                }[target_kind]
                return "\n".join([
                    f"# implement_from_paper: {method_name} (kind={target_kind})",
                    "",
                    "Follow this exact sequence — do NOT skip smoke-testing:",
                    f"  1. read_pdf({paper_path!r}) to load the paper text.",
                    "  2. Extract: architecture, inputs/outputs, loss, critical hyperparameters.",
                    f"  3. write_workspace_file(path='plugins/{method_name}.py', content=<full module>).",
                    "     The module MUST expose this exact contract:",
                    "     " + contract.replace("\n", "\n     "),
                    f"  4. smoke_test_plugin(import_path='workspace.plugins.{method_name}:build', "
                    f"kind='{target_kind}').",
                    "  5. If smoke test FAILS: read the traceback, fix the module via write_workspace_file,",
                    "     and retry smoke_test_plugin. Max 2 retries.",
                    "  6. If still failing OR the paper is ambiguous on a critical choice:",
                    "     call ask_user with a specific, answerable question.",
                    "  7. On PASS: register_plugin(id='" + method_name + f"', name='<Name>', kind='{target_kind}', "
                    f"import_path='workspace.plugins.{method_name}:build', description=<one line>).",
                    "  8. Report back to the user with the final plugin id and how to toggle it in training.",
                ])

            elif name == "make_plot":
                from pathclaw.training.plot_builder import make_plot as _make_plot
                try:
                    out = _make_plot(
                        experiment_id=arguments.get("experiment_id", ""),
                        kind=arguments.get("kind", ""),
                        spec=arguments.get("spec", ""),
                        title=arguments.get("title", ""),
                    )
                    return f"Plot saved to {out['path']} (kind={out['kind']})."
                except FileNotFoundError as e:
                    return f"make_plot failed: {e}"
                except ValueError as e:
                    return f"make_plot rejected: {e}"
                except Exception as e:
                    return f"make_plot error ({type(e).__name__}): {e}"

            else:
                import difflib as _difflib
                valid = [t.get("function", {}).get("name", "") for t in TOOLS]
                valid = [n for n in valid if n]
                close = _difflib.get_close_matches(name, valid, n=5, cutoff=0.55)
                hint = (
                    f"\n\nDid you mean: {', '.join(close)}? "
                    f"CALL THE CORRECT TOOL NAME ON YOUR NEXT TURN — do not retry '{name}'."
                ) if close else (
                    "\n\nRetry with one of the valid tool names instead. "
                    "Use list_artifacts or search_gdc / download_gdc / register_dataset as appropriate."
                )
                return f"ERROR: Unknown tool '{name}'.{hint}"

        except Exception as e:
            import traceback as _tb
            logger.error(f"Tool {name} failed:\n{_tb.format_exc()}")
            return f"Error calling {name} ({type(e).__name__}): {e}"


# ---------------------------------------------------------------------------
# Chat models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str
    session_id: str = ""


class ChatResponse(BaseModel):
    response: str
    session_id: str
    tool_calls_made: list[str] = []


# ---------------------------------------------------------------------------
# Blocking chat endpoint (kept for compatibility)
# ---------------------------------------------------------------------------

@router.post("/")
async def chat(req: ChatRequest) -> ChatResponse:
    """Send a message and await the full response (non-streaming)."""
    session_id = req.session_id or str(uuid.uuid4())[:8]
    matched_skills = _match_skills(req.message)
    skill_context = "".join(_load_skill(s) for s in matched_skills)

    if session_id not in _conversations:
        _conversations[session_id] = [
            {"role": "system", "content": _build_system_prompt(skill_context, session_id=session_id)}
        ]
    else:
        # Refresh system prompt with any newly matched skill context
        if skill_context:
            _conversations[session_id][0] = {
                "role": "system",
                "content": _build_system_prompt(skill_context, session_id=session_id),
            }

    messages = _conversations[session_id]
    messages.append({"role": "user", "content": req.message})

    tool_calls_made: list[str] = []
    provider, model = llm_providers.get_active_provider()

    for _ in range(25):  # max tool rounds
        try:
            msg = await llm_providers.chat_round(
                provider, model, messages, TOOLS, OLLAMA_BASE, NUM_CTX
            )
        except Exception as e:
            raise HTTPException(503, f"LLM error ({provider}): {e}")

        tool_calls = msg.get("tool_calls", [])
        if not tool_calls:
            content = msg.get("content", "")
            messages.append({"role": "assistant", "content": content})
            _trim_conversation(session_id)
            return ChatResponse(response=content, session_id=session_id, tool_calls_made=tool_calls_made)

        messages.append(msg)
        for tc in tool_calls:
            fn = tc.get("function", {})
            fn_name = fn.get("name", "unknown")
            fn_args = fn.get("arguments", {})
            tool_calls_made.append(fn_name)
            result = await _execute_tool(fn_name, fn_args, session_id)
            messages.append({"role": "tool", "content": result})

    return ChatResponse(
        response="I completed several tool calls but couldn't finish — please try rephrasing.",
        session_id=session_id,
        tool_calls_made=tool_calls_made,
    )


# ---------------------------------------------------------------------------
# SSE streaming endpoint
# ---------------------------------------------------------------------------

@router.post("/stream")
async def chat_stream(req: ChatRequest):
    """Stream tokens via Server-Sent Events for low-latency response."""
    session_id = req.session_id or str(uuid.uuid4())[:8]
    matched_skills = _match_skills(req.message)
    skill_context = "".join(_load_skill(s) for s in matched_skills)

    if session_id not in _conversations:
        _conversations[session_id] = [
            {"role": "system", "content": _build_system_prompt(skill_context, session_id=session_id)}
        ]
    else:
        if skill_context:
            _conversations[session_id][0] = {
                "role": "system",
                "content": _build_system_prompt(skill_context, session_id=session_id),
            }

    messages = _conversations[session_id]
    messages.append({"role": "user", "content": req.message})

    return StreamingResponse(
        _stream_generator(session_id, messages, matched_skills),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


async def _stream_generator(
    session_id: str,
    messages: list[dict],
    active_skills: list[str],
) -> AsyncGenerator[str, None]:
    tool_calls_made: list[str] = []
    full_response = ""

    def _sse(data: dict) -> str:
        return f"data: {json.dumps(data)}\n\n"

    # Emit active skills so the UI can show which skill is loaded
    if active_skills:
        yield _sse({"type": "skills", "skills": active_skills})

    provider, model = llm_providers.get_active_provider()

    import time as _time
    for round_num in range(25):
        accumulated_content = ""
        accumulated_tool_calls: list[dict] = []
        first_token = True

        try:
            async for chunk in await llm_providers.stream_round(
                provider, model, messages, TOOLS, OLLAMA_BASE, NUM_CTX
            ):
                if chunk.get("error"):
                    yield _sse({"type": "error", "message": chunk["error"]})
                    return

                msg_chunk = chunk.get("message", {})
                content_piece = msg_chunk.get("content", "")
                tool_calls_chunk = msg_chunk.get("tool_calls")

                if content_piece:
                    if first_token:
                        first_token = False
                        yield _sse({"type": "start", "session_id": session_id})
                    accumulated_content += content_piece
                    full_response += content_piece
                    yield _sse({"type": "token", "content": content_piece})

                if tool_calls_chunk:
                    accumulated_tool_calls.extend(tool_calls_chunk)

                if chunk.get("done"):
                    break

        except Exception as e:
            yield _sse({"type": "error", "message": f"LLM error ({provider}): {e}"})
            return

        # If no tool calls, this is the final response
        if not accumulated_tool_calls:
            messages.append({"role": "assistant", "content": accumulated_content})
            _trim_conversation(session_id)
            _save_chat(session_id)
            yield _sse({
                "type": "done",
                "session_id": session_id,
                "tool_calls_made": tool_calls_made,
            })
            return

        # Execute tool calls
        messages.append({
            "role": "assistant",
            "content": accumulated_content,
            "tool_calls": accumulated_tool_calls,
        })

        base = _get_backend_base()

        for tc in accumulated_tool_calls:
            fn = tc.get("function", {})
            fn_name = fn.get("name", "unknown")
            fn_args = fn.get("arguments", {})
            tool_calls_made.append(fn_name)

            # Special SSE event for code execution so frontend renders a code block
            if fn_name == "run_python":
                yield _sse({
                    "type": "code_exec",
                    "code": fn_args.get("code", ""),
                    "description": fn_args.get("description", ""),
                })

            yield _sse({"type": "tool_start", "name": fn_name, "args": fn_args})
            _t0 = _time.monotonic()

            # wait_for_job runs inline so we can emit status SSE events during polling
            if fn_name == "wait_for_job":
                jid = fn_args.get("job_id", "")
                jtype = fn_args.get("job_type", "training")
                # download_gdc returns the job id without the 'dl_' prefix the
                # queue stores internally, so normalise both forms here.
                prefix_map = {"gdc": "dl_"}
                prefix = prefix_map.get(jtype, "")
                if prefix and jid and not jid.startswith(prefix):
                    jid = f"{prefix}{jid}"
                    fn_args["job_id"] = jid
                route_map = {
                    "preprocess": f"/api/preprocess/{jid}",
                    "training": f"/api/training/{jid}",
                    "eval": f"/api/eval/{jid}",
                    "features": f"/api/features/{jid}",
                    "lora": f"/api/training/lora/{jid}",
                    "gdc": f"/api/gdc/jobs/{jid}",
                }
                poll_url = f"{base}{route_map.get(jtype, f'/api/training/{jid}')}"

                # Referential guardrail: reject fabricated/empty job_ids up front
                # so we don't burn 30 min polling a phantom. The table-driven
                # validator runs the same filesystem check as _execute_tool.
                _guard_err = validate_tool_args("wait_for_job", fn_args)
                if _guard_err is not None:
                    result = _guard_err
                    jid = ""  # skip the polling loop
                else:
                    try:
                        async with httpx.AsyncClient(timeout=10.0) as pc:
                            preflight = await pc.get(poll_url)
                        if preflight.status_code == 404:
                            result = (
                                f"ERROR: {jtype} job '{jid}' does not exist. "
                                f"You likely fabricated this job_id. "
                                f"Call list_queue or get_job_status to see real job_ids, "
                                f"or call the actual start_* tool first to create one."
                            )
                            jid = ""  # skip the polling loop
                        elif preflight.status_code != 200:
                            result = f"ERROR: cannot reach {poll_url} (HTTP {preflight.status_code})"
                            jid = ""
                    except Exception as e:
                        result = f"ERROR: pre-flight check for job {jid} failed: {e}"
                        jid = ""
                    else:
                        result = f"Job {jid}: still running after 30 min."

                consecutive_errors = 0
                for poll_i in range(180 if jid else 0):
                    elapsed = poll_i * 10
                    yield _sse({"type": "status", "message": f"Waiting for {jtype} job {jid}… {elapsed}s"})
                    await _asyncio.sleep(10)
                    try:
                        async with httpx.AsyncClient(timeout=10.0) as pc:
                            r = await pc.get(poll_url)
                        if r.status_code == 404:
                            # Job vanished mid-poll (server restart or wrong id) — abort, don't loop.
                            result = f"ERROR: job {jid} disappeared during polling (HTTP 404). Aborting wait."
                            break
                        d = r.json()
                        consecutive_errors = 0
                        status = d.get("status", "unknown")
                        if status in ("completed", "failed", "error", "partial"):
                            result = f"Job {jid}: {status}"
                            metrics = d.get("metrics", {})
                            for k, v in metrics.items():
                                result += f"\n  {k}: {v:.4f}" if isinstance(v, float) else f"\n  {k}: {v}"
                            if jtype == "gdc":
                                done_n = d.get("done", 0)
                                total_n = d.get("total", 0)
                                result += f"\n  Downloaded: {done_n}/{total_n}"
                                result += f"\n  Output: {d.get('output_dir', '?')}"
                            break
                    except Exception:
                        consecutive_errors += 1
                        if consecutive_errors >= 3:
                            result = f"ERROR: 3 consecutive poll failures for job {jid}. Aborting wait."
                            break
            else:
                result = await _execute_tool(fn_name, fn_args, session_id)

            _dur_ms = int((_time.monotonic() - _t0) * 1000)
            yield _sse({"type": "tool_result", "name": fn_name, "result": result[:2000], "duration_ms": _dur_ms})

            # Live task-plan updates for the frontend checklist
            if fn_name in ("create_task_plan", "update_task_status") and session_id:
                try:
                    from pathclaw.api.routes.tasks import load_plan as _load_plan
                    _plan_snapshot = _load_plan(session_id)
                    yield _sse({"type": "task_plan", "plan": _plan_snapshot})
                except Exception:
                    pass

            messages.append({"role": "tool", "content": result})

    yield _sse({
        "type": "done",
        "session_id": session_id,
        "tool_calls_made": tool_calls_made,
    })


# ---------------------------------------------------------------------------
# Conversation management
# ---------------------------------------------------------------------------

def _trim_conversation(session_id: str) -> None:
    """Keep conversation within context budget.
    When over limit, compress old messages into a summary instead of dropping them."""
    msgs = _conversations.get(session_id)
    if not msgs or len(msgs) <= MAX_HISTORY_MSGS:
        return

    # Preserve system prompt + last (MAX-5) messages; summarize the rest
    keep_count = MAX_HISTORY_MSGS - 5
    old_msgs = msgs[1:-(keep_count)]
    recent_msgs = msgs[-(keep_count):]

    summary_lines = ["[Summary of earlier conversation:]"]
    for m in old_msgs:
        role = m.get("role", "")
        content = str(m.get("content", ""))
        tool_calls = m.get("tool_calls", [])
        if role == "user" and content:
            summary_lines.append(f"- User: {content[:120]}")
        elif role == "assistant" and content:
            summary_lines.append(f"- Assistant: {content[:150]}")
        elif role == "assistant" and tool_calls:
            names = [tc.get("function", {}).get("name", "?") for tc in tool_calls]
            summary_lines.append(f"- Called tools: {', '.join(names)}")
        elif role == "tool" and content:
            summary_lines.append(f"- Tool result: {content[:120]}")

    summary_msg = {"role": "user", "content": "\n".join(summary_lines)}
    _conversations[session_id] = [msgs[0], summary_msg] + recent_msgs


@router.delete("/{session_id}")
async def clear_conversation(session_id: str):
    """Clear a conversation session."""
    _conversations.pop(session_id, None)
    return {"status": "ok", "session_id": session_id}


# ---------------------------------------------------------------------------
# Session lifecycle endpoints
# ---------------------------------------------------------------------------

class SessionUpdate(BaseModel):
    title: str | None = None


@router.post("/sessions")
async def create_session():
    """Create a new empty session."""
    sid = str(uuid.uuid4())[:8]
    now = _time_mod.time()
    _conversations[sid] = [{"role": "system", "content": _build_system_prompt(session_id=sid)}]
    payload = {
        "session_id": sid,
        "title": "New Session",
        "created_at": now,
        "updated_at": now,
        "messages": [],
    }
    _chat_path(sid).write_text(json.dumps(payload, indent=2))
    return {"session_id": sid, "title": "New Session"}


class RenameSessionBody(BaseModel):
    slug: Optional[str] = None
    title: Optional[str] = None


@router.post("/sessions/{session_id}/rename")
async def rename_session(session_id: str, body: RenameSessionBody):
    """Set a short slug and/or title on a session.

    Slugs are kebab-case, unique across sessions, <=40 chars. Used by Telegram
    (`/session <slug>`) and the UI sidebar."""
    actual = _resolve_session(session_id)
    if not actual:
        raise HTTPException(404, "Session not found")
    p = _chat_path(actual)
    data = json.loads(p.read_text())

    if body.slug is not None:
        new_slug = _slugify(body.slug)
        if body.slug and not new_slug:
            raise HTTPException(400, "Slug must contain letters or digits")
        if new_slug:
            for other in CHATS_DIR.glob("*.json"):
                if other == p:
                    continue
                try:
                    od = json.loads(other.read_text())
                except Exception:
                    continue
                if od.get("slug") == new_slug:
                    raise HTTPException(409, f"Slug '{new_slug}' already in use by session {od.get('session_id')}")
        data["slug"] = new_slug or None

    if body.title is not None:
        t = body.title.strip()
        if t:
            data["title"] = t[:200]

    data["updated_at"] = _time_mod.time()
    p.write_text(json.dumps(data, indent=2))
    return {"session_id": actual, "slug": data.get("slug"), "title": data.get("title")}


@router.post("/sessions/{session_id}/resume")
async def resume_session(session_id: str):
    """Load session for display and hydrate in-memory conversation from disk."""
    actual = _resolve_session(session_id)
    if not actual:
        raise HTTPException(404, "Session not found")
    session_id = actual
    p = _chat_path(session_id)
    if not p.exists():
        raise HTTPException(404, "Session not found")
    data = json.loads(p.read_text())
    # Hydrate in-memory conversation if missing (after server restart)
    if session_id not in _conversations:
        messages = [{"role": "system", "content": _build_system_prompt(session_id=session_id)}]
        full = data.get("full_messages")
        if full:
            # Restore complete tool context (tool messages, tool_calls on assistant messages)
            messages.extend(full)
        else:
            # Fallback for sessions saved before full_messages was added
            for m in data.get("messages", []):
                if m.get("role") in ("user", "assistant") and m.get("content"):
                    messages.append({"role": m["role"], "content": m["content"]})
        _conversations[session_id] = messages
    return data


# ---------------------------------------------------------------------------
# Chat history & memory endpoints
# ---------------------------------------------------------------------------

@router.get("/history")
async def list_chats():
    """List all saved chat sessions."""
    return {"chats": _list_chats()}


@router.get("/history/{session_id}")
async def get_chat(session_id: str):
    """Load a saved chat session (user+assistant messages only)."""
    p = _chat_path(session_id)
    if not p.exists():
        raise HTTPException(404, "Chat not found")
    return json.loads(p.read_text())


@router.patch("/history/{session_id}")
async def rename_chat(session_id: str, update: SessionUpdate):
    """Rename a session."""
    p = _chat_path(session_id)
    if not p.exists():
        raise HTTPException(404, "Chat not found")
    data = json.loads(p.read_text())
    if update.title is not None:
        data["title"] = update.title
        data["updated_at"] = _time_mod.time()
    p.write_text(json.dumps(data, indent=2))
    return {"status": "ok", "session_id": session_id}


@router.delete("/history/{session_id}")
async def delete_chat(session_id: str):
    """Delete a saved chat session."""
    p = _chat_path(session_id)
    if p.exists():
        p.unlink()
    notes = _notes_path(session_id)
    if notes.exists():
        notes.unlink()
    # Also clean up the session's manuscript project
    ms_dir = CHATS_DIR / f"{session_id}_manuscript"
    if ms_dir.exists():
        import shutil as _shutil
        _shutil.rmtree(ms_dir, ignore_errors=True)
    _conversations.pop(session_id, None)
    return {"status": "ok"}


@router.get("/notes/{session_id}")
async def get_session_notes(session_id: str):
    """Read per-session notebook (markdown)."""
    return {"session_id": session_id, "notes": _read_session_notes(session_id)}


class NoteAppend(BaseModel):
    topic: str
    content: str


@router.post("/notes/{session_id}")
async def append_session_notes(session_id: str, note: NoteAppend):
    """Append an entry to a session's notebook."""
    path = _append_session_note(session_id, note.topic, note.content)
    return {"status": "ok", "path": path}


@router.delete("/notes/{session_id}")
async def clear_session_notes(session_id: str):
    """Clear a session's notebook (user-invoked reset)."""
    p = _notes_path(session_id)
    if p.exists():
        p.unlink()
    return {"status": "ok"}


@router.get("/manuscript/{session_id}")
async def list_manuscript(session_id: str):
    """List files in this session's manuscript project."""
    return {
        "session_id": session_id,
        "dir": str(_manuscript_dir(session_id)),
        "files": _list_manuscript_files(session_id),
    }


@router.get("/manuscript/{session_id}/file/{filename}")
async def read_manuscript_file(session_id: str, filename: str):
    """Read a single manuscript file as plain text."""
    from fastapi.responses import PlainTextResponse
    try:
        path = _safe_manuscript_path(session_id, filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not path.exists():
        raise HTTPException(status_code=404, detail="Not found")
    return PlainTextResponse(path.read_text(errors="replace"))


class ManuscriptWrite(BaseModel):
    filename: str
    content: str
    mode: str = "write"


@router.post("/manuscript/{session_id}/file")
async def write_manuscript_file(session_id: str, body: ManuscriptWrite):
    """Create/overwrite a manuscript file (user-driven edits from the UI)."""
    try:
        path = _safe_manuscript_path(session_id, body.filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    path.parent.mkdir(parents=True, exist_ok=True)
    if body.mode == "append" and path.exists():
        with path.open("a") as f:
            f.write(body.content)
    else:
        path.write_text(body.content)
    return {"status": "ok", "size": path.stat().st_size}


@router.delete("/manuscript/{session_id}/file/{filename}")
async def delete_manuscript_file(session_id: str, filename: str):
    try:
        path = _safe_manuscript_path(session_id, filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if path.exists():
        path.unlink()
    return {"status": "ok"}


@router.post("/manuscript/{session_id}/compile")
async def compile_manuscript_endpoint(session_id: str, main_file: str = "main.tex"):
    """Compile the LaTeX project and return PDF URL or error log."""
    result = _compile_latex(session_id, main_file)
    if result["status"] == "ok":
        pdf_name = Path(result["pdf"]).name
        result["pdf_url"] = f"/api/chat/manuscript/{session_id}/pdf/{pdf_name}"
    return result


class AttachFigure(BaseModel):
    job_id: str
    job_type: str = "training"          # training | eval
    filename: str                        # plot filename (e.g. roc_curve.png)
    caption: str = ""
    label: str = ""                      # LaTeX label; if empty, derived from filename
    width: str = "0.8\\linewidth"
    insert_in_tex: bool = True           # append a figure block to main.tex


def _safe_fig_name(name: str) -> str:
    """Strip any path parts and refuse non-image extensions."""
    base = Path(name).name
    if Path(base).suffix.lower() not in {".png", ".jpg", ".jpeg", ".pdf", ".svg"}:
        raise ValueError("Only PNG/JPG/PDF/SVG figures are allowed.")
    return base


@router.post("/manuscript/{session_id}/attach-figure")
async def attach_figure(session_id: str, body: AttachFigure):
    """Copy an experiment plot into the session's manuscript figures/ folder and
    (optionally) append a \\begin{figure}...\\end{figure} block to main.tex.
    Returns the LaTeX snippet so the frontend can surface it."""
    import shutil as _shutil

    try:
        exp_dir = DATA_DIR / "experiments" / body.job_id
        src = exp_dir / "plots" / body.filename
        if not src.exists() or not src.is_file():
            available = []
            plots_dir = exp_dir / "plots"
            if plots_dir.exists():
                available = sorted(p.name for p in plots_dir.glob("*.*"))
            if not available:
                raise HTTPException(
                    status_code=404,
                    detail=f"Experiment '{body.job_id}' has no plots directory yet. "
                           "Train a model first or run `make_plot` to generate one.",
                )
            raise HTTPException(
                status_code=404,
                detail=f"Plot '{body.filename}' not found in {body.job_id}. "
                       f"Available: {', '.join(available[:10])}",
            )

        try:
            fname = _safe_fig_name(body.filename)
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=str(ve))

        fig_dir = _manuscript_dir(session_id) / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        dest = fig_dir / fname
        if dest.exists():
            try:
                if dest.read_bytes() != src.read_bytes():
                    dest = fig_dir / f"{Path(fname).stem}_{body.job_id}{Path(fname).suffix}"
            except Exception:
                pass
        _shutil.copy2(src, dest)

        label = body.label or f"fig:{Path(dest.stem).name}"
        caption = body.caption or Path(dest.stem).name.replace("_", " ")
        snippet = (
            "\\begin{figure}[ht]\n"
            "  \\centering\n"
            f"  \\includegraphics[width={body.width}]{{{dest.name}}}\n"
            f"  \\caption{{{caption}}}\n"
            f"  \\label{{{label}}}\n"
            "\\end{figure}\n"
        )

        inserted = False
        if body.insert_in_tex:
            main = _manuscript_dir(session_id) / "main.tex"
            if not main.exists():
                main.write_text(_DEFAULT_LATEX_TEMPLATE)
            text = main.read_text()
            marker = "\\end{document}"
            if marker in text:
                text = text.replace(marker, snippet + "\n" + marker, 1)
            else:
                text = text + "\n" + snippet
            main.write_text(text)
            inserted = True

        return {
            "status": "ok",
            "figure_path": f"figures/{dest.name}",
            "snippet": snippet,
            "inserted_in_main_tex": inserted,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"attach_figure failed for session={session_id} job={body.job_id}")
        raise HTTPException(status_code=400, detail=f"attach_figure failed: {type(e).__name__}: {e}")


@router.get("/manuscript/{session_id}/figure/{filename}")
async def serve_manuscript_figure(session_id: str, filename: str):
    """Serve a figure from the manuscript's figures/ subdirectory."""
    from fastapi.responses import FileResponse
    root = _manuscript_dir(session_id) / "figures"
    safe = Path(filename).name
    path = (root / safe).resolve()
    if root.resolve() not in path.parents or not path.exists():
        raise HTTPException(status_code=404, detail="Figure not found")
    ext = path.suffix.lower()
    media = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
             ".pdf": "application/pdf", ".svg": "image/svg+xml"}.get(ext, "application/octet-stream")
    return FileResponse(str(path), media_type=media)


@router.get("/manuscript/{session_id}/export")
async def export_manuscript(session_id: str):
    """Download the full manuscript project (LaTeX sources + figures + compiled PDF if present) as a zip."""
    from fastapi.responses import StreamingResponse
    import io
    import zipfile
    root = _manuscript_dir(session_id)
    if not any(root.iterdir()):
        raise HTTPException(status_code=404, detail="Manuscript project is empty.")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in root.rglob("*"):
            if p.is_file():
                zf.write(p, p.relative_to(root))
    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="manuscript_{session_id}.zip"'},
    )


@router.get("/manuscript/{session_id}/pdf/{filename}")
async def serve_manuscript_pdf(session_id: str, filename: str):
    """Serve a compiled PDF from the manuscript project."""
    from fastapi.responses import FileResponse
    try:
        path = _safe_manuscript_path(session_id, filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not path.exists() or path.suffix.lower() != ".pdf":
        raise HTTPException(status_code=404, detail="PDF not found")
    return FileResponse(str(path), media_type="application/pdf")


@router.get("/memory")
async def get_memory(session_id: str = ""):
    """Get memory facts for a session. Memory is session-scoped — pass session_id."""
    return {"memory": _load_memory(session_id), "session_id": session_id}


@router.delete("/memory/{key}")
async def delete_memory_key(key: str, session_id: str = ""):
    """Delete a specific memory key from the given session."""
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id query param required")
    mem = _load_memory(session_id)
    mem.pop(key, None)
    _memory_path(session_id).write_text(json.dumps(mem, indent=2))
    return {"status": "ok"}
