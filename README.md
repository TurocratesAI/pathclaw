<p align="center">
  <img src="docs/logo.jpeg" width="180" alt="PathClaw"><br/>
  <b>PathClaw — viberesearching</b><br/>
  <i>An AI lab where multiple scientists work in parallel on your medical-imaging problems.</i>
</p>

## About

PathClaw is a viberesearching environment for medical imaging. The first release focuses on **computational pathology**: WSI preprocessing, multiple-instance learning, genomic analysis, and per-session manuscript authoring. Future releases extend the same agent-driven workflow to radiology and other medical imaging modalities.

It is a FastAPI + browser-based research platform. An in-app LLM agent drives the full pipeline — TCGA download → WSI preprocessing → feature extraction → MIL training → evaluation → genomic analysis — through a chat interface, invoking 73 typed tools and 15 keyword-triggered skill modules. You describe the experiment in plain English; the agent runs it, confirms before destructive steps, and writes results back into a per-session workspace you control.

<!-- Drop a single platform screenshot at docs/screenshot.png (recommended: three-pane default view with sidebar + center Overview + chat panel). -->
<!-- ![PathClaw UI](docs/screenshot.png) -->

---

## Contents

- [Quickstart](#quick-setup)
- [First experiment in 5 minutes](#first-experiment-in-5-minutes)
- [Features](#what-it-does)
- [All tools (73)](#all-tools-73-total)
- [Architecture](#architecture)
- [Session model](#research-workspace--multi-agent-multi-project)
- [Plugin system](#plugin-system)
- [Troubleshooting](docs/TROUBLESHOOTING.md) · [Manual](MANUAL.md) · [Plugin dev](docs/PLUGIN_DEV.md) · [Security](docs/SECURITY.md) · [Development](docs/DEVELOPMENT.md)

---

## What it does

### Core Pipeline
- **Chat-driven pipeline**: Tell the agent what you want in plain English. It calls the right tools in sequence, confirms before big operations, and explains what it's doing in real time.
- **MIL classification**: ABMIL, TransMIL, CLAM, DSMIL, RRTMIL, WIKG — with optional MAMMOTH MoE patch embeddings (plugin)
- **Foundation model feature extraction**: any HuggingFace vision backbone by repo id — UNI, CONCH/CONCHv1.5, CTransPath, Virchow/Virchow2, GigaPath, Phikon/Phikon-v2, H-Optimus, Lunit DINO, etc. Pass `backbone=<hf/repo>` to `start_feature_extraction`; dimension is auto-detected. Six presets ship with tuned loaders for gated/SwiGLU/custom-pooling models.
- **LoRA fine-tuning**: Adapt any backbone to your cohort before re-extracting features
- **Segmentation**: SegUNet (semantic), HoVer-Net (instance), Cellpose (zero-shot cell detection)

### Genomics & Multi-Omic Analysis
- **MAF/VCF/clinical XML parsing**: Structured summaries, gene-specific queries, variant classification breakdown
- **Cohort mutation queries**: Mutation frequencies across entire MAF directories with gene/variant filters
- **Tumor Mutational Burden (TMB)**: Per-sample TMB with Low/Medium/High classification
- **Label extraction**: Slide-level labels from genomic data — MSI status, mutation status, TMB class, clinical fields — with TCGA barcode resolution and patient deduplication
- **cBioPortal integration**: Query mutations, clinical data, CNA, and MSI scores without downloading raw files
- **Survival analysis**: Kaplan-Meier curves with log-rank test, stratified by molecular subtypes or model predictions
- **Oncoplot generation**: Mutation landscape plots showing top mutated genes across a cohort
- **Biomarker discovery**: Differential mutation enrichment between groups, attention-gene correlation
- **Multi-omic label builder**: Merge MAF, clinical, expression, and model predictions into unified matrices
- **Gene expression parsing**: STAR/HTSeq/FPKM expression file analysis

### Research Workspace — multi-agent, multi-project

PathClaw is designed like a **research group**, not a single chatbot. Each chat session is an **independent agent working on its own project** — think one PhD student per session, each with their own notebook, library, and draft. You can run many at once: one session studying UCEC-MSI, another doing BRCA subtyping, another surveying foundation models — and they don't step on each other's context.

Each session gets:
- **Its own notebook** (`~/.pathclaw/chats/{sid}.notes.md`) — a running markdown log of decisions, dataset paths, job IDs, and interim results. Auto-injected into the agent's system prompt every round, so context survives trimming and compaction.
- **Its own LaTeX manuscript** (`~/.pathclaw/chats/{sid}_manuscript/`) — go from ideation → experiments → paper draft in the same chat. The agent writes `main.tex`, adds citations to `refs.bib` as it reads papers, and compiles to PDF (`tectonic` or `texlive-latex-base` required for the compile step).
- **Its own attached folders** — PDFs uploaded and attached per-session, so each project sees only its relevant literature.
- **A human-readable name (slug)** — every session has a UUID and an optional kebab-case slug you can set. Give a session a name like `chol-idh1` and you can refer to it by that name everywhere: in the sidebar, in Telegram (`/session chol-idh1`), or in-chat (ask the agent to `rename_session` to something memorable). Slugs are unique across sessions and fall back to a session-id prefix if unset.

Cross-session features:
- **Web research** from any session — PubMed search, URL fetch (with auto-routing through NCBI E-utilities for PubMed/PMC), paper-PDF ingestion.
- **Telegram bridge** — run one bot worker and ping any of your running projects from your phone. `/sessions` lists them with their slugs, `/session <slug-or-id>` binds the Telegram chat to one project (slug, full id, or any id prefix ≥4 chars works), `/new <title>` spins up a fresh one, and any plain message goes to the bound agent. Supports username whitelist + optional `/start <passcode>` gate so only you can talk to your bots.
- **Session-scoped memory** — facts you tell a session to `remember` (dataset ids, experiment ids, paths, conventions) stay in that session (`~/.pathclaw/chats/{sid}.memory.json`). Projects don't leak context into each other.

### Bring Your Own LLM (BYOL)
PathClaw is LLM-agnostic. Out of the box it supports:
- **Ollama** (local, any model you've pulled — Qwen, Llama, Gemma, Mistral, DeepSeek, Phi, etc.)
- **Anthropic** Claude (Opus/Sonnet/Haiku, API key)
- **OpenAI** GPT (4o, 4.1, o-series, API key)
- **Google** Gemini (2.5/2.0 Flash and Pro, API key)
- **Any OpenAI-compatible endpoint** via custom base URL: OpenRouter, LM Studio, vLLM, Together, Groq, Fireworks — set `openai_base` in settings and put any model tag in the model field.

Switch anytime from the gear icon (top-right) or by clicking the LLM pill in the sidebar.

### Visualization & Data Access
- **WSI viewer**: OpenSeadragon-based viewer with DZI tile streaming, attention heatmap overlay, and GeoJSON annotation overlay (QuPath export compatible)
- **GDC/TCGA integration**: Search and download slides, clinical supplements, MAF files, gene expression, copy number, and methylation data from the NCI GDC
- **Multi-provider LLM**: Ollama (local), Anthropic, OpenAI, Google Gemini — switch in settings

---

## Quick setup

### Requirements

| | |
|---|---|
| Python | 3.10+ |
| GPU | 8 GB+ VRAM (CPU works for inference; too slow for training) |
| Ollama | For local LLM — `curl -fsSL https://ollama.com/install.sh \| sh` |
| openslide | For WSI viewing — `apt install openslide-tools` or `brew install openslide` |

### Install

```bash
git clone https://github.com/devanshlalwani/PathClaw.git
cd PathClaw/backend
pip install -e ".[dev]"
```

### Pull a model (skip if using a cloud API)

```bash
ollama pull qwen3:8b        # fast, good tool use
# or
ollama pull llama3.3:8b     # stronger reasoning
```

> **Note on model size.** Smaller local models (≤7–8B, including `gemma4:e4b`, `qwen3:4b`, etc.) work for short one-off prompts but tend to drift on multi-step pipelines — hallucinated tool names, placeholder arguments, or forgetting earlier steps. If you see the agent repeatedly calling the wrong tool or inventing file IDs, try `gemma4:26b`, `qwen3:32b`, or a cloud provider (Anthropic / OpenAI / Google). For automated workflows (Telegram, overnight runs), favour a ≥20B local model or a cloud API.

### Start the server

```bash
uvicorn pathclaw.api.app:app --host 0.0.0.0 --port 8101 --reload
```

Open **http://localhost:8101** — the onboarding modal will guide you through setting a HuggingFace token (needed for gated backbones: UNI, CONCH, Virchow, GigaPath).

### Optional environment variables

```bash
export PATHCLAW_DATA_DIR=~/.pathclaw          # where data, models, logs are stored
export PATHCLAW_BACKEND_BASE=http://localhost:8101
export OLLAMA_BASE=http://localhost:11434
export HUGGINGFACE_TOKEN=hf_xxx               # or set in the UI settings modal
```

---

## First experiment in 5 minutes

In the chat box:

```
Register my slides at /path/to/my/slides as MyDataset
```

Then:

```
Preprocess MyDataset with 256px patches at 20x magnification
Extract UNI features for MyDataset
Train ABMIL with MAMMOTH on MyDataset, label column is tumor_label
What are my results?
```

The agent handles every step. It will ask for confirmation before long operations.

---

## Plugin system

PathClaw has a plugin registry at `~/.pathclaw/plugins/user_registry.json` (user overlay) and a built-in registry shipped with the backend. Built-ins: **mammoth** (MoE-LoRA patch embedding), **cellpose** (nuclei/cell instance segmentation, editable config).

Plugin contract:

```python
def build(in_dim: int, embed_dim: int, config: dict) -> nn.Module:
    # forward: (B, N, in_dim) -> (B, N, embed_dim)
    ...
```

Agent tools: `list_plugins`, `register_plugin`, `update_plugin_config`, `smoke_test_plugin`, `implement_from_paper`, `clone_repo`, `analyze_repo`. See [docs/PLUGIN_DEV.md](docs/PLUGIN_DEV.md).

---

## All tools (73 total)

### Datasets (3)
| Tool | Description |
|------|-------------|
| `list_datasets` | List all registered datasets |
| `register_dataset` | Register a dataset from local folder or single WSI file |
| `get_dataset_profile` | Quality report: class balance, slide counts, format distribution |

### GDC/TCGA (3)
| Tool | Description |
|------|-------------|
| `search_gdc` | Search GDC for slides, MAFs, clinical, expression, copy number, methylation |
| `download_gdc` | Async download via detached subprocess (survives server restarts) |
| `gdc_job_status` | Check download progress |

### Preprocessing & Features (3)
| Tool | Description |
|------|-------------|
| `start_preprocessing` | Launch Otsu tissue detection + patching job |
| `start_feature_extraction` | Extract features via foundation model (runs as detached subprocess) |
| `cancel_feature_job` | Cancel a running extraction job |

Any HuggingFace vision backbone works — pass its repo id as `backbone` and the loader
auto-detects the embedding dimension. Six presets ship with tuned weight loaders
(gated-weights handling, SwiGLU, custom pooling) so the common pathology models
plug-and-play; everything else loads via a generic `timm` / `AutoModel` path.

Presets with tuned loaders:

| Backbone | Dim | Source |
|----------|-----|--------|
| UNI | 1024 | MahmoodLab/UNI |
| CONCH | 512 | MahmoodLab/CONCH |
| CTransPath | 768 | X-zhangyang/CTransPath |
| Virchow | 1280 | paige-ai/Virchow |
| Virchow2 | 2560 | paige-ai/Virchow2 (SwiGLU, CLS+mean pooling) |
| GigaPath | 1536 | prov-gigapath/prov-gigapath |

Other HF backbones confirmed working via the generic loader: Phikon / Phikon-v2,
H-Optimus-0, Lunit DINO, Hibou, RudolfV, Kaiko pathology models, and any
`timm`/`AutoModel`-loadable ViT. Set `backbone="owner/repo"` on
`start_feature_extraction` and the dim is inferred from the first batch.

### Training (3)
| Tool | Description |
|------|-------------|
| `start_training` | Launch MIL or segmentation training job |
| `start_lora_finetuning` | LoRA fine-tune a foundation model backbone |
| `get_training_logs` | Tail training log file |

MIL methods: ABMIL, MeanPool, TransMIL, CLAM, DSMIL, RRTMIL, WIKG (all with optional MAMMOTH)

### Evaluation & Visualization (4)
| Tool | Description |
|------|-------------|
| `start_evaluation` | Run evaluation on val/test split |
| `get_eval_metrics` | AUROC, accuracy, balanced accuracy, macro/weighted F1, QWK, PR-AUC, sensitivity, specificity, confusion matrix |
| `get_eval_plots` | ROC curves, confusion matrix PNGs |
| `generate_heatmap` | Attention heatmap overlay for WSI viewer |

**Metric selection.** Both `start_training` and `start_evaluation` take `evaluation.metrics`
(a list). The agent is prompted to pause before training and ask which metric matters for
your task — `auroc` (default for binary mutation/subtype calling), `accuracy` /
`balanced_accuracy` (balanced cohorts), `macro_f1` / `weighted_f1` (class imbalance),
`qwk` (ordinal targets: grade I–IV, Gleason), or `sensitivity` / `specificity` (clinical
screening with a fixed threshold). All metrics are computed regardless; the first entry
drives early stopping and best-checkpoint selection.

### Job status includes live ETA
`GET /api/training/{id}`, `/api/features/{id}`, and `/api/preprocess/{id}` now return
`elapsed_human`, `eta_seconds`, and `eta_human` fields (ETA = `elapsed × (1 − progress) /
progress`). The jobs panel surfaces these alongside epoch/loss, and the Telegram bot prints
a `⏱ running · 42% · elapsed 3m 10s · ETA 4m 20s` line whenever the agent polls job
status — so you can ask "how far along is it?" from either UI and get a numeric answer.

### Genomics (10)
| Tool | Description |
|------|-------------|
| `parse_genomic_file` | Parse MAF/VCF/clinical XML with structured queries |
| `query_mutations` | Cohort-level mutation queries across MAF files |
| `compute_tmb` | Tumor Mutational Burden per sample with classification |
| `extract_labels_from_genomic` | Slide-level labels (MSI, mutation, TMB, clinical) with TCGA barcode resolution |
| `query_cbioportal` | Query cBioPortal REST API for mutations, clinical, CNA, MSI |
| `run_survival_analysis` | Kaplan-Meier survival analysis with log-rank test |
| `build_multi_omic_labels` | Merge MAF/clinical/predictions into unified matrix |
| `generate_oncoplot` | Mutation landscape plot (PNG + text summary) |
| `parse_gene_expression` | Parse STAR/HTSeq/FPKM expression files |
| `biomarker_discovery` | Differential mutation enrichment + attention-gene correlation |

### Workspace & Plugins (15)
| Tool | Description |
|------|-------------|
| `list_workspace_files` | List per-session workspace tree |
| `read_workspace_file` | Read a file from the session workspace |
| `write_workspace_file` | Write / overwrite a workspace file |
| `delete_workspace_file` | Delete a workspace file (no recursive rm) |
| `clone_repo` | Shallow-clone a repo (github/gitlab/huggingface/codeberg/bitbucket) into the session workspace |
| `analyze_repo` | Structured repo summary (languages, README excerpt, candidate architecture files) |
| `list_plugins` | List built-in + user-registered plugins with config + installed status |
| `register_plugin` | Register a plugin manifest (id, kind, import_path, default_config) |
| `update_plugin_config` | Persist a config override for a plugin |
| `smoke_test_plugin` | Instantiate the plugin and run a forward pass with dummy tensors |
| `implement_from_paper` | Scaffold a plugin from a paper (workflow prompt, not codegen) |
| `list_backbones` | List feature backbones (built-in + user-registered overlay) |
| `register_hf_backbone` | Register a new HF foundation model for feature extraction |
| `ask_user` | Emit an inline question card and wait for the next user message |
| `run_cellpose_segmentation` | Run cellpose on slides/patches with the current plugin config |

### Memory & Manuscript (7)
| Tool | Description |
|------|-------------|
| `remember_fact` / `recall_facts` | Cross-session memory (HF token paths, conventions, etc.) |
| `write_note` / `read_notes` | Per-session notebook (`~/.pathclaw/chats/{sid}.notes.md`) |
| `write_manuscript` / `read_manuscript` / `compile_manuscript` | Per-session LaTeX manuscript; tectonic/pdflatex compile |
| `rename_session` | Set a kebab-case slug (and optional title) on this session so Telegram / sidebar can reference it by name |

### Literature & Papers (6)
| Tool | Description |
|------|-------------|
| `search_literature` | PubMed + Semantic Scholar unified search |
| `deep_literature_review` | Multi-query literature survey with summarization |
| `get_paper_citations` | Retrieve citations for a paper ID |
| `pubmed_search` | NCBI E-utilities search with abstracts |
| `download_paper_pdf` | Download a PDF into a named folder and parse text |
| `read_pdf` | Read a PDF from an attached folder |

### System & Runtime (6)
| Tool | Description |
|------|-------------|
| `get_job_status` | Poll any job (preprocessing, feature, training, eval) |
| `wait_for_job` | Block until job completes with live status updates |
| `get_system_status` | GPU, storage, backend health |
| `fetch_url` | Fetch a URL and return cleaned text |
| `list_folders` | List user-uploaded PDF collections attached to the session |
| `run_python` | Execute arbitrary Python (pandas, numpy, pathlib pre-imported) |

### Artifacts & Training Ops (3)
| Tool | Description |
|------|-------------|
| `list_artifacts` | List all experiments with model availability |
| `compare_experiments` | Side-by-side metric comparison across experiments |
| `make_plot` | Generate a custom plot for an experiment (ROC / PR / calibration / confusion / custom matplotlib) |

### IHC scoring (3)
Rule-based — no training required. Color deconvolution (`skimage.rgb2hed`) separates
H / E / DAB; cellpose (or a morphology fallback) segments nuclei; per-cell or
membrane-band DAB is aggregated to a clinical score per slide. Writes
`~/.pathclaw/datasets/{id}/ihc_{rule}.csv` with one row per slide.

| Tool | Description |
|------|-------------|
| `list_ihc_rules` | Enumerate built-in presets with compartments + thresholds |
| `score_ihc` | Score every slide in a dataset against a preset (`rule_override` patches any field on the fly) |
| `build_ihc_patch_labels` | Emit per-patch + per-slide CSVs from the rule — supervision for a learned regressor on top of UNI/GigaPath features |

Built-in presets (extend via `register_rule` in plugin code):
| Preset | Marker | Compartment | Aggregation | Output range |
|--------|--------|-------------|-------------|-----------|
| `ki67_pi` | Ki-67 | nuclear | % positive nuclei | 0–100 |
| `er_allred` | ER | nuclear | Allred proportion + intensity | 0–8 |
| `pr_allred` | PR | nuclear | Allred proportion + intensity | 0–8 |
| `her2_membrane` | HER2 | membrane | DAB band + completeness | 0 / 1+ / 2+ / 3+ |
| `pdl1_tps` | PD-L1 | membrane | Tumor Proportion Score | 0–100 |

Dynamic rules: pass `rule_override={"dab_threshold": 0.12, "patches_per_slide": 400, ...}` to
`score_ihc` for ad-hoc adjustments without editing code. Custom markers (CK7, p53,
E-cadherin, …) register at runtime — see `docs/PLUGIN_DEV.md`.

### Task plan — prevent multi-step drift
Small local LLMs (`gemma4:26b`, `qwen3:8b`) lose long plans mid-conversation. For any
request that needs 3+ tool calls (paper generation, full pipelines, IHC cohort
scoring, genomic workflows), the agent now **commits to a plan up front**, one task
per step, via three tools:

| Tool | Description |
|------|-------------|
| `create_task_plan` | Post an ordered checklist at the start of a multi-step request |
| `update_task_status` | Flip tasks to `in_progress` / `completed` / `skipped` as it works |
| `get_task_plan` | Re-read the plan (the live plan is also injected into every system prompt so the agent never loses track) |

The plan is stored at `~/.pathclaw/sessions/{sid}/tasks.json`, rendered live in the
sidebar below Active Jobs, and ticked off in real time as each `update_task_status`
call fires. Set `pause_after: true` on a task to make the agent stop and wait for
your input before moving to the next step (default is auto-advance). SSE events
(`task_plan`) keep the frontend checklist in sync mid-turn.

## Writing custom Python scripts

The agent has a **`run_python`** tool that executes arbitrary code in a sandbox-ish worker with `pandas`, `numpy`, `pathlib`, and `subprocess` pre-imported. Use `print()` for all output — the captured stdout becomes the tool result the LLM sees.

```python
# Example the agent might run:
import pandas as pd
from pathlib import Path

labels = pd.read_csv(Path('~/.pathclaw/downloads/tcga-ucec/ucec_msi_labels_final.csv').expanduser())
print(labels['msi_status'].value_counts())
print(f"Total patients: {len(labels)}")
```

Rules:
- Every script must `print()` what it computed — return values are discarded
- Prefer one-shot scripts (load, compute, print); avoid long-running loops
- Paths use `Path('~/.pathclaw/...').expanduser()` — absolute paths from tool outputs are fine
- For anything that hits the GPU or takes > 30s, use the dedicated tools (`start_training`, `start_feature_extraction`) instead
- If you need a library that's not installed, ask the user to `pip install` it before the run

---

## Architecture

```
Browser  (index.html + app.js -- single-page, no build step)
   |  SSE streaming + REST
FastAPI backend (pathclaw.api.app)
   |  tool calls (httpx to self)
LLM provider (Ollama / Anthropic / OpenAI / Google)
   |  dispatches to
Tool handlers -> GDC API, PyTorch training, OpenSlide, genomics, filesystem
```

### Key directories

```
backend/pathclaw/
  api/
    app.py                  # FastAPI app, config endpoints, LLM status
    llm_providers.py        # Unified interface: Ollama / Anthropic / OpenAI / Google
    routes/
      chat.py               # Agent loop (6 rounds), SSE streaming, all tool definitions
      training.py           # Training job launch + segmentation + LoRA dispatch
      features.py           # Feature extraction (detached subprocess, cancel support)
      tileserver.py         # DZI tile serving, heatmap tiles, GeoJSON upload/serve
      datasets.py           # Dataset registration (directory or single file)
      evaluation.py         # Metrics, plots, heatmap generation
      gdc.py                # GDC/TCGA search + download
  training/
    trainer.py              # MIL training loop: k-fold, early stopping, metrics
    seg_trainer.py          # Segmentation training loop (UNet / HoVer-Net / Cellpose)
    lora_finetuner.py       # LoRA backbone fine-tuning via peft
    models/                 # abmil, transmil, clam, dsmil, rrtmil, wikg, seg_unet, seg_hovernet
  preprocessing/
    pipeline.py             # Otsu tissue detection, patching, coord saving
    feature_extraction.py   # All backbone inference (UNI, CONCH, CTransPath, Virchow/2, GigaPath)
    _extract_worker.py      # Standalone subprocess worker for feature extraction
  genomics/
    parsers.py              # MAF/VCF/clinical XML parsing, mutation queries, TMB
    label_extraction.py     # Slide-level label extraction with TCGA barcode resolution
    cbioportal.py           # cBioPortal REST API client
    survival.py             # Kaplan-Meier, log-rank test, survival data extraction
    visualization.py        # Oncoplot / mutation landscape rendering
    expression.py           # Gene expression file parsing (STAR, HTSeq, FPKM)
    multi_omic.py           # Multi-omic data integration
    biomarker.py            # Differential mutation and attention-gene correlation
  evaluation/
    heatmap.py              # Attention weight extraction + heatmap JSON generation
  static/
    index.html              # All HTML + CSS
    app.js                  # All JS: chat SSE, file tree, OSD viewer, GeoJSON overlay

workspace/                  # Agent knowledge base (edit to change agent behaviour)
  AGENTS.md                 # Agent identity, routing rules, tool list
  SKILLS_SUMMARY.md         # Always-loaded skill index + trigger keywords
  TOOLS.md                  # Complete tool reference with parameters
  USE_CASES.md              # 13 clinical benchmark use cases
  skills/
    gdc-tcga/               # TCGA download expertise
    dataset-intake/         # Dataset registration and validation
    wsi-preprocess/         # Patching and feature extraction
    train-config/           # Training config design
    train-exec/             # Job execution and monitoring
    evaluation/             # Metrics and evaluation
    results/                # Results interpretation
    segmentation/           # Segmentation model guidance
    lora-finetune/          # LoRA fine-tuning guidance
    data-profiling/         # Dataset QA
    data-cleaning/          # Label harmonization
    data-lifecycle/         # Storage management
    genomic-analysis/       # MAF/VCF parsing, mutations, TMB
    label-engineering/      # Genomic label extraction
    survival-biomarker/     # Survival analysis, biomarker discovery
```

---

## How the agent works

Every user message goes through this pipeline:

1. **Keyword matching** — `chat.py:_match_skills()` scans the message for trigger words and injects the matching `SKILL.md` file(s) into the system prompt
2. **System prompt** = `AGENTS.md` + `SKILLS_SUMMARY.md` + matched skill(s) — assembled fresh each turn
3. **LLM generates** — may call tools (JSON function calls) or respond directly
4. **Tool execution** — Python handlers make real API calls (GDC, filesystem, PyTorch, genomics), return structured results
5. **Up to 6 rounds** — LLM sees tool results and can call more tools before giving a final response
6. **SSE stream** — tokens, tool start/result events streamed to the browser in real time

---

## Tool-input guardrails

Small LLMs (and occasionally large ones) will happily invent `job_id`, `dataset_id`,
`experiment_id`, or `plugin_id` strings that look plausible but point at nothing.
Schema validation doesn't catch this — the shape is correct; the *reference* is
not. Left alone, `wait_for_job("feat-abc123")` burns 30 minutes polling a
non-existent endpoint.

PathClaw runs a **referential guardrail** on every tool call before the handler
body executes (`backend/pathclaw/api/validators.py`):

- Each resolver (`resolve_dataset_id`, `resolve_experiment_id`, `resolve_job_id`,
  `resolve_plugin_id`, `resolve_slide_stem`, `resolve_session_path`) does a cheap
  filesystem / registry check. On a miss it raises `ToolInputError` with a
  model-readable message: what's wrong, what's available, and which tool to call
  instead.
- The message is returned as the tool result, so the LLM sees it on the next
  round and self-corrects — no exception bubbles to the user.
- A `TOOL_VALIDATORS` table wires resolvers to high-blast-radius tools
  (`start_training`, `start_feature_extraction`, `wait_for_job`,
  `compare_experiments`, `generate_heatmap`, `update_plugin_config`, etc.).
  Read-only listing tools are intentionally exempt.
- Runs in both the streaming `wait_for_job` inline branch and the generic
  `_execute_tool` dispatcher, so the check cannot be bypassed.

To add validation to a new tool: import the appropriate resolver and add a row
to `TOOL_VALIDATORS` with `(arg_name, resolver, aux_keys)`. No change to the
tool handler is required.

---

## How to add a new skill

A skill is just a Markdown file that gets injected into the system prompt when relevant keywords are detected.

**1. Create the file:**

```
workspace/skills/my-skill/SKILL.md
```

```markdown
---
name: my-skill
description: One-line description of what this skill covers
---

# My Skill

## Role
You are an expert in ...

## Knowledge
...tables, rules, examples...

## Workflow
1. Step one
2. Step two

## API Calls
\`\`\`
tool_name({ param: "value" })
\`\`\`

## Guardrails
- NEVER ...
- Always ...
```

**2. Register trigger keywords in `chat.py`:**

```python
# backend/pathclaw/api/routes/chat.py
SKILL_TRIGGERS: dict[str, list[str]] = {
    ...
    "my-skill": ["keyword1", "keyword2", "relevant phrase"],
}
```

**3. Add a row to `SKILLS_SUMMARY.md`:**

```markdown
| `my-skill` | Domain description | keyword1, keyword2 | `tool_name` |
```

That's it. No restart needed if running with `--reload`. The skill is injected the next time a matching message is sent.

---

## How to add a new tool

Tools are Python functions the LLM can call. Two places to edit:

**1. Add the tool definition in `chat.py`** (in the `TOOLS` list):

```python
{
    "type": "function",
    "function": {
        "name": "my_tool",
        "description": "What this tool does — be specific, the LLM reads this",
        "parameters": {
            "type": "object",
            "properties": {
                "param_name": {"type": "string", "description": "What this param does"},
            },
            "required": ["param_name"],
        },
    },
},
```

**2. Add the handler in `_execute_tool()`** (same file):

```python
elif name == "my_tool":
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(f"{BACKEND_BASE}/api/my-endpoint", json=arguments)
        return resp.json().get("result", "Done")
```

**3. Add the backend endpoint** wherever makes sense (`app.py`, or a new router in `routes/`).

---

## How to edit the frontend

The UI is a single-page app with **no build step** — edit `index.html` (CSS) or `app.js` (JS) directly. The backend serves them as static files.

- **CSS** lives entirely in `<style>` tags in `index.html`
- **JS** lives entirely in `app.js`
- After editing, bump the cache-bust version: `<script src="/static/app.js?v=N">` → increment N

The app uses these patterns:
- `el('id')` — shorthand for `document.getElementById`
- `wsEl` — the main workspace content div, swapped out per tab
- `sendFromUI('message text')` — programmatically send a chat message
- `openSlideInViewer(datasetId, slideStem)` — open a slide in the OSD viewer
- `addMsg(html, 'agent'|'user')` — add a message bubble to chat

---

## LLM

I recommend using **Gemma 4** or **Sonnet 4.6**.

---

## TCGA data — important notes

TCGA filenames encode sample type. Always filter before downloading:

| Suffix | Meaning | Use for MIL? |
|--------|---------|-------------|
| `DX1`, `DX2` | Diagnostic FFPE — gold standard | Yes |
| `TS1`, `TS2` | Frozen section | Only for frozen tasks |
| `TSA`, `BS1` | Supplementary cuts | Avoid |
| `TMA` | Tissue microarray | Never |

Filter to `-01Z-...-DX` files. The agent knows this and filters automatically when you ask for "diagnostic slides".

---

## Supported use cases

See [`workspace/USE_CASES.md`](workspace/USE_CASES.md) for 13 detailed clinical benchmarks including TCGA-BRCA subtyping, TCGA-UCEC MSI prediction, TCGA-NSCLC, TCGA-RCC 3-class, EGFR mutation prediction, nucleus segmentation, and LoRA fine-tuning experiments.

---

## Troubleshooting

**"Features not found"**
```bash
ls ~/.pathclaw/features/{dataset_id}/{backbone}/*.pt | wc -l  # must be > 0
```

**"Label column not found"**
- CSV must have a column whose values match slide filenames (without extension)
- Check available columns: ask the agent to profile your dataset

**MAMMOTH ImportError**
```bash
pip install mammoth-moe>=0.1.2 einops
```

**Gated backbone 403 (UNI / CONCH / Virchow / GigaPath)**
- Request access on HuggingFace, then set your token in Settings

**OOM during training**
- Reduce `embed_dim` to 256, `num_experts` to 10, `batch_size` to 32

**openslide ImportError (viewer broken)**
```bash
apt install openslide-tools    # Linux
brew install openslide         # macOS
pip install openslide-python
```

**lifelines not found (survival analysis)**
```bash
pip install lifelines
```

---

## Configuration

Environment variables (all optional):

| Variable | Purpose |
|----------|---------|
| `PATHCLAW_DATA_DIR` | Data root (default `~/.pathclaw`) |
| `PATHCLAW_PORT` | Override server port (default 8101) |
| `OLLAMA_BASE` | Remote Ollama endpoint |
| `HUGGINGFACE_TOKEN` | For gated backbones (UNI, CONCH, Virchow, GigaPath) |

Per-session config (`~/.pathclaw/config.json`) stores: HF token, GDC token, LLM
provider + model, Anthropic / OpenAI / Google keys, disclaimer acknowledgement,
Telegram bot token. Set via the Settings modal or `POST /api/config`.

## Telegram bot

Create a bot with @BotFather, get a token, then:

```bash
curl -X POST http://localhost:8101/api/telegram/start \
  -H "Content-Type: application/json" \
  -d '{"token":"<botfather-token>","username":"<your-tg-user>","passcode":"optional"}'
```

From Telegram: `/sessions`, `/session <slug-or-id>`, `/new <title>`, `/status`.
`/sessions` shows each session's slug if set (else its id prefix). `/session` accepts
the slug, the full session id, or any id prefix ≥4 chars. Set slugs with the
`rename_session` agent tool, the sidebar rename action, or
`POST /api/chat/sessions/{id}/rename` with `{"slug":"chol-idh1","title":"CHOL IDH1"}`.
Logs at `~/.pathclaw/telegram.log`.

## Resource allocation

Multi-GPU: `POST /api/queue/resources` returns free RAM, per-GPU VRAM, and current
slot assignments. Training/feature jobs with `gpu_id` in their payload pin to a
specific device; otherwise the queue picks the first free GPU. CPU-only hosts
fall back to single-slot FIFO.

## Contributing

See [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md). Layout: FastAPI backend under
`backend/pathclaw/`, static frontend at `backend/pathclaw/static/`, agent skills
at `workspace/skills/`, plugin registry at `backend/pathclaw/plugins/`.

## Author

**Devansh Lalwani** — [devansh@turocrates.ai](mailto:devansh@turocrates.ai)

## License

MIT — see [LICENSE](LICENSE)
