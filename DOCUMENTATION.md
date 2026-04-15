# PathClaw — Technical Documentation

> *An agentic computational pathology research platform built on OpenClaw's agent infrastructure with a domain-specific MIL training backend.*

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture](#architecture)
3. [LLM Integration & Agent Model](#llm-integration--agent-model)
4. [Workspace & Skills Layer](#workspace--skills-layer)
5. [Python Backend (FastAPI)](#python-backend-fastapi)
6. [Preprocessing Pipeline](#preprocessing-pipeline)
7. [MIL Training Engine](#mil-training-engine)
8. [MAMMOTH Integration](#mammoth-integration)
9. [Evaluation & Metrics](#evaluation--metrics)
10. [Web UI](#web-ui)
11. [API Reference](#api-reference)
12. [Onboarding & Configuration](#onboarding--configuration)
13. [Data Governance & Lifecycle](#data-governance--lifecycle)
14. [Scientific References](#scientific-references)
15. [Roadmap & Future Work](#roadmap--future-work)

---

## System Overview

PathClaw is an **agentic workspace tailored for computational pathology research. It combines:

- **An agent orchestration layer** (forked from [OpenClaw](https://github.com/openclaw/openclaw)) that manages conversations, routes tasks to specialized agents, and coordinates multi-step workflows
- **A Python backend** (FastAPI) that handles domain-specific execution: WSI preprocessing, MIL model training, evaluation, and artifact management
- **A web UI** with a 3-panel layout (chat, workspace viewer, system panel) for interactive research sessions

### What It Does

1. **Ingest data** from local folders, uploads, or GDC/TCGA
2. **Profile datasets** — class balance, label quality, cohort statistics
3. **Preprocess WSIs** — Otsu tissue segmentation, patching, quality control
4. **Extract features** — via foundation model backbones (UNI, CONCH, CTransPath, Virchow, GigaPath)
5. **Train MIL models** — ABMIL, CLAM, TransMIL, DSMIL, pooling baselines
6. **Toggle MAMMOTH** — plug-and-play MoE module that replaces the linear layer
7. **Evaluate** — accuracy, AUROC, confusion matrices, ROC curves
8. **Explain results** — plain-language interpretation, run comparison, next-step recommendations
9. **Export artifacts** — reproducible `config.json`, `model.pth`, `metrics.json`, provenance records

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Browser (Web UI)                          │
│  ┌───────────┐  ┌─────────────────┐  ┌────────────────────┐ │
│  │  Chat     │  │  Workspace      │  │  System / Config   │ │
│  │  Panel    │  │  Viewer         │  │  Panel             │ │
│  │  (left)   │  │  (center)       │  │  (right)           │ │
│  └───────────┘  └─────────────────┘  └────────────────────┘ │
└──────────────────────┬──────────────────────────────────────┘
                       │  HTTP + WebSocket
┌──────────────────────▼──────────────────────────────────────┐
│         PathClaw Gateway (forked OpenClaw)                   │
│                                                              │
│  ┌──────────────────────────────────────────────────┐       │
│  │              Skills Layer (SKILL.md files)        │       │
│  │  orchestrator │ intake │ gdc │ profiling │ clean  │       │
│  │  preprocess │ train-config │ train-exec │ eval   │       │
│  │  results │ lifecycle                              │       │
│  └──────────────────────────────────────────────────┘       │
│                                                              │
│  Agent Runtime → LLM (Claude / GPT / Gemini) → Tool Calls   │
│  Session Manager │ Event Bus │ Cron │ MCP                    │
└──────────────────────┬──────────────────────────────────────┘
                       │  HTTP/REST
┌──────────────────────▼──────────────────────────────────────┐
│            Python Backend (FastAPI, port 8100)               │
│                                                              │
│  ┌──────────┐  ┌──────────┐  ┌───────────┐  ┌────────────┐ │
│  │  WSI     │  │ Feature  │  │   MIL +   │  │ Evaluation │ │
│  │  Preproc │  │ Extract  │  │  MAMMOTH  │  │ & Metrics  │ │
│  └──────────┘  └──────────┘  └───────────┘  └────────────┘ │
│  ┌──────────┐  ┌──────────┐  ┌───────────┐                 │
│  │   GDC    │  │   Job    │  │ Artifact  │                 │
│  │  Client  │  │ Scheduler│  │  Store    │                 │
│  └──────────┘  └──────────┘  └───────────┘                 │
└─────────────────────────────────────────────────────────────┘
```

### Three-Layer Design

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Presentation** | HTML/CSS/JS (single-page) | 3-panel workspace UI, chat, status |
| **Orchestration** | OpenClaw Gateway (TypeScript) | Agent sessions, LLM routing, skill dispatch, event bus |
| **Execution** | Python FastAPI | WSI preprocessing, training, evaluation, artifact store |

---

## LLM Integration & Agent Model

### Current Setup: Ollama (Local)

PathClaw currently runs **Ollama** for local, private LLM inference:

| Component | Value |
|-----------|-------|
| **Provider** | Ollama (local) |
| **Model** | `qwen3:8b` (8.2B params, Q4_K_M quantization) |
| **Port** | `http://localhost:11434` |
| **VRAM** | ~6 GB (fits on a single RTX 4500 Ada) |
| **Privacy** | All inference is on-device — no data leaves the machine |

### How to Start Ollama

```bash
# Start the Ollama server (runs in background)
ollama serve

# Verify it's running
curl http://localhost:11434/api/tags

# Test the model
ollama run qwen3:8b "What is MIL in computational pathology?"
```

### Switching Models

You can pull and use different models:

```bash
# Larger model (better reasoning, needs ~20 GB VRAM)
ollama pull qwen3:32b

# Smaller model (faster, ~4 GB VRAM)
ollama pull qwen3:4b

# Code-focused model
ollama pull codellama:13b

# List available models
ollama list
```

### Supported LLM Providers (via OpenClaw)

OpenClaw supports multiple providers. You can switch from Ollama to a cloud provider:

| Provider | Models | Config |
|----------|--------|--------|
| **Ollama (local)** | qwen3, llama3, mistral, codellama | ✅ Default — no API key needed |
| **Anthropic** | Claude Opus 4, Claude Sonnet 4 | `export ANTHROPIC_API_KEY="sk-ant-..."` |
| **OpenAI** | GPT-4.1, o3, o4-mini | `export OPENAI_API_KEY="sk-..."` |
| **Google** | Gemini 2.5 Pro, Gemini 2.5 Flash | `export GOOGLE_API_KEY="..."` |

To use a cloud provider, set the env var and configure `~/.openclaw/openclaw.json`:

```json
{
  "agent": {
    "model": "anthropic/claude-sonnet-4-20250514"
  }
}
```

### Agent Execution Model

1. **User sends a message** via the web UI chat panel
2. **OpenClaw Gateway** routes the message to the agent runtime
3. The agent loads the **system prompt** from `workspace/AGENTS.md` + `SOUL.md`
4. Relevant **skills** are loaded on-demand based on user intent
5. The **LLM generates a response** (via Ollama or cloud API) with tool calls
6. **Tools execute** — bash commands, API calls to the Python backend
7. Results are **streamed back** to the user via WebSocket
8. The **workspace viewer** updates with plots, reports, or previews

### Multi-Agent Coordination

PathClaw's 10 agents are implemented as **OpenClaw skills** with a single **orchestrator agent** routing between them:

```
User → Orchestrator (AGENTS.md)
         ├→ dataset-intake      (data ingestion & validation)
         ├→ gdc-tcga            (GDC API search & download)
         ├→ data-profiling      (class balance, quality analysis)
         ├→ data-cleaning       (label harmonization)
         ├→ data-lifecycle      (storage & retention)
         ├→ wsi-preprocess      (Otsu, patching, QC)
         ├→ train-config        (MIL/MAMMOTH config builder)
         ├→ train-exec          (training job lifecycle)
         ├→ evaluation          (metrics & visualization)
         └→ results             (interpretation & recommendations)
```

---

## Workspace & Skills Layer

### Workspace Files

| File | Purpose |
|------|---------|
| `AGENTS.md` | Master orchestrator prompt — routing rules, behavior guidelines |
| `SOUL.md` | Agent persona — tone, boundaries (never fabricate results, always confirm before deletion) |
| `TOOLS.md` | Reference for all backend API endpoints the agent can call |

### Skill Format

Each skill is a directory containing a `SKILL.md` file with YAML frontmatter:

```yaml
---
name: train-config
description: Build MIL and MAMMOTH training configurations
metadata: { "openclaw": { "requires": { "bins": ["python3"] } } }
---

# Instructions for LLM follow here...
```

**Key features:**
- **On-demand loading**: Skills are only injected into the LLM context when relevant, minimizing token usage
- **Gating**: Skills can require specific binaries (`bins`), environment variables (`env`), or config flags
- **Hierarchical precedence**: Workspace skills override global skills
- **Slash commands**: Skills can expose user-invocable commands (e.g., `/preprocess`)

### Skills Inventory

| Skill | Role | Key Capabilities |
|-------|------|-------------------|
| `dataset-intake` | Ingest data | Scan for slides (.svs/.tif/.ndpi), validate structure, register datasets |
| `gdc-tcga` | Remote data | Search GDC API, explain open vs controlled access, generate manifests |
| `data-profiling` | Analyze quality | Class distribution, label quality, file integrity, study design discussion |
| `data-cleaning` | Harmonize | Label mapping, duplicate detection, reproducible cleaning plans |
| `data-lifecycle` | Storage mgmt | Disk usage reports, retention policies, safe deletion with audit logs |
| `wsi-preprocess` | Slide prep | Otsu segmentation, patching config, QC previews, feature extraction guidance |
| `train-config` | Experiment design | Convert user goals to MIL/MAMMOTH configs, sweep generation |
| `train-exec` | Job management | Pre-flight checks, launch, monitor progress, handle failures |
| `evaluation` | Metrics & viz | Balanced accuracy, AUROC, confusion matrices, ROC curves, attention maps |
| `results` | Interpretation | Plain-language explanations, run comparison, recommendations |

---

## Python Backend (FastAPI)

### Entry Point

```python
# backend/pathclaw/api/app.py
uvicorn pathclaw.api.app:app --port 8100
```

**Base URL:** `http://localhost:8100`

### Route Modules

| Module | Prefix | Endpoints |
|--------|--------|-----------|
| `datasets.py` | `/api/datasets` | `GET /`, `POST /`, `GET /{id}`, `GET /{id}/slides`, `GET /{id}/profile` |
| `preprocess.py` | `/api/preprocess` | `POST /start`, `GET /{job_id}`, `GET /{job_id}/preview` |
| `training.py` | `/api/training` | `POST /start`, `GET /{job_id}`, `GET /{job_id}/logs`, `GET /` |
| `evaluation.py` | `/api/eval` | `POST /start`, `GET /{job_id}/metrics`, `GET /{job_id}/plots` |
| `artifacts.py` | `/api/artifacts` | `GET /`, `GET /{experiment_id}/{filename}` |
| `gdc.py` | `/api/gdc` | `POST /search`, `POST /download` |

### Data Directory Structure

All data is stored under `~/.pathclaw/` (configurable):

```
~/.pathclaw/
├── config.json              # HF token, GDC token, settings
├── datasets/                # Registered dataset metadata
│   └── {dataset_id}/
│       └── meta.json        # Slides list, label files, stats
├── preprocessed/            # Preprocessing outputs
│   └── {dataset_id}/
│       ├── patches/         # Patch coordinate JSONs
│       └── previews/        # Tissue mask overlay PNGs
├── features/                # Extracted feature tensors
│   └── {dataset_id}/
│       └── *.pt             # N×D tensors per slide
├── experiments/             # Training & eval outputs
│   └── {job_id}/
│       ├── config.json      # Training config
│       ├── model.pth        # Best model checkpoint
│       ├── metrics.json     # Final metrics
│       ├── history.json     # Loss/accuracy per epoch
│       └── plots/           # Loss curves, confusion matrices
└── jobs/                    # Job status tracking
```

---

## Preprocessing Pipeline

**File:** `backend/pathclaw/preprocessing/pipeline.py`

### Pipeline Steps

1. **Otsu Tissue Segmentation**
   - Reads WSI at a downsampled level (default: level 1) for speed
   - Converts to grayscale, applies `skimage.filters.threshold_otsu`
   - Morphological cleanup via `binary_closing` with a disk kernel
   - Returns a binary tissue mask and tissue percentage

2. **Patch Extraction**
   - Reads slide via OpenSlide at target magnification (default: 20×)
   - Iterates over the slide in a grid (default: 256×256 patches, stride=256)
   - For each patch, checks the tissue mask — keeps patches with ≥50% tissue
   - Returns patch coordinate metadata (x, y, patch_size, level, tissue_pct)

3. **Preview Generation**
   - Creates a side-by-side figure: original thumbnail + tissue mask overlay
   - Saved as PNG for QC review before committing to full preprocessing

4. **Feature Extraction** (downstream)
   - After patching, patches are passed through a foundation model backbone
   - Outputs are saved as `.pt` files (PyTorch tensors, shape N×D where D is the backbone's feature dim)

### Supported Backbones

| Backbone | Feature Dim | Source |
|----------|-------------|--------|
| UNI | 1024 | MahmoodLab/UNI |
| CONCH v1.5 | 512 | MahmoodLab/CONCH |
| CTransPath | 768 | X-zhangyang/CTransPath |
| Virchow / Virchow2 | 1280 | Paige-AI/Virchow |
| GigaPath | 1536 | Microsoft/GigaPath |

All backbones are downloaded via HuggingFace (requires `HUGGINGFACE_TOKEN`).

---

## MIL Training Engine

**File:** `backend/pathclaw/training/trainer.py`

### Implemented Models

| Model | Class | Description |
|-------|-------|-------------|
| **MeanPool** | `MeanPoolMIL` | Mean pooling baseline — averages patch embeddings, then classifies |
| **ABMIL** | `ABMIL` | Attention-Based MIL with gated attention mechanism (Tanh + Sigmoid gates) |

Both models support optional MAMMOTH via the `moe_args` parameter.

### Model Registry

```python
MODEL_REGISTRY = {
    "meanpool": MeanPoolMIL,
    "abmil": ABMIL,
    # Future: "transmil", "clam", "dsmil", "rrtmil", "wikg", "ilra", "dftd"
}

model = create_model(config)  # Factory function
```

### Training Loop

1. Load pre-extracted features (`.pt` files) from `~/.pathclaw/features/{dataset_id}/`
2. Load label mapping from `labels.json`
3. Split data 80/20 (train/val) — configurable
4. For each epoch:
   - Train: forward pass → cross-entropy loss → backprop → step optimizer
   - Validate: compute loss + accuracy on val set
   - Track best model by validation accuracy
   - Save checkpoint at best epoch
5. Generate loss/accuracy plots
6. Save outputs: `config.json`, `model.pth`, `metrics.json`, `history.json`

### Training Configuration (Pydantic Model)

```python
class TrainingConfig:
    task: str                    # "subtyping", "grading", "molecular"
    dataset_id: str              # Reference to registered dataset
    label_column: str            # Column name in label file
    feature_backbone: str        # "uni", "conch", "ctranspath"
    feature_dim: int = 1024      # Input feature dimension
    embed_dim: int = 512         # Embedding dimension for MIL
    mil_method: str = "abmil"    # MIL aggregation method
    num_classes: int = 2         # Number of output classes
    mammoth: MammothConfig       # MAMMOTH toggle + hyperparameters
    training: TrainingHyperparams  # lr, epochs, optimizer, etc.
    evaluation: EvalConfig       # Strategy (k-fold, split), metrics
```

---

## MAMMOTH Integration

### What is MAMMOTH?

**MAMMOTH** (MAtrix-factorized Mixture Module of Transformation Heads) is a parameter-efficient Mixture of Experts module from the Mahmood Lab. It replaces the standard `nn.Linear` patch embedding in MIL models with a low-rank MoE that routes each patch to a combination of experts based on its phenotype.

### How It's Integrated

```python
# Standard MIL: single linear layer
self.patch_embed = nn.Linear(in_dim, embed_dim)

# With MAMMOTH: drop-in replacement
from mammoth import Mammoth
self.patch_embed = Mammoth(
    input_dim=in_dim,         # e.g., 1024 (UNI features)
    dim=embed_dim,            # e.g., 512
    num_experts=30,           # Number of low-rank experts
    num_slots=10,             # Slots per expert for routing
    num_heads=16,             # Attention heads
    share_lora_weights=True,  # Share first low-rank layer (parameter efficient)
    auto_rank=True,           # Auto-compute appropriate rank
    dropout=0.1,
)
```

### Key Properties

| Property | Value |
|----------|-------|
| **Pip package** | `mammoth-moe` |
| **Dependencies** | PyTorch + einops only |
| **Parameter overhead** | Comparable to single linear layer (via low-rank factorization) |
| **Performance gain** | +3.8% average across 130/152 experiment configurations |
| **Toggle** | `mammoth.enabled: true/false` in training config |
| **Impact** | Often larger effect than choice of aggregation method |

### User Experience

Users don't see the internal complexity. They simply choose:
- MAMMOTH **on** (recommended, default) or **off** (baseline comparison)

The system handles all hyperparameters with validated defaults.

---

## Evaluation & Metrics

**File:** `backend/pathclaw/evaluation/evaluator.py`

### Computed Metrics

| Metric | When Used |
|--------|-----------|
| Accuracy | Always |
| Balanced Accuracy | Always (handles class imbalance) |
| AUROC | Binary and multi-class (macro, OVR) |
| F1 Score | Per-class and macro |
| Confusion Matrix | Always — saved as heatmap PNG |
| Classification Report | Per-class precision/recall/F1 |

### Generated Plots

1. **Confusion Matrix** — heatmap via seaborn
2. **ROC Curve** — per-class with AUROC value (binary tasks)
3. **Loss Curves** — train + val loss over epochs
4. **Accuracy Curve** — validation accuracy over epochs

All plots saved as PNG in `experiments/{job_id}/plots/`.

---

## Web UI

**File:** `backend/pathclaw/static/index.html`

### Layout

A VS Code-inspired 3-panel layout:

| Panel | Position | Content |
|-------|----------|---------|
| **Chat** | Left (380px) | Agent conversation, slash commands |
| **Workspace** | Center (flex) | Plots, WSI previews, reports, tables |
| **System** | Right (360px) | Backend status, GPU, storage, jobs, datasets, experiments |

### Design System

- **Theme:** Dark mode (bg #0a0e14, text #e6edf3)
- **Typography:** Inter (sans-serif) + JetBrains Mono (code)
- **Colors:** Blue accent (#58a6ff), green success (#3fb950), purple (#bc8cff)
- **Animations:** Fade-in for messages, smooth transitions

### Chat Commands

| Command | Action |
|---------|--------|
| `/help` | Show available commands |
| `/status` | System status (GPU, storage, HF token, Ollama) |
| `/datasets` | List registered datasets |
| `/register <path>` | Register a dataset folder |
| `/gdc <project>` | Search GDC (e.g., `/gdc TCGA-LUAD`) |
| `/config hf <token>` | Set HuggingFace token |

### Live Updates

- Backend status polled every 10 seconds
- Datasets and experiments refreshed every 15 seconds
- Connection indicator (Online/Offline) in top bar

---

## API Reference

### Health & Status

```
GET /api/health
→ {"status": "ok", "version": "0.1.0"}

GET /api/status
→ {"gpu": {...}, "storage": {...}, "data_dir": "..."}
```

### Datasets

```
POST /api/datasets
Body: {"name": "brca", "path": "/data/brca_slides", "description": "..."}
→ {"id": "a1b2c3d4", "slide_count": 150, "total_size_mb": 45000, ...}

GET /api/datasets/{id}/profile
→ {"slide_count": 150, "formats": {".svs": 150}, "label_candidates": [...]}
```

### Training

```
POST /api/training/start
Body: {
  "task": "subtyping",
  "dataset_id": "a1b2c3d4",
  "label_column": "subtype",
  "mil_method": "abmil",
  "mammoth": {"enabled": true, "num_experts": 30, ...},
  "training": {"epochs": 100, "lr": 1e-4, ...}
}
→ {"job_id": "train-x1y2z3", "status": "queued"}

GET /api/training/{job_id}
→ {"progress": 0.45, "epoch": 45, "metrics": {"val_accuracy": 0.87, ...}}
```

### GDC Search

```
POST /api/gdc/search
Body: {"project": "TCGA-BRCA", "data_type": "Slide Image", "access": "open"}
→ {"total": 1133, "files": [{"file_name": "...", "file_size_mb": 450, ...}]}
```

---

## Onboarding & Configuration

### First-Run Flow (Web UI)

On first visit to http://localhost:8100, a modal dialog asks for:

1. **HuggingFace Token** (required) — needed for gated foundation models (UNI, CONCH, Virchow, GigaPath). Get yours at https://huggingface.co/settings/tokens
2. **GDC Token** (optional) — only needed for controlled-access TCGA data. Open-access diagnostic slides work without it.

You can also set tokens later via the chat: `/config hf <token>`

### CLI Alternative

```bash
cd pathclaw/backend && .venv/bin/python -m pathclaw.cli onboard
```

### Configuration File

`~/.pathclaw/config.json`:
```json
{
  "huggingface_token": "hf_...",
  "gdc_token": "",
  "data_dir": "/home/user/.pathclaw",
  "onboarding_complete": true
}
```

---

## Data Governance & Lifecycle

### Credential Security

| Credential | Storage | Exposure |
|------------|---------|----------|
| HuggingFace Token | `config.json` (local file, user-only permissions) | Never logged or printed |
| GDC Token | `config.json` | Never logged; injected via env var at runtime |
| LLM API Keys | OpenClaw config (`~/.openclaw/openclaw.json`) | Managed by OpenClaw security model |

### Data Retention

- **Temporary data** (patches, intermediate features): Configurable retention window (default: 7 days)
- **Permanent artifacts** (models, configs, metrics): Protected from auto-cleanup
- **Audit log**: All deletion events recorded with timestamp, path, and reason

### Deletion Policy

- All deletions require **explicit user confirmation**
- File counts and sizes shown before deletion
- Audit trail maintained in `~/.pathclaw/audit.log`

---

## Scientific References

### MAMMOTH
> **"Mixture of Mini Experts: Overcoming the Linear Layer Bottleneck in Multiple Instance Learning"**
> Daniel Shao*, Andrew H. Song*, Faisal Mahmood
> Mahmood Lab, Harvard Medical School / MIT
> GitHub: [mahmoodlab/MAMMOTH](https://github.com/mahmoodlab/MAMMOTH)
> pip: `mammoth-moe`

### MIL Methods

| Method | Paper |
|--------|-------|
| ABMIL | Ilse et al. "Attention-based Deep Multiple Instance Learning" (ICML 2018) |
| CLAM | Lu et al. "Data-efficient and weakly supervised computational pathology" (Nature BME 2021) |
| TransMIL | Shao et al. "TransMIL" (NeurIPS 2021) |
| DSMIL | Li et al. "Dual-stream MIL" (CVPR 2021) |
| DTFD-MIL | Zhang et al. "Double-Tier Feature Distillation" (CVPR 2022) |

### Foundation Models

| Model | Paper / Source |
|-------|----------------|
| UNI | Chen et al. "A General-Purpose Self-Supervised Model for Computational Pathology" |
| CONCH | Lu et al. "A Visual-Language Foundation Model for Computational Pathology" |
| Virchow | Vorontsov et al. "Virchow: A Million-Slide Digital Pathology Foundation Model" |

---

## Current System Status (2026-03-30)

| Component | Status |
|-----------|--------|
| GPU | ✅ NVIDIA RTX 4500 Ada × 2 (CUDA 12.4) |
| PyTorch | ✅ 2.6.0+cu124 |
| Ollama | ✅ qwen3:8b (8.2B, Q4_K_M) |
| Backend | ✅ FastAPI on port 8100 |
| Storage | 917 GB free / 3.2 TB total |
| HF Token | ❌ Not yet configured |

---

## How to Start PathClaw

```bash
# Terminal 1: Start Ollama
ollama serve

# Terminal 2: Start PathClaw backend
cd ~/master/pathclaw/backend
PYTHONPATH=$(pwd) .venv/bin/uvicorn pathclaw.api.app:app --port 8100 --host 0.0.0.0

# Open browser
http://localhost:8100
```

### How to Stop

```bash
# Kill the backend
lsof -ti :8100 | xargs kill -9

# Kill Ollama
pkill ollama
```

---

## Roadmap & Future Work

### Phase 2 (Next — TCGA-LUAD Demo)
- [ ] TCGA-LUAD slide download via GDC API
- [ ] MAF file download + EGFR mutation status extraction
- [ ] Feature extraction service (UNI/CONCH backbone runner)
- [ ] OpenClaw gateway ↔ PathClaw backend integration
- [ ] Full agent-powered conversational workflow

### Phase 3
- [ ] Additional MIL models (TransMIL, CLAM, DSMIL, RRTMIL)
- [ ] K-fold cross-validation in the training loop
- [ ] Attention heatmap generation for interpretability
- [ ] GPU job queue (Celery + Redis)

### Phase 4
- [ ] Experiment comparison dashboard
- [ ] WSI tile viewer (OpenSeadragon integration)
- [ ] Deployment packaging (Docker compose)
- [ ] Publication-ready report generation
