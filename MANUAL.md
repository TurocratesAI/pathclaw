# PathClaw Manual

User manual for PathClaw v0.3. Covers installation, first-run onboarding, UI tour,
dataset workflow, agent workflows, plugin authoring, genomic workflows, Telegram
setup, multi-session operation, and CLI reference.

For a quick orientation read [README.md](README.md). For broken-state recovery read
[docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md).

---

## 1. Installation

### Prerequisites

- Python 3.10 or newer
- `openslide-tools` (Linux: `apt install openslide-tools`; macOS: `brew install openslide`)
- CUDA 12.x (optional; CPU works for inference, not practical for training)
- For local LLM: Ollama (`curl -fsSL https://ollama.com/install.sh | sh`)
- For PDF manuscript compile: `tectonic` or `texlive-latex-base` (optional)

### Install

```bash
git clone https://github.com/devanshlalwani/PathClaw.git
cd PathClaw/backend
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

### Pull a local model (skip if using an API key)

```bash
ollama pull qwen3:8b        # fast, good tool use
ollama pull gemma2:9b       # alternative
```

### Start the server

```bash
uvicorn pathclaw.api.app:app --host 0.0.0.0 --port 8101 --reload
```

Open <http://localhost:8101>.

---

## 2. First run

The onboarding modal runs in two steps:

1. **Disclaimer.** PathClaw is a research tool. Any model trained here requires
   independent clinical and statistical validation before use in patient care. You
   are responsible for regulatory compliance. Check the box and continue.
2. **Credentials + LLM provider.** Paste a HuggingFace token (required for gated
   backbones: UNI, CONCH, Virchow, GigaPath). Pick an LLM provider and model:
   - **Ollama** — no key, runs locally, pulls from your Ollama server.
   - **Anthropic** — paste API key, pick a Claude model.
   - **OpenAI** — paste API key, pick a GPT model. Optional `openai_base` for
     OpenAI-compatible endpoints (OpenRouter, LM Studio, vLLM, Together, Groq).
   - **Google** — paste API key, pick a Gemini model.

All settings persist to `~/.pathclaw/config.json`. You can change provider any time
from the gear icon.

---

## 3. UI tour

Three-column layout (resize browser, columns are fixed-width):

### Left sidebar — Explorer
- **Status bar**: Backend, GPU, LLM, HF-token indicators.
- **File tree**: `Workspace` (per-session code, repos, plugins, methods),
  `Datasets`, `Experiments`, `Folders` (attached PDF collections),
  `Uploads` (per-session).
- **Jobs panel**: live view of running/queued/finished jobs for the current session.
- **Upload button** (↑): drop any file, auto-routed by extension to
  `slides/papers/tables/genomics/user_code/misc` under the current session.

### Center — dual tab groups
- **Data Browser group**: Overview, Slides, CSV, Folders, Viewer, Plots.
- **Workspace group**: Editor (Monaco), Notebook, Manuscript, Configure, Logs.
- Each group remembers its active tab in `sessionStorage` — switching one group
  does not reset the other.

### Right — Chat panel
- Session drawer at the top: list of chat sessions, create `+ New`, rename, delete.
- Message stream with tool-call cards inline (click to expand).
- Input textarea at the bottom. Enter to send.

### Header
- Brand + version pill.
- Settings gear (opens config modal).
- Telegram button (opens bot setup modal if not configured, status if running).

---

## 4. Dataset workflow

### 4.1 Register a dataset

Option A — UI: click `+ New dataset` in the Datasets panel, point to a folder of
slides, paste a labels CSV, give it a name.

Option B — chat: *"Register my slides at `/data/UCEC-MSI/slides` as `ucec-msi`,
labels are in `/data/UCEC-MSI/labels.csv`, label column `msi_status`."*

Option C — upload button: drop `.svs`/`.ndpi`/`.tif` files; they land under
`~/.pathclaw/sessions/<sid>/uploads/slides/`. Then ask the agent to register them.

### 4.2 Preprocess

```
preprocess <dataset_id> at 20x, 256px patches, otsu threshold 0.7
```

Generates tissue masks + patch coordinates under
`~/.pathclaw/datasets/<id>/patches/`. A preview thumbnail with patch overlays
appears in the Slides tab.

### 4.3 Extract features

```
extract UNI features for <dataset_id>
```

Runs in a detached subprocess (survives server restart). Monitor via the Jobs
panel. Output: `~/.pathclaw/features/<dataset_id>/<backbone>/<slide>.pt`.

Backbones: UNI (1024d), CONCH (512d), CONCHv1.5 (768d), Virchow (1280d), Virchow2
(2560d), GigaPath (1536d), CTransPath (768d). UNI/CONCH/Virchow/GigaPath require a
HuggingFace token with access approved on the model card.

### 4.4 Train

```
train ABMIL on <dataset_id> for 10 epochs, label column msi_status
```

Optional: `with MAMMOTH`, `with cellpose segmentation`, `device cuda:1`. All
methods: ABMIL, MeanPool, TransMIL, CLAM, DSMIL, RRTMIL, WIKG.

### 4.5 Evaluate

```
evaluate train-<id> on the holdout split
generate attention heatmap for <slide> using train-<id>
make a ROC curve plot for train-<id>
```

Metrics: AUROC, balanced accuracy, F1, confusion matrix. Plots land under
`~/.pathclaw/experiments/<id>/plots/`.

---

## 5. Agent workflows (natural-language recipes)

The agent reads 15 skill modules (keyword-triggered) and calls 66 tools. A few
canonical prompts:

- *"Download 20 TCGA-UCEC diagnostic slides, preprocess, extract UNI features,
  train ABMIL on MSI status for 5 epochs."*
- *"Read the uploaded MAF file, compute TMB, and stratify survival by TMB-high
  vs TMB-low."*
- *"Clone <https://github.com/mahmoodlab/CLAM>, analyze the attention module,
  and wire it in as a plugin I can toggle."*
- *"Implement the GAMIL methodology from the attached paper as a plugin, smoke
  test it, and run a 1-epoch sanity check on 10 slides."*
- *"Use paige-ai/Virchow2 for feature extraction on the first 5 slides of
  tcga-ucec. Register the backbone if needed."*
- *"Compare train-a1b2 and train-c3d4 side by side; make a per-class ROC plot
  for the better one."*

The agent runs up to 6 tool rounds per turn, confirms before destructive
operations, and emits intermediate status in the chat stream.

---

## 6. Plugin authoring

See [docs/PLUGIN_DEV.md](docs/PLUGIN_DEV.md). Contract:

```python
# plugins/my_plugin.py
def build(in_dim: int, embed_dim: int, config: dict) -> nn.Module:
    return MyModule(in_dim, embed_dim, **config)
```

Register:

```
register plugin {
  "id": "my_plugin",
  "kind": "patch_embed",
  "import_path": "workspace.plugins.my_plugin:build",
  "default_config": {...},
  "applies_to": ["mil"]
}
```

Smoke test runs instantiation + a dummy forward pass. On success, the plugin
appears in the Training Configure tab as a toggle.

---

## 7. Genomic workflows

All tools operate on per-session uploads or attached folders.

```
parse the MAF file I uploaded and summarize per-gene mutation frequencies
compute TMB across all samples in uploads/genomics/
query cBioPortal: UCEC, MSI_SCORE_MANTIS for all patients
run survival analysis stratified by TP53 mutation status
generate an oncoplot for the top 20 most-mutated genes
discover biomarkers: differentially-mutated genes between model-predicted
  MSI-high and MSI-low slides
```

---

## 8. Telegram bot

1. Chat `@BotFather` on Telegram, create a new bot, copy the token.
2. In PathClaw UI, click the Telegram button → paste token, your Telegram
   username, optional passcode.
3. The bot starts as a detached subprocess; logs at `~/.pathclaw/telegram.log`.

Commands:
- `/start [passcode]` — authorize your chat.
- `/sessions` — list all PathClaw sessions (each entry shows the slug if set,
  otherwise a short id prefix).
- `/session <slug-or-id>` — bind this chat to a session. Accepts the session's
  slug (`chol-idh1`), its full UUID, or any id prefix ≥4 characters that
  uniquely matches one session.
- `/new <title>` — create a new session and bind.
- `/status` — current session + bot health.

Any non-command message goes to the bound session's agent — same tools, same
memory, same manuscript.

**Tip:** Telegram is much easier to use once your sessions have slugs. See
§9 for how to rename a session.

---

## 9. Multi-session operation

PathClaw treats each chat session as an isolated workspace (the "parallel PhD
student" model):

- Own `~/.pathclaw/sessions/<sid>/workspace/` — code, cloned repos, plugins, methods.
- Own `~/.pathclaw/sessions/<sid>/uploads/` — slides, PDFs, tables, genomics.
- Own `~/.pathclaw/chats/<sid>.notes.md` — running notebook auto-injected into
  the agent's system prompt.
- Own `~/.pathclaw/chats/<sid>_manuscript/` — LaTeX project with `main.tex`,
  `refs.bib`, `figures/`.
- Attached folders (shared PDF collections) are per-session attachment-list
  filtered.

Shared across sessions: registered datasets (filtered by session_id on the list
endpoint), persistent global memory (`remember_fact`), feature cache (backbones
download once, reused across all sessions).

Switch sessions from the session drawer or `?session_id=<uuid>` URL param.

### Naming sessions (slugs)

Every session has a UUID plus an optional **slug** — a short kebab-case name
(`a-z`, `0-9`, `-`, up to 40 chars) that must be unique across sessions. Slugs
are how you refer to a session without copy-pasting its UUID — in the sidebar,
on Telegram (`/session chol-idh1`), and in agent chat history.

Three ways to set a slug:

1. **From chat (recommended):** ask the agent — *"rename this session to
   `chol-idh1`"*. The agent calls the `rename_session` tool with
   `{"slug":"chol-idh1"}` (and an optional human `title`). The slug is validated
   for format and uniqueness; on conflict the tool returns an error and the
   agent will ask for a different name.
2. **From the sidebar:** session drawer → rename action → enter a slug and/or
   title.
3. **Directly via HTTP:**
   ```bash
   curl -X POST http://localhost:8101/api/chat/sessions/<session_id>/rename \
     -H "Content-Type: application/json" \
     -d '{"slug":"chol-idh1","title":"TCGA-CHOL IDH1 mutation"}'
   ```
   Returns `409` on slug conflict, `400` on bad format, `404` if the session
   doesn't exist.

Slugs are resolved everywhere session ids are accepted: chat resume
(`?session_id=<slug>`), Telegram `/session <slug>`, and the agent tool
`resume_session`.

---

## 10. Advanced

### Multi-GPU

Jobs auto-dispatch to the first free GPU. Pin a specific device:

```
train ABMIL on <dataset_id> with device cuda:1
```

`GET /api/queue/resources` returns free RAM, per-GPU VRAM, and current slot
assignments.

### Memory caps

Feature extraction sizes the batch from free VRAM at startup (4 GB → 16 patches,
8 GB → 64, 16+ GB → 128). Training has per-slide OOM recovery — if one slide
OOMs it is skipped, logged, and training continues.

### Session export/import

```bash
tar -czf session-<sid>.tar.gz \
  ~/.pathclaw/sessions/<sid>/ \
  ~/.pathclaw/chats/<sid>.notes.md \
  ~/.pathclaw/chats/<sid>_manuscript/ \
  ~/.pathclaw/chats/<sid>.json
```

Restore by untarring onto a new install at the same paths.

---

## 11. CLI reference

```bash
pathclaw onboard       # interactive: disclaimer, HF token, LLM provider
pathclaw server        # start FastAPI (uvicorn under the hood)
pathclaw --help        # all subcommands
```

CLI onboarding writes to the same `~/.pathclaw/config.json` as the UI modal, so
the two are interchangeable.

---

## 12. Where things live

```
~/.pathclaw/
  config.json                # tokens, LLM config, disclaimer state
  server.port                # current port (written on startup)
  telegram.log               # bot process logs
  datasets/                  # registered dataset metadata
  features/<dataset>/<bb>/   # extracted features (.pt)
  experiments/<id>/          # training runs (model, metrics, plots, logs)
  jobs/                      # on-disk job status (features, downloads)
  folders/                   # shared PDF collections
  plugins/user_registry.json # user-registered plugins + config overrides
  chats/<sid>.json           # chat history
  chats/<sid>.notes.md       # session notebook
  chats/<sid>_manuscript/    # per-session LaTeX project
  sessions/<sid>/
    workspace/{user_code,repos,plugins,methods}/
    uploads/{slides,papers,tables,genomics,user_code,misc}/
```
