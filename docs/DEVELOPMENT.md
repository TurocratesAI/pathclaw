# Development Guide

Orientation for contributors modifying PathClaw itself (not users extending via
plugins — see [PLUGIN_DEV.md](PLUGIN_DEV.md) for that).

---

## Layout

```
backend/pathclaw/
  api/
    app.py                  # FastAPI app, route wiring, startup hooks
    llm_providers.py        # Ollama / Anthropic / OpenAI / Google adapters
    routes/
      chat.py               # Agent loop, tool dispatch, SSE stream, skill triggers
      datasets.py           # Dataset registration, preview, profile
      training.py           # Training submission, status, heatmaps
      gdc.py                # TCGA search + download
      artifacts.py          # Plots, manuscript figures
      folders.py            # Attached PDF collections
      queue.py              # Task queue, multi-GPU dispatch, resources endpoint
      telegram.py           # Bot setup, session binding
      # ...15+ route modules total
  preprocessing/
    feature_extraction.py   # Backbone loader + patch encoding subprocess
    tissue.py               # Otsu, HED, Macenko stain norm
  training/
    trainer.py              # MIL training loop, OOM recovery, checkpointing
    mil_methods/            # ABMIL, CLAM, TransMIL, DSMIL, RRTMIL, WIKG
  plugins/
    registry.json           # Built-in plugin registry (mammoth, cellpose)
    mammoth.py
    cellpose.py
  genomics/                 # MAF/VCF parsing, TMB, oncoplot, cBioPortal, survival
  static/
    index.html              # Single-page app shell
    app.js                  # ~3k LOC frontend (no build step)
  telegram_bot.py           # Bot runtime (runs as detached subprocess)
workspace/
  skills/                   # 15 keyword-triggered skill modules
```

---

## Route wiring

`api/app.py` creates the FastAPI app and includes each router:

```python
from pathclaw.api.routes import chat, datasets, training, ...
app.include_router(chat.router, prefix="/api/chat")
app.include_router(datasets.router, prefix="/api/datasets")
# ...
```

Adding a new route module:

1. Create `routes/<name>.py` with `router = APIRouter()`.
2. Add handlers with `@router.get(...)` etc.
3. Register in `app.py` with a URL prefix.
4. If the module runs background work, add a startup hook in `app.py`.

---

## Agent tools

Tools are defined in `api/routes/chat.py` as JSON schema entries passed to the
LLM, plus matching handler functions. The loop:

1. LLM returns a `tool_use` block with name + arguments.
2. `dispatch_tool(name, args, session)` looks up the handler, runs it.
3. Result is appended as a `tool_result` block and the LLM continues.
4. Max 6 rounds per turn.

To add a tool:

1. Add the JSON schema to the tools list (search for `TOOLS = [` in `chat.py`).
2. Add a handler function, typically `async def tool_my_name(args, session):`.
3. Register in the `DISPATCH` dict.
4. If the tool needs a new skill module, add it under `workspace/skills/` and
   key it off a keyword in the injection table.

---

## Skill injection

`chat.py` lines ~343–415 scan the user's message for keywords and prepend the
matched skill's `SKILL.md` content into the system prompt for that turn. Skills
are small markdown cheat sheets the agent consults when a topic comes up.

To add a skill:

1. `mkdir workspace/skills/<skill-name>/`
2. Write `SKILL.md` with usage guidance the agent should see.
3. Add an entry to the trigger table in `chat.py` keyed off relevant keywords.

---

## Frontend

No build step. `static/app.js` is served raw. Cache-bust by bumping the `?v=N`
query parameter on the script tag in `static/index.html` after any `app.js`
change — otherwise users will hit stale cached versions.

Helpers to use:

- `apiJson(url, opts)` — fetch wrapper that throws readable errors on non-2xx
- `showToast(msg, kind)` — ephemeral bottom-right notification
- `confirmModal(msg)` — async confirm dialog returning `Promise<boolean>`
- `showError(msg)` — shortcut for `showToast(msg, 'error')`

Do not reintroduce `alert()` or `confirm()` — they block the event loop and
look like native browser UI in a way that's jarring next to the rest of the
app.

---

## Running locally

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
uvicorn pathclaw.api.app:app --reload --port 8101
```

`--reload` picks up backend changes. Frontend changes need a hard refresh
(Ctrl+Shift+R) or a cache version bump.

---

## Tests

Smoke test:

```bash
pathclaw --help
```

Plugin smoke test (per plugin):

```bash
curl -X POST http://localhost:8101/api/plugins/smoke_test -d '{"plugin_id":"mammoth"}'
```

There is no formal pytest suite yet — contributions welcome. Any new suite
should avoid GPU requirements in CI.

---

## Release checklist

1. Bump version in `backend/pyproject.toml`.
2. Bump `v=` cache param in `static/index.html` if `app.js` changed.
3. Update `CHANGELOG.md`.
4. Verify `grep -rE 'Turocrates AI|founders@' .` is empty.
5. Tag `git tag v0.X.Y && git push --tags`.
