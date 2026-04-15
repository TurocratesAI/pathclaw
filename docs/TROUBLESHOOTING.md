# Troubleshooting

Reference for common failure modes. If your issue is not here, check
`~/.pathclaw/telegram.log`, the uvicorn stderr, or the Jobs panel `logs` tab of
the failing job.

---

## Installation & startup

### Backend won't start — "Address already in use"

```bash
lsof -i :8101
kill <pid>
```

Or change port: `PATHCLAW_PORT=8102 uvicorn pathclaw.api.app:app --port 8102`.

### `ImportError: openslide`

```bash
apt install openslide-tools        # Linux
brew install openslide             # macOS
pip install openslide-python
```

OpenSlide needs both the system library **and** the Python binding — they are
separate packages.

### `ImportError: mammoth-moe`

```bash
pip install mammoth-moe>=0.1.2 einops
```

MAMMOTH is optional. If you don't need it, the import is lazy — the plugin
registry flags it as unavailable and the MAMMOTH toggle in the training form is
disabled.

### `ImportError: lifelines` (survival analysis)

```bash
pip install lifelines
```

---

## Data & feature extraction

### HuggingFace 403 on gated backbone (UNI / CONCH / Virchow / GigaPath)

1. On HuggingFace, visit the model page and click "Request access".
2. Wait for approval (usually minutes to a day).
3. Create a read token at <https://huggingface.co/settings/tokens>.
4. Paste it into PathClaw Settings → HuggingFace token, or
   `export HUGGINGFACE_TOKEN=hf_...` before starting the server.

### "Features not found at /home/.../features/<dataset>/<backbone>"

Feature extraction never ran or was cancelled. Check:

```bash
ls ~/.pathclaw/features/<dataset>/<backbone>/*.pt | wc -l
```

If 0, re-run extraction. If fewer than the slide count, the job crashed mid-way
— check `~/.pathclaw/jobs/feat-*/status.json` for the traceback.

### OpenSlide can't read `.mrxs`

`.mrxs` slides are a directory — make sure the entire `SlideFolder.mrxs` tree
is present, not just the `.mrxs` file. Some `.mrxs` slides also need the
`libopenslide-dev` variant of the openslide package.

### CUDA OOM during feature extraction

Lower the per-batch patch count — feature extraction auto-sizes based on free
VRAM at startup, but a concurrent job can starve it. Either wait for the other
job to finish or pass `batch_size` explicitly.

### CUDA OOM during training

trainer.py skips an individual slide on OOM and continues. Repeated skips on the
same slides suggest the slide is too large for your GPU — lower `embed_dim` to
256, `num_experts` to 10, or use a smaller backbone (UNI over Virchow2).

---

## GDC / TCGA

### `search_gdc` returns 0 hits

JSON filter encoding was fixed in commit `dc25571`. If your version is older,
update. Also check the filter cookbook in `workspace/skills/gdc-tcga/SKILL.md` —
common mistake is using `project.project_id` when the filter field is
`cases.project.project_id`.

### Download stops partway through

Downloads restart from checkpoints in commit `c2851b8` and survive server
restart. If a job is stuck in `running` state with no progress, check
`~/.pathclaw/jobs/dl_*.json` for the subprocess pid; if the process is dead
the queue marks it failed on next poll.

### Downloaded filenames are opaque UUIDs

TCGA ships data with opaque filenames; the GDC JSON return includes the
`file_name` field with the human name. Ask the agent to build a mapping or check
the clinical supplement.

---

## Training

### Training immediately fails with "Features not found"

The training config points to a feature backbone that was never extracted. Run
feature extraction first, or point `feature_backbone` to one you have on disk.
Check available features:

```bash
ls ~/.pathclaw/features/<dataset>/
```

### "Label column not found"

The labels CSV must have a column whose values match slide filenames (without
extension). Ask the agent: *"profile dataset <id>"* to see available columns
and slide-to-row match rate.

### Trained model but metrics look random

A known historical bug: `trainer.py` used to silently generate random labels if
`labels.json` was missing. Fixed. If you hit this on an old checkpoint, re-run
training with the correct labels.

---

## UI

### "Unexpected token 'I', \"Internal S\"... is not valid JSON"

The server returned 500 and the frontend was parsing HTML as JSON. Fixed in
v0.3 via the `apiJson` helper — errors now surface as readable toasts. If you
still see this, your browser cached an old `app.js` — hard-refresh (Ctrl+Shift+R)
or bump the `?v=` param in `index.html`.

### Sidebar tree flickers or loses expansion state

Fixed in v0.3 via hash-gated diff rendering. Clear browser cache if you see it
again.

### Viewer tiles show 404

OpenSeadragon is asking for tiles the backend can't serve. Most common cause:
the slide file moved after the dataset was registered. Re-register, or fix the
`path` in `~/.pathclaw/datasets/<id>/metadata.json`.

---

## Plugins

### `smoke_test_plugin` fails with a shape error

Your plugin's `build(in_dim, embed_dim, config)` signature must return a module
whose `forward(x)` maps `(B, N, in_dim) -> (B, N, embed_dim)`. Check both the
in/out shapes and that the call signature matches exactly. See
[docs/PLUGIN_DEV.md](PLUGIN_DEV.md).

### `implement_from_paper` produces unusable code

The tool is a workflow prompt, not codegen. The agent still has to author the
code via `write_workspace_file`. If the generated plugin is wrong, read the
paper sections the agent couldn't resolve and call `ask_user` for clarification.

---

## Telegram

### Bot not responding

1. Check status: `curl http://localhost:8101/api/telegram/status`.
2. Check logs: `tail -f ~/.pathclaw/telegram.log`.
3. Verify token: your BotFather token has `bot` as the first segment
   (`<id>:<hash>`).
4. If you set a passcode, users must send `/start <passcode>` first.

### Messages from wrong user

Set the `username` allowlist on `/api/telegram/start` — only that username is
accepted. If you don't set one, any Telegram user who knows the bot can message.

---

## Multi-GPU

### Second GPU not utilized

Multi-GPU dispatch landed in v0.3. Earlier versions pinned GPU 0. Check:

```bash
curl http://localhost:8101/api/queue/resources
```

Should list every GPU. If only one shows up, check
`CUDA_VISIBLE_DEVICES` in the env of your server process.

### Want to pin a specific GPU

Pass `device` in the training config or add `gpu_id` to the queue payload.
Examples:

```
train ABMIL on <dataset_id> with device cuda:1
```

---

## Agent behaviour

### Agent runs wrong dataset

Usually a label-column mismatch — the agent inferred a column that exists but
doesn't mean what it thought. Ask for a profile first:
*"profile dataset <id>"* — check class balance and column meanings before
committing to training.

### Agent gets stuck in a tool loop

Max 6 rounds per turn. If the agent isn't converging, send a message that
breaks the loop: *"stop, summarize what you tried, then ask me what to do."*

### Agent can't find files I uploaded

Uploads are per-session. Check:

```bash
ls ~/.pathclaw/sessions/<current_sid>/uploads/
```

Cross-session file access requires attaching folders (`attach_to_session`) or
registering the file as a dataset/plugin in the target session.
