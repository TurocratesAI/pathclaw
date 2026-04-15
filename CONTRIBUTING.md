# Contributing

Thanks for looking at PathClaw. Before opening a PR please read
[docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) for layout, routing, agent-tool
conventions, and how skills inject into the chat prompt.

## Ground rules

- **No marketing prose in code or docs.** Technical tone only.
- **Don't commit `~/.pathclaw/`** — it holds tokens, large binaries, and
  per-user state. The `.gitignore` excludes it; verify your branch doesn't
  reintroduce it.
- **Frontend cache** — if you edit `backend/pathclaw/static/app.js`, bump the
  `?v=N` query param in `static/index.html` so users don't hit a stale cache.
- **No `alert()` / `confirm()`** — use `showToast()` / `confirmModal()` in
  `app.js`.
- **No `fetch().then(r => r.json())`** — use the `apiJson()` helper so
  non-2xx responses surface as readable errors.

## Dev setup

```bash
git clone https://github.com/devanshlalwani/PathClaw.git
cd PathClaw/backend
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
uvicorn pathclaw.api.app:app --reload --port 8101
```

## Commit style

- Present-tense imperative ("Add multi-GPU dispatch", not "Added ...").
- One logical change per commit. Keep mechanical renames separate from
  behaviour changes.
- Reference an issue number if one exists (`#42`).

## PR checklist

- [ ] Affected routes start and return expected shapes (`curl` them).
- [ ] Any new agent tool has a JSON schema, a handler, and a dispatch entry.
- [ ] Any new skill has a `SKILL.md` and a keyword trigger in `chat.py`.
- [ ] `CHANGELOG.md` updated under the unreleased section.
- [ ] No secrets in the diff (`git diff | grep -E 'hf_|sk-|ghp_|xoxb-'`).

## Reporting bugs

File an issue via the [bug report template](.github/ISSUE_TEMPLATE/bug_report.md).
Include the server log tail, your Python version, GPU vendor/model, and the
exact command or agent prompt that reproduced the problem.

## Security

Do not open a public issue for security findings. Email
`devansh@turocrates.ai` instead. See [docs/SECURITY.md](docs/SECURITY.md).
