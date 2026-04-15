# Security Notes

PathClaw is a research tool intended for single-user or trusted-group
deployments. It has no built-in authentication on the HTTP API and should not
be exposed to the public internet without a reverse proxy that enforces access
control.

---

## Secrets on disk

`~/.pathclaw/config.json` stores the following in plaintext:

- HuggingFace token
- Anthropic / OpenAI / Google API keys
- Telegram bot token
- Optional passcode

Recommended: `chmod 600 ~/.pathclaw/config.json` after first run.

Do not commit `~/.pathclaw/` into any repo. The `.gitignore` in the project
root excludes it; if you clone elsewhere, verify the exclusion.

---

## Cloned repositories

The `clone_repo` agent tool fetches arbitrary code from GitHub / GitLab /
HuggingFace / Codeberg / Bitbucket into
`~/.pathclaw/sessions/<sid>/workspace/repos/`. Registered plugins can execute
that code inside the server process.

Before enabling a user-cloned repo as a plugin:

1. Read the plugin source.
2. Run the smoke test in an isolated conda/venv if the plugin pulls heavy
   dependencies.
3. Inspect the `default_config` — a plugin that shells out or writes outside
   the session workspace is a red flag.

The agent will not auto-enable an unreviewed plugin — registration requires
either an explicit tool call or a registry edit.

---

## Telegram bot

The bot is an unauthenticated chat gateway unless you configure access control:

- **`username` allowlist** — only the listed Telegram usernames can message.
  Set this at `/api/telegram/start`.
- **Passcode** — users must send `/start <passcode>` before the bot accepts
  prompts. Combines with the username allowlist.

If you set neither, any Telegram user who guesses the bot's handle can invoke
the agent with full tool access on the bound session. Always set at least one.

---

## LLM API key egress

When you pick Anthropic, OpenAI, or Google as provider, every agent turn sends
the system prompt, conversation history, attached files (as text), and tool
call arguments to that provider. Do not paste PHI or patient-identifiable data
into chat unless your agreement with the provider permits it.

Ollama runs locally and sends nothing external.

---

## Network exposure

Default bind is `0.0.0.0:8101` — accessible on the LAN. For a multi-user
deployment:

- Bind to `127.0.0.1` and front with nginx or Caddy
- Add basic auth at the proxy
- Terminate TLS at the proxy (PathClaw speaks plain HTTP)

The tile server and manuscript compile endpoints read from and write to the
local filesystem with the server process's permissions. A compromised session
can read anything that user can read.

---

## Reporting vulnerabilities

Email `devansh@turocrates.ai` with a description and reproduction steps. Do
not open a public GitHub issue for security findings.
