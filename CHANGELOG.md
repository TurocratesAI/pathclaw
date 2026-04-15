# Changelog

All notable changes to PathClaw are documented here. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.3.0] — 2026-04-15

First open-source release.

### Added
- Multi-GPU dispatch in the task queue: per-GPU slot tracker, first-free
  allocation, optional pinning via `payload.gpu_id`.
- `GET /api/queue/resources` — host RAM, per-GPU VRAM, slot assignments.
- Frontend `apiJson()` helper with structured error toasts; replaces
  `fetch().then(r=>r.json())` calls that silently parsed HTML 500 pages.
- `confirmModal()` async dialog replacing native `confirm()`.
- Genomics module: 10 tools covering MAF/VCF parsing, TMB, oncoplot,
  cBioPortal, KM survival, multi-omic labels, biomarker discovery.
- Plugin system: built-in `mammoth` (MoE-LoRA) and `cellpose` (nuclear
  segmentation), user registry overlay at `~/.pathclaw/plugins/user_registry.json`.
- Telegram bot: multi-session binding (`/sessions`, `/session <id>`,
  `/new`, `/status`), username allowlist, optional passcode.
- Documentation set: `MANUAL.md`, `docs/PathClaw-Manual.docx`,
  `docs/TROUBLESHOOTING.md`, `docs/PLUGIN_DEV.md`, `docs/SECURITY.md`,
  `docs/DEVELOPMENT.md`, `docs/AGENT_TOOLS.md`.

### Changed
- README rewritten — removed marketing prose, accurate tool count (66, was
  35), full feature/skills/configuration sections.
- Author consolidated to Devansh Lalwani <devansh@turocrates.ai>.
- Trainer per-slide OOM recovery: skip the offending slide, log, continue.
- Frontend cache version bumped to `v=43`.

### Fixed
- `search_literature` `NameError` (`os` shadow as `_os`).
- `search_gdc` JSON filter encoding (filters now serialize correctly).
- GDC downloads survive server restart via on-disk checkpoints.
- Sidebar tree flicker via hash-gated diff rendering.
- 14 `alert()` sites replaced with structured toasts so backend errors are
  readable instead of raw JSON parse messages.

### Security
- Documented Telegram bot exposure, config.json plaintext token storage,
  cloned-repo trust posture, LLM API egress.
