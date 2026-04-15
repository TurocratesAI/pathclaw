# Plugin Development

PathClaw plugins extend the training pipeline without touching core code. Two
built-in plugins ship in the repo (MAMMOTH MoE-LoRA, Cellpose nuclear
segmentation); users can register more via `~/.pathclaw/plugins/user_registry.json`
or at runtime via the agent.

---

## Contract

A plugin is a Python module exposing a single factory function:

```python
# workspace/plugins/my_plugin.py
import torch.nn as nn

def build(in_dim: int, embed_dim: int, config: dict) -> nn.Module:
    return MyModule(in_dim, embed_dim, **config)

class MyModule(nn.Module):
    def __init__(self, in_dim, embed_dim, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, embed_dim),
        )

    def forward(self, x):
        # x: (B, N, in_dim)  →  (B, N, embed_dim)
        return self.net(x)
```

Rules:

- `build` receives the feature dim of the chosen backbone (`in_dim`), the
  downstream MIL embed dim (`embed_dim`), and a free-form `config` dict.
- The returned module's `forward(x)` must map `(B, N, in_dim)` to
  `(B, N, embed_dim)`. `B` is batch (usually 1 for MIL), `N` is number of
  patches per slide.
- No side effects at import time. All GPU allocation happens inside `build`.
- `config` defaults come from registration; users can override per training run.

---

## Registration

Two ways:

### A. User registry file

Edit `~/.pathclaw/plugins/user_registry.json`:

```json
{
  "plugins": [
    {
      "id": "my_plugin",
      "kind": "patch_embed",
      "import_path": "workspace.plugins.my_plugin:build",
      "default_config": {"hidden": 512},
      "applies_to": ["mil"],
      "description": "Two-layer MLP patch embedder"
    }
  ]
}
```

### B. Via the agent

```
register plugin {
  "id": "my_plugin",
  "kind": "patch_embed",
  "import_path": "workspace.plugins.my_plugin:build",
  "default_config": {"hidden": 512},
  "applies_to": ["mil"]
}
```

The agent writes the same JSON entry and reloads the registry.

---

## Smoke test

Before wiring a plugin into a real training run, run the smoke test:

```
smoke_test_plugin my_plugin
```

This instantiates the module with synthetic `(in_dim=1024, embed_dim=512)`,
feeds a random `(1, 256, 1024)` tensor, and asserts the output shape is
`(1, 256, 512)`. Failures surface the traceback in the chat stream.

---

## Using a plugin in training

Once registered and smoke-tested, the plugin appears as a toggle in the
Training Configure tab. Or from chat:

```
train ABMIL on tcga-ucec for 10 epochs, use my_plugin as patch embedder
```

Config override at training time:

```
train ABMIL on tcga-ucec with my_plugin(hidden=1024)
```

---

## Debugging tips

- Import errors: check `import_path` uses colon separator (`module:function`),
  not dot.
- Shape mismatches: run the smoke test, read the assertion message.
- Silent failures: plugins run inside the training subprocess; logs go to
  `~/.pathclaw/experiments/<id>/logs/stderr.log`.
- Config plumbing: add `print(config)` in `build` and re-run the smoke test to
  confirm what the trainer actually passes.

---

## Built-in plugins for reference

- **mammoth** — MoE-LoRA patch embedder (`backend/pathclaw/plugins/mammoth.py`)
- **cellpose** — Nuclear segmentation preprocessor (`backend/pathclaw/plugins/cellpose.py`)

Read these to see the full contract in use.
