#!/usr/bin/env python3
"""Dump agent tool registry from chat.py as markdown.

Usage:
    python scripts/dump_tools.py > docs/AGENT_TOOLS.md

Parses the TOOLS = [...] literal in backend/pathclaw/api/routes/chat.py and
emits a markdown table. For per-tool JSON schemas, the file is the source of
truth — this script is a convenience for the top-level reference doc.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

CHAT_PY = Path(__file__).resolve().parent.parent / "backend/pathclaw/api/routes/chat.py"


def extract_tools() -> list[dict]:
    src = CHAT_PY.read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == "TOOLS":
                    return [_to_dict(e) for e in node.value.elts if isinstance(e, ast.Dict)]
    return []


def _to_dict(node: ast.Dict) -> dict:
    out = {}
    for k, v in zip(node.keys, node.values):
        if not isinstance(k, ast.Constant):
            continue
        try:
            out[k.value] = ast.literal_eval(v)
        except Exception:
            out[k.value] = "<unparseable>"
    return out


def main() -> int:
    tools = extract_tools()
    if not tools:
        print("No TOOLS array found", file=sys.stderr)
        return 1
    print(f"# Agent Tools ({len(tools)} total)\n")
    print("Auto-generated from `backend/pathclaw/api/routes/chat.py`.\n")
    print("| Tool | Description |")
    print("|------|-------------|")
    for t in tools:
        name = t.get("name", "?")
        desc = t.get("description", "")
        desc = desc.replace("\n", " ").replace("|", "\\|")[:200]
        print(f"| `{name}` | {desc} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
