"""Server-side guardrails that compensate for small-LLM agent-loop failures.

Gemma4:26b (and many similarly-aligned 7–27B local models) drift predictably:

- **Tool-name hallucination.** Emits ``pathfinding:list_artifacts`` or
  ``gemma:plan`` because MCP-style namespaced names dominated pretraining.
- **Repetition loops** under grammar-constrained decoding
  (`"attachment of the attachment of the ..."`, ollama #15502).
- **Safety meltdown** after a single tool-call correction — the turn
  collapses into *"I am programmed to be a helpful and harmless AI
  assistant"* and never recovers.

Switching models would be the easy answer, but hospitals running PathClaw
on-prem often have gemma4 pulled and working. The server is the right
layer to absorb this flakiness:

- Be **liberal in what tool names we accept** (fuzzy match + prefix strip).
- **Detect repetition** in streamed text and kill the stream before the
  model finishes looping.
- **Detect safety meltdowns** so the executor can nudge instead of
  treating the refusal as a final answer.

These are cheap heuristics, not a replacement for a better model — but
they turn gemma4 from "agent loop collapses half the time" into "runs a
multi-step plan to completion".
"""
from __future__ import annotations

import difflib
import re


# --- (1) Tool-name normalizer ------------------------------------------------

# Namespace prefixes observed in the wild from gemma4/qwen2/llama3 when they
# misremember the MCP convention. Always strip — never part of a real tool.
_TOOL_PREFIX_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{1,24}[:/]")


def _canon(s: str) -> str:
    """Canonical form for matching: lowercase + strip non-alphanum. Turns
    ``CreateTaskPlan``, ``create-task-plan``, ``create_task_plan`` all into
    ``createtaskplan``, which collapses the common small-model casing/delim
    drift."""
    return re.sub(r"[^a-z0-9]", "", s.lower())


def normalize_tool_name(name: str, valid_names: list[str]) -> str | None:
    """Map a hallucinated tool name onto a real one, or ``None`` if no
    unambiguous match.

    Strategy (in order):
        1. Exact match → keep as-is.
        2. Strip any ``prefix:`` or ``prefix/`` namespace → exact match.
        3. Case-insensitive exact match.
        4. Canonical form (lowercase, punctuation-stripped) exact match —
           handles ``CreateTaskPlan`` / ``waitForJob`` → ``wait_for_job``.
        5. difflib close match on the canonical form (cutoff 0.85) — only if
           a single top candidate clears the threshold (prevents silent
           routing to the wrong tool).

    Returns the canonical name, or ``None`` if the caller should surface the
    unknown-tool error to the model as before.
    """
    if not name:
        return None
    if name in valid_names:
        return name

    stripped = _TOOL_PREFIX_RE.sub("", name)
    if stripped != name and stripped in valid_names:
        return stripped

    lowered_map = {v.lower(): v for v in valid_names}
    lc = stripped.lower()
    if lc in lowered_map:
        return lowered_map[lc]

    canon_map = {_canon(v): v for v in valid_names}
    cn = _canon(stripped)
    if cn and cn in canon_map:
        return canon_map[cn]

    # Fuzzy over canonical forms to absorb minor typos (extra/missing char).
    canon_keys = list(canon_map)
    close = difflib.get_close_matches(cn, canon_keys, n=2, cutoff=0.85)
    if len(close) == 1:
        return canon_map[close[0]]
    if len(close) >= 2 and difflib.SequenceMatcher(
        None, cn, close[0]
    ).ratio() - difflib.SequenceMatcher(
        None, cn, close[1]
    ).ratio() >= 0.1:
        return canon_map[close[0]]

    return None


# --- (2) Stream-level repetition killer -------------------------------------

def has_repetition_loop(text: str, *, min_reps: int = 5) -> bool:
    """Return True if the tail of ``text`` is any small substring (3–32 chars)
    repeated ``min_reps`` times in a row.

    Checks multiple span widths so we catch both char-level loops
    (``aaaaaaaa``) and phrase-level loops (``attachment of the attachment of
    the ...``) — the latter is the actual gemma4 failure mode under
    constrained decoding. Only looks at the tail, so running this after
    every streamed chunk is cheap.
    """
    if len(text) < 3 * min_reps:
        return False
    for span in (3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 24, 28, 32):
        if len(text) < span * min_reps:
            continue
        tail = text[-span * min_reps:]
        first = tail[:span]
        if all(tail[i * span : (i + 1) * span] == first for i in range(min_reps)):
            return True
    return False


# --- (3) Safety-meltdown interceptor ----------------------------------------

# Gemma-family melt-down openings. Matching is case-sensitive and anchored at
# the start because we only care about meltdowns as the *entire* response —
# if the model produces real work first and then says "I cannot fulfill
# this request" in passing, that's different (and rarer).
_MELTDOWN_PREFIXES = (
    "I cannot fulfill this request",
    "I am programmed to be a helpful and harmless",
    "I am programmed to be helpful and harmless",
    "My safety guidelines prohibit",
    "I am unable to provide",
    "I'm sorry, but I cannot",
    "I'm not able to help with that",
    "As a large language model,",
    "As an AI language model,",
)


def is_safety_meltdown(text: str) -> bool:
    """Return True if ``text`` looks like a refusal monologue rather than a
    real response. Only fires on short (<800 char) responses that START with
    one of the canonical opener phrases."""
    s = (text or "").lstrip()
    if len(s) > 800:
        return False
    return s.startswith(_MELTDOWN_PREFIXES)


__all__ = [
    "normalize_tool_name",
    "has_repetition_loop",
    "is_safety_meltdown",
]
