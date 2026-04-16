"""Dedicated planner call — constrained-decoded JSON task plan.

Small local LLMs (gemma4, qwen3, llama3) routinely ignore "call create_task_plan
first" instructions when the user's request is long and enumerated. They drift
straight into tool-calling whatever looks first.

The fix that actually works (per LangGraph's plan-and-execute pattern + the
α-UMi "weak tool learners" paper) is a separate LLM call whose ONLY job is to
emit a JSON plan. Output shape is enforced via:

  - Ollama: `format` param → llama.cpp GBNF grammar-constrained decoding
  - Anthropic: native tool_use with forced tool_choice
  - OpenAI: response_format json_schema (strict)
  - Google: generationConfig responseSchema

After the plan is generated we persist it via the existing tasks module so the
executor loop sees it pinned at the top of every system prompt.
"""
from __future__ import annotations

import json
import re
from typing import Any

import httpx


PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "tasks": {
            "type": "array",
            "minItems": 1,
            "maxItems": 12,
            "items": {
                "type": "object",
                "properties": {
                    # maxLength caps prevent the known gemma4 repetition-loop
                    # bug during grammar-constrained decoding of free-text
                    # fields (ollama issues #15502, #15260).
                    "title": {"type": "string", "minLength": 3, "maxLength": 80},
                    "description": {"type": "string", "minLength": 3, "maxLength": 240},
                    "pause_after": {"type": "boolean"},
                },
                "required": ["title", "description"],
            },
        },
    },
    "required": ["tasks"],
}


PLANNER_SYSTEM = """You convert a user's multi-step request into an ordered JSON task plan.

Rules:
- 3-10 tasks. Each task is one logical chunk of work (typically 1-3 tool calls).
- Titles: short imperative, <60 chars (e.g. "Download CHOL slides", "Train ABMIL 5-fold").
- Descriptions: one sentence stating the concrete outcome or tools involved.
- Preserve the user's order exactly. Do NOT add steps the user didn't ask for.
- Set pause_after=true ONLY when the user explicitly wants to review results
  before proceeding (e.g. "show me the plan", "confirm X before continuing").
- Default pause_after=false so the agent auto-advances through the plan.

Return ONLY the JSON object matching the schema — no prose, no code fences."""


_NUMBERED_STEP_RE = re.compile(r"(?m)^\s*\d+[.)]\s+\S")
_BULLET_RE = re.compile(r"(?m)^\s*[-*]\s+\S")
_CONNECTORS = ("first", "then", "after that", "finally", "next", "afterwards", "lastly")


def should_plan(user_message: str) -> bool:
    """Heuristic: does this request have ≥3 discrete steps?

    Conservative on purpose — we'd rather skip planning for a borderline prompt
    than waste a planner call. False positives cost ~1-2s; false negatives cost
    drift on the real request.
    """
    m = user_message or ""
    if len(m) < 120:
        return False
    if len(_NUMBERED_STEP_RE.findall(m)) >= 3:
        return True
    if len(_BULLET_RE.findall(m)) >= 3:
        return True
    connectors = sum(
        1 for w in _CONNECTORS
        if re.search(rf"\b{re.escape(w)}\b", m, re.I)
    )
    if connectors >= 3:
        return True
    return False


def _clean_tasks(raw: Any) -> list[dict]:
    """Normalize planner output. Gemma4's grammar-constrained decoder sometimes
    emits empty/short titles ("", ":") — fall back to the description's first
    words so the task still shows up usably."""
    out: list[dict] = []
    if not isinstance(raw, list):
        return out
    for t in raw[:12]:
        if not isinstance(t, dict):
            continue
        title = (t.get("title") or "").strip().lstrip(":").strip()[:80]
        desc = (t.get("description") or "").strip().lstrip(":").strip()[:400]
        if len(title) < 5:
            # Derive title from description when the LLM leaves title blank.
            first_sentence = re.split(r"[.!?:]", desc, maxsplit=1)[0].strip()
            title = first_sentence[:60] if first_sentence else title
        if len(title) < 3:
            continue
        out.append({
            "title": title,
            "description": desc,
            "pause_after": bool(t.get("pause_after")),
        })
    return out


# ---------------------------------------------------------------------------
# Per-provider planner calls
# ---------------------------------------------------------------------------

async def _plan_ollama_once(
    model: str, user_message: str, ollama_base: str, num_ctx: int,
    seed: int | None = None,
) -> list[dict]:
    options: dict[str, Any] = {"num_ctx": num_ctx, "temperature": 0.1}
    if seed is not None:
        options["seed"] = seed
    async with httpx.AsyncClient(timeout=90.0) as client:
        resp = await client.post(
            f"{ollama_base}/api/chat",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": PLANNER_SYSTEM},
                    {"role": "user", "content": user_message},
                ],
                "stream": False,
                "format": PLAN_SCHEMA,
                "options": options,
            },
        )
        resp.raise_for_status()
        content = resp.json().get("message", {}).get("content", "") or ""
    try:
        return _clean_tasks((json.loads(content) or {}).get("tasks"))
    except json.JSONDecodeError:
        return []


async def _plan_ollama(
    model: str, user_message: str, ollama_base: str, num_ctx: int,
) -> list[dict]:
    """Call the active model with grammar-constrained decoding.

    If the first pass returns 0 usable tasks (known gemma4 repetition-loop
    under constrained decoding — ollama #15502, capped by maxLength in
    PLAN_SCHEMA so output still parses), retry once with a different seed.
    The executor-level first-turn interception is the final fallback when
    neither pass produces a plan.
    """
    tasks = await _plan_ollama_once(model, user_message, ollama_base, num_ctx)
    if tasks:
        return tasks
    return await _plan_ollama_once(model, user_message, ollama_base, num_ctx, seed=7)


async def _plan_anthropic(model: str, user_message: str, api_key: str) -> list[dict]:
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload = {
        "model": model,
        "max_tokens": 2048,
        "system": PLANNER_SYSTEM,
        "messages": [{"role": "user", "content": user_message}],
        "tools": [{
            "name": "emit_plan",
            "description": "Emit the ordered task plan for this request.",
            "input_schema": PLAN_SCHEMA,
        }],
        "tool_choice": {"type": "tool", "name": "emit_plan"},
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            "https://api.anthropic.com/v1/messages", json=payload, headers=headers,
        )
        resp.raise_for_status()
        data = resp.json()
    for block in data.get("content", []) or []:
        if block.get("type") == "tool_use" and block.get("name") == "emit_plan":
            return _clean_tasks((block.get("input") or {}).get("tasks"))
    return []


async def _plan_openai(model: str, user_message: str, api_key: str, base: str) -> list[dict]:
    url = (base or "https://api.openai.com/v1").rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": PLANNER_SYSTEM},
            {"role": "user", "content": user_message},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "task_plan", "schema": PLAN_SCHEMA, "strict": True},
        },
        "temperature": 0.1,
    }
    headers = {"Authorization": f"Bearer {api_key}", "content-type": "application/json"}
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
    return _clean_tasks((json.loads(content) or {}).get("tasks"))


async def _plan_google(model: str, user_message: str, api_key: str) -> list[dict]:
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        f"?key={api_key}"
    )
    payload = {
        "systemInstruction": {"parts": [{"text": PLANNER_SYSTEM}]},
        "contents": [{"role": "user", "parts": [{"text": user_message}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": PLAN_SCHEMA,
            "temperature": 0.1,
        },
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(url, json=payload)
        resp.raise_for_status()
        text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
    return _clean_tasks((json.loads(text) or {}).get("tasks"))


async def generate_plan(
    provider: str,
    model: str,
    user_message: str,
    *,
    ollama_base: str = "http://localhost:11434",
    num_ctx: int = 8192,
    anthropic_key: str = "",
    openai_key: str = "",
    openai_base: str = "",
    google_key: str = "",
) -> list[dict]:
    """Dispatch to the active provider's planner. Returns [] on any failure so
    the caller can fall back to the in-loop MANDATORY instruction."""
    try:
        if provider == "ollama":
            return await _plan_ollama(model, user_message, ollama_base, num_ctx)
        if provider == "anthropic" and anthropic_key:
            return await _plan_anthropic(model, user_message, anthropic_key)
        if provider == "openai" and openai_key:
            return await _plan_openai(model, user_message, openai_key, openai_base)
        if provider == "google" and google_key:
            return await _plan_google(model, user_message, google_key)
    except Exception:
        return []
    return []
