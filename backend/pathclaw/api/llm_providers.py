"""Multi-provider LLM abstraction for PathClaw.

Supports: Ollama (local), Anthropic, OpenAI (GPT), Google (Gemini).

Each provider's stream_round() yields Ollama-compatible chunk dicts so that
chat.py can consume them without needing provider-specific parsing logic.

Chunk format yielded by stream_round():
    {"message": {"content": str, "tool_calls": list}, "done": bool}

Blocking chat_round() returns:
    {"content": str, "tool_calls": list}
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, AsyncGenerator

import httpx

PATHCLAW_DATA_DIR = Path(os.environ.get("PATHCLAW_DATA_DIR", "~/.pathclaw")).expanduser()
CONFIG_PATH = PATHCLAW_DATA_DIR / "config.json"

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    if CONFIG_PATH.exists():
        return json.loads(CONFIG_PATH.read_text())
    return {}


def get_active_provider() -> tuple[str, str]:
    """Return (provider_name, model) from config, falling back to Ollama defaults."""
    cfg = _load_config()
    provider = cfg.get("llm_provider") or os.environ.get("PATHCLAW_PROVIDER", "ollama")
    model = cfg.get("llm_model") or os.environ.get("PATHCLAW_MODEL", "qwen3:8b")
    return provider, model


def list_provider_models(provider: str) -> list[str]:
    """Return known models for a provider (UI hints — free-text input also accepted).

    For Ollama, users may type any model tag they've pulled locally; this list is
    just a suggestion. The same applies to OpenAI-compatible endpoints (OpenRouter,
    Together, Groq, LM Studio, vLLM) when pointing `openai_base` at a custom URL.
    """
    models = {
        "ollama": [
            "qwen3:8b", "qwen3:14b", "qwen3:32b",
            "qwen2.5:7b", "qwen2.5:14b", "qwen2.5-coder:7b",
            "llama3.3:8b", "llama3.3:70b", "llama3.1:8b",
            "gemma3:4b", "gemma3:12b", "gemma3:27b",
            "deepseek-r1:7b", "deepseek-r1:14b", "deepseek-r1:32b",
            "mistral:7b", "mixtral:8x7b",
            "codellama:7b", "codellama:13b",
            "phi4:14b",
        ],
        "anthropic": [
            "claude-opus-4-6",
            "claude-sonnet-4-6",
            "claude-haiku-4-5",
            "claude-opus-4-20250514",
            "claude-sonnet-4-20250514",
            "claude-haiku-4-5-20251001",
        ],
        "openai": [
            "gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini",
            "o4-mini", "o3-mini",
        ],
        "google": [
            "gemini-2.5-flash", "gemini-2.5-pro",
            "gemini-2.0-flash", "gemini-2.0-flash-lite",
        ],
    }
    return models.get(provider, [])


# ---------------------------------------------------------------------------
# Tool format converters
# ---------------------------------------------------------------------------

def _tools_to_anthropic(tools: list[dict]) -> list[dict]:
    """Convert OpenAI-style tool list to Anthropic format."""
    result = []
    for t in tools:
        fn = t.get("function", {})
        result.append({
            "name": fn.get("name", ""),
            "description": fn.get("description", ""),
            "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
        })
    return result


def _tools_to_google(tools: list[dict]) -> list[dict]:
    """Convert OpenAI-style tool list to Gemini format."""
    declarations = []
    for t in tools:
        fn = t.get("function", {})
        declarations.append({
            "name": fn.get("name", ""),
            "description": fn.get("description", ""),
            "parameters": fn.get("parameters", {"type": "object", "properties": {}}),
        })
    return [{"functionDeclarations": declarations}] if declarations else []


# ---------------------------------------------------------------------------
# Message format converters
# ---------------------------------------------------------------------------

def _messages_to_anthropic(messages: list[dict]) -> tuple[str, list[dict]]:
    """Convert Ollama-format messages to Anthropic format.

    Returns (system_content, conversation_messages).
    Handles: system, user, assistant (with/without tool_calls), tool results.
    Tool IDs are assigned positionally for round-trip consistency.
    """
    system_content = ""
    conv = []
    tool_call_id_map: list[str] = []  # index → fake_id for pending tool results

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "") or ""
        tool_calls = msg.get("tool_calls", [])

        if role == "system":
            system_content = content
            continue

        if role == "user":
            # Could be a plain user message or a tool result
            # Tool results come after assistant messages with tool_calls
            if tool_call_id_map:
                # This is a tool result message
                tool_use_id = tool_call_id_map.pop(0)
                block = {"type": "tool_result", "tool_use_id": tool_use_id, "content": content}
                if conv and conv[-1]["role"] == "user" and isinstance(conv[-1]["content"], list):
                    conv[-1]["content"].append(block)
                else:
                    conv.append({"role": "user", "content": [block]})
            else:
                conv.append({"role": "user", "content": content})
            continue

        if role == "assistant":
            blocks: list[dict] = []
            if content:
                blocks.append({"type": "text", "text": content})
            for i, tc in enumerate(tool_calls):
                fn = tc.get("function", {})
                fake_id = f"tool_{len(conv)}_{i}"
                tool_call_id_map.append(fake_id)
                args = fn.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {}
                blocks.append({
                    "type": "tool_use",
                    "id": fake_id,
                    "name": fn.get("name", ""),
                    "input": args,
                })
            conv.append({"role": "assistant", "content": blocks if blocks else content})
            continue

        if role == "tool":
            # Bare tool result (Ollama appends these after tool_calls assistant message)
            if tool_call_id_map:
                tool_use_id = tool_call_id_map.pop(0)
            else:
                tool_use_id = f"tool_unknown_{len(conv)}"
            block = {"type": "tool_result", "tool_use_id": tool_use_id, "content": content}
            if conv and conv[-1]["role"] == "user" and isinstance(conv[-1]["content"], list):
                conv[-1]["content"].append(block)
            else:
                conv.append({"role": "user", "content": [block]})

    return system_content, conv


def _messages_to_google(messages: list[dict]) -> tuple[str, list[dict]]:
    """Convert Ollama-format messages to Gemini format.

    Returns (system_instruction, conversation_messages).
    """
    system_instruction = ""
    conv = []
    pending_tool_calls: list[dict] = []

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "") or ""
        tool_calls = msg.get("tool_calls", [])

        if role == "system":
            system_instruction = content
            continue

        if role == "user":
            conv.append({"role": "user", "parts": [{"text": content}]})
            pending_tool_calls.clear()
            continue

        if role == "assistant":
            parts: list[dict] = []
            if content:
                parts.append({"text": content})
            for tc in tool_calls:
                fn = tc.get("function", {})
                args = fn.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {}
                parts.append({"functionCall": {"name": fn.get("name", ""), "args": args}})
                pending_tool_calls.append(fn.get("name", ""))
            conv.append({"role": "model", "parts": parts if parts else [{"text": ""}]})
            continue

        if role == "tool":
            fn_name = pending_tool_calls.pop(0) if pending_tool_calls else "unknown"
            # Gemini tool results must be in a "user" role
            if conv and conv[-1]["role"] == "user":
                conv[-1]["parts"].append({
                    "functionResponse": {"name": fn_name, "response": {"result": content}}
                })
            else:
                conv.append({"role": "user", "parts": [{
                    "functionResponse": {"name": fn_name, "response": {"result": content}}
                }]})

    return system_instruction, conv


def _anthropic_content_to_tool_calls(content: list[dict]) -> list[dict]:
    """Extract Ollama-compatible tool_calls from Anthropic content blocks."""
    result = []
    for block in content:
        if block.get("type") == "tool_use":
            result.append({
                "function": {
                    "name": block.get("name", ""),
                    "arguments": block.get("input", {}),
                }
            })
    return result


# ---------------------------------------------------------------------------
# Ollama provider
# ---------------------------------------------------------------------------

async def stream_round_ollama(
    model: str, messages: list[dict], tools: list[dict],
    ollama_base: str, num_ctx: int,
) -> AsyncGenerator[dict, None]:
    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream(
            "POST",
            f"{ollama_base}/api/chat",
            json={
                "model": model,
                "messages": messages,
                "tools": tools,
                "stream": True,
                "options": {"num_ctx": num_ctx, "temperature": 0.2},
            },
        ) as resp:
            if resp.status_code != 200:
                yield {"done": True, "error": f"Ollama error {resp.status_code}"}
                return
            async for raw_line in resp.aiter_lines():
                if not raw_line.strip():
                    continue
                try:
                    chunk = json.loads(raw_line)
                    yield chunk
                except json.JSONDecodeError:
                    continue


async def chat_round_ollama(
    model: str, messages: list[dict], tools: list[dict],
    ollama_base: str, num_ctx: int,
) -> dict:
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            f"{ollama_base}/api/chat",
            json={
                "model": model,
                "messages": messages,
                "tools": tools,
                "stream": False,
                "options": {"num_ctx": num_ctx, "temperature": 0.2},
            },
        )
        if resp.status_code != 200:
            raise httpx.HTTPStatusError(
                f"Ollama error {resp.status_code}", request=resp.request, response=resp
            )
        data = resp.json()
        return data.get("message", {})


# ---------------------------------------------------------------------------
# Anthropic provider
# ---------------------------------------------------------------------------

ANTHROPIC_API_BASE = "https://api.anthropic.com/v1"
ANTHROPIC_VERSION = "2023-06-01"


async def stream_round_anthropic(
    model: str, messages: list[dict], tools: list[dict], api_key: str,
) -> AsyncGenerator[dict, None]:
    system_content, conv = _messages_to_anthropic(messages)
    anthr_tools = _tools_to_anthropic(tools)

    payload: dict[str, Any] = {
        "model": model,
        "max_tokens": 4096,
        "stream": True,
        "messages": conv,
    }
    if system_content:
        payload["system"] = system_content
    if anthr_tools:
        payload["tools"] = anthr_tools

    headers = {
        "x-api-key": api_key,
        "anthropic-version": ANTHROPIC_VERSION,
        "content-type": "application/json",
    }

    accumulated_text = ""
    tool_blocks: list[dict] = []
    current_tool: dict | None = None
    current_tool_json = ""

    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream("POST", f"{ANTHROPIC_API_BASE}/messages",
                                 json=payload, headers=headers) as resp:
            if resp.status_code != 200:
                body = await resp.aread()
                yield {"done": True, "error": f"Anthropic error {resp.status_code}: {body.decode()[:200]}"}
                return

            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if data_str == "[DONE]" or not data_str:
                    continue
                try:
                    ev = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                ev_type = ev.get("type", "")

                if ev_type == "content_block_start":
                    block = ev.get("content_block", {})
                    if block.get("type") == "tool_use":
                        current_tool = {
                            "function": {
                                "name": block.get("name", ""),
                                "arguments": {},
                            }
                        }
                        current_tool_json = ""

                elif ev_type == "content_block_delta":
                    delta = ev.get("delta", {})
                    if delta.get("type") == "text_delta":
                        text = delta.get("text", "")
                        accumulated_text += text
                        yield {"message": {"content": text, "tool_calls": None}, "done": False}
                    elif delta.get("type") == "input_json_delta":
                        current_tool_json += delta.get("partial_json", "")

                elif ev_type == "content_block_stop":
                    if current_tool is not None:
                        try:
                            current_tool["function"]["arguments"] = json.loads(current_tool_json) if current_tool_json else {}
                        except json.JSONDecodeError:
                            current_tool["function"]["arguments"] = {}
                        tool_blocks.append(current_tool)
                        current_tool = None
                        current_tool_json = ""

                elif ev_type == "message_stop":
                    break

    # Emit final chunk with tool calls only — content already streamed as deltas
    yield {
        "message": {
            "content": "",
            "tool_calls": tool_blocks if tool_blocks else None,
        },
        "done": True,
    }


async def chat_round_anthropic(
    model: str, messages: list[dict], tools: list[dict], api_key: str,
) -> dict:
    system_content, conv = _messages_to_anthropic(messages)
    anthr_tools = _tools_to_anthropic(tools)

    payload: dict[str, Any] = {
        "model": model,
        "max_tokens": 4096,
        "stream": False,
        "messages": conv,
    }
    if system_content:
        payload["system"] = system_content
    if anthr_tools:
        payload["tools"] = anthr_tools

    headers = {
        "x-api-key": api_key,
        "anthropic-version": ANTHROPIC_VERSION,
        "content-type": "application/json",
    }

    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(f"{ANTHROPIC_API_BASE}/messages", json=payload, headers=headers)
        if resp.status_code != 200:
            raise httpx.HTTPStatusError(
                f"Anthropic error {resp.status_code}: {resp.text[:200]}",
                request=resp.request, response=resp
            )
        data = resp.json()

    content_blocks = data.get("content", [])
    text = " ".join(b.get("text", "") for b in content_blocks if b.get("type") == "text")
    tool_calls = _anthropic_content_to_tool_calls(content_blocks)
    return {"content": text, "tool_calls": tool_calls}


# ---------------------------------------------------------------------------
# OpenAI provider
# ---------------------------------------------------------------------------

OPENAI_API_BASE_DEFAULT = "https://api.openai.com/v1"


def _openai_base() -> str:
    """Allow overriding OpenAI base URL for OpenRouter / LM Studio / vLLM / Together / Groq."""
    cfg = _load_config()
    return (
        cfg.get("openai_base")
        or os.environ.get("OPENAI_BASE")
        or OPENAI_API_BASE_DEFAULT
    )


def _messages_to_openai(messages: list[dict]) -> list[dict]:
    """Convert internal Ollama-style messages to OpenAI format.

    OpenAI requires:
    - assistant tool_calls: [{id, type:"function", function:{name, arguments:str}}]
    - tool results: {role:"tool", tool_call_id, content}
    """
    conv = []
    # Map positional index → call_id for pairing results with calls
    pending_ids: list[str] = []

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "") or ""
        tool_calls = msg.get("tool_calls") or []

        if role == "system":
            conv.append({"role": "system", "content": content})
            continue

        if role == "user":
            conv.append({"role": "user", "content": content})
            continue

        if role == "assistant":
            oai_tcs = []
            for i, tc in enumerate(tool_calls):
                fn = tc.get("function", {})
                args = fn.get("arguments", {})
                if not isinstance(args, str):
                    args = json.dumps(args)
                call_id = tc.get("id") or f"call_{len(conv)}_{i}"
                oai_tcs.append({
                    "id": call_id,
                    "type": "function",
                    "function": {"name": fn.get("name", ""), "arguments": args},
                })
                pending_ids.append(call_id)
            entry: dict = {"role": "assistant", "content": content or None}
            if oai_tcs:
                entry["tool_calls"] = oai_tcs
            conv.append(entry)
            continue

        if role == "tool":
            call_id = pending_ids.pop(0) if pending_ids else f"call_unknown_{len(conv)}"
            conv.append({"role": "tool", "tool_call_id": call_id, "content": content})
            continue

    return conv


async def stream_round_openai(
    model: str, messages: list[dict], tools: list[dict], api_key: str,
) -> AsyncGenerator[dict, None]:
    payload = {
        "model": model,
        "messages": _messages_to_openai(messages),
        "stream": True,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    accumulated_args: dict[int, str] = {}
    tool_calls_acc: list[dict] = []

    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream("POST", f"{_openai_base()}/chat/completions",
                                 json=payload, headers=headers) as resp:
            if resp.status_code != 200:
                body = await resp.aread()
                yield {"done": True, "error": f"OpenAI error {resp.status_code}: {body.decode()[:200]}"}
                return

            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                choice = (chunk.get("choices") or [{}])[0]
                delta = choice.get("delta", {})
                content = delta.get("content") or ""
                tc_delta = delta.get("tool_calls") or []

                if content:
                    yield {"message": {"content": content, "tool_calls": None}, "done": False}

                for tc in tc_delta:
                    idx = tc.get("index", 0)
                    if idx >= len(tool_calls_acc):
                        fn = tc.get("function", {})
                        tool_calls_acc.append({
                            "function": {"name": fn.get("name", ""), "arguments": ""}
                        })
                        accumulated_args[idx] = ""
                    fn_delta = tc.get("function", {})
                    if "name" in fn_delta and fn_delta["name"]:
                        tool_calls_acc[idx]["function"]["name"] = fn_delta["name"]
                    if "arguments" in fn_delta:
                        accumulated_args[idx] = accumulated_args.get(idx, "") + fn_delta["arguments"]

    # Parse accumulated args
    for i, tc in enumerate(tool_calls_acc):
        try:
            tc["function"]["arguments"] = json.loads(accumulated_args.get(i, "{}"))
        except json.JSONDecodeError:
            tc["function"]["arguments"] = {}

    yield {
        "message": {
            "content": "",
            "tool_calls": tool_calls_acc if tool_calls_acc else None,
        },
        "done": True,
    }


async def chat_round_openai(
    model: str, messages: list[dict], tools: list[dict], api_key: str,
) -> dict:
    payload: dict[str, Any] = {
        "model": model,
        "messages": _messages_to_openai(messages),
        "stream": False,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(f"{_openai_base()}/chat/completions", json=payload, headers=headers)
        if resp.status_code != 200:
            raise httpx.HTTPStatusError(
                f"OpenAI error {resp.status_code}: {resp.text[:200]}",
                request=resp.request, response=resp,
            )
        data = resp.json()

    msg = (data.get("choices") or [{}])[0].get("message", {})
    content = msg.get("content") or ""
    raw_tcs = msg.get("tool_calls") or []
    tool_calls = []
    for tc in raw_tcs:
        fn = tc.get("function", {})
        args = fn.get("arguments", "{}")
        try:
            args = json.loads(args) if isinstance(args, str) else args
        except Exception:
            args = {}
        tool_calls.append({"function": {"name": fn.get("name", ""), "arguments": args}})
    return {"content": content, "tool_calls": tool_calls}


# ---------------------------------------------------------------------------
# Google Gemini provider
# ---------------------------------------------------------------------------

GOOGLE_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"


async def stream_round_google(
    model: str, messages: list[dict], tools: list[dict], api_key: str,
) -> AsyncGenerator[dict, None]:
    system_instruction, conv = _messages_to_google(messages)
    gemini_tools = _tools_to_google(tools)

    payload: dict[str, Any] = {"contents": conv}
    if system_instruction:
        payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}
    if gemini_tools:
        payload["tools"] = gemini_tools

    url = f"{GOOGLE_API_BASE}/{model}:streamGenerateContent?key={api_key}"

    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream("POST", url, json=payload) as resp:
            if resp.status_code != 200:
                body = await resp.aread()
                yield {"done": True, "error": f"Google error {resp.status_code}: {body.decode()[:200]}"}
                return

            # Gemini streams as a JSON array: [chunk, chunk, ...]
            # Parse incrementally as bytes arrive so tokens stream to the user.
            buffer = ""
            async for raw in resp.aiter_bytes(1024):
                buffer += raw.decode("utf-8", errors="replace")
                # Extract and yield complete top-level JSON objects as they accumulate
                while True:
                    # Skip array wrapper characters and whitespace
                    stripped = buffer.lstrip(" \t\r\n,[")
                    if not stripped or not stripped.startswith("{"):
                        buffer = stripped  # consume leading junk
                        break
                    # Find the matching closing brace using depth tracking
                    depth = 0
                    found = -1
                    in_str = False
                    escape = False
                    for i, ch in enumerate(stripped):
                        if escape:
                            escape = False
                        elif ch == "\\" and in_str:
                            escape = True
                        elif ch == '"':
                            in_str = not in_str
                        elif not in_str:
                            if ch == "{":
                                depth += 1
                            elif ch == "}":
                                depth -= 1
                                if depth == 0:
                                    found = i
                                    break
                    if found < 0:
                        break  # incomplete object, wait for more bytes
                    obj_str = stripped[:found + 1]
                    # Advance buffer past the consumed object
                    buffer = stripped[found + 1:]
                    try:
                        chunk = json.loads(obj_str)
                        candidates = chunk.get("candidates", [{}])
                        content = (candidates[0] or {}).get("content", {})
                        parts = content.get("parts", [])
                        for part in parts:
                            if "text" in part:
                                yield {"message": {"content": part["text"], "tool_calls": None}, "done": False}
                            elif "functionCall" in part:
                                fc = part["functionCall"]
                                args = fc.get("args", {})
                                yield {
                                    "message": {
                                        "content": "",
                                        "tool_calls": [{"function": {
                                            "name": fc.get("name", ""),
                                            "arguments": args,
                                        }}],
                                    },
                                    "done": False,
                                }
                    except json.JSONDecodeError:
                        pass

    yield {"message": {"content": "", "tool_calls": None}, "done": True}


async def chat_round_google(
    model: str, messages: list[dict], tools: list[dict], api_key: str,
) -> dict:
    system_instruction, conv = _messages_to_google(messages)
    gemini_tools = _tools_to_google(tools)

    payload: dict[str, Any] = {"contents": conv}
    if system_instruction:
        payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}
    if gemini_tools:
        payload["tools"] = gemini_tools

    url = f"{GOOGLE_API_BASE}/{model}:generateContent?key={api_key}"

    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(url, json=payload)
        if resp.status_code != 200:
            raise httpx.HTTPStatusError(
                f"Google error {resp.status_code}: {resp.text[:200]}",
                request=resp.request, response=resp,
            )
        data = resp.json()

    candidates = data.get("candidates", [{}])
    parts = (candidates[0] or {}).get("content", {}).get("parts", [])
    text = " ".join(p.get("text", "") for p in parts if "text" in p)
    tool_calls = []
    for p in parts:
        if "functionCall" in p:
            fc = p["functionCall"]
            tool_calls.append({
                "function": {"name": fc.get("name", ""), "arguments": fc.get("args", {})}
            })
    return {"content": text, "tool_calls": tool_calls}


# ---------------------------------------------------------------------------
# Unified dispatch
# ---------------------------------------------------------------------------

async def stream_round(
    provider: str,
    model: str,
    messages: list[dict],
    tools: list[dict],
    ollama_base: str = "http://localhost:11434",
    num_ctx: int = 16384,
) -> AsyncGenerator[dict, None]:
    """Dispatch stream_round to the correct provider."""
    cfg = _load_config()

    if provider == "anthropic":
        api_key = cfg.get("anthropic_api_key") or os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            async def _err():
                yield {"done": True, "error": "Anthropic API key not set. Run /api/config to configure."}
            return _err()
        return stream_round_anthropic(model, messages, tools, api_key)

    elif provider == "openai":
        api_key = cfg.get("openai_api_key") or os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            async def _err():
                yield {"done": True, "error": "OpenAI API key not set. Run /api/config to configure."}
            return _err()
        return stream_round_openai(model, messages, tools, api_key)

    elif provider == "google":
        api_key = cfg.get("google_api_key") or os.environ.get("GOOGLE_API_KEY", "")
        if not api_key:
            async def _err():
                yield {"done": True, "error": "Google API key not set. Run /api/config to configure."}
            return _err()
        return stream_round_google(model, messages, tools, api_key)

    else:  # default: ollama
        return stream_round_ollama(model, messages, tools, ollama_base, num_ctx)


async def chat_round(
    provider: str,
    model: str,
    messages: list[dict],
    tools: list[dict],
    ollama_base: str = "http://localhost:11434",
    num_ctx: int = 16384,
) -> dict:
    """Dispatch chat_round to the correct provider."""
    cfg = _load_config()

    if provider == "anthropic":
        api_key = cfg.get("anthropic_api_key") or os.environ.get("ANTHROPIC_API_KEY", "")
        return await chat_round_anthropic(model, messages, tools, api_key)

    elif provider == "openai":
        api_key = cfg.get("openai_api_key") or os.environ.get("OPENAI_API_KEY", "")
        return await chat_round_openai(model, messages, tools, api_key)

    elif provider == "google":
        api_key = cfg.get("google_api_key") or os.environ.get("GOOGLE_API_KEY", "")
        return await chat_round_google(model, messages, tools, api_key)

    else:
        return await chat_round_ollama(model, messages, tools, ollama_base, num_ctx)
