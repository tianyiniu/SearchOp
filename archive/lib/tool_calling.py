"""A minimal tool-calling loop for an OpenAI-compatible chat model.

The loop: send the conversation + tool schemas to the model; if it asks to call
tools, run them and feed the results back; stop when it returns a final answer
(or after it spends its ``max_tool_calls`` budget). Works with the OpenAI SDK
pointed at OpenAI, a local
vLLM server, or any OpenAI-compatible endpoint.

    from openai import OpenAI
    from tools import build_tools
    from tool_calling import run_with_tools

    client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
    registry, schemas = build_tools()
    result = run_with_tools(client, "Qwen/Qwen3-14B",
                            "You are a helpful assistant. Use tools, then answer.",
                            "Who composed the music for the film Priest (1994)?",
                            schemas, registry)
    print(result.answer)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    """One tool invocation the model made, with its result."""
    name: str
    arguments: dict
    result: str


@dataclass
class AgentResult:
    answer: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    num_model_calls: int = 0


def _parse_arguments(raw: str) -> dict:
    """Tool-call arguments arrive as a JSON string; parse leniently."""
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else {"query": str(obj)}
    except Exception:
        return {"query": str(raw)}


def _query_of(args: dict) -> str:
    """All three tools take a single string. Accept common key names, else the
    first string value."""
    for key in ("query", "url", "code", "q", "input"):
        v = args.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    for v in args.values():
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def run_with_tools(
    client: Any,
    model: str,
    system_prompt: str,
    user_prompt: str,
    tool_schemas: list[dict],
    tool_registry: dict,
    max_tool_calls: int = 20,
    temperature: float = 0.7,
    max_tokens: int = 2048,
) -> AgentResult:
    """Drive ``model`` through tool use until it answers or it spends its budget
    of ``max_tool_calls`` tool calls, then return its final answer plus the tool
    calls it made."""
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    traces: list[ToolCall] = []
    n_calls = 0

    # Each model round may request several tool calls; the loop bound is a safety
    # net while ``len(traces) < max_tool_calls`` is the real budget check.
    for _ in range(max_tool_calls + 2):
        if len(traces) >= max_tool_calls:
            break  # spent the tool budget -> fall through to a final-answer request
        n_calls += 1
        response = client.chat.completions.create(
            model=model, messages=messages, tools=tool_schemas,
            temperature=temperature, max_tokens=max_tokens,
        )
        msg = response.choices[0].message

        # No tool calls -> the model gave its final answer.
        if not msg.tool_calls:
            return AgentResult(msg.content or "", traces, n_calls)

        # Echo the assistant turn (with its tool_calls) back into the history;
        # the API requires this before the matching tool result messages.
        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ],
        })

        # Run each requested tool and append its result. Every tool_call_id must
        # get a reply, so once the budget is spent we answer the rest with a note
        # instead of running them.
        for tc in msg.tool_calls:
            name = tc.function.name
            args = _parse_arguments(tc.function.arguments)
            if len(traces) >= max_tool_calls:
                result = "[tool-call budget exhausted; give your final answer now]"
            else:
                query = _query_of(args)
                tool = tool_registry.get(name)
                if tool is None:
                    result = f"[unknown tool: {name}. Available: {sorted(tool_registry)}]"
                elif not query:
                    result = f"[empty arguments for {name}: {args}]"
                else:
                    try:
                        result = tool(query)
                    except Exception as exc:  # don't let a tool failure kill the loop
                        result = f"[tool {name} failed: {exc}]"
                traces.append(ToolCall(name=name, arguments=args, result=result))
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    # Budget spent (or loop bound hit): ask once for a plain-text final answer.
    n_calls += 1
    messages.append({"role": "user", "content": "Stop using tools and give your final answer now."})
    response = client.chat.completions.create(
        model=model, messages=messages, temperature=temperature, max_tokens=max_tokens,
    )
    return AgentResult(response.choices[0].message.content or "", traces, n_calls)
