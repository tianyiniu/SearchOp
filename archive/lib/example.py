"""End-to-end demo: let a model answer a question using the three tools.

Requires an OpenAI-compatible chat endpoint (OpenAI, or a local vLLM server)
and a Serper key for the web tools:

    pip install openai requests
    export SERPER_API_KEY=...
    export OPENAI_BASE_URL=http://localhost:8000/v1   # your vLLM/OpenAI endpoint
    export OPENAI_API_KEY=EMPTY                        # 'EMPTY' for local vLLM
    export MODEL=Qwen/Qwen3-14B
    python example.py
"""

import os

from openai import OpenAI

from tools import build_tools
from tool_calling import run_with_tools

client = OpenAI(
    base_url=os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1"),
    api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
)
MODEL = os.environ.get("MODEL", "Qwen/Qwen3-14B")

SYSTEM = (
    "You are a careful research assistant. Use the available tools to gather "
    "evidence, then give a short final answer prefixed with 'ANSWER: '."
)
QUESTION = "In 1994, Linus Roache starred in Priest. Who composed the music on his next film?"

registry, schemas = build_tools()  # all three tools; pass enabled=[...] to restrict
result = run_with_tools(client, MODEL, SYSTEM, QUESTION, schemas, registry)

print(f"Model calls: {result.num_model_calls}")
for call in result.tool_calls:
    print(f"  {call.name}({call.arguments})  ->  {call.result[:100]!r}")
print()
print(result.answer)
