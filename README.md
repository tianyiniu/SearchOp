# Tool-calling kit

A small, self-contained extraction: three agent tools, a minimal tool-calling
loop, and two benchmark downloaders. No dependencies on the parent project.

## Contents

| File | What it is |
| --- | --- |
| `tools.py` | The three tools — `search_info` (web search), `fetch_url` (page fetch), `code_compute` (run Python). Each is a `tool(query) -> str` callable with an OpenAI tool schema. `build_tools()` returns a `{name: tool}` registry + schema list. |
| `tool_calling.py` | `run_with_tools(...)` — a minimal loop that lets an OpenAI-compatible model call the tools and feed results back until it answers. |
| `download_frames.py` | Fetch the FRAMES benchmark → `data/frames/test.jsonl`. |
| `download_sealqa.py` | Fetch the SealQA `seal_hard` split → `data/sealqa/test.jsonl`. |
| `example.py` | End-to-end demo wiring the tools + loop together. |

## Install

```
pip install openai requests datasets
```

## Setup

- `SERPER_API_KEY` — required by `search_info` / `fetch_url` (https://serper.dev).
- An OpenAI-compatible chat endpoint for the loop (OpenAI, or a local vLLM server).

```
export SERPER_API_KEY=...
export OPENAI_BASE_URL=http://localhost:8000/v1   # vLLM / OpenAI endpoint
export OPENAI_API_KEY=EMPTY                        # 'EMPTY' for local vLLM
export MODEL=Qwen/Qwen3-14B
```

## Use

```
python download_frames.py     # -> data/frames/test.jsonl  (824 questions)
python download_sealqa.py     # -> data/sealqa/test.jsonl
python example.py             # model answers a question using the tools
```

Each JSONL row is `{"id", "question", "ground_truth", "source"}`.

## Notes

- `code_compute` runs Python in a subprocess with a timeout — it is **not** a
  security sandbox. Run only trusted code, or replace the executor.
- The loop is deliberately minimal (no caching, retries, or context management).
  It targets the OpenAI/vLLM tool-calling API and stops after `max_rounds`.
