"""Which mechanism actually stops Qwen3.5 from thinking, and where does the
answer land? Tries each candidate against a served model and reports content,
reasoning_content, finish reason and token counts."""
import sys
from openai import OpenAI

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:18103/v1"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "Qwen/Qwen3.5-9B"
MAXTOK = int(sys.argv[3]) if len(sys.argv) > 3 else 3072

SYS = ("You are a Solver answering a hard graduate-level multiple-choice question. "
       "Reason step by step from your own knowledge, then commit. Reason step by step, "
       "then end with exactly one line: 'ANSWER: <letter>'. Always choose exactly one "
       "letter; never abstain.")
USER = ("Which surface molecule on tumor cells, when upregulated, stimulates T cell "
        "proliferation?\nA) CD28\nB) CTLA-4\nC) B7-1 (CD80)\nD) PD-L1\n\nGive your response.")

CASES = {
    "1. plain (no kwargs)": {},
    "2. enable_thinking=False (what our code sends)":
        {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
    "3. thinking=False": {"extra_body": {"chat_template_kwargs": {"thinking": False}}},
    "4. reasoning_effort=none": {"extra_body": {"reasoning_effort": "none"}},
    "5. chat_template_kwargs enable_thinking=False + /no_think in user":
        {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}, "_suffix": " /no_think"},
}

client = OpenAI(base_url=BASE, api_key="EMPTY", timeout=180, max_retries=0)
print(f"model={MODEL}  base={BASE}  max_tokens={MAXTOK}\n")
for name, cfg in CASES.items():
    cfg = dict(cfg)
    suffix = cfg.pop("_suffix", "")
    try:
        r = client.chat.completions.create(
            model=MODEL, temperature=0.7, max_tokens=MAXTOK,
            messages=[{"role": "system", "content": SYS},
                      {"role": "user", "content": USER + suffix}], **cfg)
    except Exception as exc:
        print(f"{name}\n   REJECTED: {type(exc).__name__}: {str(exc)[:160]}\n")
        continue
    m = r.choices[0].message
    content = m.content or ""
    reasoning = getattr(m, "reasoning_content", None) or ""
    has_answer = "ANSWER:" in content.upper()
    print(f"{name}")
    print(f"   finish={r.choices[0].finish_reason}  completion_tokens={r.usage.completion_tokens}")
    print(f"   content: {len(content)} chars, ANSWER line present: {has_answer}")
    print(f"   reasoning_content: {len(reasoning)} chars"
          + (f", ANSWER present: {'ANSWER:' in reasoning.upper()}" if reasoning else ""))
    print(f"   content starts: {content[:110]!r}")
    if has_answer:
        print(f"   content ends:   {content[-70:]!r}")
    print()
