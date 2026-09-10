# Onboarding: FRAMES search-side work

> Handoff document for continuing the search-side work in a fresh session. Paste it as context.

## Goal

Raise answer accuracy on `datasets/frames_qwen_answerable.json` (266 questions) toward ~100%.

Each question in this set was checked in advance: Qwen3-14B answers it correctly when given its
gold documents (the Wikipedia pages the benchmark says contain the answer). So reasoning is not the
bottleneck. If search collects the right documents, the answer follows. This work is about
collecting the right evidence.

## Where the code is

- `scripts/eval_search_step.py` — the evaluator. Search presets are in `PRESETS` (~line 76).
  Search plans: `run_iterative`, `run_single`, `run_decompose`, `run_decompose_react`.
  Prompts: `SEARCH_SYSTEM`, `DECOMPOSE_SYSTEM`.
- `tools.py` — `search_info` (web search through Serper, a paid API) and `fetch_url` (page
  download). Both hit live Serper by default, which costs money. `LOCAL_SCRAPE_URL` routes page
  downloads to the local Wikipedia server instead (see below).
- `scripts/doc_qa.py` — `gold_urls`, `normalize_url`, `pack_documents`, `answer_with_docs`.
  The last one is the same downstream call that was used to build the question split.
- `scripts/llm_judge.py` — `judge_answer` grades answers with gpt-5.4-mini.
- `Wikipedia_cache/` — the 1,052 gold Wikipedia pages, served locally by `scripts/wiki_backend.py`.
- `scripts/evidence.py` and `scripts/store_tools.py` — newer, uncommitted. `EvidenceStore`
  collects everything a search plan finds, indexed by document. `store_tools.py` wraps the tools
  so the searching agent sees a short digest of each page while the final answerer gets the full
  text. The older plans keep their own recording logic so their outputs stay byte-identical;
  `scripts/test_evidence.py` checks this.

## Free page downloads: the local Wikipedia server

`scripts/wiki_backend.py` loads the 1,052 cached gold pages into memory and mimics Serper's page
API at `http://127.0.0.1:5000/` (the `LOCAL_SCRAPE_URL` in `tools.py`). Point `fetch_url` at it
and every page download is free:

```bash
python3 scripts/wiki_backend.py &                                   # start the server on :5000
$PY scripts/eval_search_step.py --scrape-url http://127.0.0.1:5000/ # route fetches to it
```

Page downloads are the bulk of the Serper cost, so this saves most of the money. Two caveats:

- `search_info` (web search) still uses paid Serper. The local server serves page content, not
  search results.
- The cache holds only the gold pages. Fetching any other URL returns nothing (404 → empty). So
  during open-web search the agent can only successfully read gold pages. That changes behavior,
  not just cost. Fully free and faithful runs are the ones that fetch gold URLs directly, with no
  open-web discovery — that is how the debate and relabel steps ran.

## Current results

From `outputs/search_step_summary.json`:

| config | n | %gold found | %gold fetched | %all-found | **downstream** | ans-in-evid | acc\|evid | refusal |
|---|--|--|--|--|--|--|--|--|
| single_shot | 266 | 12.7% | 11.5% | 0.0% | 14.7% | 20.7% | 43.6% | 69.2% |
| decompose | 262 | 16.4% | 12.2% | 1.5% | 11.8% | 20.2% | 30.2% | 69.1% |
| iterative_snippets | 266 | 37.6% | — | 10.5% | **32.3%** | 20.3% | 53.7% | 30.5% |
| react_baseline | 258 | 35.9% | 5.7% | 10.5% | 29.5% | 24.4% | 47.6% | 48.8% |
| decompose_react | 248 | **43.9%** | 12.0% | **15.7%** | 29.4% | 26.6% | 54.5% | 57.3% |

Column meanings: "%gold found" — share of gold URLs the agent surfaced in search results.
"%gold fetched" — share it actually downloaded. "downstream" — final answer accuracy.
"ans-in-evid" — how often the answer text appears in the retrieved evidence. "acc|evid" —
accuracy on the questions where the answer was in the evidence.

Best downstream accuracy is 32% against a ~100% ceiling. Almost all of the gap is in retrieval.

## What we learned (start here)

1. **Agents find gold URLs but do not download them. This is the biggest fixable loss.**
   `react_baseline` surfaces 35.9% of gold URLs in its search results but downloads only 5.7%.
   The agent throws away about 84% of the gold URLs it already found (it averages 0.6–1.3
   downloads per question).
2. **Retrieval is the ceiling.** The answer appears in the retrieved evidence only 20–27% of the
   time. Fix that and downstream accuracy should follow, because these questions are answerable by
   construction.
3. **Search snippets alone go surprisingly far.** `iterative_snippets` never downloads a page
   (`fetch_k=0`), yet it has the best downstream accuracy (32.3%) and the fewest refusals (30.5%).
   The "—" in its fetched column is a side effect of never fetching, not a failure.
4. **Plain `decompose` is the worst plan.** Most downloads, lowest precision (5.7%), most noise.
   Splitting a question into sub-queries only helps when paired with selective, step-by-step
   fetching (`decompose_react`).
5. Secondary: even when the answer is in the evidence, accuracy is only 44–55%. That is a
   synthesis problem and belongs to the aggregation work, not this prong. Focus here on getting
   the answer into the evidence.

## What to try next

- Make agents download the gold-candidate URLs they surface, especially Wikipedia links. Raise
  `fetch_k`, or add a rule that always downloads the top surfaced results. Biggest expected win.
- Bias search toward Wikipedia and download through the local server. The gold documents are
  Wikipedia pages and already cached, so downloads are free and reliable while iterating.
- Improve query planning. `decompose_react` leads on retrieval; iterate on `DECOMPOSE_SYSTEM` and
  its budgets.
- Track `answer_in_evidence_rate` as the primary target, not just downstream accuracy.

## How to run

```bash
export PY=/nas-ssd2/tianyin4/cache/venvs/vllm-host/bin/python3   # only interpreter with deps
# needs: SERPER_API_KEY (search; page downloads free via the local server),
#        OPENAI_API_KEY (judge), vLLM serving Qwen3-14B on :7472
python3 scripts/wiki_backend.py &                                # optional: free page downloads
$PY scripts/eval_search_step.py --limit 10                       # debug first (search is paid)
$PY scripts/eval_search_step.py --configs decompose_react        # one config, full run
```

## Gotchas

- Live Serper search costs money. Always debug with `--limit` first.
- `n` differs per config (248–266) because some questions error out. Compare configs on the common
  subset of questions.
- Per-config caches: `outputs/search_*_cache.jsonl`. Summary: `outputs/search_step_summary.json`.
- Do not overwrite `datasets/frames_qwen_answerable.json`. It is the search input and matches the
  existing results.

## Status and caveats

- The numbers above are read from the existing `outputs/search_step_summary.json`. Nothing was
  re-run for this document.
- Finding #1 (found-but-not-fetched) is a hypothesis, not a proven fix. Validate it before
  building on it.

## Recommendation

Fix the found-but-not-fetched gap first. Run `decompose_react` with a rule that downloads every
surfaced gold-candidate URL (or simply the top few results), route downloads through the local
Wikipedia server, and check whether `answer_in_evidence_rate` climbs from ~27% toward the ceiling.
It is the cheapest change with the largest expected gain, and it directly tests the leading
diagnosis.
