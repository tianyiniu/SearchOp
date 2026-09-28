#!/usr/bin/env bash
# End-to-end pipeline: data collection -> search-side -> debate-side.
#
#   ./run_pipeline.sh              # everything
#   ./run_pipeline.sh data         # just the splits
#   ./run_pipeline.sh search       # just the search-side eval
#   ./run_pipeline.sh debate       # just the debate-side eval
#   LIMIT=5 ./run_pipeline.sh      # smoke test (outputs get a _limitN suffix)
#
# Prereqs, both must already be running:
#   vLLM serving Qwen/Qwen3-14B on :7472
#   python3 scripts/wiki_backend.py &          on :5000
# API keys are read from .env by each script (OPENAI_API_KEY, SERPER_API_KEY).

set -euo pipefail
cd "$(dirname "$0")"

PY=${PY:-/nas-ssd2/tianyin4/cache/venvs/vllm-host/bin/python3}
STAGE=${1:-all}
LIMIT=${LIMIT:-}
LIM_ARG=""
[ -n "$LIMIT" ] && LIM_ARG="--limit $LIMIT"

hr() { printf '\n=== %s ===\n' "$1"; }

# --- preflight ------------------------------------------------------------
# A dead service does not fail loudly downstream: fetch_url swallows the
# connection error and silently drops the document, so questions get thin
# contexts and land in the wrong split. Check before spending any API budget.
hr "preflight"
[ -x "$PY" ] || { echo "no interpreter at $PY (set PY=...)"; exit 1; }
"$PY" - <<'EOF'
import socket, sys
bad = False
for name, port in [("vLLM", 7472), ("wiki_backend", 5000)]:
    s = socket.socket(); s.settimeout(2)
    up = s.connect_ex(("127.0.0.1", port)) == 0
    s.close()
    print(f"  {name}:{port} {'ok' if up else 'DOWN'}")
    bad |= not up
if bad:
    sys.exit("services down — start vLLM and scripts/wiki_backend.py first")
EOF
[ -f .env ] || echo "  warning: no .env — OPENAI_API_KEY / SERPER_API_KEY must be exported"
echo "  interpreter: $PY"
[ -n "$LIMIT" ] && echo "  LIMIT=$LIMIT (smoke test; outputs suffixed _limit$LIMIT)"

# --- 1. data collection ---------------------------------------------------
# Step 1 (get_guaranteed_answerable.py) is NOT re-run: it is unchanged, costs
# GPT-5.4 calls over all 824 FRAMES questions, and its output is already on disk.
# Delete datasets/frames_guaranteed_answerable.json to force a rebuild.
if [ "$STAGE" = all ] || [ "$STAGE" = data ]; then
  hr "1a. guaranteed_answerable (GPT-5.4: fails alone, succeeds with docs)"
  if [ -f datasets/frames_guaranteed_answerable.json ]; then
    echo "  exists, skipping — delete it to rebuild"
  else
    "$PY" scripts/get_guaranteed_answerable.py $LIM_ARG
    "$PY" scripts/cache_web_links.py      # download gold pages for the new split
  fi

  hr "1b. qwen_answerable / unanswerable / borderline (k=3, GPT compression + verify)"
  "$PY" scripts/get_agent_answerable.py $LIM_ARG
fi

# --- 2. search-side -------------------------------------------------------
# Runs on qwen_answerable: these questions are answerable IF search finds the
# right documents, so search quality is measurable. Uses LIVE Serper — costs money.
if [ "$STAGE" = all ] || [ "$STAGE" = search ]; then
  hr "2. search-step eval (retrieval overlap + downstream answer)"
  echo "  note: hits live Serper, this one costs money"
  "$PY" scripts/eval_search_step.py $LIM_ARG
fi

# --- 3. debate-side -------------------------------------------------------
# Runs on qwen_unanswerable: retrieval is solved by construction (gold docs are
# handed over), so any gain is pure aggregation.
if [ "$STAGE" = all ] || [ "$STAGE" = debate ]; then
  hr "3a. debate-step eval (fixed shape library)"
  "$PY" scripts/eval_debate_step.py $LIM_ARG

  hr "3b. evolve per-question debate schemas"
  "$PY" scripts/evolve_debate.py $LIM_ARG

  hr "3c. test-time routing (reads outputs/debate_step_cache.jsonl from 3a)"
  "$PY" scripts/route_debate.py
fi

hr "done"
echo "  splits    -> datasets/frames_qwen_{answerable,unanswerable,borderline}.json"
echo "  search    -> outputs/search_step_summary.json"
echo "  debate    -> outputs/debate_step_summary.json"
echo "  evolve    -> outputs/evolve_debate_summary.json"
echo "  routing   -> outputs/route_debate_summary.json"
