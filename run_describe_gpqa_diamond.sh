#!/usr/bin/env bash
# Describe, embed and route GPQA-Diamond (198 questions, test only) into the SuperGPQA groups
# (outputs/describe_v3/clusters_600_train.json), so a SuperGPQA search's programs can run on it
# (run_eval_gpqa_cluster.sh). No new groups are made: GPQA has no train split, and its questions go
# to the SuperGPQA group whose medoid is nearest, by the rule the SuperGPQA test questions follow.
#
#     bash run_describe_gpqa_diamond.sh 2>&1 | tee -a outputs/describe_v3/run_gpqa_diamond.log
#
# Options: --gpu N        the card the embedding model runs on (~2 GB); left out: CPU (a few minutes)
#          --python PATH  the Python to use (default: the vllm-updated venv)
#          --workers N    describer calls in flight (default 8)
#
# The describer is the one that made the SuperGPQA groups (scripts/describe_supergpqa_v3.py, its
# prompt, anchors and vocabulary). Cost: about $3 of describer calls (OPENAI_API_KEY from .env).
# Embedding and routing are local. Resumable: every step skips work already done.
# Writes only these files in outputs/describe_v3/:
#     templates_gpqa_diamond_test.jsonl   the records of the 198 questions
#     vectors_gpqa_diamond_test.npz       their vectors
#     routes_gpqa_diamond_test.json       their routes (eval_routed_dev.py --routes)
set -euo pipefail
cd "$(dirname "$0")"

PYTHON=/nas-ssd2/tianyin4/cache/venvs/vllm-updated/bin/python
[[ -x "$PYTHON" ]] || PYTHON=python3
GPU=""
WORKERS=8
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu) GPU="$2"; shift 2 ;;
        --python) PYTHON="$2"; shift 2 ;;
        --workers) WORKERS="$2"; shift 2 ;;
        -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
        *) echo "unknown option $1 (see --help)" >&2; exit 1 ;;
    esac
done
if [[ -n "$GPU" ]] && ! CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" -c \
        "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "--gpu $GPU: no CUDA card $GPU is visible to $PYTHON" >&2; exit 1
fi
OUT=outputs/describe_v3
DESCRIBER=describe_supergpqa_v3
TEST=datasets/gpqa_diamond_test.json
CLUSTERS=$OUT/clusters_600_train.json          # repo-relative: eval_routed_dev.py compares it with the run's
RECORDS=$OUT/templates_gpqa_diamond_test.jsonl
VECTORS=$OUT/vectors_gpqa_diamond_test.npz
ROUTES=$OUT/routes_gpqa_diamond_test.json
stamp() { date "+%Y-%m-%d %H:%M:%S"; }

# --- 1. describe -----------------------------------------------------------------------------
missing() {  # prints how many questions of $TEST lack a record of this describer's prompt
    "$PYTHON" - "$TEST" "$RECORDS" "$DESCRIBER" <<'EOF'
import json, os, sys
sys.path.insert(0, "scripts")
d = __import__(sys.argv[3])
ids = {r["id"] for r in json.load(open(sys.argv[1]))}
done = set()
if os.path.exists(sys.argv[2]):
    for line in open(sys.argv[2]):
        if line.strip():
            r = json.loads(line)
            if r.get("prompt_version") == d.PROMPT_VERSION and r.get("model") == d.MODEL and "error" not in r:
                done.add(r["id"])
print(len(ids - done))
EOF
}
for attempt in 1 2 3; do
    left=$(missing)
    [[ "$left" == 0 ]] && break
    echo "[$(stamp)] step 1: describing $left questions (attempt $attempt)"
    "$PYTHON" scripts/$DESCRIBER.py --anchors "$OUT/anchors.json" --workers "$WORKERS" \
        --dataset "$TEST" --out "$RECORDS"
done
left=$(missing)
if [[ "$left" != 0 ]]; then
    echo "$left questions of $TEST still have no description; run this script again" >&2; exit 1
fi
echo "[$(stamp)] step 1: every question of $TEST has a record"

# --- 2. embed (redone only when the records are newer) -----------------------------------------
if [[ -f "$VECTORS" && "$VECTORS" -nt "$RECORDS" ]]; then
    echo "[$(stamp)] step 2: $VECTORS is up to date"
else
    echo "[$(stamp)] step 2: embedding (GPU '${GPU:-none, CPU}')"
    CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" scripts/embed_questions.py --describer "$DESCRIBER" \
        --templates "$RECORDS" --dataset "$TEST" --out "$VECTORS"
fi

# --- 3. route into the SuperGPQA train groups --------------------------------------------------
echo "[$(stamp)] step 3: routing"
"$PYTHON" scripts/route_questions.py --clusters "$CLUSTERS" \
    --train-vectors "$OUT/vectors_600_train.npz" --test-vectors "$VECTORS" --out "$ROUTES"
"$PYTHON" - "$ROUTES" "$CLUSTERS" "$TEST" <<'EOF'
import json, sys
from collections import Counter
data = json.load(open(sys.argv[1]))
data["clusters"] = sys.argv[2]                 # repo-relative, so the routes work in any clone
json.dump(data, open(sys.argv[1], "w"), indent=1)
n = len(json.load(open(sys.argv[3])))
assert len(data["routes"]) == n, f"{len(data['routes'])} routes, expected {n}"
print(f"step 3: {n} questions routed; per group {dict(sorted(Counter(r['group'] for r in data['routes'].values()).items()))}")
EOF
echo "[$(stamp)] done: $ROUTES"
