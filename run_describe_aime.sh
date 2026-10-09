#!/usr/bin/env bash
# Describe, embed, cluster and route AIME 2022-2025 (scripts/prepare_aime.py: 60 train, 60 test) with
# the MATH describer as it stands (scripts/describe_math_v1.py: its prompt, its step/challenge
# vocabulary discovered on MATH, and the MATH anchors, outputs/describe_math_v1/anchors.json), so
# AIME questions are described in the same terms as MATH ones. The groups are made on the 60 train
# questions and the 60 test questions are routed to them. Every output goes to outputs/describe_aime_v1/.
#
#     mkdir -p outputs/describe_aime_v1
#     bash run_describe_aime.sh 2>&1 | tee -a outputs/describe_aime_v1/run.log
#
# Options: --gpu N         the card the embedding model runs on (~2 GB); left out: CPU (a few minutes)
#          --python PATH   the Python to use (default: the vllm-updated venv)
#          --workers N     describer calls in flight (default 8)
#          --min-size N    smallest group allowed when the number of groups is chosen (default 20:
#                          2 groups, 26 and 34 train questions; with 5, 6 groups of 5-18, too small for
#                          a search to tell programs apart). Clustering and routing cost seconds and
#                          no API call, so rerunning with another --min-size only redoes them
#
# Cost: describing the 120 questions, about $2 (OPENAI_API_KEY from .env). Embedding, clustering and
# routing are local. Resumable: after an interruption or API errors, run it again.
set -euo pipefail
cd "$(dirname "$0")"

PYTHON=/nas-ssd2/tianyin4/cache/venvs/vllm-updated/bin/python
[[ -x "$PYTHON" ]] || PYTHON=python3
GPU=""
WORKERS=8
MIN_SIZE=20
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu) GPU="$2"; shift 2 ;;
        --python) PYTHON="$2"; shift 2 ;;
        --workers) WORKERS="$2"; shift 2 ;;
        --min-size) MIN_SIZE="$2"; shift 2 ;;
        -h|--help) sed -n '2,21p' "$0"; exit 0 ;;
        *) echo "unknown option $1 (see --help)" >&2; exit 1 ;;
    esac
done
if [[ -n "$GPU" ]] && ! CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" -c \
        "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "--gpu $GPU: no CUDA card $GPU is visible to $PYTHON" >&2; exit 1
fi
VERSION=math-v1
DESCRIBER=describe_math_v1
OUT=outputs/describe_aime_v1
ANCHORS=outputs/describe_math_v1/anchors.json   # the MATH anchors: the same few-shot examples as MATH
TRAIN=datasets/aime_2022_2025_train.json       # the groups are made on these
TEST=datasets/aime_2022_2025_test.json         # and these are routed to them
mkdir -p "$OUT"
stamp() { date "+%Y-%m-%d %H:%M:%S"; }

for f in "$TRAIN" "$TEST" "$ANCHORS"; do
    [[ -f "$f" ]] || { echo "$f is missing (python scripts/prepare_aime.py writes the AIME files)" >&2; exit 1; }
done
if ! got=$("$PYTHON" -c "import sys; sys.path.insert(0, 'scripts'); import $DESCRIBER as d; print(d.PROMPT_VERSION)"); then
    echo "cannot import scripts/$DESCRIBER.py with $PYTHON (pass --python)" >&2; exit 1
fi
if [[ "$got" != "$VERSION" ]]; then
    echo "scripts/$DESCRIBER.py is at prompt '$got', not $VERSION" >&2; exit 1
fi

# --- 1. describe (a failed call is an error line; a rerun redoes the gaps; a gap left is fatal) ---
missing() {  # dataset records -> prints how many questions lack a record
    "$PYTHON" - "$1" "$2" "$DESCRIBER" <<'EOF'
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
describe() {  # dataset split
    local data=$1 records=$OUT/templates_$2.jsonl
    for attempt in 1 2 3; do
        [[ "$(missing "$data" "$records")" == 0 ]] && return 0
        echo "[$(stamp)] step 1: describe $2 (attempt $attempt)"
        "$PYTHON" scripts/$DESCRIBER.py --anchors "$ANCHORS" --workers "$WORKERS" --dataset "$data" --out "$records"
    done
    [[ "$(missing "$data" "$records")" == 0 ]] && return 0
    echo "$(missing "$data" "$records") questions of $data still have no description; run again" >&2; exit 1
}
describe "$TRAIN" train
describe "$TEST" test
echo "[$(stamp)] step 1: every question has a description"

# --- 2. embed (redone only when the records are newer) -------------------------------------------
for split in train test; do
    vectors=$OUT/vectors_$split.npz
    if [[ $split == train ]]; then data=$TRAIN; else data=$TEST; fi
    if [[ -f "$vectors" && "$vectors" -nt "$OUT/templates_$split.jsonl" ]]; then
        echo "[$(stamp)] step 2: $vectors is up to date"; continue
    fi
    echo "[$(stamp)] step 2: embedding $split (GPU '${GPU:-none, CPU}')"
    CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" scripts/embed_questions.py --describer "$DESCRIBER" \
        --templates "$OUT/templates_$split.jsonl" --dataset "$data" --out "$vectors"
done

# --- 3. cluster the train questions ---------------------------------------------------------------
echo "[$(stamp)] step 3: clustering the train questions (groups of at least $MIN_SIZE)"
"$PYTHON" scripts/cluster_questions.py --rep description --vectors "$OUT/vectors_train.npz" \
    --templates "$OUT/templates_train.jsonl" --dataset "$TRAIN" \
    --out "$OUT/clusters_train.json" --min-size "$MIN_SIZE"

# --- 4. route the test questions to the train groups ---------------------------------------------
echo "[$(stamp)] step 4: routing the test questions"
"$PYTHON" scripts/route_questions.py --clusters "$OUT/clusters_train.json" \
    --train-vectors "$OUT/vectors_train.npz" --test-vectors "$OUT/vectors_test.npz" \
    --out "$OUT/routes_test.json"
"$PYTHON" - "$OUT/routes_test.json" "$OUT/clusters_train.json" <<'EOF'
import json, sys
data = json.load(open(sys.argv[1]))
data["clusters"] = sys.argv[2]                 # repo-relative, so the routes work in any clone
json.dump(data, open(sys.argv[1], "w"), indent=1)
EOF

# --- tripwires: descriptions must not discuss whether a question is well-posed, and must be in English
"$PYTHON" - "$OUT" <<'EOF'
import json, re, sys
flaw = re.compile(r"ambigu|underspecif|well-specified|unique (answer|choice|selection)|"
                  r"supports? a unique|inconsistent|more than one (option|choice|answer) (is|could)", re.I)
script = re.compile(r"[Ѐ-ӿ֐-ۿऀ-ॿ぀-ヿ一-鿿가-힯]")
for split in ("train", "test"):
    recs = [json.loads(l) for l in open(f"{sys.argv[1]}/templates_{split}.jsonl") if l.strip()]
    recs = [r for r in recs if "error" not in r]
    f = sum(bool(flaw.search(r["template"])) for r in recs)
    s = sum(bool(script.search(r["template"])) for r in recs)
    print(f"tripwire {split}: {f} of {len(recs)} templates discuss ambiguity or uniqueness"
          + ("  <-- check these before using the groups" if f > 0.02 * len(recs) else ""))
    print(f"tripwire {split}: {s} of {len(recs)} templates are not in English"
          + ("  <-- check these before using the groups" if s else ""))
EOF
echo "[$(stamp)] done: $OUT"
