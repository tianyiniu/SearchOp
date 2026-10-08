#!/usr/bin/env bash
# Describe, embed and route the SuperGPQA 2k TEST split (1,000 questions) into the existing groups
# of the 600 subset's train split (outputs/describe_v3/clusters_600_train.json). No new groups are
# made: the groups, their medoids and the train vectors are the ones the 600 runs searched on.
#
#     bash run_describe_supergpqa_2k_test.sh --gpu 3 2>&1 | tee -a outputs/describe_v3/run_2k_test.log
#
# Options: --gpu N        the card the embedding model runs on (~2 GB); left out: CPU (about 15 min)
#          --python PATH  the Python to use (default: the vllm-updated venv)
#          --workers N    describer calls in flight (default 8)
#
# The 600 test split is inside the 2k test split, and its 300 questions already have records
# (templates_test.jsonl). Those are copied into the new record file first, so only the other 700
# are sent to the describer (scripts/describe_supergpqa_v3.py, the prompt and vocabulary that made
# the groups). Cost: about $5-12 of describer calls (OPENAI_API_KEY from .env). Embedding and
# routing are local.
#
# Resumable: every step skips work already done, so after an interruption or API errors, run it
# again. Writes only these files in outputs/describe_v3/:
#     templates_2k_test.jsonl   the records of all 1,000 questions
#     vectors_2k_test.npz       their vectors
#     routes_2k_test.json       their routes (eval_routed_dev.py --routes); the 300 questions of
#                               the 600 test split keep their routes_600_test.json route
set -euo pipefail
cd "$(dirname "$0")"

PYTHON=/nas-ssd2/tianyin4/cache/venvs/vllm-updated/bin/python
GPU=""
WORKERS=8
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu) GPU="$2"; shift 2 ;;
        --python) PYTHON="$2"; shift 2 ;;
        --workers) WORKERS="$2"; shift 2 ;;
        -h|--help) sed -n '2,25p' "$0"; exit 0 ;;
        *) echo "unknown option $1 (see --help)" >&2; exit 1 ;;
    esac
done
if [[ -n "$GPU" ]] && ! CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" -c \
        "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "--gpu $GPU: no CUDA card $GPU is visible to $PYTHON" >&2; exit 1
fi
OUT=outputs/describe_v3
DESCRIBER=describe_supergpqa_v3
TEST=datasets/supergpqa_2k_test.json
RECORDS=$OUT/templates_2k_test.jsonl
VECTORS=$OUT/vectors_2k_test.npz
ROUTES=$OUT/routes_2k_test.json
stamp() { date "+%Y-%m-%d %H:%M:%S"; }

# --- 0. the 300 records already made for the 600 test split ----------------------------
if [[ ! -f "$RECORDS" ]]; then
    cp "$OUT/templates_test.jsonl" "$RECORDS"
    echo "[$(stamp)] step 0: started $RECORDS from $OUT/templates_test.jsonl ($(wc -l < "$RECORDS") records)"
fi

# --- 1. describe the rest -----------------------------------------------------------------
missing() {  # prints how many questions of $TEST lack a record of this describer's prompt
    "$PYTHON" - "$TEST" "$RECORDS" "$DESCRIBER" <<'EOF'
import json, sys
sys.path.insert(0, "scripts")
d = __import__(sys.argv[3])
ids = {r["id"] for r in json.load(open(sys.argv[1]))}
done = set()
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

# --- 2. embed (redone only when the records are newer) ------------------------------------
if [[ -f "$VECTORS" && "$VECTORS" -nt "$RECORDS" ]]; then
    echo "[$(stamp)] step 2: $VECTORS is up to date"
else
    echo "[$(stamp)] step 2: embedding (GPU '${GPU:-none, CPU}')"
    CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" scripts/embed_questions.py --describer "$DESCRIBER" \
        --templates "$RECORDS" --dataset "$TEST" --out "$VECTORS"
fi

# --- 3. route into the 600 train groups -----------------------------------------------------
echo "[$(stamp)] step 3: routing"
"$PYTHON" scripts/route_questions.py --clusters "$OUT/clusters_600_train.json" \
    --train-vectors "$OUT/vectors_600_train.npz" --test-vectors "$VECTORS" --out "$ROUTES"

# --- 4. the 300 questions routed before keep their stored route -------------------------------
# Embedding on other hardware (CPU, another GPU) moves the vectors very slightly (cosine 0.9999 on
# CPU), which can flip a question that sits on a group boundary (2 of the 300 on CPU). Those 300 keep
# the route the 600 runs were evaluated with, so the two test sets agree on every shared question.
"$PYTHON" - "$OUT/routes_600_test.json" "$ROUTES" <<'EOF'
import json, sys
old = json.load(open(sys.argv[1]))["routes"]
data = json.load(open(sys.argv[2]))
new = data["routes"]
assert len(new) == 1000, f"{len(new)} routes, expected 1000"
assert set(old) <= set(new), "a question of the 600 test split is missing"
flips = [q for q in old if new[q]["group"] != old[q]["group"]]
for q in flips:
    print(f"  {q}: routed to group {new[q]['group']} here, {old[q]['group']} before "
          f"(margin {old[q]['margin']}); kept {old[q]['group']}")
new.update(old)
data["kept_from"] = {"file": sys.argv[1], "questions": len(old), "group_changes_undone": len(flips)}
json.dump(data, open(sys.argv[2], "w"), indent=1)
print(f"step 4: the {len(old)} questions of the 600 test split keep their stored routes "
      f"({len(old) - len(flips)} had come out the same)")
EOF
echo "[$(stamp)] done: $ROUTES"
