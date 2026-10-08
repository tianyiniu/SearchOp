#!/usr/bin/env bash
# Describe, embed, cluster and route the 600-question SuperGPQA subset with the
# describer's prompt v3 (scripts/describe_supergpqa_v3.py). Every output goes to
# outputs/describe_v3/; nothing outside it is written. ("v3" here is the
# describer's prompt version, unrelated to run_pipeline_cluster.sh.)
#
#     mkdir -p outputs/describe_v3 && bash run_describe_v3.sh 2>&1 | tee -a outputs/describe_v3/run.log
#
# Resumable: every step skips work already done, so after an interruption or
# API errors, run it again.
#
# Anchors are drafted by the model at step 1 and used as drafted. To correct
# them by hand first: STOP_AFTER_ANCHORS=1 bash run_describe_v3.sh, edit
# outputs/describe_v3/anchors.json, then run again without it.
#
# Cost: ~$4 for describing the 600 questions if prompt caching works, ~$10 if
# not. Embedding (GPU $GPU, ~2 GB), clustering and routing are local.
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python}"
GPU="${GPU:-0}"
WORKERS="${WORKERS:-8}"
VERSION=v3
OUT="${OUT:-outputs/describe_$VERSION}"
ANCHORS=$OUT/anchors.json
mkdir -p "$OUT"
stamp() { date "+%Y-%m-%d %H:%M:%S"; }

# the describer must be at the prompt version this directory is named for
if ! got=$("$PYTHON" -c "import sys; sys.path.insert(0, 'scripts'); import describe_supergpqa_v3 as d; print(d.PROMPT_VERSION)"); then
    echo "cannot import scripts/describe_supergpqa_v3.py with $PYTHON (set PYTHON=...)" >&2; exit 1
fi
if [[ "$got" != "$VERSION" ]]; then
    echo "scripts/describe_supergpqa_v3.py is at prompt '$got', not $VERSION" >&2; exit 1
fi

# Every question of a dataset needs a record of this prompt version. The
# describer writes a failed call as an error line and moves on, so a gap is
# rerun (a rerun only redoes the gaps), and still a gap after that is fatal.
missing() {  # dataset records -> prints how many questions lack a record
    "$PYTHON" - "$1" "$2" <<'EOF'
import json, sys
sys.path.insert(0, "scripts")
import describe_supergpqa_v3 as d
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

describe() {  # dataset-name split
    local data=datasets/supergpqa_$1.json records=$OUT/templates_$2.jsonl
    for attempt in 1 2 3; do
        echo "[$(stamp)] describe $1 (attempt $attempt)"
        "$PYTHON" scripts/describe_supergpqa_v3.py --anchors "$ANCHORS" --workers "$WORKERS" \
            --dataset "$data" --out "$records"
        if [[ "$(missing "$data" "$records")" == 0 ]]; then return 0; fi
    done
    echo "$(missing "$data" "$records") questions of $data still have no description" >&2; exit 1
}

# --- 1. anchors: drafted from the 600 train split --------------------------------
if [[ -f "$ANCHORS" ]]; then
    echo "[$(stamp)] step 1: $ANCHORS exists, using it"
else
    echo "[$(stamp)] step 1: drafting anchors"
    "$PYTHON" scripts/describe_supergpqa_v3.py --draft-anchors 12 --anchors "$ANCHORS" \
        --dataset datasets/supergpqa_600_train.json
    # a draft call that fails is skipped, not retried: never go on with a short set
    n=$("$PYTHON" -c "import json, sys; print(len(json.load(open(sys.argv[1]))))" "$ANCHORS")
    if [[ "$n" != 12 ]]; then
        mv "$ANCHORS" "$ANCHORS.short"
        echo "only $n of 12 anchors were drafted (kept as $ANCHORS.short); run again" >&2; exit 1
    fi
    if [[ "${STOP_AFTER_ANCHORS:-0}" == 1 ]]; then
        echo "[$(stamp)] stopped after drafting: correct $ANCHORS by hand, then run again"; exit 0
    fi
fi

# --- 2. describe the 600 subset (300 train, 300 test) ------------------------------
describe 600_train train
describe 600_test test

# --- 3. embed (local; redone only when the records are newer) ----------------------
for size in 600; do for split in train test; do
    vectors=$OUT/vectors_${size}_${split}.npz
    if [[ -f "$vectors" && "$vectors" -nt "$OUT/templates_$split.jsonl" ]]; then
        echo "[$(stamp)] step 3: $vectors is up to date"; continue
    fi
    echo "[$(stamp)] step 3: embedding $size $split"
    CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" scripts/embed_questions.py --describer describe_supergpqa_v3 \
        --templates "$OUT/templates_$split.jsonl" --dataset "datasets/supergpqa_${size}_${split}.json" \
        --out "$vectors"
done; done

# --- 4. cluster the train split (300 questions, so groups of 30 or more) ---------------
echo "[$(stamp)] step 4: clustering"
"$PYTHON" scripts/cluster_questions.py --rep description --vectors "$OUT/vectors_600_train.npz" \
    --templates "$OUT/templates_train.jsonl" --dataset datasets/supergpqa_600_train.json \
    --out "$OUT/clusters_600_train.json" --min-size 30

# --- 5. route the test split to the train groups -----------------------------------
echo "[$(stamp)] step 5: routing"
for size in 600; do
    "$PYTHON" scripts/route_questions.py --clusters "$OUT/clusters_${size}_train.json" \
        --train-vectors "$OUT/vectors_${size}_train.npz" --test-vectors "$OUT/vectors_${size}_test.npz" \
        --out "$OUT/routes_${size}_test.json"
done

# --- tripwire: templates should never discuss whether a question is well-posed ------
"$PYTHON" - "$OUT" <<'EOF'
import json, re, sys
pat = re.compile(r"ambigu|underspecif|well-specified|unique (answer|choice|selection)|"
                 r"supports? a unique|inconsistent|more than one (option|choice) (is|could)", re.I)
for split in ("train", "test"):
    recs = [json.loads(l) for l in open(f"{sys.argv[1]}/templates_{split}.jsonl") if l.strip()]
    hits = [r["id"] for r in recs if "error" not in r and pat.search(r.get("template", ""))]
    print(f"tripwire {split}: {len(hits)} of {len(recs)} templates discuss ambiguity or uniqueness"
          + ("  <-- check these before using the groups" if len(hits) > 0.02 * len(recs) else ""))
EOF
echo "[$(stamp)] done: $OUT"
