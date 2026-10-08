#!/usr/bin/env bash
# Describe, embed, cluster and route the HLE text splits (800 train, 200 test)
# with the describer's prompt v4 and a step/challenge vocabulary discovered on
# the HLE train split. Every output goes to outputs/describe_hle_v4/, except the
# discovery output (outputs/vocab_discovery_hle_text_train_800.*) and the
# converted datasets (datasets/hle_text_*.json).
#
#     mkdir -p outputs/describe_hle_v4 && bash run_describe_hle.sh 2>&1 | tee -a outputs/describe_hle_v4/run.log
#
# Run it twice. The first run converts the data, runs vocabulary discovery and
# stops: paste the printed STEPS and CHALLENGES over the ones in
# scripts/describe_hle_v4.py (leave PROMPT_VERSION = "v4"). The second run
# checks the paste and does the rest. Resumable: after an interruption or API
# errors, run it again.
#
# Anchors are drafted by the model and used as drafted. To correct them by hand
# first: STOP_AFTER_ANCHORS=1 bash run_describe_hle.sh, edit
# outputs/describe_hle_v4/anchors.json, then run again without it.
#
# Cost: discovery ~$2-3; describing the 1,000 questions ~$9 if prompt caching
# works, ~$21 at list price. Embedding (GPU $GPU, ~2 GB), clustering and routing
# are local.
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python}"
GPU="${GPU:-0}"
WORKERS="${WORKERS:-8}"
VERSION=v4
OUT="${OUT:-outputs/describe_hle_$VERSION}"
ANCHORS=$OUT/anchors.json
TRAIN=datasets/hle_text_train_800.json
TEST=datasets/hle_text_test_200.json
VOCAB="${VOCAB:-outputs/vocab_discovery_hle_text_train_800.json}"   # written by the discovery step
MIN_SIZE="${MIN_SIZE:-80}"          # smallest group allowed: 10% of the train split, as in the SuperGPQA runs
mkdir -p "$OUT"
stamp() { date "+%Y-%m-%d %H:%M:%S"; }

if ! got=$("$PYTHON" -c "import sys; sys.path.insert(0, 'scripts'); import describe_hle_v4 as d; print(d.PROMPT_VERSION)"); then
    echo "cannot import scripts/describe_hle_v4.py with $PYTHON (set PYTHON=...)" >&2; exit 1
fi
if [[ "$got" != "$VERSION" ]]; then
    echo "scripts/describe_hle_v4.py is at prompt '$got', not $VERSION" >&2; exit 1
fi

# --- 0a. convert the HLE jsonl files to the pipeline's layout ---------------------
if [[ ! -f "$TRAIN" || ! -f "$TEST" ]]; then
    echo "[$(stamp)] step 0: converting the HLE files"
    "$PYTHON" scripts/prepare_hle.py
fi

# --- 0b. vocabulary: discover on the train split, then stop for the paste ----------
# The scan fails on any step or challenge about defects in the question itself:
# the questions and keys are taken as correct.
flaw_scan() {  # file-with-vocabulary-json-or-"code"
    "$PYTHON" - "$1" <<'EOF'
import json, re, sys
sys.path.insert(0, "scripts")
import describe_hle_v4 as d
pat = re.compile(r"ambigu|underspecif|well-specified|well-posed|sufficien|inconsisten|defect|flaw|"
                 r"damaged|garbled|contradict|unique answer|no correct|missing information", re.I)
if sys.argv[1] == "code":
    items = {**d.STEPS, **d.CHALLENGES}
else:
    v = json.load(open(sys.argv[1]))
    items = {i["key"]: i["definition"] for kind in ("steps", "challenges") for i in v[kind]}
hits = {k: m.group(0) for k, t in items.items() if (m := pat.search(k + " " + t))}
for k, w in hits.items():
    print(f"  question-defect wording in {k!r}: {w!r}")
sys.exit(1 if hits else 0)
EOF
}

if [[ ! -f "$VOCAB" ]]; then
    echo "[$(stamp)] step 0: discovering the vocabulary on $TRAIN"
    "$PYTHON" scripts/describe_hle_v4.py --discover --dataset "$TRAIN" --out "$VOCAB" \
        --n-batches 20 --batch-size 40 --workers "$WORKERS"
    if ! flaw_scan "$VOCAB"; then
        echo "the discovered vocabulary above has items about question defects: drop or reword them when pasting" >&2
    fi
    echo "[$(stamp)] stopped: paste the STEPS and CHALLENGES printed above over the ones in"
    echo "  scripts/describe_hle_v4.py (keep PROMPT_VERSION = \"$VERSION\"), then run this script again"
    exit 0
fi
# the lists in the code must be the discovered ones (definitions may be edited, keys not)
if ! "$PYTHON" - "$VOCAB" <<'EOF'
import json, sys
sys.path.insert(0, "scripts")
import describe_hle_v4 as d
v = json.load(open(sys.argv[1]))
ok = True
for kind, code in (("steps", d.STEPS), ("challenges", d.CHALLENGES)):
    found = {i["key"] for i in v[kind]}
    if set(code) != found:
        print(f"{kind} in describe_hle_v4.py do not match {sys.argv[1]}: "
              f"missing {sorted(found - set(code))}, extra {sorted(set(code) - found)}", file=sys.stderr)
        ok = False
sys.exit(0 if ok else 1)
EOF
then
    echo "paste the discovered STEPS and CHALLENGES into scripts/describe_hle_v4.py first" >&2; exit 1
fi
if ! flaw_scan code; then
    echo "scripts/describe_hle_v4.py has steps or challenges about question defects; remove them" >&2; exit 1
fi

# Every question needs a record of this prompt version. The describer writes a
# failed call as an error line and moves on, so a gap is rerun (a rerun only
# redoes the gaps), and still a gap after that is fatal.
missing() {  # dataset records -> prints how many questions lack a record
    "$PYTHON" - "$1" "$2" <<'EOF'
import json, sys
sys.path.insert(0, "scripts")
import describe_hle_v4 as d
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

describe() {  # dataset split
    local data=$1 records=$OUT/templates_$2.jsonl
    for attempt in 1 2 3; do
        echo "[$(stamp)] describe $2 (attempt $attempt)"
        "$PYTHON" scripts/describe_hle_v4.py --anchors "$ANCHORS" --workers "$WORKERS" \
            --dataset "$data" --out "$records"
        if [[ "$(missing "$data" "$records")" == 0 ]]; then return 0; fi
    done
    echo "$(missing "$data" "$records") questions of $data still have no description" >&2; exit 1
}

# --- 1. anchors: drafted from the train split -------------------------------------
if [[ -f "$ANCHORS" ]]; then
    echo "[$(stamp)] step 1: $ANCHORS exists, using it"
else
    echo "[$(stamp)] step 1: drafting anchors"
    "$PYTHON" scripts/describe_hle_v4.py --draft-anchors 12 --anchors "$ANCHORS" --dataset "$TRAIN"
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

# --- 2. describe ----------------------------------------------------------------------
describe "$TRAIN" train
describe "$TEST" test

# --- 3. embed (local; redone only when the records are newer) ------------------------
for split in train test; do
    vectors=$OUT/vectors_$split.npz
    if [[ $split == train ]]; then data=$TRAIN; else data=$TEST; fi
    if [[ -f "$vectors" && "$vectors" -nt "$OUT/templates_$split.jsonl" ]]; then
        echo "[$(stamp)] step 3: $vectors is up to date"; continue
    fi
    echo "[$(stamp)] step 3: embedding $split"
    CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" scripts/embed_questions.py --describer describe_hle_v4 \
        --templates "$OUT/templates_$split.jsonl" --dataset "$data" --out "$vectors"
done

# --- 4. cluster the train split ------------------------------------------------------
echo "[$(stamp)] step 4: clustering"
"$PYTHON" scripts/cluster_questions.py --rep description --vectors "$OUT/vectors_train.npz" \
    --templates "$OUT/templates_train.jsonl" --dataset "$TRAIN" \
    --out "$OUT/clusters_train.json" --min-size "$MIN_SIZE"

# --- 5. route the test split to the train groups --------------------------------------
echo "[$(stamp)] step 5: routing"
"$PYTHON" scripts/route_questions.py --clusters "$OUT/clusters_train.json" \
    --train-vectors "$OUT/vectors_train.npz" --test-vectors "$OUT/vectors_test.npz" \
    --out "$OUT/routes_test.json"

# --- tripwires: templates must not discuss whether a question is well-posed, and
#     must be in English ------------------------------------------------------------
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
