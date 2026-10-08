# Shared setup of the cluster pipeline (sourced, never run on its own): the model family and dataset
# from the arguments, every path and option, the server check, and steps 1 and 2 (the external
# baselines on the test split and on the search questions, unless EXTERNAL_BASELINES=0 (HLE and MATH:
# then only on the comparison's questions), and the dev split). Both
# run_compare_external_cluster.sh and run_pipeline_cluster.sh source it, so they always share the settings.
# The first argument is the model family (gptoss, qwen, qwen9b or qwen4b), the second the dataset
# (supergpqa, hle or math).

FAMILY="${1:?gptoss, qwen, qwen9b or qwen4b}"
DATASET="${2:-supergpqa}"
RUNNAME="run3"                                       # the SuperGPQA run directory
SUPERGPQA_OPTS=(--high-cost 3 --turn-cap 20 --judge-persona --any-round-width)   # run3's search options
N_DEV_ASKED="${N_DEV:-}"                             # N_DEV=... from the environment, if given
N_DEV="${N_DEV_ASKED:-0}"                            # train questions held out as a dev split (0: none)
TIE=()                                               # the search's and champion step's tie rule (none: exact ties)
REUSE_SEEDS_FROM=""                                  # a run directory whose seeds.json this run starts from
REUSE_TRAIN_ALL=0                                    # 1: the train baselines start from the whole-train files
MID_TEST=1                                           # 0: no look at the test split during the search (step 7)
COMPARE_EXTERNAL=0                                   # 1: run_compare_external_cluster.sh, then the search needs its pass
EXTERNAL_BASELINES=1                                 # 0: none on the test split or the search questions (steps 5 and 9 show none)
COMPARE_PER_GROUP=""                                 # the comparison's questions: the first N search questions per group (empty: all)
COMPARE_RUNS=""                                      # the comparison's external baselines: runs per question (empty: K)
SR_ARGS=(--max-tokens 24576 --feedback-max-tokens 16384 --recover)   # the external Self-Refine's limits and recovery
PROTOCOLS="direct,self_refine,mad,direct_high,self_refine_high"   # in-executor rows of the test table
case "$FAMILY" in
    gptoss)
        MODEL="openai/gpt-oss-20b"; BASE_URL="${BASE_URL:-http://localhost:7472/v1}"
        FTAG="gptoss"; WORKERS="${WORKERS:-128}"
        EXECUTOR=(--visible-reasoning)               # gpt-oss thinks in a hidden channel
        BTAG="gptoss20b_high"; BASELINE_ARGS=(--reasoning-effort high)
        if [[ "$DATASET" == supergpqa ]]; then        # SuperGPQA run4 (2026-10-07): qwen9b run2's design and settings
            # and the same prompts: no --visible-reasoning (every speaker is asked to write out its reasoning
            # since 2026-10-06, and a summary of a reply whose reasoning is hidden is made from that reasoning)
            EXECUTOR=()
            RUNNAME="run4"
            SUPERGPQA_OPTS=(--plain-instruction --last-round-vote --count-read-summaries
                            --high-cost 3 --turn-cap 15 --total-cap 21)
            N_DEV="${N_DEV_ASKED:-100}"; TIE=(--tie-questions 1); REUSE_TRAIN_ALL=1; MID_TEST=0; COMPARE_EXTERNAL=1
            REUSE_SEEDS_FROM="../pipeline_cluster_qwen9b/run2"   # the same seeds as qwen9b run2
            PROTOCOLS="direct_high,self_refine_high"
        fi ;;
    qwen)
        MODEL="Qwen/Qwen3.5-35B-A3B-FP8"; BASE_URL="${BASE_URL:-http://localhost:7473/v1}"
        FTAG="qwen35b"; WORKERS="${WORKERS:-64}"
        EXECUTOR=()
        BTAG="qwen35_35b_think"; BASELINE_ARGS=(--family qwen --temperature 1.0 --top-p 0.95) ;;
    qwen9b)                                          # see "qwen9b" above: run2 (run1's settings are listed there)
        MODEL="Qwen/Qwen3.5-9B"; BASE_URL="${BASE_URL:-http://localhost:7472/v1}"
        FTAG="qwen9b"; WORKERS="${WORKERS:-128}"
        EXECUTOR=(--plain-instruction --last-round-vote --count-read-summaries)
        BTAG="qwen35_9b_think"; BASELINE_ARGS=(--family qwen --temperature 1.0 --top-p 0.95)
        RUNNAME="run2"; SUPERGPQA_OPTS=(--high-cost 3 --turn-cap 15 --total-cap 21); N_DEV="${N_DEV_ASKED:-100}"
        TIE=(--tie-questions 1); REUSE_SEEDS_FROM=""; REUSE_TRAIN_ALL=1; MID_TEST=0; COMPARE_EXTERNAL=1
        PROTOCOLS="direct_high,self_refine_high" ;;   # the in-executor matches of the external Direct CoT and Self-Refine
    qwen4b)                                          # Qwen3.5-4B (2026-10-07): qwen9b run2's design, settings and prompts
        MODEL="Qwen/Qwen3.5-4B"; BASE_URL="${BASE_URL:-http://localhost:7473/v1}"
        FTAG="qwen4b"; WORKERS="${WORKERS:-128}"
        EXECUTOR=(--plain-instruction --last-round-vote --count-read-summaries)
        BTAG="qwen35_4b_think"; BASELINE_ARGS=(--family qwen --temperature 1.0 --top-p 0.95)
        RUNNAME="run1"; SUPERGPQA_OPTS=(--high-cost 3 --turn-cap 15 --total-cap 21); N_DEV="${N_DEV_ASKED:-100}"
        TIE=(--tie-questions 1); REUSE_SEEDS_FROM="../pipeline_cluster_qwen9b/run2"; REUSE_TRAIN_ALL=0
        MID_TEST=0; COMPARE_EXTERNAL=1
        PROTOCOLS="direct_high,self_refine_high" ;;
    *) echo "family must be gptoss, qwen, qwen9b or qwen4b" >&2; exit 1 ;;
esac
case "$DATASET" in
    supergpqa)
        TRAIN="${TRAIN:-datasets/supergpqa_600_train.json}"
        TEST="${TEST:-datasets/supergpqa_600_test.json}"
        CLUSTERS="${CLUSTERS:-outputs/describe_v3/clusters_600_train.json}"
        ROUTES="${ROUTES:-outputs/describe_v3/routes_600_test.json}"
        OUT="${OUT:-outputs/pipeline_cluster_$FTAG}"
        RUN="$OUT/$RUNNAME"
        PER_GROUP="${PER_GROUP:-0}"                  # every question of every group is a search question
        REUSE_TEST_FROM=""                           # gptoss run3 started from run2's test debates (with run1's)
        [[ "$FAMILY:$RUNNAME" == gptoss:run3 ]] && REUSE_TEST_FROM="$OUT/run2/test_eval"
        JUDGE=(); ANSWERS=()
        SEARCH_OPTS=("${SUPERGPQA_OPTS[@]}")
        QTAG="$([[ "$PER_GROUP" == 0 ]] && echo all || echo "cap$PER_GROUP")"
        (( N_DEV > 0 )) && QTAG="train$N_DEV"        # the search questions: the groups' questions not held out
        SEARCH_Q="$OUT/search_questions_$QTAG.json"
        BNAME_TRAIN="${BTAG}_$(basename "$CLUSTERS" .json)_search_$QTAG"                  # ..._clusters_600_train_search_all
        REUSE_TRAIN_FROM="${BTAG}_$(basename "$CLUSTERS" .json)_search_cap50"           # run2's 150 search questions
        # qwen9b run2: its 200 search questions are among the 300 that qwen9b run1 ran them on
        if (( REUSE_TRAIN_ALL )); then REUSE_TRAIN_FROM="${BTAG}_$(basename "$CLUSTERS" .json)_search_all"; fi ;;
    hle)                                             # HLE (2026-10-07): the design of SuperGPQA qwen9b run2 / gptoss run4
        TRAIN="${TRAIN:-datasets/hle_text_train_800.json}"
        TEST="${TEST:-datasets/hle_text_test_200.json}"
        CLUSTERS="${CLUSTERS:-outputs/describe_hle_v4/clusters_train.json}"
        ROUTES="${ROUTES:-outputs/describe_hle_v4/routes_test.json}"
        OUT="${OUT:-outputs/pipeline_cluster_hle_$FTAG}"
        RUNNAME="run1"                               # gptoss: run2 (its run1, 2026-09-29, had the earlier design; kept)
        [[ "$FAMILY" == gptoss ]] && RUNNAME="run2"
        RUN="$OUT/$RUNNAME"
        # 100 of the 800 train questions are held out as a dev split (the champion step); the search
        # questions are the first 50 of each group's other questions, in the clusters file's order
        # (as run1's 50 per group, less the dev questions among them); the rest are not used
        PER_GROUP="${PER_GROUP:-50}"
        N_DEV="${N_DEV_ASKED:-100}"
        REUSE_TEST_FROM=""
        JUDGE=(--judge-model gpt-6-luna --judge-cache "$OUT/judge_gpt-6-luna.jsonl")
        ANSWERS=(--answers open "${JUDGE[@]}")      # 'ANSWER: <answer>', compared normalised, graded by the judge
        # one design and one set of prompts for every family (gpt-oss with no visible-reasoning
        # sentence, as in SuperGPQA run4)
        EXECUTOR=(--plain-instruction --last-round-vote --count-read-summaries)
        SEARCH_OPTS=(--high-cost 3 --turn-cap 15 --total-cap 21)
        TIE=(--tie-questions 1); MID_TEST=0
        # the external baselines are run separately (2026-10-07): none on the test split or the 200 search
        # questions, and steps 5 and 9 show no external rows; the comparison before the search runs on the
        # first 25 search questions of each group (100), with the external baselines on those alone, one
        # run each, against one run of ours (generation 0 runs our second)
        COMPARE_EXTERNAL=1; EXTERNAL_BASELINES=0; COMPARE_PER_GROUP=25; COMPARE_RUNS=1
        # the comparison's external Self-Refine as the collaborator runs it (baselines/run_baselines.py:
        # 28,672-token answers, 24,576-token feedback, a cut-off feedback recovered); the 16,384-token
        # feedback left 44% of gpt-oss's HLE feedback turns empty. Direct CoT is already the same.
        SR_ARGS=(--max-tokens 28672 --feedback-max-tokens 24576 --recover --recover-feedback)
        PROTOCOLS="direct_high,self_refine_high"
        # every family starts from the qwen9b HLE run's seeds (as on SuperGPQA and MATH)
        REUSE_SEEDS_FROM=""
        [[ "$FAMILY" != qwen9b ]] && REUSE_SEEDS_FROM="../pipeline_cluster_hle_qwen9b/run1"
        QTAG="cap${PER_GROUP}_dev$N_DEV"
        SEARCH_Q="$OUT/search_questions_$QTAG.json"
        BNAME_TRAIN="${BTAG}_$(basename "$TRAIN" .json)_search_$QTAG"     # ..._hle_text_train_800_search_cap50_dev100
        # gptoss: its run1 baselines on run1's search questions (50 per group, no dev split) hold most of
        # these, so only the questions new to the search are run (the same code reads them identically;
        # used only with EXTERNAL_BASELINES=1)
        REUSE_TRAIN_FROM="${BTAG}_$(basename "$TRAIN" .json)_search_cap$PER_GROUP" ;;
    math)                                            # MATH Level 5 (2026-10-07): the design of SuperGPQA qwen9b run2 / gptoss run4
        TRAIN="${TRAIN:-datasets/math_l5_300_train.json}"   # the 300-question train subset the groups are made on
        TEST="${TEST:-datasets/math_l5_test.json}"
        CLUSTERS="${CLUSTERS:-outputs/describe_math_v1/clusters_train.json}"
        ROUTES="${ROUTES:-outputs/describe_math_v1/routes_test.json}"
        OUT="${OUT:-outputs/pipeline_cluster_math_$FTAG}"
        RUN="$OUT/run1"
        PER_GROUP=0                                  # the dev split decides what is held out
        REUSE_TEST_FROM=""
        JUDGE=(); ANSWERS=(--answers math)           # answers in \boxed{}, compared and graded by math-verify
        # one design and one set of prompts for every family (gpt-oss with no visible-reasoning
        # sentence, as in run4)
        EXECUTOR=(--plain-instruction --last-round-vote --count-read-summaries)
        SEARCH_OPTS=(--high-cost 3 --turn-cap 15 --total-cap 21)
        N_DEV="${N_DEV_ASKED:-100}"; TIE=(--tie-questions 1); MID_TEST=0
        # as on HLE (2026-10-08): the external baselines are run separately, so none on the test split or
        # the 200 search questions, and steps 5 and 9 show no external rows; the comparison before the
        # search runs on the first 25 search questions of each group, one external run against one of
        # ours, with the collaborator's Self-Refine (baselines/run_baselines.py, the same on every dataset)
        COMPARE_EXTERNAL=1; EXTERNAL_BASELINES=0; COMPARE_PER_GROUP=25; COMPARE_RUNS=1
        SR_ARGS=(--max-tokens 28672 --feedback-max-tokens 24576 --recover --recover-feedback)
        PROTOCOLS="direct_high,self_refine_high"
        # every family starts from the qwen9b MATH run's seeds (as gptoss run4 started from qwen9b run2's)
        REUSE_SEEDS_FROM=""
        [[ "$FAMILY" != qwen9b ]] && REUSE_SEEDS_FROM="../pipeline_cluster_math_qwen9b/run1"
        QTAG="train$N_DEV"
        SEARCH_Q="$OUT/search_questions_$QTAG.json"
        BNAME_TRAIN="${BTAG}_$(basename "$TRAIN" .json)_search_$QTAG"     # ..._math_l5_300_train_search_train100
        REUSE_TRAIN_FROM="" ;;
    *) echo "dataset must be supergpqa, hle or math" >&2; exit 1 ;;
esac
# MATH and HLE: a search of another family starts from the qwen9b seeds of the dataset, so they must
# exist first (the comparison step, run_compare_external_cluster.sh, needs no seeds)
if [[ ( "$DATASET" == math || "$DATASET" == hle ) && -n "$REUSE_SEEDS_FROM"
      && "$(basename "$0")" == run_pipeline_cluster.sh
      && ! -f "$RUN/seeds.json" && ! -f "$OUT/$REUSE_SEEDS_FROM/seeds.json" ]]; then
    echo "$OUT/$REUSE_SEEDS_FROM/seeds.json does not exist: run the qwen9b $DATASET pipeline first" \
         "(its step 3 writes the seeds every family starts from)" >&2; exit 1
fi
EXECUTOR+=("${ANSWERS[@]}" "${SEARCH_OPTS[@]}")      # seeds, search and test evaluation alike
CMP_Q="$SEARCH_Q"; BNAME_CMP="$BNAME_TRAIN"           # the comparison's questions and their external baselines
if [[ -n "$COMPARE_PER_GROUP" ]]; then               # the first N of each group's search questions
    CMP_Q="$OUT/search_questions_cap${COMPARE_PER_GROUP}_dev$N_DEV.json"
    BNAME_CMP="${BTAG}_$(basename "$TRAIN" .json)_search_cap${COMPARE_PER_GROUP}_dev$N_DEV"   # ..._search_cap25_dev100
fi
SEEDS="$RUN/seeds.json"
SPLITS="$RUN/splits.json"
DEV_ARGS=()
(( N_DEV > 0 )) && DEV_ARGS=(--dev-split "$SPLITS")
GENERATIONS="${GENERATIONS:-10}"
MID="${MID:-5}"                                      # the search pauses after this generation for a test run
WINDOW="${WINDOW:-32768}"                            # the window the server must report
K=3                                                  # runs per question: baselines and test replicates
CMP_K="${COMPARE_RUNS:-$K}"                          # runs per question of the comparison's external baselines
VENV_PYTHON=/nas-ssd2/tianyin4/cache/venvs/vllm-updated/bin/python   # this server's environment
[[ -x "$VENV_PYTHON" ]] || VENV_PYTHON=python3        # elsewhere (a rented GPU): the python3 on the PATH
PYTHON="${PYTHON:-$VENV_PYTHON}"
TAGNAME="$("$PYTHON" -c "import sys; sys.path.insert(0, 'scripts'); import program_space as P; print(P.model_tag('$MODEL'))")"
CACHE="$OUT/rounds_$TAGNAME.jsonl"
BRES="baselines/results"
BNAME_TEST="${BTAG}_$(basename "$TEST" .json)"                                # ..._supergpqa_600_test

mkdir -p "$OUT" "$RUN" "$BRES"
stamp() { date "+%Y-%m-%d %H:%M:%S"; }

echo "[$(stamp)] checking that $BASE_URL serves $MODEL with a $WINDOW-token window"
SERVED="$(curl -sf --max-time 10 "$BASE_URL/models" || true)"
LEN="$(echo "$SERVED" | "$PYTHON" -c "
import json, sys
try:
    d = json.load(sys.stdin)
except ValueError:
    sys.exit(0)
print(next((m.get('max_model_len', '') for m in d.get('data', []) if m.get('id') == '$MODEL'), ''))")"
if [[ -z "$LEN" ]]; then
    echo "no server at $BASE_URL serving $MODEL (is this the right machine and port?)" >&2; exit 1
fi
if [[ "$LEN" != "$WINDOW" ]]; then
    echo "the server's window is $LEN tokens, not $WINDOW: restart it with --max-model-len $WINDOW" \
         "(or set WINDOW=$LEN to run with it; the window is part of every recording's key)" >&2; exit 1
fi
for f in "$TRAIN" "$TEST" "$CLUSTERS" "$ROUTES"; do
    [[ -f "$f" ]] || { echo "$f is missing (run_describe_v3.sh, run_describe_hle.sh or run_describe_math.sh makes" \
                            "the clusters and routes)" >&2; exit 1; }
done

# --- the external baselines: one question set at a time -----------------------------------------
missing_samples() {  # dataset results [k] -> how many of the k (default K) samples per question are not done
    "$PYTHON" - "$1" "$2" "${3:-$K}" <<'EOF'
import json, sys
data, path, k = sys.argv[1], sys.argv[2], int(sys.argv[3])
ids = {r["id"] for r in json.load(open(data))}
done = set()
try:
    for line in open(path):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get("error") is None and r.get("id") in ids and r.get("sample_idx", k) < k:
            done.add((r["id"], r["sample_idx"]))
except FileNotFoundError:
    pass
print(len(ids) * k - len(done))
EOF
}
run_baselines() {  # dataset name [k] -> direct and Self-Refine, k (default K) runs each, on exactly the dataset's questions
    local data="$1" name="$2" runs="${3:-$K}" left_d left_r left_s attempt B k
    local common=(--data "$(readlink -f "$data")" --endpoints "${BASE_URL%/v1}" --model "$MODEL" --k "$runs"
                  "${BASELINE_ARGS[@]}")
    for attempt in 1 2 3; do
        left_d="$(missing_samples "$data" "$BRES/direct_$name.jsonl" "$runs")"
        left_r="$(missing_samples "$data" "$BRES/direct_${name}_rec.jsonl" "$runs")"
        left_s="$(missing_samples "$data" "$BRES/selfrefine_${name}_rec.jsonl" "$runs")"
        if [[ "$left_d" == 0 && "$left_r" == 0 && "$left_s" == 0 ]]; then break; fi
        echo "[$(stamp)]   attempt $attempt on $data; to run: direct $left_d, direct with recovery $left_r," \
             "Self-Refine with recovery $left_s samples"
        "$PYTHON" baselines/generate.py --out "$BRES/direct_$name.jsonl" "${common[@]}" --max-tokens 28672
        # the direct samples as generated, with every one cut off before its answer recovered
        "$PYTHON" baselines/recover.py --data "$(readlink -f "$data")" --results "$BRES/direct_$name.jsonl" \
            --out "$BRES/direct_${name}_rec.jsonl" --endpoints "${BASE_URL%/v1}" --model "$MODEL" \
            "${BASELINE_ARGS[@]}"
        "$PYTHON" baselines/selfrefine.py --out "$BRES/selfrefine_${name}_rec.jsonl" "${common[@]}" \
            "${SR_ARGS[@]}"
    done
    left_d="$(missing_samples "$data" "$BRES/direct_$name.jsonl" "$runs")"
    left_r="$(missing_samples "$data" "$BRES/direct_${name}_rec.jsonl" "$runs")"
    left_s="$(missing_samples "$data" "$BRES/selfrefine_${name}_rec.jsonl" "$runs")"
    if [[ "$left_d" != 0 || "$left_r" != 0 || "$left_s" != 0 ]]; then
        echo "external baselines on $data still incomplete (direct $left_d, direct with recovery $left_r," \
             "Self-Refine with recovery $left_s samples): run again" >&2
        exit 1
    fi
    # scored on the dataset's questions only, over all the runs and on the first run alone: the
    # recovered files, and the files of the method as published where they are complete
    for B in "direct_${name}_rec" "selfrefine_${name}_rec" "direct_$name" "selfrefine_$name"; do
        [[ -f "$BRES/$B.jsonl" && "$(missing_samples "$data" "$BRES/$B.jsonl" "$runs")" == 0 ]] || continue
        for k in $(printf "%s\n" "$runs" 1 | sort -un); do
            "$PYTHON" baselines/score.py --data "$data" --results "$BRES/$B.jsonl" --k "$k" \
                --save "$BRES/${B}_k$k.json" "${JUDGE[@]}"
        done
    done
}
external() {  # name -> the label=file pairs the reports read: with recovery, then as published if scored
    (( EXTERNAL_BASELINES )) || { echo ""; return; }     # none in this run: the reports have no external rows
    local pairs="self-refine=$BRES/selfrefine_$1_rec_k$K.json,direct=$BRES/direct_$1_rec_k$K.json"
    [[ -f "$BRES/selfrefine_$1_k$K.json" ]] && pairs+=",self-refine no recovery=$BRES/selfrefine_$1_k$K.json"
    [[ -f "$BRES/direct_$1_k$K.json" ]] && pairs+=",direct no recovery=$BRES/direct_$1_k$K.json"
    echo "$pairs"
}

# --- 1. external baselines on the test split ---------------------------------------------------
if (( EXTERNAL_BASELINES )); then
    echo "[$(stamp)] step 1: external baselines on $TEST"
    run_baselines "$TEST" "$BNAME_TEST"
else
    echo "[$(stamp)] step 1: skipped: no external baselines in this run (they are run separately)"
fi

# --- 2. external baselines on the train search questions --------------------------------------
# With a dev split, the split is made first (a fixed draw: the same file on every rerun). The search
# questions follow from the clusters, the cap and the split, so they are written out every time
# (the same file unless those change) and the baselines are run and scored on them alone.
if (( N_DEV > 0 )); then
    echo "[$(stamp)] step 2: dev split of $TRAIN ($N_DEV dev questions, seed 0)"
    "$PYTHON" scripts/split_train_dev.py --dataset "$TRAIN" --n-dev "$N_DEV" --seed 0 --out "$SPLITS"
fi
echo "[$(stamp)] step 2: $( (( EXTERNAL_BASELINES )) && echo "external baselines on the search questions" || echo "the search questions, with no external baselines" ) ($( (( N_DEV > 0 )) && echo "the groups' train-split questions" || { [[ "$PER_GROUP" == 0 ]] && echo "every question" || echo "at most $PER_GROUP per group"; }))"
"$PYTHON" scripts/train_baselines.py --export --clusters "$CLUSTERS" --per-group "$PER_GROUP" \
    "${DEV_ARGS[@]}" --dataset "$TRAIN" --out "$SEARCH_Q"
# an earlier run's samples on some of these questions are kept: the scripts skip every (question,
# sample) already in their output, and a sample's seed depends only on its question and index
if (( EXTERNAL_BASELINES )); then
    for f in "direct_%s" "direct_%s_rec" "selfrefine_%s_rec"; do
        old="$BRES/$(printf "$f" "$REUSE_TRAIN_FROM").jsonl" new="$BRES/$(printf "$f" "$BNAME_TRAIN").jsonl"
        if [[ -n "$REUSE_TRAIN_FROM" && ! -f "$new" && -f "$old" ]]; then
            echo "[$(stamp)]   $new starts from a copy of $old"
            cp "$old" "$new"
        fi
    done
    run_baselines "$SEARCH_Q" "$BNAME_TRAIN"
fi
# the comparison's own questions (COMPARE_PER_GROUP), a part of the search questions, with the
# external baselines on them alone (once they are complete a rerun only scores them again)
if (( COMPARE_EXTERNAL )) && [[ "$CMP_Q" != "$SEARCH_Q" ]]; then
    echo "[$(stamp)] step 2: external baselines on the comparison's questions (the first $COMPARE_PER_GROUP" \
         "search questions of each group), $CMP_K run(s) each"
    "$PYTHON" scripts/train_baselines.py --export --clusters "$CLUSTERS" --per-group "$COMPARE_PER_GROUP" \
        "${DEV_ARGS[@]}" --dataset "$TRAIN" --out "$CMP_Q"
    "$PYTHON" - "$CMP_Q" "$SEARCH_Q" <<'EOF'
import json, sys
part, whole = ({r["id"] for r in json.load(open(p))} for p in sys.argv[1:])
if not part <= whole:
    sys.exit(f"{sys.argv[1]}: {len(part - whole)} questions are not search questions")
print(f"  {len(part)} of the {len(whole)} search questions")
EOF
    run_baselines "$CMP_Q" "$BNAME_CMP" "$CMP_K"
fi

COMMON=(--model "$MODEL" --base-urls "$BASE_URL" --live-cache "$CACHE" --clusters "$CLUSTERS"
        --dataset "$TRAIN" --per-group "$PER_GROUP" "${DEV_ARGS[@]}" --workers "$WORKERS" "${EXECUTOR[@]}")
