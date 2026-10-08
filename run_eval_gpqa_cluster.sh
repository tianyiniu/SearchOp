#!/usr/bin/env bash
# A finished SuperGPQA cluster search's programs on GPQA-Diamond (198 questions, test only;
# 2026-10-08). GPQA has no train split, so there is no search on it: each GPQA question goes to its
# SuperGPQA group (outputs/describe_v3/routes_gpqa_diamond_test.json, made by
# run_describe_gpqa_diamond.sh with the describer that made the groups) and runs that group's
# program. The evaluation is step 9 of run_pipeline_cluster.sh on these questions: the routed
# champions, beside them the routed slot-A holders, the global champion, and the in-executor
# protocols (direct_high, self_refine_high), 3 replicates. No external baselines (run_baselines.py
# runs those). Run it after run_pipeline_cluster.sh <family> supergpqa has finished (it needs the
# run's champions, step 8b), with the same model server:
#
#     bash run_eval_gpqa_cluster.sh qwen4b 2>&1 | tee -a outputs/pipeline_cluster_qwen4b/run1/test_eval_gpqa.log
#
# Writes <run>/test_eval_gpqa/results_k3.md and .json. Resumable: finished debates are kept
# (<run>/test_eval_gpqa/rounds_<model>.jsonl), so after an interruption run it again.
set -euo pipefail
cd "$(dirname "$0")"
FAMILY="${1:?the model family of the SuperGPQA run (gptoss, qwen9b or qwen4b)}"
set -- "$FAMILY" supergpqa
source pipeline_cluster_setup.sh                    # the run's settings (and steps 1-2, already done)

GPQA=datasets/gpqa_diamond_test.json
GPQA_ROUTES=outputs/describe_v3/routes_gpqa_diamond_test.json
EVAL="$RUN/test_eval_gpqa"
for f in "$GPQA" "$GPQA_ROUTES" "$RUN/summary.json"; do
    [[ -f "$f" ]] || { echo "$f is missing" >&2; exit 1; }
done
champs=(--no-champions)
if (( N_DEV > 0 )); then
    [[ -f "$RUN/champions.json" ]] || { echo "$RUN/champions.json is missing: finish" \
        "bash run_pipeline_cluster.sh $FAMILY supergpqa first (its step 8b chooses the champions)" >&2; exit 1; }
    champs=()
fi
mkdir -p "$EVAL"
echo "[$(stamp)] GPQA-Diamond: the routed programs of $RUN on $GPQA, $K replicates -> $EVAL"
"$PYTHON" scripts/eval_routed_dev.py --run "$RUN" --routes "$GPQA_ROUTES" --dataset "$GPQA" "${champs[@]}" \
    --out "$EVAL" --live-cache "$EVAL/rounds_$TAGNAME.jsonl" --model "$MODEL" --base-urls "$BASE_URL" \
    --workers "$WORKERS" --reps "$K" --baselines "$PROTOCOLS" "${EXECUTOR[@]}" 2>&1 | tee -a "$EVAL.log"
echo "[$(stamp)] done: $EVAL/results_k$K.md"
