#!/usr/bin/env bash
# A finished SuperGPQA cluster search's programs on GPQA-Diamond (198 questions, test only;
# 2026-10-08). GPQA has no train split, so there is no search on it: each GPQA question goes to its
# SuperGPQA group (outputs/describe_v3/routes_gpqa_diamond_test.json, made by
# run_describe_gpqa_diamond.sh with the describer that made the groups) and runs that group's
# program. The evaluation is step 9 of run_pipeline_cluster.sh on these questions: the routed slot-A
# holders (each group's strongest grid program) and the global slot holder, 3 replicates; with
# --champions, the routed champions (step 8b), beside them the routed slot-A holders, and the global
# champion. Each group's program runs only on its own group's questions, the global program on every
# question, and no in-executor protocols run (since 2026-10-09); the 3 replicates run in one pool. No external baselines (run_baselines.py
# runs those). Run it after run_pipeline_cluster.sh <family> supergpqa has finished (with --champions:
# after it ran with --champions, which makes the champions), with the same model server:
#
#     bash run_eval_gpqa_cluster.sh qwen4b 2>&1 | tee -a outputs/pipeline_cluster_qwen4b/run1/test_eval_gpqa.log
#     bash run_eval_gpqa_cluster.sh qwen4b --champions     (the run's champions; off by default, 2026-10-09)
#
# Writes <run>/test_eval_gpqa/results_k3.md and .json. Resumable: finished debates are kept
# (<run>/test_eval_gpqa/rounds_<model>.jsonl), so after an interruption run it again.
set -euo pipefail
cd "$(dirname "$0")"
PICK_CHAMPIONS=0                                     # --champions: the run's champions (its step 8b)
ARGS=()
for a in "$@"; do
    if [[ "$a" == --champions ]]; then PICK_CHAMPIONS=1; else ARGS+=("$a"); fi
done
FAMILY="${ARGS[0]:?the model family of the SuperGPQA run (gptoss, qwen9b or qwen4b)}"
set -- "$FAMILY" supergpqa
source pipeline_cluster_setup.sh                    # the run's settings (and steps 1-2, already done)

GPQA=datasets/gpqa_diamond_test.json
GPQA_ROUTES=outputs/describe_v3/routes_gpqa_diamond_test.json
EVAL="$RUN/test_eval_gpqa"
for f in "$GPQA" "$GPQA_ROUTES" "$RUN/summary.json"; do
    [[ -f "$f" ]] || { echo "$f is missing" >&2; exit 1; }
done
champs=(--no-champions)
if (( N_DEV > 0 && PICK_CHAMPIONS )); then
    [[ -f "$RUN/champions.json" ]] || { echo "$RUN/champions.json is missing: finish" \
        "bash run_pipeline_cluster.sh $FAMILY supergpqa --champions first (its step 8b chooses the champions)" >&2; exit 1; }
    champs=()
fi
mkdir -p "$EVAL"
echo "[$(stamp)] GPQA-Diamond: the routed programs of $RUN on $GPQA, $K replicates -> $EVAL"
"$PYTHON" scripts/eval_routed_dev.py --run "$RUN" --routes "$GPQA_ROUTES" --dataset "$GPQA" "${champs[@]}" \
    --out "$EVAL" --live-cache "$EVAL/rounds_$TAGNAME.jsonl" --model "$MODEL" --base-urls "$BASE_URL" \
    --workers "$WORKERS" --reps "$K" --baselines "$PROTOCOLS" --routed-only --pool-replicates "${EXECUTOR[@]}" \
    2>&1 | tee -a "$EVAL.log"
echo "[$(stamp)] done: $EVAL/results_k$K.md"
