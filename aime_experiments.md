# AIME experiments: all three models on one rented GPU server

AIME 2022-2025 replaces MATH level 5 for Qwen 3.5 9B, Qwen 3.5 4B and gpt-oss-20b. Each model has
three parts, run in this order:

1. **The comparison:** our Direct CoT and Self-Refine against the external ones, on the 40 search
   problems, 1 run each. Check its report before step 3.
2. **The external baselines:** Direct CoT, self-consistency (5 samples), Self-Refine (2 rounds) and
   MAD (3 agents, 2 rounds), on the 60 test problems, 3 runs each. AFlow and MaAS are not run here.
3. **The search and the test:** 10 generations on the 40 search problems, then the 60 test problems,
   3 runs each, with each group's strongest search program. The 20 dev problems are used only with
   `--champions`, which first chooses each group's champion on them.

Everything the runs need is in the repo: the problems, the 2 groups, the routes, the dev split and
the seeds (written once with the 9B setup; the 4B and gpt-oss copy them). AIME is graded locally with
`math-verify`, so no API key is needed.

## Setup (once)

```bash
git clone git@github.com:tianyiniu/SearchOp.git && cd SearchOp
pip install -r requirements.txt
```

Use `tmux`: every run below takes hours. Every command resumes: after a stop, run the same command
again and finished work is kept.

## The model servers

Run one model at a time. The 9B and gpt-oss both use port 7472. The `_rental` scripts serve each
model exactly as our own servers do (model name, port, window, parsers, chat options). They use GPU
0 by default and put the weights in `/workspace/models` when `/workspace` exists. On a machine with
more GPUs, pass the list, such as `0,1`, for one copy of the model per GPU.

| model | start the server | port | family name |
|---|---|---|---|
| Qwen 3.5 9B | `bash Model_hosting/deploy_qwen35_9b_rental.sh` | 7472 | `qwen9b` (baselines: `qwen35-9b`) |
| Qwen 3.5 4B | `bash Model_hosting/deploy_qwen35_4b_rental.sh` | 7473 | `qwen4b` (baselines: `qwen35-4b`) |
| gpt-oss-20b | `bash Model_hosting/deploy_gpt_oss_20b_rental.sh` | 7472 | `gptoss` (baselines: `gptoss-20b`) |

The server is ready when `curl localhost:<port>/v1/models` answers. Keep the 32,768-token window
the scripts set: every run checks it.

## The commands, for each model

With the model's server up, run steps 1-3 with its names from the table. For the 9B:

```bash
# 1. comparison (then read outputs/pipeline_cluster_aime_qwen9b/run1/external_baselines.md)
mkdir -p outputs/pipeline_cluster_aime_qwen9b/run1
bash run_compare_external_cluster.sh qwen9b aime 2>&1 | tee -a outputs/pipeline_cluster_aime_qwen9b/run1/compare.log

# 2. external baselines on the 60 test problems
mkdir -p baselines/results
python baselines/run_baselines.py --model qwen35-9b --data datasets/aime_2022_2025_test.json 2>&1 | tee -a baselines/results/aime_qwen35_9b.log

# 3. search and test (it starts only after a passed comparison)
bash run_pipeline_cluster.sh qwen9b aime 2>&1 | tee -a outputs/pipeline_cluster_aime_qwen9b/run1/pipeline.log
```

For the 4B, replace `qwen9b` with `qwen4b` and `qwen35-9b` with `qwen35-4b`. For gpt-oss, replace
them with `gptoss` and `gptoss-20b`.

**Checking the comparison (step 1).** The report must say `PASS`, with every row of its **Checks**
table `yes`. With 40 problems, the accuracy check catches only a gap of about 10 points or more, so
also read the numbers in **Ours against external**. If ours is clearly lower, do not start step 3.
Steps 1 and 2 are independent: step 2 can run while you check step 1.

## Results

- `baselines/results/table_<tag>_aime_2022_2025_test_rec.md`: the four external baselines (avg@1,
  vote@3, avg@3, pass@3, tokens).
  - The tags are `qwen35_9b_think`, `qwen35_4b_think` and `gptoss20b_high_explain`.
  - gpt-oss's first prompt also asks it to write its reasoning in the reply, as on the other
    datasets.
- `outputs/pipeline_cluster_aime_<family>/run1/test_eval/results_k3.md`: our routed programs on the
  60 test problems, with the strongest search program over both groups.

To move a run to another server, copy `outputs/pipeline_cluster_aime_<family>/` (without its `.lock`
file) and the model's `baselines/results/*aime*` files, then run the same commands there.
