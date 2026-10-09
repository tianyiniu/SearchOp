# Qwen 3.5 4B experiments: how to run them

This guide runs our method (the cluster pipeline) with **Qwen/Qwen3.5-4B** on four datasets. It
lists every command in the order to run them. The external baselines (Direct CoT,
self-consistency, Self-Refine, MAD) are **not** part of this work. Tianyi runs those.

## The experiments

| # | Experiment | Search on | Test on | Commands |
|---|---|---|---|---|
| 1 | SuperGPQA | 200 of the 300 train questions (3 groups) | the 1,000-question test split | [section 3](#3-supergpqa-search-and-the-1000-question-test) |
| 2 | GPQA-Diamond | no search: it uses the SuperGPQA programs of experiment 1 | the 198 questions | [section 4](#4-gpqa-diamond-the-supergpqa-programs-on-gpqa) |
| 3 | HLE (text only) | 200 of the 800 train questions (4 groups) | the 200 test questions | [section 5](#5-hle-search-and-test) |
| 4 | AIME 2022-2025 | 40 of the 60 train problems (2 groups) | the 60 test problems | [section 6](#6-aime-search-and-test) |

MATH level 5 is dropped: every model scored 95-98% on it, which leaves no room to show a
difference. AIME replaces it for all three models (Qwen 3.5 4B, Qwen 3.5 9B and gpt-oss-20b).

**Already done for you:**
- **Question groups and routes, for all four datasets.** The groups were made from the train
  questions. Every test question is routed to the group of the train questions most like it. Your
  runs make no new groups.
- **Seed programs.** Each search starts from the seeds of the earlier Qwen 3.5 9B run on the same
  dataset, so step 3 copies them and makes no API call.
- **The HLE comparison for the 4B.** It passed, so the HLE search can start at once (section 5).

## 1. Setup

### 1.1 The code and the private data

```bash
git clone git@github.com:tianyiniu/SearchOp.git
cd SearchOp
tar -xzf /path/to/qwen4b_hle_gpqa_data.tar.gz
```

**The repo is public.** HLE and GPQA ask that their questions never appear online, where models
could be trained on them. So their files are not in the repo. Tianyi sends them to you separately
as `qwen4b_hle_gpqa_data.tar.gz`. Unpack it in the repo's top folder, as shown above. It holds 12
files:
- the HLE and GPQA-Diamond question files (`datasets/`);
- the 4B HLE comparison (`outputs/pipeline_cluster_hle_qwen4b/`);
- that comparison's external-baseline replies (`baselines/results/`).

**Never commit or push these files, or any HLE or GPQA output.** `.gitignore` keeps them out, so
never use `git add -f`.

### 1.2 Python

Use Python 3.12. One environment serves both the model server and the pipeline:

```bash
pip install -r requirements.txt
```

The scripts run `python3` from your `PATH`, so activate this environment before every command.

### 1.3 The API key (HLE only)

Only HLE uses a paid API. Its answers are free text, so a judge model (`gpt-6-luna`, through the
OpenAI API) grades them. The other datasets need no key:
- SuperGPQA and GPQA-Diamond are multiple choice, so their letters are graded locally.
- AIME is graded locally with `math-verify` (its answers are integers in `\boxed{}`).

Nothing else calls the API: the groups, routes and seeds are already made.

Make a file named `.env` in the repo's top folder that holds one line:

```
OPENAI_API_KEY=sk-...
```

`.gitignore` keeps `.env` out of git, so never commit it. To check the key, run the code below. It
makes a free call that lists the models the key can use:

```bash
python - <<'EOF'
from dotenv import load_dotenv; load_dotenv(".env")
from openai import OpenAI
print("gpt-6-luna available:", "gpt-6-luna" in {m.id for m in OpenAI().models.list()})
EOF
```

It must print `True`. The judge grades each distinct (question, answer) pair once. Its verdicts are
saved in `outputs/pipeline_cluster_hle_qwen4b/judge_gpt-6-luna.jsonl`, so a rerun does not pay
again.

### 1.4 The model server

Start it on the machine that runs the scripts. It runs one copy of the model on each GPU you list:

```bash
bash Model_hosting/deploy_qwen35_4b.sh 0,1,2,3
```

- It serves `Qwen/Qwen3.5-4B` on port **7473** with a **32,768-token window**.
- The first start downloads about 9 GB of weights to the Hugging Face cache.
- It is ready when `curl http://localhost:7473/v1/models` answers.
- Do not change the model name, the port or the window. Every script checks the name and the
  window, and the window is part of every saved debate's key.
- Keep the server up for all the runs. All four experiments use the same server.

`requirements.txt` pins vLLM 0.18.0, which the HLE comparison ran on. vLLM 0.29 also works with
these scripts.

The vLLM log will show lines such as `VLLMValidationError: This model's maximum context length is
32768 tokens. However, you requested 28672 output tokens ...`. **These are expected.** The external
baselines first ask for 28,672 reply tokens. When a prompt is too long for that, vLLM refuses, and
the script counts the prompt and asks again for the room that is left.

### 1.5 Long runs

- **Use `tmux` or `screen`.** A search runs for many hours.
- **Every script resumes.** After an interruption (a closed session, a stopped server, a reboot),
  start the server again and run the **same command** again. Finished work is kept.
- **One run per server at a time.** Each run keeps 128 debates in flight, so two runs on one
  server slow each other down. With several servers, you can run different datasets at the same
  time, one per server.
- **Run every command from the repo's top folder.**
- **Do not edit the code or the settings while you run these experiments.** Each search checks
  that its comparison ran under the same settings and prompts. The HLE comparison was made with
  this exact code.

## 2. The rule: the comparison first, then a manual check, then the search

Each dataset starts with a short comparison, `run_compare_external_cluster.sh`. It checks that our
pipeline's own Direct CoT (`direct_high`) and Self-Refine (`self_refine_high`) behave like the
external Direct CoT and Self-Refine on about 100 search questions, with 1 run of each. It writes
`<run folder>/external_baselines.md` and ends. The search script refuses to start until that
comparison has passed under the same settings.

**Check `external_baselines.md` yourself before you start the search:**
1. The title line says `PASS`.
2. Every row of the **Checks** table says `yes`. The checks cover:
   - every question ran;
   - accuracy, ours against external;
   - how often a run ends with no answer;
   - how often a reply runs out of room;
   - that the low-effort solver ran.
3. In **Ours against external**, look at the accuracy differences. With 100 questions and 1 run,
   the automatic check only catches a gap of about 6 points or more. If ours is clearly lower
   than the external one, stop and tell Tianyi, even when the check passed.

If the comparison fails, or a number looks wrong, **do not start the search.** Send Tianyi
`external_baselines.md` and `compare.log` from that run folder.

## 3. SuperGPQA: search and the 1,000-question test

```bash
mkdir -p outputs/pipeline_cluster_qwen4b/run1
bash run_compare_external_cluster.sh qwen4b supergpqa 2>&1 | tee -a outputs/pipeline_cluster_qwen4b/run1/compare.log
```

- The comparison uses 102 questions: the first 34 search questions of each of the 3 groups.
- Check `outputs/pipeline_cluster_qwen4b/run1/external_baselines.md` as section 2 says.
- Then start the search:

```bash
bash run_pipeline_cluster.sh qwen4b supergpqa 2>&1 | tee -a outputs/pipeline_cluster_qwen4b/run1/pipeline.log
```

What the run does:
- It holds back 100 of the 300 train questions as a dev split.
- It searches for 10 generations on the other 200.
- It runs each group's strongest search program on the **1,000 test questions**, 3 times each
  (step 9).
- It does not use the dev questions by default. To choose each group's champion on them first
  (step 8b), add `--champions` to the command.

**Result:** `outputs/pipeline_cluster_qwen4b/run1/test_eval/results_k3.md`.

## 4. GPQA-Diamond: the SuperGPQA programs on GPQA

GPQA-Diamond has no train split, so there is no search on it. Its 198 questions were routed into
the SuperGPQA groups by the method the SuperGPQA test questions use: 164 to group 0, 34 to group 1
and none to group 2. Each question runs its group's strongest search program from section 3.

Run this **after the SuperGPQA run has finished**, with the same server:

```bash
bash run_eval_gpqa_cluster.sh qwen4b 2>&1 | tee -a outputs/pipeline_cluster_qwen4b/run1/test_eval_gpqa.log
```

It is the test step of section 3 (step 9) on these questions, 3 runs each.

**Result:** `outputs/pipeline_cluster_qwen4b/run1/test_eval_gpqa/results_k3.md`.

## 5. HLE: search and test

**Skip the comparison: it already passed.** It ran on the 4B on 2026-10-08, on 100 search
questions with 1 run each, and its results came with the tarball:

| | ours | external |
|---|---|---|
| Direct CoT | 17.0% | 10.0% |
| Self-Refine | 14.0% | 15.0% |

The report is `outputs/pipeline_cluster_hle_qwen4b/run1/external_baselines.md`. Check that it is
there and says `PASS`. Then, with `.env` in place (section 1.3), start the search:

```bash
bash run_pipeline_cluster.sh qwen4b hle 2>&1 | tee -a outputs/pipeline_cluster_hle_qwen4b/run1/pipeline.log
```

- The search's generation 0 reuses the comparison's debates (`rounds_qwen35_4b.jsonl`, from the
  tarball).
- The search uses 200 search questions: 50 from each of the 4 groups, after 100 dev questions are
  held back.
- It runs 10 generations, then the 200 test questions 3 times each. The 100 dev questions are not
  used (they are used only with `--champions`).
- The judge grades every new answer, through the API key.

**Result:** `outputs/pipeline_cluster_hle_qwen4b/run1/test_eval/results_k3.md`.

## 6. AIME: search and test

AIME 2022-2025 has 120 problems, split 60 train and 60 test with 15 of each year on each side
(`scripts/prepare_aime.py`). The 60 train problems form 2 groups: geometry (26) and counting,
number theory and algebra (34). 10 problems of each group are held back as the dev split, so the
search runs on 40. The seeds were written with the Qwen 3.5 9B setup and are copied, so every
model starts from the same programs.

```bash
mkdir -p outputs/pipeline_cluster_aime_qwen4b/run1
bash run_compare_external_cluster.sh qwen4b aime 2>&1 | tee -a outputs/pipeline_cluster_aime_qwen4b/run1/compare.log
```

- The comparison uses all 40 search problems, with 1 run each.
- Check `outputs/pipeline_cluster_aime_qwen4b/run1/external_baselines.md` as section 2 says. With
  only 40 problems, the automatic accuracy check catches only a gap of about 10 points or more.
- Then start the search:

```bash
bash run_pipeline_cluster.sh qwen4b aime 2>&1 | tee -a outputs/pipeline_cluster_aime_qwen4b/run1/pipeline.log
```

AIME replies are long. In a check with the 9B, finished replies used 13,000-22,000 tokens, and 2
of 8 ran out of the window (the recovery step found an answer for one of them). Expect each AIME
debate to take longer than one on the other datasets.

**Result:** `outputs/pipeline_cluster_aime_qwen4b/run1/test_eval/results_k3.md`, which covers the 60
test problems.

## 7. What a search run does

`run_pipeline_cluster.sh` runs these steps in order and writes its progress to `pipeline.log`:

1. **Dev split and search questions:** the dev split is a fixed draw, the same on every rerun.
   This step also checks the comparison.
2. **Seeds:** copied from the Qwen 3.5 9B run of the same dataset.
3. **Generation 0:** every seed runs on every search question, twice.
4. **Generations 1 to 10:** each generation makes 2 waves of children. A wave makes one child for
   each slot. There are two slots for each group and one for the whole set: 7 slots with 3 groups,
   9 with 4. A new best program is confirmed with a second run before it is kept.
5. **The test (step 9):** each test question runs its group's strongest search program 3 times.
   The table also shows the strongest search program over all groups, and our `direct_high` and
   `self_refine_high` as reference rows.

The champion step (8b) runs only with `--champions`. Then the 5 strongest distinct programs of each
group run on that group's dev questions, twice, and the best becomes the group's champion. The test
then runs the champions, and the table shows them beside the strongest search programs.

**Main files in the run folder:**
- `test_eval/results_k3.md` and `.json`: the test results;
- `champions.json`: the chosen programs (only with `--champions`);
- `summary.json` and `generations.jsonl`: the search;
- `archive.jsonl`: every program tried.

**For planning, the Qwen 3.5 9B runs on our server** (2 RTX PRO 6000 cards):
- **SuperGPQA:** the search took 11 hours. The champion step (not run by default) took 2 hours. The test on 300
  questions took 4 hours, so expect about 3 times that on 1,000.
- **HLE:** generation 0 took 7 hours, then about 3.3 hours per generation.

A run folder grows to as much as 1 GB.

## 8. Sending the results back

When a run ends, send Tianyi its whole output folder:
- `outputs/pipeline_cluster_qwen4b/`, which holds SuperGPQA and GPQA-Diamond;
- `outputs/pipeline_cluster_hle_qwen4b/`;
- `outputs/pipeline_cluster_aime_qwen4b/`;
- the comparisons' external-baseline files, `baselines/results/*qwen35_4b_think*search_cap*`.

**Send them directly (for example `tar` and `scp`), never through GitHub.** The HLE and GPQA
outputs contain their questions.

## 9. Troubleshooting

| Message | What to do |
|---|---|
| `no server at http://localhost:7473/v1 serving Qwen/Qwen3.5-4B` | Start the server (section 1.4) or wait until it answers. The server and the scripts must run on the same machine. |
| `the server's window is N tokens, not 32768` | Restart the server with `deploy_qwen35_4b.sh`, which sets the window. |
| `no passed comparison ... run bash run_compare_external_cluster.sh ... first` | Run the comparison and check it (section 2). For HLE, the tarball is missing or was not unpacked in the repo's top folder. |
| `OPENAI_API_KEY is not set` | Make `.env` (section 1.3). Only HLE needs it. |
| `.../seeds.json does not exist: run the qwen9b ... pipeline first` | The seed files come with the repo (`outputs/pipeline_cluster_*qwen9b/run*/seeds.json`). Run `git pull`, and check that nothing deleted them. |
| `... still incomplete ...: run again`, or a server that stopped | Restart the server if needed, then run the same command again. |
| `VLLMValidationError: ... maximum context length` in the vLLM log | Expected (section 1.4). |

If anything else stops a run, send Tianyi the last 50 lines of its log.
