# Per-cluster prompt optimization: experiment plan (5 October 2026)

**Decisions (user, 5 October):**
- Test on the 300 routed test questions only (option a), with 3 runs per question.
- No guard and no control.
- The paid GPT-6 calls are approved.

**Built:** `scripts/prompt_opt_v3.py`, with tests in `tests/test_prompt_opt_v3.py`.

**Command** (both stages, chained; resumable): `bash run_prompt_opt_v3.sh`. The stages one at a time:

    python scripts/prompt_opt_v3.py --stage optimize --plain-instruction --last-round-vote
    python scripts/prompt_opt_v3.py --stage test --plain-instruction --last-round-vote

## Question

Can GPT-6 improve the best program of each cluster by rewriting its persona prompts from examples of its debates?

The main test compares each program with its new prompts against the same program with its original prompts. Both run on the same questions, so the comparison is paired per question. The picker plan (option B in findings_and_options.md) is on hold.

## Terms

| Term | Meaning |
|---|---|
| Cluster | One of the 3 question groups of the v3 Qwen run (train sizes 144, 99, 57). |
| Program | The best (slot A) program of a cluster in the v3 Qwen run. |
| Persona prompt | The system prompt of one speaker role (solver, critic, expert). Only its role text changes. |
| Trace | One program run on one question: each speaker's visible reply, the summary later speakers read, and its answer letter. |
| Iteration | GPT-6 reads traces and writes new prompts; we run the program with them on the same 6 questions. |
| P0 to P3 | The prompts at iterations 0 (original) to 3. |

## What stays fixed

- The model: Qwen3.5-9B, thinking on for high effort.
- The settings of the cluster-pipeline run: 32,768-token window, plain instruction, last-round vote, high effort counts 5 turns, 16-turn cap, 500-word summaries.
- Each program's rounds, efforts, visibility and stop rule.
- The user message: the question and options, and later the "Prior responses" digest.
- The summary and commit prompts.
- The answer line, which the code adds after every persona prompt: "End with exactly one line: 'ANSWER: <letter>'. Always choose exactly one letter; never abstain."

## The programs and the prompts that change

| Cluster | Program, as it ran | Prompts to rewrite |
|---|---|---|
| 0 | self_refine_high: solver (high), then critic (high), then solver (high) | solver, critic |
| 1 | g9_0A1, as it ran: 2 solvers (high), then critic (high); the critic's answer is final | solver, critic |
| 2 | g8_1A2: 2 solvers (high), then 1 blind solver (low), then expert and solver (low) | solver, expert |

- **Cluster 1:** the program as written has a 4th round, a solver, that the turn cap never let run. We use the program as it ran, without that round. Its behavior and its recorded scores stay the same.
- **One prompt per persona per cluster.** The same text applies in every round and at both efforts. Cluster 2's solver speaks at high effort in round 1 and at low effort in rounds 3 and 4.
- **The expert prompt keeps its {field} slot.** The code fills in each question's field.

**The prompts at iteration 0** (P0, from the recorded run):
- Solver: "You are a Solver answering a hard graduate-level multiple-choice question. Reason step by step from your own knowledge, then commit."
- Critic (from the earlier prompt evolution E7): "You are a Critic. Your job is to challenge the leading answer, not to restate it. Check for hidden flaws: misapplied rules, unstated assumptions, overlooked constraints, or an option that better fits the question. Be concise but explicit about why the leading answer may be wrong and what alternative is better supported. If you find a real flaw, change the answer; otherwise keep it."
- Expert: "You are a leading expert in {field}, answering a hard graduate-level multiple-choice question inside your own specialty. Reason step by step from your expert knowledge of the field, then commit."

## The 6 example questions per cluster

- They come from the cluster's train questions only, never from test.
- They come from the program's recorded run 0, so the traces for P0 cost nothing.
- 2 are right and 4 are wrong. They are shown in a seeded random order, the same in every iteration.
- **The 4 wrong ones must be reachable.** The right answer must appear in at least one of our 6 independent recorded replies to that question.
  - In most failures, no speaker in the debate ever gives the right answer. Within the debate itself, this held for 43 of 44 failures in cluster 0, 39 of 44 in cluster 1 and 15 of 24 in cluster 2.
  - A prompt cannot fix those, and GPT-6 would be pushed to write domain facts into the prompts.
  - Reachable failures available: 11 of 44 (cluster 0), 16 of 44 (cluster 1), 13 of 24 (cluster 2).

## What GPT-6 sees

One call per cluster per iteration, to gpt-6-sol at medium effort (the same settings as the program guide). It receives:
- The program in plain words, the fixed parts, and the current prompts.
- The cluster profile from describe_v3: the template, typical steps and failure risks.
- For each of the 6 questions:
  - the question and its options, the right letter, the program's final answer, and whether it was right;
  - for each speaker in order: persona, effort, what it saw (only the question, or the debate), its visible reply, the summary later speakers read, and its letter.
- From iteration 2: the earlier prompts and traces, the new prompts with their new traces on the same 6 questions, and the list of changes so far with their outcomes.

The hidden thinking is not in our recordings. A speaker's thinking runs 8,000 to 30,000 tokens. The traces show what later speakers read, and the prompts mainly shape that.

**Rules for GPT-6:**
1. Return one prompt for each listed persona, as JSON, with one sentence of reasons per change.
2. Write domain-free prompts. Use no facts, terms, numbers, names or option text from the shown questions. A prompt must help any question in the cluster.
3. Never name an answer letter or hint at an answer.
4. Do not mention the quality of a question or its answer key (for example, "the question may be ambiguous").
5. Do not set a reply length.
6. Do not write the ANSWER line, because the code adds it.
7. Keep {field} in the expert prompt.
8. Results on 6 questions are noisy: a question changes outcome between two runs of the same prompts about 13% of the time. Judge a change by the reasoning in the traces, not by the score.

**Checks on GPT-6's output** (on failure, ask once more, then keep the earlier prompts):
- valid JSON with exactly the listed personas;
- no ANSWER line;
- {field} present in the expert prompt;
- no 3-word phrase and no rare term taken from the 6 questions or their options.

## One iteration

1. GPT-6 writes prompts P_i.
2. The program runs with P_i on the same 6 questions: 1 run, fresh replies, about 30 model calls per cluster.
3. Everything is saved: the request, the reply, the prompts, the traces and the scores.

There are 3 iterations. The final prompts are P3.

**Optional guard (recommended):**
- Score P0 to P3 once on the cluster's other train questions (138, 93 and 51), about 0.6 GPU-hours per iteration.
- Keep the best of P0 to P3.
- With noise of ±4 to 7 points, this only catches a prompt that clearly hurts.

## Evaluation on the 1k test

- The 1k test (datasets/supergpqa_2k_test.json) contains our 300 test questions. It shares no question with the 300 search questions.
- Only those 300 have cluster routes.
- The describer prompt that made the clusters (v3) is no longer in the code. The code is now at v4, and git HEAD has v1.

| Option | What runs | GPU time, 1 run | Paid calls | Smallest difference it can show |
|---|---|---:|---|---|
| a. Routed, 300 routed questions | P3 programs only; P0 already has 3 recorded runs | about 1 h | none | about 3.6 points |
| b. Each program on all 1k, no routing | P0 on the 700 new questions, and P3 on all 1k, for each program | about 11 h | none | about 3.2 per program; about 2 pooled over the 3 programs |
| c. Describe and route the 700 new questions | P0 on 700 and P3 on 1k, routed | about 3.5 h | about 700 describer calls (gpt-6-sol) | about 3.2 |

- With 3 runs per question, each option triples in GPU time. The smallest difference then drops to about 2 points.
- Option c also needs the v3 describer prompt rebuilt, or a check of v4 routes against v3 routes on the 300 questions they share.
- **Optional control:** the same programs with the official SuperGPQA instruction as the solver's role text. This shows whether GPT-6's prompts beat a simple, known fix.
- The Qwen3.5-9B paper baselines on the 1k test (direct, self-consistency, Self-Refine, MAD) do not exist yet. They need about 20 GPU-hours, and the paper needs them anyway.

## Code changes (small; off by default)

- `run_program(..., prompt_overrides=None)` merges per-persona prompts into the per-question prompts, filling {field} per question.
  - The round cache key already hashes these prompts, so new prompts get new recordings.
  - With no overrides, nothing changes: the keys stay the same and the v3 test numbers replay exactly.
- A new script, `scripts/prompt_opt_v3.py`:
  - draws the 6 questions;
  - builds traces from the cache;
  - calls GPT-6 and checks its output;
  - runs the 6 questions and loops 3 iterations;
  - runs the optional guard, then the test evaluation.
  - Output goes to `outputs/prompt_opt_v3/cluster_<g>/iter_<i>/`.
- Tests:
  - the override reaches the executor;
  - the cache key changes with the prompts;
  - with no overrides, results replay unchanged;
  - the output checks catch each rule.

## Cost

- **GPT-6:** 9 calls (3 clusters × 3 iterations), of about 25,000 to 60,000 input tokens each. This uses the paid key.
- **GPU:**
  - iterations: about 30 minutes in all;
  - guard: about 2 hours;
  - test: 1 to 11 hours with 1 run, depending on the option.

## Expectations and risks

- **The expected gain is small.** With reasoning on, prompting techniques differ by at most 0.51 points (ReasonLab). Prompt optimization on Qwen3.5-9B gave an average +1.4 (MAS-PromptBench). Our own prompt effect, about 3 points, is our Solver prompt against the official prompt, and it is the same in every cluster.
- **Overfitting:** 6 questions and 3 rounds of edits can fit noise. The guard and the paired test on 1k questions address this.
- **What GPT-6 cannot see:** with only visible replies, it cannot see where the hidden reasoning went wrong.
