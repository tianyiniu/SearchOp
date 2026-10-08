# Pipeline v4: findings and plan (3 October 2026)

## Part 1. The project in short

### The goal

We want a method that answers hard questions better than one model alone. The method uses several calls to the same model. The calls talk to each other in rounds. This is a "debate".

The goal is to find debate plans that beat a strong simple method, "self-refine", by 3 to 5 points of accuracy.

### The data and the model

- **Model:** gpt-oss-20b, an open model. All speakers in a debate are calls to this model.
- **Effort:** gpt-oss-20b has a setting for how long it reasons before it answers. "High effort" reasons for a long time. "Low effort" reasons for a short time.
- **SuperGPQA:** graduate-level multiple-choice questions, usually with 10 options. Up to v3, we used a train split of 300 questions and a test split of 300 questions.
- **HLE (Humanity's Last Exam):** very hard questions. Most of them need a free-text answer. We use a train split of 800 questions and a test split of 200 questions. A separate judge model (gpt-6-luna) grades the free-text answers.

### Terms

| Term | Meaning |
|---|---|
| Speaker | One call to the model in a debate. A speaker has a role, for example "solver" (answers the question) or "critic" (reviews the earlier answers and gives its own answer). |
| Round | A set of speakers that speak at the same time. Speakers in later rounds can see the earlier answers. |
| Program | A plan for one debate. It has a list of rounds, a list of rules, and a final read. |
| Rule | A condition and an action. Example: "If all answers in round 1 are the same, stop." |
| Final read | The method that selects the final answer when the program stops. Example: "vote" takes the most common answer. |
| Turn | The unit of cost. A high-effort speaker counts as 3 turns. A low-effort speaker counts as 1 turn. One question can use at most 15 turns in v4 (20 in run3). |
| Self-refine | A simple method: one solver answers, one critic reviews the answer, and the solver answers again. |
| Seed | A program that the search starts with. Some seeds come from the literature. A strong model (gpt-6-sol) writes other seeds. |
| Search | An evolutionary search. In each generation, it changes the best programs to make new programs. It runs the new programs on the train split and keeps the best ones. |
| Train split | The questions that the search uses to score programs. |
| Dev split | Questions that the search never sees. We use them only to select the final programs. |
| Test split | Questions that we use only to measure the final result. |
| Test accuracy | The percentage of correct answers on the test split, averaged over 3 runs. |
| ± | The standard error of a difference, paired by question. A difference smaller than 2 times this number can be noise. |

### The pipeline up to v3

1. **Describe:** a model writes a description of each train question.
2. **Cluster:** the descriptions become number vectors. Similar vectors go into one group. SuperGPQA has 3 groups. HLE has 4 groups.
3. **Search:** the search finds the best program for each group, and one best program for all questions.
4. **Route:** at test time, each test question goes to the nearest group. It runs the program of that group.

## Part 2. What we found

### Finding 1. The search does not beat self-refine

| Method (SuperGPQA, test) | Test accuracy | Tokens per question |
|---|---:|---:|
| Run3 (cluster search, routed) | 46.9 | 42k |
| Run2 (cluster search, routed, smaller budget) | 48.3 | 12k |
| Self-refine (run separately from the search code) | 48.1 | 29k |
| Self-refine, high effort (in the search code) | 48.0 | 28k |
| 1 high-effort solver | 46.6 | 11k |

- On HLE, the search got 12.7%. Self-refine with high effort got 14.0%.
- Run2 gets the same accuracy as self-refine with about 40% of its tokens.

### Finding 2. The correct answer is often in the debate, but the debate does not select it

- In run3, the correct answer appears at least one time in 58.1% of the debates. The final answer is correct in only 46.9%.
- Most of this difference has one cause. Only one speaker gave the correct answer, and two or more speakers agreed on one wrong answer.

### Finding 3. A better judge gives only a small gain (the judge experiments)

A "chooser" is a method that selects the final answer from the answers of the speakers. We tested choosers on recorded answers, so every chooser saw the same answers.

| Chooser | What it does | Result |
|---|---|---|
| Vote | Takes the most common answer. | The baseline. |
| Judge | One high-effort call reads all answers and selects one. | About +2 points over the vote. About 49% test accuracy. Not consistent across question sets. |
| Pairwise | Compares two answers at a time, in both orders. | About +2 points over the vote. Consistent. Costs about 3 times more tokens than the judge. |
| Pairwise, top 2 answers, one order | Compares only the two most common answers, in one call. | The same gain as full pairwise, at the cost of one call. |
| Low-effort pairwise | Pairwise with low-effort calls. | No gain. Correct in only 54 to 59% of comparisons. |
| Blind judge, re-solve, verify each answer | Other designs. | No gain over the judge. |

- The gain over self-refine is about 1 point. This is within noise.
- When the high-effort speaker is wrong and the low-effort speaker is correct, the judge selects the correct answer in only 18% of cases.

### Finding 4. When only a minority has the correct answer, the debate text does not show it

We read 100 debates in detail. In each debate, the correct answer was present.

- When most speakers gave the correct answer, a careful reader could find it in 38 of 56 debates.
- When only a minority gave the correct answer, a careful reader could find it in only 4 of 44 debates.
- Half of the wrong answers (67 of 136) came from a wrong fact or formula. A judge that is the same model cannot supply a fact that the model does not know.
- Some signals marked wrong answers. These signals were strong only when the majority was already correct:
  - The result of the speaker matches a different option.
  - The speaker says that another option is also correct.
  - The speaker adds data that the question does not give, or ignores data that the question gives.

### Finding 5. The final read had a flaw

- Some run3 programs stopped when 3 of 4 solvers agreed. Then they took the answer of the last speaker, not the majority answer.
- In 50 of 166 such debates, the last speaker was the one who disagreed.
- A repair: "last answer" means the majority answer of the last round. With one speaker in the last round, nothing changes.
- With the repair, run3 gets 48.0% instead of 46.9% (+1.1 ± 0.7). Self-refine does not change.

### Finding 6. The text groups do not help

- Routing by group was not better than a random group: −0.6 ± 0.6 points on SuperGPQA, and −1.2 ± 0.7 on HLE.
- Some questions really do suit some programs better. These "outcome groups" repeat between two runs.
- But nothing visible before the answer predicts them:
  - The question text predicts them no better than always guessing the largest group.
  - The first round of the debate predicts them no better either.
  - Routing by the first-round pattern gave +0.0 ± 0.6 points.
- The first round predicts how hard a question is. When all 4 solvers agree, programs are correct about 65% of the time. When the answers are split, programs are correct 20 to 45% of the time.
- Earlier experiments agree:
  - Hand-written cases ("Method 3") gave 14.8% against 14.5%.
  - A separate search for each question found paths that mostly did not repeat.

### Finding 7. Expensive programs give little

- On easy questions, cheap programs are as good as expensive programs.
- 1 high-effort solver: 46.6% at 11k tokens. Run3's best program: 47.1% at 48k tokens.

### Finding 8. The search scores are too high

| Program | Selected by | Train | Test |
|---|---|---:|---:|
| grid_1 | the search | 54.2 | 47.1 |
| grid_0 | the search | 53.7 | 47.4 |
| grid_2 | the search | 51.0 | 48.0 |
| self_refine_high | fixed seed | 47.7 | 48.0 |
| direct_high | fixed seed | 47.2 | 46.6 |

- The search compares many programs on the same questions. It selects the program that was lucky. On new questions, the luck goes away.
- A simulation with a held-out dev split reduced the error of the reported score from 4 points to 2 points. It also selected slightly better programs: 52.0 against 51.2 on unseen questions.

## Part 3. The v4 changes

Status: **Decided** means the user agreed. **Proposed** means the user did not decide yet.

| # | Change | Reason (finding) | Status |
|---|---|---|---|
| 1 | Remove the describer, the clustering, the routing and the per-group slots. | 6 | Decided |
| 2 | The search makes programs for all questions. The rules after round 1 make the groups. | 6, 7 | Decided |
| 3 | Keep the best programs at 3 cost levels: cheap, medium and expensive. | 7 | Decided |
| 4 | Measure cost in turns: high effort = 3 turns, low effort = 1 turn. | 7 | Decided |
| 5 | Level limits use the average turns per train question. Cheap: up to 4.5. Medium: up to 10.5. Expensive: more than 10.5 (the cap is 15). | 7 | Decided |
| 6 | Split the 300 train questions into a new train split (200) and a dev split (100). The search never sees the dev split. | 8 | Decided |
| 7 | At the end, for each level: run the top 2 programs and the best seed on the dev split. Keep the best. | 8 | Proposed |
| 8 | Repair the final read: "last answer" means the majority answer of the last round. The repair applies in the search and in the test. It is a setting: on for v4, off for older runs. | 5 | Decided |
| 9 | Remove the judge speaker and the 2 judge seeds. | 3 | Decided |
| 10 | The seed writer sees 10 example questions from the train split, with no group profile. | 1 | Proposed |
| 11 | Each generation makes 5 new programs for each level (15 in total). | 3 | Proposed |
| 12 | No second run on the train split. The dev split replaces it. | 8 | Proposed |
| 13 | Report accuracy, tokens, and the share and accuracy of the questions that took each rule. | 2 | Proposed |
| 14 | No new chooser rounds (no high-effort judge, no pairwise). | 3, 4 | Proposed |
| 15 | Run v4 on SuperGPQA only. HLE waits for a later run. | 1 | Decided |
| 16 | Lower the turn cap from 20 to 15 turns per question. In run3, the turns past 15 bought nothing: replayed under a cap of 15, the 18 programs that went past 15 lost 0.2 points on train, and the final programs did not change on test. | 7 | Decided |

### What does not change

- The program grammar, the types of changes, and crossover.
- The literature seeds.
- The 32k context window for every model.
- The summaries between rounds.
- 10 generations.
- The test protocol: 3 runs on the 300 questions of the test split.

## Part 4. The global-pipeline experiment, step by step

### Step 1. Make the splits

1. Take the 300 questions of the v3 SuperGPQA train split.
2. Select 100 at random, with a set random seed. These are the dev split.
3. The other 200 are the new train split.
4. The search never sees the dev split.
5. The test split (300 questions) does not change.

### Step 2. Make the seeds

1. Use the same 8 literature seeds as v3. Five are high-effort: direct answer, self-refine, self-consistency, `verify_then_decide`, and new solvers on disagreement. Three are low-effort: debate (mad), early exit on agreement, and expert first.
2. Remove the judge seeds.
3. The seed writer (gpt-6-sol) writes 3 more seeds, one for each cost level. It sees 10 example questions drawn at random from the train split. It does not see answers, accuracy or difficulty.
4. Every seed must run at least one round, and it must stay below the turn cap. Reject a seed that does not.

### Step 3. Score the seeds

1. Run each seed one time on the train split (200 questions).
2. Record its accuracy and its average turns per question.
3. Put each seed into its cost level by its average turns:
   - Cheap: up to 4.5 turns.
   - Medium: more than 4.5 and up to 10.5 turns.
   - Expensive: more than 10.5 turns (at most 15).

### Step 4. Run the search (10 generations)

In each generation:
1. For each cost level, take the best program of that level as the parent.
2. Make 5 new programs from each parent with the usual changes. Examples: add or remove a round, change a rule, change a speaker, change the effort, or crossover.
3. Run each new program one time on the train split.
4. Measure its accuracy and its average turns.
5. Put it into the level of its average turns. A change can move a program to a different level.
6. Each level keeps its best program as the next parent.

### Step 5. Select the final programs on the dev split

1. For each level, take the top 2 programs by train accuracy and the best seed of that level. That is 9 programs.
2. Run them one time on the dev split (100 questions).
3. In each level, keep the program with the best dev accuracy.
4. The result is 3 final programs: one cheap, one medium and one expensive.

### Step 6. Test

1. Run the 3 final programs on the test split, 3 times each.
2. Also run the baselines with the same settings: 1 high-effort solver, self-refine with high effort, and self-refine run separately from the search code.
3. Use the repaired final read for all programs. The search also used it, so the search and the test score programs in the same way. The repair applies to both "last" reads: `stop:last_commit` and `stop:last_speaker`.

### Step 7. Report

For each final program and each baseline:
- Test accuracy and its standard error.
- The paired difference from self-refine.
- Tokens and turns per question.
- For each rule of the program: the share of test questions that took the rule, and their accuracy. These are the groups that the search found.

Also report one graph: test accuracy against tokens, with one point for each final program and each baseline.

Do not report the train or dev accuracy as results. They contain selection luck.

## Part 5. What we expect, and how we judge the result

### Expected result

- Accuracy about the same as self-refine, between 47 and 49%. No method in our tests went higher with this model.
- The cheap program should cost much less than self-refine. Run2 already showed 40% of the tokens at the same accuracy.

### The experiment succeeds if one of these is true

1. One final program reaches the accuracy of self-refine (within noise) with fewer tokens.
2. One final program beats self-refine by more than 2 standard errors.

### The experiment tells us something even if it fails

- If the cheap program is much worse than self-refine, cheap debates do not work with this model.
- If the rules send almost all questions down one path, the search did not find useful groups.

## Part 6. Open questions and risks

### Decisions made on 3 October 2026

1. Dev split: 100 questions.
2. The final-read repair: yes, in the search and in the test.
3. The judge speaker and the judge seeds: removed.
4. HLE: not in v4. A later run can use more train questions than run1 (50 per group was too few).

### Decisions still open

The user did not decide changes 7 and 10 to 14 in Part 3 yet. If nobody objects, v4 uses them as written.

### Risks

- **Noise:** a train split of 200 questions gives noisy scores. The dev split reduces the luck, but it cannot remove it.
- **The model's limit:** half of the wrong answers are wrong facts. No debate structure can fix these with one model.
- **Turns and tokens:** turns make low-effort programs look more expensive than they are. A high-effort speaker uses about 19 times more tokens than a low-effort speaker, but counts only 3 times more. Thus, low-effort programs can be placed one level too high.
- **Safety of the code:** new options must be off by default, so earlier runs replay without change.

### Ideas kept for later (not in v4)

- An extraction step with low-effort calls. It asks two questions about each argument. First: "Which option does its own result match?" Second: "Does it say that another option is also correct?" Expected gain: 1 point or less.
- Pairwise comparison with medium effort. Not tested.
- Speakers written for a type of question ("specialists"). This needs groups that come from the text, and v4 does not use text groups.

## Part 7. Implementation (3 October 2026)

### How to run

```bash
mkdir -p outputs/pipeline_global_gptoss/run1
bash run_pipeline_global.sh gptoss 2>&1 | tee -a outputs/pipeline_global_gptoss/run1/pipeline.log
```

Run it inside tmux on the server that hosts gpt-oss-20b, so it keeps going if the SSH session closes. The seed stage calls gpt-6-sol, so `OPENAI_API_KEY` must be in `.env`.

If the run stops, run the same command again. The script keeps finished work. A step that cannot finish stops the script with a message, for example when the model server is down. Fix the cause, then run the command again.

### Files

| File | Content |
|---|---|
| `run_pipeline_global.sh` | The whole v4 experiment, steps 1 to 7. |
| `scripts/split_train_dev.py` | The train and dev splits. |
| `scripts/program_seeds_global.py` | The seed stage. |
| `scripts/evolve_pipeline_global.py` | The search, and the final choice on the dev split (`--select-dev`). |
| `scripts/eval_pipeline_global.py` | The test evaluation, the paths of each final program, and the graph. |
| `tests/test_pipeline_global.py` | The offline tests. |
| `scripts/evolve_program_mcq.py`, `scripts/program_space.py` | The final-read repair (`--last-round-vote`, off by default). |
| `scripts/program_guide.py` | The seed writer for cost levels. |

### Outputs (in `outputs/pipeline_global_gptoss/run1/`)

| File | Content |
|---|---|
| `splits.json` | The train and dev splits. |
| `seeds.json`, `seeds.md` | The seeds. |
| `archive.jsonl`, `generations.jsonl`, `summary.json` | The search. |
| `final.json`, `final.md` | The final programs and their dev scores. |
| `test_eval/results_k3.md`, `test_eval/accuracy_vs_tokens.svg` | The test results and the graph. |

### Tests before the first run

- The 55 new offline tests pass. The 8 existing test suites pass.
- With the repair off, the run2 and run3 test evaluations replay exactly, and their gap reports are byte-identical.
- Under the v4 settings, the 8 literature seeds replay from the v3 cache on all 200 train questions. The search starts with them at no cost.
- Under the v4 settings, the test baselines replay from the run3 test cache with the same marks as before.
