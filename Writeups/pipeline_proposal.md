# Pipeline proposal: a debate program per kind of question

Last updated 2026-09-18. Steps 1 to 5 are built and running; steps 6 and 7
are designed but not built. Related work is in `literature_review.md`.

## The idea in one paragraph

A multi-agent debate is a sequence of rounds in which copies of one model
play roles: solvers answer, critics attack, verifiers check. Something has to
decide after each round what happens next: another round, a different role, or
stop and read off the answer. We write that decision-maker as a short, readable
list of if-then rules (a "program") and search for good programs by trying many
and keeping the best. Our earlier search looked for one program for every
question. The winner was the one with the best average, the average is
dominated by the most common kind of question, and the search collapsed: its
top five programs were the same program written five ways. The change here is
to sort questions by how they are solved, and to keep the best program for
each kind.

## The data

SuperGPQA multiple-choice questions, up to ten options, that Qwen3.5-27B got
wrong three times out of three. They are hard by construction.

| Split | Questions | Used for |
|---|---|---|
| Train | 1,515 | describing, grouping, searching, champion picking |
| Dev | 1,515 | the final comparison (step 7), never for selection |
| Strict test | 1,000 | run once, at the very end |

## Overview

| Step | What it does | Status |
|---|---|---|
| 1 | A strong model describes how each question is solved | done |
| 2 | Each description becomes a list of numbers | done |
| 3 | Questions are sorted into six groups | done |
| 4 | 50 questions per group are chosen for the search; the rest are held out | done |
| 5 | One search keeps the best program per group and cost level | running on two models |
| 6 | A new question is sent to its group's program | designed |
| 7 | The routed programs are compared with fair baselines | designed |

---

## Step 1. Describe how each question is solved

`scripts/describe_questions.py`, model gpt-5.6-terra at medium reasoning,
OpenAI Responses API with a fixed output form.

For each of the 1,515 training questions the model fills in:

- **moves**: which of ten thinking moves a careful solver needs, for example
  recalling an exact fact, setting up a calculation, checking each part of a
  multi-part statement;
- **risk**: the single most likely way a good student gets it wrong, from
  seven, for example confusing two similar facts, or applying the wrong model;
- **knowledge**: whether the answer is recalled, derived, or both;
- **template**: one sentence describing the solution path with every subject
  word removed.

Two things keep the labels consistent. The lists of moves and risks were not
written by us: the model was shown 25 batches of 40 questions from across all
fields, asked what separates how they are solved, and its proposals were merged
into ten moves and seven risks. And ten fixed reference examples are included
in every call, so every question is labelled against the same yardstick.

The descriptions are used only for sorting. The debating model never sees
them, so nothing the strong model knows can leak into a debate.

Output: `outputs/question_templates_train.jsonl` (1,515 records, no errors).

## Step 2. Turn each description into numbers

`scripts/embed_questions.py`.

- **Checklist**: twenty 0/1 entries, one per move, risk and knowledge type.
- **Template meaning**: the one-sentence template passed through the
  embedding model `microsoft/harrier-oss-v1-0.6b`, so two templates that mean
  the same in different words come out close.
- **Raw question**: the question text through the same model. Not used for
  grouping, because it carries the topic; kept for routing in step 6.

Output: `outputs/question_vectors_train.npz`.

## Step 3. Sort the questions into groups

`scripts/cluster_questions.py`. Questions with similar checklists and
templates go together. Each group's centre is a real question, so a group can
be read, not just counted.

We tried every number of groups from 2 to 30 and measured, for each: whether
the groups stay the same when the data is resampled, whether any group is too
small to use, and whether a group can be recognised from the raw question
alone. Six was the largest number that stayed stable (0.87 agreement under
resampling) with every group at 100 or more questions. At twenty groups
stability fell to 0.68; at thirty, most groups had under fifty questions.

| Group | Size | What it is | Typical risk |
|---|---|---|---|
| 0 | 381 | recall an exact fact | confusing similar facts |
| 1 | 188 | set up and carry out a calculation | arithmetic or unit slip |
| 2 | 548 | recall a concept and match it to a category | near-miss category |
| 3 | 113 | apply a rule with conditions, or check every part of a compound statement | overlooked qualifier |
| 4 | 100 | combine several clues, as in a clinical case | mixed |
| 5 | 185 | choose the right model, then calculate | wrong model |

The grouping never saw subject or difficulty labels, yet the groups split
cleanly by dataset difficulty and mix subjects freely. That is our evidence
that they capture how a question is solved and not what it is about.

Output: `outputs/clusters_train_both.json` and a readable `.md` report.

## Step 4. Choose the search questions

Each group's questions are listed in a fixed order: the centre question first,
then alternately the question farthest from everything chosen so far and a
random one. The search uses the **first 50** of each list, 300 questions in
all, an 80% cut from the 1,515. Because the order does not depend on how many
are taken, the search can later be widened to 75 or 100 per group and every
recorded round is reused.

Everything else in a group is held out for champion picking: 331, 138, 498,
63, 50 and 135 questions.

---

## Step 5. The search

`scripts/program_seeds.py`, `scripts/evolve_program_clusters.py`, with
`scripts/program_space.py` (programs, mutation, distances) and
`scripts/program_guide.py` (the guide model). Offline test:
`tests/test_program_clusters.py`.

### 5.1 What a program is

An opening plan of one to five rounds, an ordered list of rules, and a default
way of stopping. After every round the rules are checked from the top and the
first that matches decides the next action.

- **Conditions** look only at the debate so far, never at the answer key:
  rounds run, size of the opening round's majority, number of distinct letters,
  whether the last two speakers agree on a switch, whether the last round
  agreed, what ran last.
- **Actions**: next planned round, critic, verifier, two fresh solvers who see
  nothing, a blind expert, an expert-solver pair, synthesizer, or stop.
- **Stop reads**: the most recent letter, the last round's letter, or a
  plurality over every letter committed.

Cost is counted in speaker turns, capped at 16 per question.

### 5.2 How a speaker call is made (the context fix)

The earlier pipeline lost about one reply in five to truncation. Now:

- each reply may be up to 6,144 tokens;
- after a long reply, a short second call asks the same speaker to summarise
  its reasoning in 120 words and restate its letter; later speakers are shown
  the summary. If the main reply was cut before committing, the summary's
  letter is the commitment;
- a short reply that already commits is its own summary and is shown in full
  (first 300 and last 900 characters if long);
- speakers within one round are called at the same time, since they never see
  each other.

For models that reason in a hidden channel (gpt-oss), an off-by-default switch
adds one sentence asking for the reasoning in the visible reply. Without it the
visible reply was a bare "ANSWER: X" and critics had nothing to read.

All of these settings are part of each recording's identity, so rounds recorded
under different settings can never be replayed as if they were the same.

### 5.3 Scoring by replay

A recorded round is identified by the question, the exact sequence of round
types run so far, and the replicate number. Not by the program. Any two
programs that walk the same sequence on a question share those rounds, and a
child pays only on questions where its edit changed the path, and only from the
round where the paths split.

No recording from any earlier experiment is used. Each model has its own cache
file, started empty.

A new program is first replayed for free. If it has gaps:

- with no cached path at all (a new opening round, a random immigrant) it gets
  a first look of up to 300 speaker turns, spread across the groups;
- otherwise it is filled only if its projected score in some group is within
  two points of that group's current best at its cost level;
- partly scored programs are reconsidered at the start of each of the next
  five generations ("top-up"). Missing questions count as wrong.

### 5.4 Two replicates

Sampling means the same program on the same question can end differently on a
rerun. Replicate 0 screens everything. Replicate 1, a fully fresh rerun, is
spent only where a decision depends on it:

- any program that would be a cell's best, or in a group's top five, is rerun
  on that group's 50 questions;
- any program that is one of at most three to solve some question is rerun on
  that question.

A question's mark is the mean over its replicates, so 0, ½ or 1.

### 5.5 The seeds: 20 programs

- **8 protocols from the literature**, written fresh: direct answer;
  self-consistency (four solvers, vote); multi-agent debate (three solvers,
  three rounds, vote); self-refine (solver, critic, solver, critic, solver);
  verify then decide; early exit on agreement; fresh voices on disagreement;
  expert first.
- **4 written by gpt-5.6-luna** in one call, shown the eight and the six group
  profiles and told to differ from both.
- **8 random**, from a pool of 100: a ten-question sanity run drops programs
  that stop at once or run to the cap (mean turns outside 2 to 12), then the
  eight farthest in structure from the twelve fixed seeds are taken.

Nothing from earlier experiments is a seed.

### 5.6 What is kept: cells, not a leaderboard

A cell is a group crossed with a cost level: cheap (under 4 turns per
question), medium (4 to 8), expensive (over 8). Eighteen cells, each holding
its best program. Every program ever scored stays in the archive; nothing is
deleted. Two programs that took the same path and gave the same letter on all
300 questions count as one individual, and the later one is never selected.

### 5.7 Each generation: 18 children

| Slots | Parent chosen by | Purpose |
|---|---|---|
| 6 | cell bests, rotating through the 18 cells | quality per group and cost |
| 5 | lexicase: shuffle the questions, keep only the programs best on the first, then the second, and so on | keeps specialists that win a few questions nobody else wins |
| 5 | farthest, in structure and behaviour combined, from the parents already chosen, among programs within 8 points of a cell best | diversity |
| 2 | fresh random programs, farthest in structure from the archive | new material |

No more than a quarter of a generation's parents may descend from one seed.

Distance between two programs is the average of two parts. **Structural**: how
little their building blocks overlap (opening rounds, kinds of condition,
actions, stop reads). **Behavioural**: the share of questions where they took
different paths, averaged with the share where they gave different letters.

### 5.8 Mutation

Seven kinds of edit: change a rule's action, add a rule, change a condition,
change the default stop, drop a rule, swap two rules, or change the opening
plan.

Two thirds of children get a random edit. One third get a guided one: the
search draws the kind of edit first, then gpt-5.6-luna is shown the parent, its
score in every group, the target group's profile, and digests of five failures
and two successes (round by round, who said which letter, and the correct
one). Failures that were right after the opening round and lost later are
shown first. Because the search picks the kind of edit, the model cannot
simply add rounds; when the draw is "drop a rule", its job is to choose which.
A returned edit of the wrong kind is rejected and replaced by a random one.

A child that behaves exactly like a program already scored is mutated again,
up to three times. This costs nothing, since identical behaviour means every
round was already recorded.

### 5.9 Stopping and champions

Up to 40 generations; the search stops early if no cell improves for 10.

Then, for each group: its cell bests plus the next best, five finalists with
distinct behaviour, each run on up to 150 of the group's held-out questions at
two replicates. The champion is the best held-out mean, ties to fewer turns. A
global champion is picked the same way over all 300, for the comparison in
step 7.

### 5.10 Outputs and resuming

Per run directory: `archive.jsonl` (every program with per-question results at
both replicates), `generations.jsonl` (per generation: children by kind of
edit, guided against random, neutral edits, new bests by source, time, calls),
`summary.json`, `champions.json`. Every step resumes from disk. Widening to
more questions per group is a resume with a larger `--per-group`.

---

## Step 6. Sending a new question to its program (to build)

Two routes, both to be measured:

- **Exact**: describe the new question with the same strong model and form as
  step 1, and assign it to the nearest group centre. One extra short call per
  question.
- **Cheap**: a small classifier on the raw question's embedding. On the
  training questions it recovers the group about 70% of the time, against 36%
  for always guessing the largest group.

## Step 7. The comparison (to build)

On the 1,515 dev questions, fresh runs, every method at the same number of
calls:

1. one global champion for every question;
2. six group champions with the router;
3. six group champions with questions sent to a **random** group;
4. six group champions with the **true** group (what a perfect router gives);
5. the named protocols from the seed set, direct answer included.

The central number is the gap between 2 and 3: it separates "having six
programs helps" from "sending the question to the right one helps". The strict
test set is run once, after everything else is fixed.

---

## Where things stand (2026-09-18)

Two searches are running side by side, with nothing shared on disk except the
read-only questions, groups, and a copy of the same 20 seeds.

| | Qwen3.5-35B-A3B-FP8 | gpt-oss-20b (low reasoning) |
|---|---|---|
| Generation | 4 | 4 |
| Time per generation | 25 to 60 minutes | about 10 minutes |
| Direct answer, 300 questions | 24.3% | 14.3% |
| Range across the 20 seeds | 23.3% to 27.0% | 10.7% to 16.0% |

What the seeds already say:

- **On Qwen, no standard protocol is distinguishable from a single call.**
  With 300 questions the noise is about ±2.5 points, and all twenty seeds fall
  inside it.
- **gpt-oss-20b is close to chance** (about 10% with ten options). Medium
  reasoning was tested on 90 questions: four times the tokens, 7% of replies
  cut off, the same accuracy. This run serves as a negative control.
- **The context fix works**: no failed summary calls; 11% of Qwen's main
  replies ended without a letter and were recovered by the summary.
- **The search behaves as designed**: parents from a dozen lineages per
  generation, guided edits valid and removing rules on net, neutral edits
  caught for free.

## Open questions

1. Whether any group's champion beats direct answer on held-out questions by
   more than the noise. If none does, the honest result is that debate does not
   help this model on these questions, broken down by kind of question.
2. Whether the two calculation groups (1 and 5) need different programs.
3. Whether group 4, the smallest, is a real kind or a leftover.
4. Whether the cheap router is good enough, or the extra call is needed.
5. Whether 50 questions per group is enough to tell programs apart; widening to
   100 is one command and reuses everything.

## Commands

```bash
# steps 1-4 (done)
python scripts/describe_questions.py --out outputs/question_templates_train.jsonl
python scripts/embed_questions.py
python scripts/cluster_questions.py --rep both --out outputs/clusters_train_both.json

# step 5: seeds, search, champions in one resumable script
bash run_cluster_search.sh          # Qwen,    outputs/cluster_search/
bash run_cluster_search_gptoss.sh   # gpt-oss, outputs/cluster_search_gptoss/

# widen later, reusing every recording
python scripts/evolve_program_clusters.py --out outputs/cluster_search/run1 --resume --per-group 100

# offline test, no model needed
python tests/test_program_clusters.py
```
