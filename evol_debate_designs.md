# Debate-side designs: measurements, algorithms, and current results

> Written 2026-08-09 as a companion to `evol_lit_search.md` (which reviewed the literature and
> ranked candidates C1–C10). Updated 2026-08-24 and later reorganized so the current state comes
> first.
>
> **How this document is ordered.** The summary and glossary come first. **Part 5 is the current
> state of the project** — the two per-question experiments, every result, and what to run next.
> Parts 0–4 follow as the record of the earlier global-recipe phase, kept with their original
> numbering so cross-references like §0.9.1 still resolve. New readers: read the summary, then
> Part 5. Read Parts 0–4 only when you need to know why a decision was made.

---

## Where the project stands (summary, 2026-08-24)

The goal: a small model (Qwen3-14B) fails hard multiple-choice questions. We make it debate with
itself — several calls with different instructions, reading each other — to rescue answers it
gets wrong. Early work searched for **one debate recipe that works best on average**. The best
such recipe scores **14.5%** on our 3,030-question fail set (questions the model got wrong 3 out
of 3 times; asking directly scores ~3.9%).

The project's real thesis is that **different questions need different debates**. Two experiments
now test that directly:

- **Experiment A (`scripts/treegrow_mcq.py`)** builds a debate for each question from scratch,
  one round at a time, using tree search steered by the answer key. Every winning recipe is
  re-run 3 times to weed out luck. Result: **19.2–19.7%** of questions have a *verified* recipe
  of their own, versus 14.5% for the best single recipe — and half of those wins are questions
  the single recipe cannot solve. Without the luck check the number would have been a misleading
  49.9%.
- **Experiment B (`scripts/adaptive_debate_mcq.py`)** runs the best single recipe but lets simple
  rules trim it short or extend it per question, based only on the answers committed so far — no
  answer key needed at run time. Result: **14.8% vs 14.5%** overall (a wash), but one rule (stop
  early once a corrected answer is confirmed) is a clear win, and one rule (append a verifier to
  undecided debates) clearly hurts and should be removed.

The deployable version of experiment A — a controller that picks debate moves without the answer
key, trained on 150,530 recorded search decisions — is built and tested but **not yet run**. That
is the main open number.

## Glossary

Terms used throughout, especially in the historical parts:

| term | plain meaning |
|---|---|
| schema / genome | a debate recipe: which speakers talk, in what order, seeing what |
| departure | a run "departs" when its final letter differs from the base model's answer |
| departure rate (D) | how often a recipe changes the model's original answer |
| departure precision (P) | when it changes the answer, how often the change is correct |
| the master equation | accuracy ≈ D × P, on this fail-only dataset |
| execution | one full run of a recipe on one question |
| paired / prefix batches | every candidate recipe is tested on the same questions, so scores compare fairly |
| LCB | lower confidence bound — a score discounted for small sample size, so lucky small runs don't win |
| PCFG / EDA / CEM (E5) | a search that learns a probability table over recipe parts and samples from it |
| MAP-Elites (E3) | a search that keeps the best recipe in each cell of a behavior grid, not one global best |
| reflective persona-text evolution (E7) | a search where an LLM reads failed debates and rewrites one speaker's instructions |
| architect | an LLM that proposes edits to a recipe |
| retention pool | questions the model already answers correctly — used to check a recipe does no harm |

---

## Part 5 — The per-question phase (2026-08-22 → 08-24) — CURRENT

Everything in this part was measured on `datasets/supergpqa_strict_train.json`: 3,030 questions
the model (Qwen3-14B, thinking off, temperature 0.7) answered wrong 3 out of 3 times during
filtering. Asking directly scores ~3.9%.

### 5.1 The pivot, and the contrarian ban

Two decisions ended the global-recipe phase:

1. **The project's thesis is a custom recipe per question.** Every search so far had produced one
   global recipe. The new experiments had to be per-question by construction.
2. **The contrarian persona was banned.** The contrarian is forbidden from keeping the current
   answer. On a fail-only dataset that is free points; on real questions the model gets right, it
   is forced error. Every recipe leaning on it (the E3 winner at 25.0%, both extended-E5 winners)
   was disqualified. The closest production-legal substitute is the E7 rewritten critic, whose
   prompt ends "if you find a real flaw, change the answer; otherwise keep it."

### 5.2 New machinery

- **Two new personas** in `debate_mcq.py`. The **verifier** does not re-solve; it tests each
  committed letter against the exact wording of the question, then keeps or switches. The
  **expert** is a solver whose system prompt names the question's field (from dataset metadata) —
  the only per-question prompt. Neither was added to the old searches' gene pools, so their
  caches stay valid.
- **A round-level cache** (`RoundRunner` in `schema_fitness.py`), keyed by (question, exact
  sequence of rounds run so far, replicate number), storing raw responses. Tree branches share
  their parent's rounds for free; a trimmed recipe is a cache prefix of the full one; killed runs
  resume mid-question; replicate numbers give independent re-runs of the same recipe.
- The executor was split into run-one-round and read-off-the-answer pieces so debates can execute
  incrementally. A 69-check offline test suite (`scripts/test_treegrow_adaptive.py`) proves the
  refactor reproduces the legacy executor exactly, and scripts both experiments' decision logic
  end to end with a stubbed model.

### 5.3 Baselines on this split (from `outputs/bestofn_strict_cache.jsonl`)

| method | calls | needs answer key? | accuracy |
|---|---|---|---|
| ask once | 1 | no | 3.7% |
| majority vote over 7 samples | 7 | no | 2.3% — *below* asking once; the model's consensus is wrong by construction |
| classic multi-agent debate | 7 | no | 3.1% |
| best-of-3 / 5 / 7 (key picks the winner) | 3/5/7 | yes | 8.9% / 12.9% / 16.9% |
| best global recipe (E7 winner) | 8 | no | **14.5%** |

The 14.5% recipe — 4 solvers (blind, parallel) → 1 solver who reads all four → 3 rewritten
critics, answer = last letter — is the bar every per-question method must beat, and experiment
B's starting point. It scored 16.7% on the 192 questions it was selected on: ~2 points of
selection optimism, now measured.

### 5.4 Experiment B — one master recipe, trimmed or stretched live (`adaptive_debate_mcq.py`)

**The method.** Run the master recipe one round at a time. After every round, plain code (no
model calls, no answer key) reads the sequence of committed letters and decides: stop, continue,
or append extra rounds. Fully deployable as-is — the answer key is used only to grade results.

- *Baseline letter* = the most common letter among round 1's four solvers.
- *Stop early:* the last two speakers (both after round 1) agree on a letter that differs from
  the baseline — a corrected answer, confirmed. Output it; later critics, who might wander off
  it, never run.
- *If the recipe ends undecided:* **parked** (the answer never moved) → append fresh voices that
  never saw the stuck answer: a blind eliminator, then an expert + solver who see only its
  survivor list (if they agree — even with the baseline — that's the answer), then a verifier if
  they disagree. **Churning** (the answer moved but was never confirmed) → append a verifier
  (stop if it backs any already-committed letter), then a synthesizer forced to pick among the
  candidates.
- The un-adapted 8-call recipe runs afterwards as a paired control; through the shared cache it
  only pays for rounds the adaptive pass trimmed.

**Results** (all 3,030 questions, 25,959 calls, average 8.2 calls/question, 0 errors):

Overall: **adaptive 14.8% vs fixed 14.5%** — 43 wins / 33 losses head-to-head, not significant.
The per-rule breakdown carries the information:

| track | n | adaptive | fixed | head-to-head | verdict |
|---|---|---|---|---|---|
| settled early (trimmed) | 1,682 | 21.3% | 20.3% | +26 / −10 | **the stop rule works** (significant) |
| settled on final round | 203 | 23.6% | 23.6% | 0 / 0 | identical by construction |
| parked (fresh voices) | 669 | 2.4% | 0.9% | +12 / −2 | small real gain; stuck mostly stays stuck |
| churning (judges) | 476 | 5.7% | 9.7% | +3 / −22 | **hurts — delete this rule** |

The churn failure: on 22 questions the fixed recipe's last critic was right, and the appended
verifier overrode it by "backing" an earlier wrong letter. Removing that one rule projects to
~15.4% vs 14.5% — but that projection was made by looking at graded results, so it must be
re-earned on the held-out split. Closers used: fresh-voices agreement 518, verifier-backed 464,
verifier 151, synthesizer 12.

Bonus finding: the letter pattern is a free difficulty signal. It splits the set into a 62% slice
scoring ~21.5% and a 38% slice scoring 2–6%, using no labels.

**Two autopsies with deployment consequences:**

- *The blind eliminator is still bad.* Across its 574 parked-track runs, the correct answer
  survived its cut only 22% of the time, and 60% of its "survivor lists" contain a single
  option — it answers while claiming not to. It nets positive here only because parked questions
  were headed for a wrong answer anyway.
- *When the parked answer was already correct* (6 cases — rare only because this split is
  fail-only): the extension kept it 4 times and destroyed it twice. Both destructions came
  through the fresh-voices agreement rule (the pair agreed inside the eliminator's poisoned,
  gold-free context); all four saves came through the verifier, which sees the full debate. On
  real question mixtures, easy questions park on correct answers constantly, so this failure
  mode scales. **Rule fix before any deployment claim: never send the rescue crew into a debate
  that was unanimous from start to finish.**

### 5.5 Experiment A — build a debate per question with tree search (`treegrow_mcq.py`)

**The method.** A tree node is a state: the entire executed debate so far, real text included. An
edge appends one move. Moves: add a critic (the rewritten one); add a verifier; add two blind
independent solvers ("fresh"); add a blind field expert; eliminate-then-solve (an eliminator
crosses out options, a solver picks from the survivors); or stop, taking the last committed
letter. Rounds are atomic — speakers within a round can't see each other (as in standard
multi-agent debate), so there is no state to branch on inside one.

Training (answer key steers): start with one solver; try all 5 moves in separate copies; keep the
best 2 wrong branches, chosen for *diversity of answers*; grow to depth 4. **Every hit is
verified:** the winning recipe is re-run from scratch 3 times (fresh samples, cached as
replicates 1–3) and counts only if at least 2 reruns are also correct. A lucky hit is demoted to
unsolved and keeps growing — the search routes around flukes. Every expansion is recorded as a
lesson (state, move, outcome): 150,530 lessons total.

Deployment (built, tested, **not yet run**): a controller call after each round picks the next
move, shown the live transcript plus ~12 lessons from similar training situations. ~6–8 calls per
question, no key.

**Results** (all 3,030 questions, 224,446 calls, 0 errors):

- **Raw solve rate (some rollout hit the right answer): 49.9%. Verified: 19.7%** (596 questions;
  19.2% under the stricter rule below). Of 10,354 raw hits, 49% passed zero reruns and 31%
  passed one — four of five raw hits were luck. Verification was the single most consequential
  design decision of the phase.
- **Per-question structure is real:** 77 distinct winning recipes; the most common
  (solver → critic) covers only 40% of winners; runners-up: solver→verifier→critic (53),
  solver→fresh→critic (25), solver→expert (24), solver→critic→critic (23). Verified solves by
  depth: 19 / 280 / 169 / 91 / 37.
- **Winning recipes are short:** average 3.0 calls, median 2 (distribution: 1 call ×19, 2 ×266,
  3 ×122, 4 ×104, 5 ×58, 6 ×26, 7 ×1). That is deployment cost *if you knew the recipe*; the
  search itself spent ~74 calls/question.
- **Overlap with the global recipe:** 306 questions are solved *only* by their custom verified
  recipe; 152 only by the global recipe's single run; 290 by both; 2,282 (75%) by neither.
  Union: 24.7%.
- **Against resampling:** raw 49.9% at ~28 tries/question is far above extrapolated best-of-28
  plain resampling (~27% from the best-of-k curve). And the verified 19.2% is held to a *harder*
  bar (works 2 of 3) than best-of-7's 16.9% (worked once).
- **Move quality** (rate a move's node was a verified solve, and what fraction of its raw hits
  survived reruns): critic 4.6% / 34% survive; expert 1.2% / 24%; fresh 0.5% / 14%; verifier
  0.3% / 5%; eliminate 0.3% / 5%. Share of a move's verified nodes that passed reruns 3-for-3:
  critic 43%, expert 42%, fresh 19%, verifier 15%, eliminate 15%.

**The eliminate autopsy (a bias caught by the rerun statistics).** Verification's strength
depends on the chance rate, and a move can manipulate the chance rate: the eliminator narrows 10
options to a few, so a downstream guess passes "2 of 3 reruns" far too easily. Measured over
33,216 eliminator rounds: it left exactly **one survivor 68% of the time** (answering while
claiming not to) and **deleted the correct answer 81% of the time** — it eliminates with the same
faulty knowledge that made the question hard. 27 of its 28 winning recipes passed verification at
the minimum 2-of-3. Demoting winners resting solely on a 2/3 eliminate- or verifier-final recipe:
**19.7% → 19.2%** (the strict headline; the finding is robust). For future runs: drop the
eliminate move, or require 3-of-3 for any recipe containing an option-narrowing step.

### 5.6 What the two experiments agree on

1. **The rewritten critic does the real work.** Best verified rate, highest rerun survival,
   present in most winning recipes (final speaker in 404 of 596).
2. **The verifier is a bad decider.** 5% rerun survival in A; the churn-closer that cost 4 points
   in B. It has value only mid-recipe (solver→verifier→critic is A's #2 winning recipe) — and, in
   one small sample, as a full-context protector of already-correct answers (§5.4).
3. **The eliminator should be retired as a scorer.** It kills the right answer roughly 4 times in
   5, blind or not.
4. **Stuck questions stay stuck.** B's parked track: 2.4%. A, with ~74 oracle-steered calls per
   question: 75% of the split has no verified recipe at all. The wins concentrate where the
   model's knowledge exists but its first answer is wrong.
5. **Every unverified success number in this project should be distrusted.** The measured
   optimism ladder: E7's 16.7% → 14.5% off its selection batch; A's 49.9% → 19.7% under reruns;
   19.7% → 19.2% under chance-adjusted reruns.

### 5.7 Current scoreboard (3,030-question fail split)

| method | calls/q | key at run time? | accuracy |
|---|---|---|---|
| ask once | 1 | no | 3.7% |
| best global recipe (fixed) | 8 | no | 14.5% |
| experiment B (adaptive rules) | 8.2 | no | 14.8% (~15.4% with the churn rule deleted — unconfirmed) |
| best-of-7 | 7 | **yes** | 16.9% |
| experiment A, verified ceiling | ~74 to find, 3.0 to run | **yes** | 19.2–19.7% |
| A + global recipe union | — | **yes** | 24.7% |
| experiment A controller | ~6–8 | no | **not yet run** — the open number |

### 5.8 What to run next, in order

| # | what | cost | what it settles |
|---|---|---|---|
| 1 | **The controller** (`treegrow_mcq.py --mode controller`) | ~30k calls | The deployable per-question number. Target: 14.5%. Ceiling: 19.2%. |
| 2 | **Learn B's rules from data instead of hand-writing them.** Any trimming rule reads only letter sequences, so candidate rules replay against the cached transcripts at zero model cost, scored with the key on train, validated split-half. Includes the three known fixes: delete the churn extension, never extend unanimous debates, condition on baseline strength. | ~0 to search; ~10–25k to confirm | Whether learned trim/extend rules beat both 14.5% and 14.8%. |
| 3 | **Held-out confirmation** of every surviving claim on `supergpqa_strict_test.json` | ~30k | Removes all remaining selection optimism. |
| 4 | **Retention scoring** of the winners on the answerable pool | ~10k each | The deployability check Parts 0–4 kept warning about; §5.4's 2-of-6 breakage makes it urgent. |
| 5 | Config for any future tree runs: drop the eliminate move (or require 3-of-3 for narrowing recipes) | free | Closes the chance-rate loophole. |

### 5.9 Engineering notes

- Caches: `outputs/treegrow_rounds_cache.jsonl` (486 MB, 194k rounds),
  `outputs/adaptive_rounds_cache.jsonl` (40 MB); lessons in `outputs/treegrow_lessons.jsonl`;
  summaries in `outputs/treegrow_train_summary.json` and `outputs/adaptive_debate_summary.json`;
  per-question records in `outputs/adaptive_debate_records.jsonl`. Backups frozen as
  `*_backup_aug23.jsonl`.
- A disk-full crash mid-run left two ~0.5 MB null-byte lines in the treegrow cache. Both cache
  loaders now skip unreadable lines (reporting the count) instead of crashing, and the two lost
  rounds simply re-ran on resume. Nothing else was lost — the run resumed at round granularity.
- Experiment A's search decisions are deterministic given the cached transcripts, so resumed runs
  reconstruct the same trees. The controller's decisions are live model calls and are not cached;
  a controller restart re-pays them.

---

## Part 6 — Evolved control programs (2026-09-01)

### 6.1 The idea

Experiment B (§5.4) trims or extends the master recipe using a hand-written set of rules: if a
switch is confirmed, stop; if parked, bring in fresh voices; if churning, call a verifier. Read
as a small program — a list of if-then rules over the debate's state so far — instead of as a
one-off script. Rather than hand-write that program, search for a better one.

A program still needs no answer key at run time: every condition it can check reads only what
happened during the run (how many speakers have spoken, whether they agree, whether a switch has
been confirmed). One program is deployed for every question, but because it branches on what
actually happens, each question takes its own path through it. This gives per-question behavior
from a single deployable artifact, which the fixed recipe cannot do and the answer-key-hungry
tree search (experiment A) cannot do without training a separate controller.

### 6.2 The rule language and the replay engine

Built in `scripts/evolve_program_mcq.py`. A program is:

- a **plan**: the sequence of rounds it runs by default (the master recipe's five rounds, in
  this experiment);
- a list of **rules**, each a set of conditions and one action, checked in order;
  a **default** action if no rule matches.

Conditions read observable state: how many rounds or extra actions have run, whether the last
two speakers agree, whether a switch has been confirmed, whether the debate is parked, what the
last action was, whether an action has run at all. Actions are the existing debate moves:
continue the plan, run a named extra round (critic, verifier, fresh solvers, expert, the
eliminator pair, a synthesizer), or stop and read off an answer.

A program is scored by **replay**, not by calling the model. Every round experiment A and
experiment B ever ran was recorded in two cache files, keyed by the exact sequence of rounds
that came before it. To score a program on one question, the replay engine asks the program what
to do, looks up the recorded round for that exact state, feeds the result back to the program,
and repeats until the program stops. No model calls happen. A question is marked **off-cache**
if the program reaches a state nobody recorded a round for.

Before trusting this, the engine was validated: experiment B's own rules were encoded as a
program and replayed against experiment B's own round cache. The replay reproduced experiment
B's recorded result exactly — same final letter, same track label, same closer, on all 3,030
questions, with zero off-cache questions. The fixed 8-round recipe, replayed the same way,
matched its recorded 14.49% exactly as well. Only after this passed was the replay engine used
for anything new.

Coverage was checked next against the much larger treegrow cache (265,678 recorded rounds, from
experiment A). A handful of simple hand-built programs — solver then critic, solver then two
critics, and so on — were replayed against it. Coverage ran 90% to 100% depending on program
length, confirming there is enough recorded data to search over without constant off-cache
failures.

### 6.3 Data

Everything in this experiment comes from questions already in `datasets/supergpqa_strict_train.json`
— the same 3,030-question fail set as the rest of Part 5. No new questions and no new model
calls were introduced. What is new is only the *order and stopping pattern* in which existing
recorded rounds get walked.

The 3,030 questions were split by a seeded shuffle (seed 0) into two disjoint halves of 1,515
questions each, called **train** and **dev**. Both halves are covered by the same round caches;
the split controls only which questions the search is allowed to look at while choosing which
program to keep. Train accuracy was used for selection during the run. Dev accuracy was computed
once, at the end, on programs the search never scored against dev — so it is honest about
whether a program overfit to quirks of the train batch, but it is still a replay over old
recordings, not a fresh live run. §6.6 explains what that limits.

### 6.4 The search

Population-based random mutation, no LLM proposing edits yet: a mutation changes one rule's
action, adds or removes a rule, changes or adds one of a rule's conditions, or changes the
default. The population was seeded with experiment B's program, the fixed-recipe program, and
several hand-built probe programs (solver+critic, solver+critic+critic, and similar).

Run with `--generations 40 --population 16 --offspring 4`. Each generation, every surviving
program is mutated 4 times; every new mutant is scored (by replay) on train; the best 16 survive
to the next generation. 1,877 distinct programs were evaluated in total over the run. The whole
run cost zero model calls — only local computation reading the two cache files.

### 6.5 Results

Five near-identical top programs came out of the search (they differ mainly in rules that never
fire — dead code left over from mutation). Both halves, replayed:

| program | train accuracy | train n correct | dev accuracy | dev n correct | avg calls/question |
|---|---|---|---|---|---|
| evolved winner | 17.82% | 270 / 1515 | **15.58%** | 236 / 1515 | 6.4 |
| experiment B (hand-written) | 16.24% | 246 / 1515 | 13.40% | 203 / 1515 | 8.2 |
| fixed 8-round master recipe | 15.71% | 238 / 1515 | 13.27% | 201 / 1515 | 8.0 |
| plain solver → critic | 16.30% | 247 / 1515 | **15.58%** | 236 / 1515 | 2.0 |

Coverage was 100% on both halves for every program above, except plain solver → critic (3
off-cache questions on dev, 0 on train).

The winner leads on train by construction — it was chosen using train accuracy — so the train
column is expected to favor it and is not by itself evidence of anything. What the train column
is useful for is the *gap* to dev: every program drops 2 to 3 points from train to dev (17.82% →
15.58% for the winner, 16.24% → 13.40% for experiment B, 16.30% → 15.58% for solver → critic).
That drop is close to uniform across programs that were not selected on train at all (experiment
B and solver → critic were not touched by this search), so most of it is the ordinary difference
between two disjoint question samples, not the search overfitting train specifically. The
winner's own train-to-dev drop (2.24 points) is in the same range.

Head-to-head on dev (questions only one side solves, same underlying recordings): against
experiment B, +48 / −15; against the fixed recipe, +48 / −13. Both are clear wins, not noise,
since they are measured on identical cached samples. On train the same comparisons are +43 / −19
and +49 / −17 — the same direction and a similar margin, which is another point in favor of the
result being real rather than a train-half artifact. Against plain solver → critic the aggregate
is a statistical tie on both halves (dev: +131 / −132; train: +146 / −123 — a train-half edge
that does not survive to dev), but the two programs solve different questions throughout: they
agree on the final letter on only 38% of dev questions. Their union would exceed either alone —
a question this experiment did not chase, but worth noting for later.

**What the winning program actually does.** Reading its four action paths on dev:

| path | share of dev | accuracy | calls |
|---|---|---|---|
| root round → reader solver → critic, then stop | 48.5% (735 q) | 22.2% | 6 |
| root round → reader solver, then stop | 19.7% (298 q) | 12.4% | 5 |
| one round further before stopping | 7.6% (115 q) | 21.7% | 7 |
| the full 5-round plan, nothing ever confirmed | 24.2% (367 q) | 3.0% | 8 |

Tracing the actual rule logic against these paths: the reader solver (round 2) and the first
critic (round 3) both commit to the same letter, different from round 1's majority — a
**two-voice confirmed departure** — the program stops immediately, and this is its strongest and
most common bucket (22.2% on nearly half of dev). If only the reader solver departs and nothing
confirms it, the program's default rule still takes that single, unconfirmed departure as final
— weaker (12.4%) but still far above doing nothing. If nothing ever departs from round 1's
majority through all five rounds, the program runs the whole recipe and returns the majority
answer, at 3.0% — this bucket is the same "stuck questions stay stuck" finding as the rest of
Part 5, and no trimming rule reaches it.

So the program the search found reduces to a small idea, not a complicated one: trust the first
departure once a second independent speaker confirms it, stop immediately rather than letting
later rounds talk the debate out of a correct correction, and give up early on debates that
never depart at all. This matches §0.2's finding that the *first* edit is the only one that
beats the random-departure null — the search rediscovered that finding and built a policy
directly on top of it.

### 6.6 Example programs

To make "a program searched for this" concrete, two single-rule changes to experiment B's own
program, scored on the first 600 train questions (experiment B itself scores 18.5% on this
subset). Both come from trying one random mutation 80 times and keeping only mutants whose paths
stayed well inside the cache, so the accuracy numbers below are not distorted by large off-cache
counts.

**An unsuccessful change.** Adding one rule — `if acts>=3 -> stop:last_speaker` — drops accuracy
from 18.5% to 16.8% (n=600, no off-cache questions). The reason is visible directly in the
program's behavior: every single one of the 600 questions now takes the exact same path (root
round → reader solver → critic, then stop), where experiment B's original program branches into
several different paths depending on what happened. The new rule fires as soon as 3 actions have
run, regardless of whether a departure was confirmed, and because it is unconditional it
pre-empts every one of experiment B's more specific rules the moment they would otherwise
matter — the "did two speakers actually confirm a switch" check, the parked-track extension, all
of it. One overly broad rule silently deleted the rest of the program's logic. This is the
concrete version of something the search's fitness would normally catch and select against; it
is shown here specifically because it did *not* survive selection.

**A successful change.** Adding a different rule — `if ran:pair_expert_solver and last:verifier
-> verifier` — raises accuracy from 18.5% to 19.6%, but only on 561 of the 600 questions (39 went
off-cache). This rule fires late, in the parked-question extension, at the point where experiment
B's original program stops after one verifier round. The mutant instead asks a *second* verifier
round before closing. On the questions where that state was actually recorded, it helps. It goes
off-cache on the rest because a second verifier round at that exact position in the debate was
never something the original treegrow or adaptive runs happened to try, so there is no recorded
transcript to look up. This is a concrete example of the replay engine's central limitation: a
program can only be honestly scored on states someone already ran live, and a mutation that
reaches into unexplored territory is scored on fewer and fewer questions the further it strays.
It is also a small preview of why the evolved winner's headline number (§6.5) needs live
confirmation before it is trusted at full strength — some of the space a search wants to explore
is thinner in the cache than the well-trodden parts near experiment B and the fixed recipe.

### 6.7 Caveats

- **This is entirely a replay of existing recordings**, not a fresh run. Every round the program
  reads off was recorded once, at temperature 0.7, during earlier experiments. No new model calls
  happened in this run. The result shows what already-recorded transcripts would answer under a
  different stopping pattern, not what a fresh live run answers. Sampling luck is a known risk at
  every stage of this project — treegrow's raw 49.9% collapsed to a verified 19.2% once reruns
  were checked (§5.5) — and this experiment has not yet been checked the same way.
- **The tie with plain solver → critic is the most important open question.** If it survives live
  confirmation, the honest conclusion is that all of the trimming program's value is already
  present in one solver plus one rewritten critic, and the extra rounds only change *which*
  questions get solved, not how many.
- Programs cannot currently see which plan round just ran — only the action name ("continue")
  is visible, not which round of the plan it was. This made mid-plan persona extensions
  effectively unreachable during this search. A future run that names plan rounds individually
  would open up a part of the space this run could not explore.
- Fitness during the search was plain accuracy, with no cost term. A cost-aware fitness might
  have converged directly on the cheap solver-critic policy instead of the 6.4-call one.

### 6.8 What to run next

The decisive next step is a **live** run on `datasets/supergpqa_strict_test.json` — a question
set neither treegrow nor experiment B ever touched, so replay would mostly miss the cache and
real model calls are required. Execute, not replay, three or four arms at matched fresh samples:
the evolved winner, experiment B, the fixed recipe, and plain solver → critic. This settles
whether the dev-half tie with solver → critic is real, and gives the first accuracy numbers in
this section that are not replays of old recordings.

---

# History: the global-recipe phase (Parts 0–4)

Everything below is the record of the phase that searched for one recipe to use on every
question (2026-08-09 → 08-11). It is kept because its measurements still constrain the current
work, and because Part 5's decisions cite it. Numbering is unchanged.

---

## Part 0 — The measurements that drove the design

### 0.1 The master equation

The strict corpus is, by construction, questions Qwen3-14B gets wrong. So a recipe can only be
right if it **moves off the base model's answer**. Define, for one run of schema `s` on question
`i`:

- **departure** — the committed letter differs from the minimal schema's letter on the same
  question;
- **departure rate** `D(s)` — the probability of departing;
- **departure precision** `P(s)` — the probability of being correct, given a departure.

Because staying almost never yields a correct answer on this corpus (the base answer is wrong by
construction):

```
    accuracy(s)  ≈  D(s) × P(s)
```

This is not a modelling assumption. It is an identity that holds numerically in every cache.

### 0.2 Every arm decomposes cleanly, and the picture is not what the search assumed

Computed from `outputs/evolve_mcq_strict_cache.jsonl`, restricted to trajectories whose step 0
was wrong (so the identity applies exactly). The **null** column is the accuracy of departing to
a *uniformly random other option*: `E[1/(n_options−1)]`, ≈ 11.8% at a mean of 9.66 options.

| step | executions | departure rate | departure precision | random-departure null | lift |
|---|---|---|---|---|---|
| 1 | 3869 | 43.0% | **14.9%** | 11.8% | **+3.1** |
| 2 | 3250 | 39.3% | 10.3% | 11.7% | −1.4 |
| 3 | 2799 | 38.3% | 8.4% | 11.8% | −3.4 |
| 4 | 2550 | 36.9% | 10.1% | 11.7% | −1.6 |
| 5 | 2361 | 36.9% | 7.8% | 11.7% | −3.9 |
| 6 | 2186 | 37.5% | 6.7% | 11.8% | −5.1 |
| **pooled, all edited steps** | **6644 dep.** | — | **10.4%** | **11.8%** | **−1.4** |

Read the last row carefully. **Pooled over every edit the architect ever made, the debate
recipes' departures are less accurate than throwing a dart at the remaining options.** Only the
first edit — adding one critic — clears the null, and only by 3.1 points.

The same decomposition by structure and by edit type:

| rounds in schema | departure rate | departure precision |    | edit action | departure rate | departure precision |
|---|---|---|---|---|---|---|
| 2 | 44.4% | 14.1% |  | `add_round` (n=12740) | 39.8% | 10.8% |
| 3 | 39.8% | 9.4% |  | `modify_round` (n=3473) | 36.5% | 9.7% |
| 4 | 38.9% | 9.1% |  | `set_final` (n=801) | 38.1% | 5.9% |
| 5 | 38.1% | 7.6% |  | | | |
| 6 | 38.6% | 6.8% |  | | | |

**Adding rounds does not increase departure at all** — the rate is flat at 37–44% from 2 rounds
to 6. What it does is destroy departure precision: 14.1% → 6.8%.

This comparison survives the obvious objection. Later steps see only questions where earlier
steps failed, so their mix is harder — but the random-departure null is `1/(n_options−1)`, which
does **not** depend on question difficulty. The null holds on every subset, so "worse than
random" is not a survivorship artifact. *(But see the correction in §0.9.5: the conclusion holds
for the architect's edits, not for depth itself.)*

### 0.3 The likely mechanism — it is in the executor, not the search

`debate_mcq.execute_schema` passes `_digest(prior)` — every previous persona's letter and 700
characters of its reasoning — to **every** later persona. There is no way to express "answer
without seeing the others." So each added round is another chance to anchor on a wrong committed
letter. The measurement says exactly that: departures stop being informative as soon as there is
a chain of prior commitments to anchor on.

This matters for algorithm design: the genome is missing the one axis (visibility) that controls
the quantity the objective actually factors through.

### 0.4 The critic's whole contribution is the flip, and it is small

Held-out split, `outputs/retrieve_*_strict_test_cache.jsonl`, n=1000:

| | departure rate | accuracy given departure | accuracy given no departure | overall |
|---|---|---|---|---|
| `always_critic` vs the base answer | 44.5% | **15.1%** | 2.2% | 7.9% |
| **null:** two independent `single_pass` draws | 33.2% | **9.6%** | ~1.1% | 3.9% |

The critic buys +11.3pp of departure rate and +5.5pp of departure precision over simply
resampling. That is the entire +4pp of `always_critic` over `single_pass`, and it is the entire
+3pp the whole evolution loop has over best-of-N. There is nothing else in the results.

### 0.5 Agreement predicts wrongness on this corpus — and the caveat that matters

Six selection arms on the same 1000 held-out questions, one run each:

| arm | solve rate |
|---|---|
| `single_pass` | 3.2% |
| `fixed_debate` | 4.1% |
| `self_critique` | 6.3% |
| `retrieve_copy` | 7.3% |
| `retrieve_synth` | 7.8% |
| `always_critic` | **7.9%** |
| **oracle over the 6 answer letters** | **21.9%** |

The gold letter is somewhere in the six arms' answers on 21.9% of questions — nearly 3× the best
single arm. Now look at how that coverage is distributed, and what happens to standard
aggregation:

| aggregation over the 6 arms' letters (no ground truth used) | accuracy |
|---|---|
| plurality vote | **3.0%** |
| best single arm (`always_critic`) | 7.9% |
| pick a letter chosen by exactly one arm | 8.5% |
| `always_critic`, but take a different letter when all six agree | 10.0% |
| **least-frequent letter** | **13.6%** |

| how many of 6 arms produced the most common letter | n | accuracy of that letter |
|---|---|---|
| 6/6 (unanimous) | 262 | **0.0%** |
| 5/6 | 196 | 1.5% |
| 4/6 | 204 | 3.9% |
| 3/6 | 250 | 7.6% |
| ≤2/6 | 88 | 0.0% |

**Unanimity is a perfect predictor of being wrong: 0 correct out of 262.** And when the gold
letter is produced by some arm, it is produced by exactly one arm in 128 of 219 cases. Consensus
is anti-correlated with truth; the signal lives in the minority.

**Caveat, and it is serious.** This is a property of the *filter*, not of reasoning. The corpus
keeps only questions Qwen fails 3/3, so the model's stable answer is a stably wrong answer. Any
method that exploits anti-consensus directly is exploiting the benchmark construction, and it
will invert badly on unfiltered data. §1.5 makes the co-objective that prevents this a structural
part of every design below. It is also why `datasets/supergpqa_qwen_answerable.json` (5,216 rows,
never evaluated with these arms) is a required part of the fitness function, not an optional
check.

### 0.6 Evolution and resampling are genuinely complementary

Full strict set, n=4029, matched at 7 runs:

|  | solved |
|---|---|
| best-of-7 (`single_pass`) | 688 (17.1%) |
| GT evolution | 845 (21.0%) |
| both | 364 |
| **evolution only** | **481** |
| **best-of-7 only** | **324** |
| union | 1169 (**29.0%**) |

The disagreement is large and asymmetric, so this is not noise. On the 3,341 questions where the
minimal schema is 0-for-7 — where resampling is by definition hopeless — evolution still solves
14.4%. The union at 29.0% versus 21.0% and 17.1% is the measured size of the wider-vs-deeper
prize.

### 0.7 Sweep results (`scripts/portfolio_sweep.py`, run 2026-08-09)

Offline over the cached letters, n=1000 `strict_test`. Base letter = the most common letter among
the 7 cached minimal-schema samples (a much better estimate of "the model's answer" than the
single draw used in §0.4, so these departure numbers supersede it).

| arm | accuracy | calls | departure rate `D` | precision `P` | `P` − null |
|---|---|---|---|---|---|
| `retrieve_copy` | 7.3% | 3.2 | 35.5% | **18.6%** | +6.6 |
| `always_critic` | 7.9% | 2.0 | 40.8% | 17.4% | +5.4 |
| `retrieve_synth` | 7.8% | 2.6 | **43.9%** | 15.9% | +3.9 |
| `self_critique` | 6.3% | 3.0 | 39.1% | 14.3% | +2.3 |
| `fixed_debate` | 4.1% | 7.0 | 22.5% | 12.0% | **+0.0** |
| `single_pass` | 3.2% | 1.0 | 23.4% | 10.7% | −1.3 |
| *random-departure null* | — | — | — | *12.0%* | — |

**The ceiling result.** Since `accuracy = D·P + (1−D)·P_stay` and `P_stay ≈ 1.4%`, a recipe that
departed on *every* question would score exactly `P`. The best departure precision anywhere in
the project is **18.6%**. So:

> **No single-run recipe in this genome can exceed ≈18.6% unless departure precision itself
> rises.** Structure can only move `D`.

That bound sits right on top of everything else measured — GT evolution at 7 runs is 21.0%, the
6-arm oracle is 21.9%, best-of-7 is 17.1%. Every prong is hitting the same wall, and the wall is
**how good a departure is**, not how many attempts or what structure.

Note also that `fixed_debate` lands *exactly* on the null: 3 solvers × 2 rounds + synthesizer
produces departures indistinguishable from a dart throw. That confirms the §0.3 anchoring story
on held-out data, using the most-structured arm in the library.

**Portfolio value at matched calls** (200 split-half repeats; subset, aggregator, arm priority
and vote weights all chosen on the fit half, scored on the eval half):

| calls | portfolio (CV) | sd | `single_pass@k` | best homog. rule | pass@k (oracle) | modal config |
|---|---|---|---|---|---|---|
| 2 | 7.9% | 0.8 | 5.2% | `drop_base_first` | 7.0% | `first: always_critic` |
| 3 | 8.2% | 1.0 | 6.6% | `drop_base_anti` | 9.7% | `drop_base_first: always_critic+single_pass` |
| 5 | **10.7%** | 1.0 | 8.3% | `anti_plurality` | 14.1% | `drop_base_first: always_critic+retrieve_synth` |
| 7 | 10.4% | 1.0 | 10.2% | `anti_plurality` | 17.6% | `drop_base_first: always_critic+retrieve_synth+single_pass` |
| 8 | **12.0%** | 1.0 | 10.2% | `anti_plurality` | 17.6% | `drop_base_anti: always_critic+retrieve_synth+self_critique` |
| 12 | 12.5% | 1.1 | 10.2% | `anti_plurality` | 17.6% | 5-arm `drop_base_anti` |
| 16 | 12.5% | 1.1 | 10.2% | `anti_plurality` | 17.6% | 5-arm `drop_base_anti` |

> ⚠️ **The "+2pp for the portfolio" reading of this table is WRONG and was retracted on
> 2026-08-11.** Its homogeneous comparator was `single_pass@k` at 1 call/sample, because no
> repeated-sample data existed for any other arm. With the real control measured (§0.9.1) the
> portfolio loses at every budget. The rest of the table — the per-arm decomposition, the 18.6%
> ceiling, the `drop_base_first` rule — stands.

### 0.9 Run results, 2026-08-09 → 08-11

Three runs: `repeat_schema_mcq.py` (always_critic ×7 on `strict_test`), the retention split plus
four fixed arms on the answerable pool, and `evolve_cem_mcq.py` (E5, 5 iterations, 90k calls).

#### 0.9.1 The portfolio loses to repeated `always_critic` — E4 is dead

| calls | heterogeneous portfolio (CV) | `always_critic` ×k, `drop_base_anti` |
|---|---|---|
| 4 | 7.9% | **10.4%** |
| 6 | 10.5% | **11.9%** |
| 8 | 12.0% | **12.9%** |
| 10 | 11.7% | **13.7%** |
| 12 | 12.5% | **14.3%** |

The mechanism is visible: two `always_critic` samples disagree **46.5%** of the time versus 33.2%
for two `single_pass` samples. **Resampling one good recipe decorrelates better than combining
six different recipes.** Portfolio-of-recipes has no headroom on this task.

#### 0.9.2 At matched calls, GT evolution loses to trivial resampling

GT evolution spends **18.2 calls per question**, not 7 — recipes grow as they are edited. The
original "+3.9pp over best-of-7" was measured with a 2.6× budget advantage.

| calls | `single_pass` pass@k | `always_critic` pass@k | GT evolution |
|---|---|---|---|
| 4 | 11.8% | **13.2%** | 10.1% |
| 6 | **15.5%** | **17.1%** | 12.9% |
| 8 | — | **20.2%** | 13.4% |
| 12 | — | **25.1%** | 15.3% |
| 14 | — | **27.0%** | 15.9% |
| 18 | — | — | 17.7% |

Evolution's 21.0% ceiling costs 18+ calls; `always_critic` pass@7 reaches 27.0% at 14. Evolution
was framed as an oracle upper bound on routing, but an upper bound sitting below a non-oracle
baseline at matched cost has stopped bounding anything. This is the honest headline of the whole
global-recipe phase.

#### 0.9.3 Retention: the best recovery arm is net-harmful in the wild

| arm | calls | recovery | retention | break-even *f* |
|---|---|---|---|---|
| `single_pass` | 1 | 3.2% | 76.7% | — |
| `self_critique` | 3 | 6.3% | 70.4% | 67.0% |
| `fixed_debate` | 7 | 4.1% | **79.1%** | always |
| `always_critic` | 2 | **7.9%** | **65.9%** | 69.7% |

*f* = the fraction of questions the base model must fail for the arm to beat `single_pass` on the
mixture. `always_critic` needs 69.7%; the actual pools give 5027/12841 = **39.1%**, so it is
≈ −4.8pp net. McNemar test on retention vs `single_pass`: p=3.2e-10. `fixed_debate` is the only
arm better on both axes (+0.9 recovery, +2.4 retention, p=0.081) — the arm that had been written
off as weakest.

#### 0.9.4 E5 broke the 18.6% precision ceiling — unconfirmed, but the one positive result

`evolve_cem_mcq.py`, 5 iterations, 18 fully-raced candidates, paired on the same 384-question
batch:

| schema | accuracy | departure rate | **departure precision** | calls |
|---|---|---|---|---|
| `solver; critic` (= `always_critic`) | 8.1% | 42.7% | 16.5% | 2 |
| `solver; critic; critic` | 10.9% | 41.9% | **23.6%** | 3 |
| `4×solver; solver; 3×critic` | 11.5% | 44.5% | **24.0%** | 8 |

9 of 18 candidates cleared 18.6%, and the top-precision recipes share one signature:
**consecutive critics**. The gain is entirely in `P`, not `D` — the axis §0.7 identified as
binding.

**Not yet significant.** McNemar p=0.135 on accuracy, two-proportion z=1.61 on precision. At
N=384 the minimum detectable difference is ~3.2pp and the observed gain is 2.8pp; and since this
recipe won a race over 18 candidates, the estimate is selection-inflated. Needs ≥1000 held-out
questions.

The CEM's model entropy also went **up**, 4.68 → 8.77: the warm-start prior built from the
existing GT-evolution corpus was over-concentrated on `solver; critic -> last`, and the real
elites are deeper and more varied. The existing corpus is a misleading prior.

#### 0.9.5 Correction to §0.2

§0.2 concluded that adding rounds destroys departure precision (14.1% → 6.8%). That was
confounded: it measured what *the architect chose to add*, conditioned on failure, with
survivorship across steps. A clean repeated critic, measured paired on a fixed batch, does the
opposite (+7.1pp). The defensible claim is that **the architect's** added rounds destroyed
precision — not that depth does.

### 0.10 What this implied for algorithm design

1. **The objective to evolve is `D × P`, not accuracy.** Both factors are separately measurable,
   and `D` needs **no ground truth** — the base letters for all 4,030 questions are already
   cached in `outputs/bestofn_strict_cache.jsonl`. That converts a 3.9%-base-rate yes/no fitness
   into a dense continuous one for free. This is the single most useful fact in this document for
   making an evolutionary search affordable.
2. **The genome is missing its most important gene.** Nothing expresses persona *visibility*, and
   visibility is the mechanism that controls anchoring, hence `P`.
3. ~~Individual schema accuracy is nearly saturated (7.9%); the unit of evolution should be an
   ensemble plus its aggregator.~~ **Retracted 2026-08-11** (§0.9.1): a portfolio of distinct
   recipes loses to repeated samples of one recipe at every matched budget. Replacement:
   individual accuracy was **not** saturated — §0.9.4 raised it from 7.9% to 10.9% by evolving
   structure alone. The unit of evolution stays the single recipe; what has to grow is the
   **genome**, not the ensemble.
4. **A retention co-objective is mandatory**, or every result is filter-hacking.
5. **Statistical power is the binding constraint.** Disagreement between two nearby arms is
   d ≈ 0.10, so the minimum detectable paired difference is ≈ `1.96·√(d/N)`: 2.8pp at N=500,
   2.0pp at N=1000, 1.4pp at N=2000. Any search whose selection steps need to resolve <2pp is not
   affordable and must be redesigned to make bigger moves.
6. **Correction to `evol_lit_search.md` C6:** the trajectory caches do **not** contain
   transcripts. `steps[k]` holds `{step, action, schema, answer, correct, diagnosis, rationale}`
   — letters only. Training a verifier from them is not free; it needs a re-run with transcript
   logging.

---

## Part 1 — The shared substrate (`scripts/schema_fitness.py`)

Every design in Part 2 needs the same four things. Building them once, first, is what made the
rest affordable. This is infrastructure, not an algorithm.

### 1.1 Instrument the executor

`execute_schema` originally returned one letter. Make it return a **behavior record**:

```python
{ "final": "H",
  "round_letters": [["C"], ["H"], ["H","H"]],   # per round, per persona
  "n_calls": 4,
  "departed": True,          # final != cached base letter for this question
  "churn": 1,                # number of times the running letter changed
  "n_distinct": 2,
  "final_rule_flipped": False }
```

Nothing else changes; `execute_schema` gains an optional `return_trace=True`. One run now yields
a vector instead of one bit, and `departed`/`churn`/`n_distinct`/`n_calls` all need **no ground
truth**.

### 1.2 Prefix-shared paired execution (common random numbers)

Recipes in a population often start with the same first rounds. With a fixed seed, round 1 of
`[solver]` and round 1 of `[solver][critic]` are byte-identical calls. Memoize on
`(qid, seed_idx, tuple_of_rounds_so_far, persona_index) → response`.

Two payoffs, both large:
- **Cost.** In a population where most individuals share a `solver` prefix, this saves roughly
  30–50% of calls per generation.
- **Variance.** Two recipes compared on the same question with the same seed differ only in the
  part that actually differs. This is the Fishtest pairing insight made exact, and it is what
  makes the power numbers in §0.10(5) achievable rather than optimistic.

### 1.3 The evaluation cascade

Modelled on AlphaEvolve's evaluator cascade; each stage is cheap and kills most candidates.

| stage | cost | signal | kills |
|---|---|---|---|
| 0 | free | seen before? behavior fingerprint within ε of an evaluated individual? cost over budget? | duplicates (89% of the old first edits) |
| 1 | 64 questions × cost(s) | **departure rate + churn, no ground truth** | anything whose `D` is outside the target band, or that anchors (churn ≈ 0) |
| 2 | 256 questions, stratified | accuracy, ranked by lower confidence bound | the bottom half, successive-halving style |
| 3 | 1024 + 256 answerable | accuracy + **retention**, paired vs the incumbent | promotes to the archive only on a passed paired test |

Stage 1 is the novel part, and it is what the master equation buys: a dense, continuous,
label-free filter in front of a 4%-base-rate yes/no one.

### 1.4 Statistics

- Beta posterior per (schema, batch); rank by `LCB = BetaPPF(0.25, α, β)` so a lucky single roll
  cannot hold an elite slot.
- Paired promotion by exact McNemar test on discordant pairs (`scripts/compare_oracle_arms.py`
  already has `mcnemar_exact`; lift it into the shared module).
- Successive halving within a generation; a sequential test only for the final
  incumbent-vs-challenger call.
- Every candidate carries `n_calls`; **all reporting is per call, never per generation.**

### 1.5 The corpus is a mixture, always

Fitness batches are drawn from two pools and reported as two numbers:

- **recovery** on `supergpqa_strict_train` (the fail set) — what we want to go up;
- **retention** on a matched sample of `supergpqa_qwen_answerable` — what must not go down.

Selection is over the pair. A recipe that gets recovery by maximizing departure will visibly
destroy retention, and the Pareto front over (recovery, retention) is the honest deliverable —
likely the paper's central figure. Note the retention pool had never been evaluated with any of
these arms; that gap needed closing before any of Part 2 ran.

---

## Part 2 — Five evolutionary designs

Ordered by what I would actually run, not by ambition. Final outcomes are in Part 3.

---

### E1 — Departure-precision evolution: add the genes the objective factors through

**The idea.** Evolve directly on the decomposition. The genome gains the genes that control `D`
and `P`, and selection is multi-objective over `(D, P)` rather than one accuracy number.

**Genome extension.** Two additions, both small changes to `debate_mcq.py`:

1. **A visibility gene per round** — the missing gene from §0.3:

   ```python
   {"personas": [...], "sees": "all" | "none" | "letters_only" | "last_round" | "no_letters"}
   ```

   `none` = an independent solver that never sees the prior chain (breaks anchoring).
   `no_letters` = sees the prior *reasoning* with the committed letters stripped (debate the
   argument, not the commitment). `letters_only` = sees the tally without the reasoning.
   This is ~15 lines in `_digest` plus a validator case.

2. **Departure-forcing personas** — new entries in `PERSONA_PROMPTS`:

   | persona | prompt intent |
   |---|---|
   | `contrarian` | may not re-select the letter the previous round committed to; must justify a different one |
   | `eliminator` | rules out the most likely wrong options with reasons; does **not** answer |
   | `ranker` | emits a ranked top-3 instead of one letter (gives aggregators something to work with) |
   | `independent` | a solver forced to `sees: none` |

**Fitness.** Multi-objective selection (NSGA-II) on three objectives: recovery (`D×P` on the fail
pool), retention (accuracy on the answerable pool), cost (`n_calls`). Stage 1 of the cascade
prescreens on `D` alone.

**Why this first.** It is the only design that tests the mechanism §0.3 identifies. And its
cheapest form is not an evolutionary run at all: evaluate `[solver][critic(sees=no_letters)]`,
`[solver][independent][critic]`, `[solver][contrarian]`, `[solver][eliminator][solver(sees=none)]`
as **fixed arms** on 1000 questions each. That is ~8k calls total, and it says whether the
anchoring hypothesis is right before spending anything on a population.

**Must beat.** `always_critic` at 7.9% and, more informatively, its decomposition (D=44.5%,
P=15.1%). A recipe at D=70%, P=15% would score 10.5%; at D=60%, P=20%, 12%.

**A negative result would mean:** every visibility setting lands at P ≈ 12% (the null). That
would say the executor cannot make an informed departure at all on this corpus — which bounds
every structural method, and is worth publishing.

---

### E2 — Island genetic algorithm with cascade racing, over the E1 genome

**The idea.** The workhorse population algorithm. Fitness is solve rate over shared question
batches, not one question — the reframing that turns a 1-sample hill climb into something with a
real fitness.

**Configuration.**
- Population 24, four islands of 6. The best individual migrates between adjacent islands every 5
  generations. Every 10 generations, cull the worst island and reseed it from the global archive
  (FunSearch's diversity mechanism).
- **Variation operators**, chosen by an adaptive bandit (Fialho et al.) with credit assigned by
  how often each operator's offspring survive stage 2:
  - `llm_mutate` — the architect, but shown **aggregate** statistics of the parent (its D, its P,
    its per-field profile, three failed transcripts) rather than one question's history;
  - `structural_crossover` — splice rounds between two parents, take persona-set unions or
    intersections, inherit the final rule. No LLM needed. **There was no crossover at all
    before;**
  - `llm_crossover` — show the architect two parents and their behavior records, ask for a child;
  - `simplify` — delete a round or a persona. The old loop could add but never remove
    (`remove_round` ≈ 0% of actions); given §0.2, a shrink operator may be the single most
    valuable one in the set.
- **Behavioral novelty rejection** (after ShinkaEvolve, improved): reject an offspring whose
  stage-1 fingerprint `(D, churn, n_distinct, cost)` is within ε of an already-evaluated
  individual. This is behavioral rather than textual, so it is better targeted than embedding
  the serialized recipe — and stage 1 is cheap enough to compute before deciding.
- **Selection**: non-dominated sorting on (recovery, retention, cost); within a front, the lower
  confidence bound.

**Budget.** ~24 individuals × 10 generations. With the cascade, most die at stage 0/1: expect
~24×64 (stage 1) + ~8×256 (stage 2) + ~3×1280 (stage 3) ≈ 7.4k question-evaluations per
generation, times mean cost ~3 calls, minus ~35% from prefix sharing ≈ **~14k calls/generation**.
Ten generations ≈ 140k calls. Real but not absurd on a local 14B — and the honest comparison is
best-of-k at the same total, which is why §1.4 insists on per-call reporting.

**Must beat.** `always_critic` at 1 run, on `strict_test`, with retention reported alongside.

---

### E3 — MAP-Elites over (departure rate × churn × cost), with noise-aware cells

**The idea.** The behavior descriptors here are not arbitrary — they are the factors of the
objective, which is exactly the condition under which MAP-Elites reveals something meaningful.

- **Descriptors**: `D` in 10 bins × `churn` in 5 bins × `n_calls` in 5 bins (250 cells). All
  three come free from stage 1, with no ground truth.
- **Cell contents**: a *deep grid* (Flageat & Cully) of up to 8 individuals with pooled Beta
  posteriors; the cell's elite is the highest **lower confidence bound**, not the highest point
  estimate. Each time a cell is selected as a parent, one of its residents is re-evaluated, so
  the estimates sharpen and a lucky roll cannot squat.
- **Parent selection**: sample a cell with probability proportional to how often its offspring
  have improved a cell, not uniformly.
- **Variation**: E2's operator set.

**Why it is worth its own run.** The output is the empirical frontier `P*(D)` — the best
departure precision achievable at each departure rate. That single curve answers the project's
central question directly:

- if `P*(D)` is flat at ~12% (the null) for all `D`, the grammar is empty and no search will
  help;
- if `P*` decays with `D`, there is a real precision/recall tradeoff and the deployable optimum
  is an interior point you can locate;
- if some cell holds `D=0.7, P=0.18`, that is a 12.6% recipe and the whole thesis is alive.

Any of the three is a publishable answer — the property you want from an expensive run.

**Must beat.** Not an accuracy bar — this is the illumination run. Its deliverable is the
frontier plus a diverse archive that seeds E4.

---

### E4 — ~~Coevolve a recipe portfolio and its aggregator~~ — **KILLED 2026-08-11**

> Falsified by §0.9.1 before it was built. A portfolio of distinct recipes loses to repeated
> samples of a single recipe at every matched call budget, because repeats of one good recipe
> decorrelate *more* (46.5% pairwise disagreement) than six different recipes do. The premise —
> that recipe diversity produces error decorrelation that resampling cannot — is false on this
> task. The design below is kept only as a record of what was tested and why it failed.
>
> Salvage: the *aggregator* half still matters, but over repeated samples of ONE recipe — a
> selection problem, not an evolutionary one.

#### (original design, superseded)

**The idea.** §0.5: the best individual arm is 7.9%, the 6-arm oracle is 21.9%, and a
hand-written one-liner ("take the least-frequent letter") already extracts 13.6% of it. So evolve
the ensemble, not the member.

**Two populations, evaluated only in combination** (Potter & De Jong cooperative coevolution):

- **Population S** — recipes, seeded from E2/E3's archive.
- **Population A** — aggregators. Genome:
  ```python
  {"rule": "plurality" | "anti_plurality" | "borda" | "reliability_weighted" | "drop_base_then_plurality" | "llm_judge",
   "weights": {schema_id: float},          # learned on train, or evolved
   "drop_base_letter": bool,               # exclude the base model's answer from the ballot
   "tie_break": "highest_reliability" | "rarest" | "judge",
   "judge_prompt": "<evolved text, only for rule=llm_judge>"}
  ```

- **Fitness of a pair.** A portfolio of k recipes, run once each, aggregated by `A`, scored on
  the mixture corpus at a fixed **call** budget. Portfolio choice inside the loop is a coverage
  problem — greedy plus local swaps gives a (1−1/e) guarantee, and it selects for **marginal
  complementarity**, a different pressure from mean accuracy.
- **Credit assignment.** A recipe's fitness is its marginal contribution to the portfolio's
  score (leave-one-out), which is what stops the population from collapsing onto k copies of
  `always_critic`.

**The core control, stated up front.** A 4-recipe portfolio at 2 calls each is 8 calls. Its
control is `always_critic` sampled 4× and aggregated the same way, at the same 8 calls. If the
diverse portfolio beats the homogeneous one at matched calls, error decorrelation is real. That
control had never been run — and when it was run (§0.9.1), it killed the design.

**Zero-GPU version first.** With 6 arms × 1000 questions of cached letters, every 2-, 3-, and
4-subset and every aggregator rule can be enumerated offline. That is what
`scripts/portfolio_sweep.py` did.

---

### E5 — Cross-entropy method over a schema grammar — **IMPLEMENTED AND RUN ✅**

> `scripts/evolve_cem_mcq.py`, run 2026-08-11 (5 iterations, 90k calls, 18 candidates fully
> raced). **The only new method tested in this phase, and the only positive result.** It raised
> single-run accuracy 8.1% → 10.9% and departure precision 16.5% → 23.6% over `always_critic` on
> a paired batch, breaking the 18.6% ceiling §0.7 identified (§0.9.4). Not yet significant at
> N=384; needs held-out confirmation.

#### (design)

**The idea.** Instead of mutating individuals, learn and refine a probability distribution over
the grammar: sample recipes from it, keep the best fraction, refit, repeat. This is Jin &
Branke's "implicit averaging" — noise robustness comes from the population, not from re-running
each candidate — the right family when every fitness sample costs LLM calls.

**Model.**
```
P(n_rounds)                            # categorical over 1..6
P(personas | round_position, n_rounds) # categorical over persona multisets
P(sees | round_position)               # the E1 visibility gene
P(final)
```
Factorized, ~200 parameters, all conjugate with Dirichlet priors.

**Loop.** Sample 40 recipes → cascade-evaluate on a shared batch → take the top 20% by lower
confidence bound → refit with smoothing → blend `θ ← (1−ρ)θ_old + ρ θ_new` with ρ≈0.3 to avoid
premature collapse.

**Why include it.** Three reasons: it is ~60 lines; it warm-starts for free by fitting the
initial distribution to the 845 solved recipes already in `evolve_mcq_strict_cache.jsonl`; and
its output is a *sampler*, which is what E4's portfolio construction would have wanted. It is
also the cheapest test of whether the grammar has any exploitable structure at all — if the
fitted distribution after 10 iterations matches its warm start, the space is flat.

**Must beat.** E2 at matched calls. If a 60-line method matches a full island GA, that is a real
finding about the size of the search space and should be reported as one.

---

### E6 — (Deferred) Per-question wider-vs-deeper allocation

Listed for completeness because §0.6 says the prize is real (29.0% union vs 21.0% / 17.1%), but
it depended on E4's library existing first.

At test time, with a k-run budget, decide per question whether to spend the next call **wider**
(another member of the portfolio) or **deeper** (an edit to the current best), by Thompson
sampling from Beta posteriors — recipe-level priors fitted on train, updated per question by the
observed departure/agreement state (the only reward available without ground truth). The learned
wider/deeper ratio by difficulty would be publishable even if accuracy did not move.

The blocker is §0.5: the natural per-question reward is agreement, and agreement is
anti-correlated with correctness here. So E6 needed E4's aggregator to supply a usable reward
first.

---

### E7 — Reflective evolution of persona TEXT, with Pareto-front parent selection *(added 2026-08-11)*

**Why this design exists.** §0.9.4 changed the picture. Structure was thought saturated at 7.9%;
evolving it took single-run accuracy to 10.9% and departure precision to 23.6%. The gain came
entirely from `P` — from *what a critic does when it fires*, not from how often it fires. But
`PERSONA_PROMPTS` was three frozen strings, and a second critic helped only because it re-ran the
same instruction. The text is the axis with the most room and the least attention — exactly what
MASS predicts (topology alone is the weakest of its three optimization stages).

**Genome.** `(topology, persona_prompts)` — the existing grammar plus the text of each persona's
system prompt. The topology half is already handled by E5/E2; this adds the text half.

**Variation = reflection, not random mutation.** GEPA's operator, adapted: sample a batch of
*failed* runs of the parent, show an LLM the question, the persona outputs, and whether the final
answer departed from the base answer (never the gold letter — the firewall holds). Ask it to
diagnose in plain language why the critic's objection did not land, then rewrite ONE persona's
prompt. Reflective text edits make moves of the size the batch can actually resolve, which is the
practical reason they beat token-level or random edits at ~3pp resolution (§0.10(5)).

**Selection = Pareto front over per-question outcomes,** not a mean score. With yes/no outcomes
at an 8% base rate, an individual joins the front if it *uniquely* solves some question. That
preserves specialists a mean would discard, and since the arms correlate weakly (§0.5), the front
will be wide. GEPA reports this beats mean-based selection at up to 35× fewer rollouts.

**Fitness.** The E5/E3 cascade unchanged, but with the retention pool as a hard second
objective — this is the design most able to cheat by writing a prompt that says "assume the first
answer is wrong," and §0.9.3 shows what that costs.

**Must beat.** `solver; critic; critic` at matched calls, on held-out data, with retention
reported.

**What a negative result means.** If reflectively evolved critic prompts cannot push `P` past
~24%, then the executor genuinely cannot make an informed departure on questions it fails, and
the wall is the model, not the search. That is the paper either way.

---

## Part 3 — Final status of the global-recipe designs

**Superseded 2026-08-24.** All three global-recipe searches were built and run; outcomes below.
The project then pivoted to per-question methods — see Part 5 for current work and ordering. This
section closes the record of the global-recipe phase.

| design | code | run | outcome |
|---|---|---|---|
| **E5** CEM over a schema grammar | `scripts/evolve_cem_mcq.py` | ✅ 90k + 110k calls | First run 10.9% (384 q). Extended run (E1 genome): best 21.9% at 11 calls, runner-up 20.6% at 5 calls — **both lean on the contrarian, which was then banned** (see §5.1), disqualifying them. Its contrarian-free *structures* seeded E7. |
| **E3-lite** MAP-Elites over (D × cost) | `scripts/mapelites_mcq.py` | ✅ | Winner 25.0% on 96 q — also contrarian-reliant, also disqualified. |
| **E1** genome extension (visibility + new personas) | ✅ in `debate_mcq.py` | ✅ | Built; used by the extended E5/E3 runs and by everything in Part 5. |
| **E7** reflective persona-text evolution | `scripts/evolve_persona_text_mcq.py` | ✅ 24k calls | **Produced the project's global recipe:** 4 solvers → reader solver → 3 rewritten critics. 16.7% on its 192-question selection batch; **14.5% on the full 3,030** (~2 pp selection optimism, measured). This recipe is experiment B's master. |
| **E2** island GA + racing | ✗ | ✗ | never built; overtaken by the per-question pivot |
| **E4** portfolio + aggregator | ✗ | ✗ | killed by §0.9.1 |
| **E6** wider-vs-deeper allocation | ✗ | ✗ | never built |

The old ordering advice in this part is obsolete; the current ordering is §5.8.

---

## Part 4 — Risks (as assessed during the global-recipe phase)

1. **Anti-consensus is filter-exploitation.** §0.5's 13.6% is real on this corpus and will invert
   on unfiltered data. §0.9.3 measured the cost concretely: `always_critic` needs 69.7% of
   questions to be failures before it beats `single_pass`, and the real rate is 39.1%. Every
   winner must be re-scored on the retention pool, or the result will not survive review.
2. ~~Departure precision may be capped at the null.~~ **Partly resolved (§0.9.4):** E5 reached
   23.6% against an 11.8% null, roughly double. The ceiling is higher than §0.7 claimed; the open
   question is where it sits, not whether it exists. The remaining risk is milder — that ~24% is
   the wall, in which case single-run accuracy caps near 24% and the deployable story has to come
   from elsewhere.
3. **Power.** d ≈ 0.10, so differences under 2pp need N > 2000 paired questions. Design every
   selection step to make moves larger than its batch can resolve, and prefer racing (kill the
   obviously bad) over ranking (order the nearly equal).
4. **Prefix caching changes the sampling distribution.** Reusing round-1 outputs across a
   population makes members correlated, which reduces *ensemble* diversity even as it reduces
   variance. Use it for paired A/B comparisons; disable it when measuring an ensemble's coverage.
5. **The retention pool is unmeasured.** 5,216 answerable rows, zero arms run on them. Every
   claim about deployability is currently one-sided. *(Still true in Part 5 — see §5.8 item 4.)*

---

## Recommendation

Run the controller (`treegrow_mcq.py --mode controller`, ~30k calls). It is the project's open
number: the deployable per-question method, with a measured floor (14.5%, the global recipe) and
a measured ceiling (19.2%, the verified oracle). While it runs, replay learned trim/extend rules
for experiment B against the cached transcripts — that search is free. Then confirm every
surviving claim on the held-out split and score the winners on the retention pool before making
any deployment claim.
