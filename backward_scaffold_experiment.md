# Plan: teaching a small model with hint sheets, then taking the hints away

Written 2026-09-23, revised the same day for the reduced size (100 + 100 questions, five
tries). One dataset (SuperGPQA), one small model (gpt-oss-20b). Nothing here uses the
cluster-and-route pipeline. The code is in `scripts/hint_study/`, the driver is
`run_hint_study.sh`, and the numbers land in `outputs/hint_study/`.

## 1. The idea in plain words

Some questions the small model gets right on its own. Some it cannot get right even when we
hand it most of the answer. Neither kind teaches us anything. The interesting questions are the
ones in the middle: the model fails alone, but succeeds when a stronger model writes it a hint
sheet.

For those middle questions we do two things.

First, we take the hints away one at a time, starting from the end of the sheet, and each time
we ask: which way of working lets the small model still finish the job? Does it need to check
its own work? Does it need a second opinion? Does it need to try several times and vote? Does it
need to plan first? From the answers we build a recipe, and then we test that recipe on new
questions with no hints at all.

Second, for questions that are about remembering facts rather than working things out, the
hints are the facts. We check which facts the model actually has in memory. Then we take the
facts away one at a time and search for a way of asking that makes the model bring the fact up
by itself. At the end, no facts are given, and we test whether the way of asking still helps.

## 2. Two kinds of questions

The plan treats two kinds of questions differently. Real examples from our training pool:

**A working-out question** (Space physics). "Assuming solids are divided into two categories,
100% iron meteorites and silicate rocks containing 10% iron, and assuming the solar system has
the same metal content as Earth, what proportion of total meteorite mass is iron to stone?" Ten
numeric choices; the right one is 0.75.

To answer it you have to know one fact (Earth is about one third iron by mass) and then do
three things with it: write the mixing equation, solve it, and read off the stone share. A hint
sheet for it would be:

1. Earth is roughly 32% iron by mass, and the question says the solar system's solids match that.
2. Let x be the share of iron meteorites. Then all of x is iron, and one tenth of the rest is iron, so x + 0.1 (1 − x) = 0.32.
3. Solving gives x = 0.25.
4. (kept back, never shown) The stone share is therefore 1 − 0.25 = 0.75.

**A remembering question** (Physical chemistry of metallurgy). "Surface tension is related to
the state of combination between liquid particles. Which of the following relationships is
correct?" Seven orderings of metallic, ionic and molecular liquids; the right one is
metallic > ionic > molecular.

There is nothing to work out here. You either know the three facts or you do not:

- Surface tension is higher when the particles in the liquid are held together more strongly.
- Liquid metals are held together most strongly (mercury's surface tension is about seven times water's).
- Molten salts come next; liquids made of ordinary molecules are held together least.

Many questions are a mix: a few facts to remember, then a few steps to work through. We handle
those as facts first, then steps. Our earlier labelling pass tagged every question as
remembering, working-out, or mixed, and that tag is what the study uses to balance the draw.
(The big model, when it writes a sheet, calls almost every working-out question "mixed",
because it always lists the facts a solver must know first. That is a difference in convention,
not a disagreement about the question.)

## 3. Words used below

- **Small model**: gpt-oss-20b, the model we are trying to help. It runs on our own servers, so
  its calls cost time but no money. It always runs with its full thinking on unless a way of
  working says otherwise.
- **Big model**: GPT-5.6-terra, which writes the hint sheets and, when needed, judges answers. Its
  calls cost money.
- **Hint sheet**: what the big model writes for a question. For a working-out question it is a
  numbered list of steps with the final step held back. For a remembering question it is a list
  of facts. The sheet must never name or quote an answer choice.
- **Five tries**: whenever we test whether the model can do something, we ask it five times
  with randomness on and count how many times it is right. One try tells you little; five tell
  you whether it can do the thing reliably, sometimes, or not at all.
- **Fails alone**: right at most 1 time out of 5 with no hints.
- **Succeeds with hints**: right at least 4 times out of 5 with the full hint sheet.
- **Finishes**: a way of working "finishes" from a given number of hints if it is right at least
  3 times out of 5.
- **Ways of working**: the things we can ask the small model to do besides "just answer". The
  list is in step 5.

## 4. Size of the study

Two hundred questions: 100 for developing (train) and 100 for the final test, drawn at random
from the 1,815 questions that already carry a label from our earlier labelling pass. No new
labelling calls. Each hundred has 34 remembering, 33 working-out and 33 mixed questions, so
each kind gets the same weight. The draw is fixed by a seed and written to
`datasets/hint_study_train.json` and `datasets/hint_study_test.json`.

The later steps shrink these numbers. The model fails most of these questions alone, but some
sheets will give the answer away, and the big model will disagree with the official key on
some. In the smoke test the big model disagreed with the key on 9 of 15 questions, including
every working-out question, and on the three we checked by hand the key was wrong every time
(a strain energy missing its one half; a fall time of 4 s given as 2√2 s; a power factor of
√3/2 given as 1/2). That is not a surprise: the pool these questions come from was built by
keeping questions another model got wrong, and a question with a wrong key is always "got
wrong". The funnel in step 3 counts this. If the count is large, that is a finding in itself.
Two ways to handle it, to be decided before the real run:

- exclude those questions and top up the draw from the same labelled pool, or
- ask the big model a second time, independently; if it gives the same letter twice, treat the
  key as wrong and use the big model's letter, flagged in every table.

## 5. The plan, step by step

Every step keeps every reply it gets in a cache, so a step can be rerun after an interruption
and continues where it stopped, and a later step that asks the same thing gets the stored reply.

### Step 1: draw the questions (`draw_questions.py`, done)

As in section 4. The manifest with the mix of kinds, disciplines and SuperGPQA difficulty tags
is in `outputs/hint_study/draw_manifest.json`.

We record SuperGPQA's own easy/medium/hard tag but never use it to choose questions. In our
data that tag mostly says what kind of question it is, not how hard it is for our model: 62% of
"remembering" questions are tagged easy, and the model still fails them.

### Step 2: find the questions the small model cannot do alone (`unaided.py`)

Every question, five tries, no hints, full thinking. Keep the ones it gets right at most once.
We expect this to be most of the 200.

One thing to watch: with full thinking on, some replies run out of room before giving a letter.
We count those as wrong but keep a separate tally, so we can tell "did not know" from "ran out
of space".

Cost: 1,000 small-model calls, the longest ones in the study.

### Step 3: hint sheets, and the give-away check (`hint_sheets.py`, `leak_check.py`)

For each question the model failed, the big model does four things in one call:

- Answers the question itself. If its answer disagrees with the official one, the question is
  put aside as "possibly mislabelled" and counted (see section 4).
- Says whether the question is remembering, working-out, or mixed.
- Writes the hint sheet. For working-out: three to eight steps, each one thing you could check,
  each tagged with what kind of step it is (set up the quantities, pick the law, recall a
  constant, algebra, arithmetic, split into cases, sanity-check, match to a choice). The last
  step, the one that produces the answer, is stored but never shown. For remembering: two to
  six facts, each a plain sentence, plus for each fact a short question that asks for that fact
  on its own, and a wrong version of the fact for a later check.
- Never mentions a choice letter or repeats the text of a choice. A text check flags any sheet
  that does anyway.

Then the give-away check. We show the small model the hint sheet and the answer choices, but not
the question itself, five tries. If it still picks the right letter three or more times, the
sheet gives the answer away. We ask the big model to rewrite it once with a stricter
instruction. If it still gives the answer away, we drop the question and count it.

Cost: about 200 big-model calls; 1,000 short small-model calls for the check.

### Step 4: keep the middle questions, and trim each sheet (`gate_with_sheet.py --trim`)

Each question with its full hint sheet, five tries. Keep the ones the model now gets right at
least four times. These are the middle questions, the only ones the rest of the plan uses.

Then trim. Remove one hint at a time, five tries each. A hint whose removal does not hurt is a
candidate to drop. We drop candidates one by one, most harmless first, and re-test after each,
because two hints that each look unnecessary can turn out to be needed together. What is left
is the shortest sheet that still works.

Decision point. If a kind of question has fewer than 15 middle questions in a split, we top up
that kind from the labelled pool before going on.

Cost: about 7,000 small-model calls, most of them short.

### Step 5: working-out questions, take the hints away from the end (`hints_needed.py`)

Here is the list of ways of working we test. Each one is given the question plus the first few
hints, and has to finish from there.

| way of working | what the model is asked to do | what it tells us |
|---|---|---|
| just continue | finish from the hints in one go, full thinking | the baseline |
| continue, light thinking | the same, with the model's thinking turned down | whether thinking hard matters |
| try three times and vote | three separate "just continue" attempts, majority letter | whether more tries help, without new information |
| critic first | one call lists what could go wrong from here; a second call finishes, having read that | whether a second opinion helps |
| check first | one call re-does the given steps and says whether they hold; a second call finishes | whether checking helps |
| plan first | one call writes the remaining steps as a plan; a second call carries it out | whether structure helps |
| start over as an expert | ignores the hints except as background and solves from scratch in an expert voice | whether a fresh start helps |

For each middle question and each way of working, we find the fewest hints it needs. We do not
test every number of hints one by one. We first give all the hints (if that fails, this way of
working cannot finish the question at all). Then we give half and see whether it finishes. If
yes, we try a quarter; if no, three quarters; and so on. Three or four rounds pin down the
smallest number of hints that still works. For "just continue" we also test every number of
hints, so we get its full curve.

Two things come out of this:

- For each way of working, how many hints it needs on average, and what share of the sheet that
  is. If "critic first" needs fewer hints than "just continue", the critic is doing real work.
  If every way of working needs the same number, none of them adds anything, and that is the
  result.
- A table of step kinds against ways of working: for arithmetic steps, which way of working
  finishes them most often; for pick-the-law steps, which; and so on. To fill it in properly we
  cannot only look at the final letter, so on a sample the big model also reads the
  continuation and says whether the next step in it is actually right. (This judged part is not
  written yet; it waits for the first real numbers.)

The same search runs on remembering questions with the facts as the hints. That gives, for
free, the "how many facts can we hide before plain asking fails" number that step 8 starts from.

Cost: about 25,000 small-model calls.

### Step 6: working-out questions, build a recipe and test it with no hints (not written yet)

From step 5 we know, for each question, the cheapest chain of ways of working that gets from
"no hints" to the answer: for example "plan first" to get through the set-up, then "check
first" through the algebra, then "just continue" to the end. From the table of step kinds we
turn those chains into one general recipe: plan first, name the kind of each step in the plan,
and use the way of working that the table says is best for that kind.

Then the real test. The recipe runs on the 100 test questions, with no hint sheet at all, five
tries, against: just answer; try three times and vote; the single best way of working; and the
self-refine method we already have numbers for.

We also report one honest number: how much worse things get when the model's own earlier
steps replace the big model's. In step 5 the hints were always right. In real use, the "hints"
are whatever the model wrote for itself a minute ago, and those can be wrong.

### Step 7: remembering questions, find out which facts the model actually has (`fact_probe.py`)

You cannot prompt a model into stating a fact it never learned. So before trying to, we check
each fact on each middle question's sheet in two ways:

- Ask for it directly, using the short question the big model wrote for that fact, five tries.
  The big model judges whether the reply states the fact.
- Show the true fact next to its wrong version and ask which is right, in both orders, twice
  each.

Each fact ends up in one of three bins: the model can state it when asked; the model can only
recognise it; the model does not have it. A question that needs a fact from the third bin is
put in a "needs a lookup" pile and reported separately. Only questions whose facts the model can
at least recognise go on to step 8.

Cost: about 2,500 small-model calls and about 1,200 big-model judging calls.

### Step 8: remembering questions, take the facts away and search for the right way of asking (not written yet)

Level one: give all facts but one. The one we hide is the fact the model was least likely to
state on its own in step 7. Measure how often plain asking gets the question right. Then search
for a way of asking that does better. The search is a loop: a way of asking is tried on a batch
of questions, the big model reads the failures and proposes a revised wording, and the revised
wording is tried on the next batch. What counts as success is, first, whether the model brings
up the hidden fact, and only second whether it picks the right letter.

The search does not write free text from scratch. It rearranges and rewords a short menu of
moves:

- write down everything you know about the topic before answering;
- explain each answer choice in one line;
- state the general rule before the specific case;
- cross out choices that contradict something you have already said;
- ask the question again under a different wording;
- ask under three different wordings and vote.

Level two hides two facts, and so on, until the last level hides every fact. Whatever survives
the last level is a way of asking that helps with nothing given.

Three guards against fooling ourselves:

- The wordings are developed on the train questions and scored on the test questions.
- The big model reads every wording and rejects any that contains a name, a number, or a claim
  that belongs to a particular training question. A wording that only works because it
  smuggles the facts in is a cheat, and we report it if we find it.
- Final test with no facts given, on the test questions, against plain asking and against
  "try three times and vote".

### Step 9: the report (`report.py`)

The funnel and the tables of section 6 from whatever files exist, written to
`outputs/hint_study/report.md`. Run it at any point.

## 6. What we will be able to show

- A funnel: how many questions were drawn, how many the model failed alone, how many the big
  model agreed with the key on, how many sheets gave the answer away, how many the model solved
  with the sheet. Broken down by question kind.
- For each way of working, how many hints it needs on average and the share of the sheet that
  is; for "just continue", the curve of "how often right" against "how many hints given".
- The table of step kinds against ways of working. This is the heart of the working-out half.
- The final scores on the test questions, with the cost in tokens per question.
- For remembering questions: what share of the needed facts the model can state, can only
  recognise, or does not have at all, and how that share predicts whether the question gets
  solved.
- For remembering questions: how often right against how many facts were hidden, for plain
  asking and for the best way of asking we found.
- The gap between "given correct hints" and "given its own earlier steps".

## 7. Cost and time

Small model: about 35,000 calls for steps 2 to 5 and 7 together, roughly 100 to 200 million
tokens generated. At 20,000 tokens per second across the servers that is a few hours of
generation. Big model: about 200 hint sheets, about 1,200 short judging calls, and a few hundred
calls for proposing wordings in step 8.

## 8. How to run it

Inside tmux, with a gpt-oss-20b server up (`Model_hosting/deploy_gpt_oss_20b.sh`):

```
bash run_hint_study.sh                  # steps 1, 2, 3, 4, 5, 7 and the report, in order
STEPS="2" bash run_hint_study.sh        # one step
ENDPOINTS=http://localhost:7472,http://localhost:7474 bash run_hint_study.sh   # several servers
```

Each step can also be run on its own with `--split train`, `--limit N` for a few questions,
or `--k 1` for a one-try smoke test, and `--out-dir` to keep a trial run apart from the real
one. The smoke test of 2026-09-23 ran every written step on a handful of questions with one try
each, in `outputs/hint_study_smoke/`.

The same study with Qwen3.5-9B as the small model:

```
bash run_hint_study_qwen9b.sh                                    # on the machine that serves the 9B
ENDPOINTS=http://<that machine>:7472 bash run_hint_study_qwen9b.sh   # or from here
```

The 9B is served on another machine, on port 7472. The script checks that the endpoint serves
Qwen3.5-9B before it starts. Its step 2 can run at any time; its step 3 adds to the shared
sheet file, so run steps 3 onwards after the gpt-oss run has finished its step 3
(`STEPS="2"` first, `STEPS="3 4 5 7 8"` later).

It writes to `outputs/hint_study_qwen35_9b/` and shares the hint sheets with the gpt-oss run
(`--teacher-dir outputs/hint_study`), so a sheet is written once and both models are tested on
the same sheets. The give-away check, the middle set and everything after are per model. For
Qwen, "full thinking" is thinking on and "light thinking" is thinking off; sampling follows the
Qwen baselines (top-p 0.95, top-k 20, presence penalty 1.5).

## 9. What could go wrong, and what we do about it

- The model runs out of room before answering. We tally it separately. If it dominates step 2,
  we repeat step 2 with thinking set to medium and report both.
- The big model disagrees with the key on many questions. Section 4.
- The big model's steps are not the steps the small model would take. Trimming the sheet in
  step 4 and measuring "how many hints" instead of "which step" both soften this.
- The critic and checker just agree with whatever they are shown. We saw this in the old
  pipeline. Step 5's judged sample and a count of "the checker repeated the mistake" will show
  it if it happens again.
- A recipe that works with correct hints may fall apart on its own steps. That is why step 6
  tests with no hints and reports the gap.
- A way of asking that has quietly memorised the facts. The train/test split and the big
  model's read-through in step 8.
- The model is near guessing on this pool: 13% right with ten choices. Every score is reported
  together with "right at least once in five tries", and with the number of questions it rests
  on.
- Five tries is a coarse ruler. "Right 4 of 5" can happen by luck for a question the model gets
  right 60% of the time. The test questions are the guard: nothing is claimed from the train
  numbers alone.

## 10. Results of the first gpt-oss run (2026-09-23)

The run took 9.8 hours, 5,418 small-model calls and 60 million generated tokens, far less than
planned, because the funnel collapsed. Of 200 questions, 13 reached the middle set.

| stage | train | test |
|---|---|---|
| drawn | 100 | 100 |
| fails alone (right at most 1 of 5) | 80 | 87 |
| big model agrees with the key | 25 | 33 |
| sheet did not give the answer away (after one rewrite) | 13 | 18 |
| middle (right at least 4 of 5 with the whole sheet) | 5 | 6 |

Three things caused the collapse, in order of size.

**Most of the keys are wrong.** The big model disagreed with the official answer on 125 of the
176 questions it wrote sheets for: 80% of working-out questions, 60% of mixed, 55% of
remembering. On those 125, the small model's own most common letter matched the big model 71
times and the key 6 times; Qwen 35B with thinking on, on the ones it had answered before,
matched the big model 37 times and the key never. Three models against the key. The three
disagreements checked by hand were all wrong keys. The pool was built by keeping questions a
model got wrong, and a question with a wrong key is always "got wrong", so the pool is where
the bad keys collected.

**The give-away check removes remembering questions by design.** 36 of the 58 sheets the big
model agreed on let the small model pick the answer from the sheet alone; the stricter rewrite
fixed 20. For a remembering question the facts are the answer: "Armstrong's first teacher, at a
juvenile home, made him the band leader" cannot be written without pointing at Peter Davis, and
a sheet that avoids the name does not help (the model still answers Joe Oliver). The check makes
sense for working-out questions, where a step can reveal the number, and not for remembering.

**The whole-sheet gate fails for reasons other than ability.** Of 31 usable sheets, 14 scored 0
of 5 with every hint shown. Three of those were five replies that ran out of room. Several are
the model being right in substance: on the money-supply question it answers "$50.4 billion" nine
hints in a row, and the key is the option "both $50.4 billion and $30 + $20.4 billion". Replies
ran out of room in 20% of all tries, and in 24 to 38% of tries on working-out and mixed
questions, against 2 to 5% on remembering ones; 38 of the 167 "fails alone" questions had three
or more such tries, so for them "fails" and "ran out of room" cannot be told apart.

What the 13 middle questions say, with the caution that 13 is too few for any conclusion:

- With every hint shown, plain "just continue" is right 4.9 of 5 times. Every two-call way of
  working is worse: critic first 3.5, check first 4.1, plan first 4.2, start over as an expert
  3.6, light thinking 3.6. Extra speakers add noise, as in the old pipeline.
- Trying three times and voting is the exception: 5.0 of 5 with all hints, the fewest hints
  needed (55 to 58% of the sheet, against 76 to 92% for the others), and 2.0 of 5 with no hints
  where plain continue gets 0.85.
- Sheets trim from 7 or 8 items to 2 or 3: most of what the big model writes is not needed.
- On the 10 remembering or mixed middle questions, of 32 facts the model can state 18, can only
  recognise 4, and does not have 10; 7 of the 10 questions need a fact it does not have.

Nine sheets were written for questions that did not fail alone (4 train, 5 test), in a second
pass after the first leak check. The funnel above leaves them out. Harmless, but the cause is
not clear from the files.

Changes before the next run:

1. Corrected keys. Where the big model disagrees with the key and the small model's own most
   common letter (or Qwen) agrees with the big model, use the big model's letter as the key and
   flag it; ask the big model a second time for the rest. This roughly doubles the usable pool.
2. No give-away check for remembering questions; keep the text flag and the fact checks of
   step 7, which are the real study for that kind. Keep the check for working-out questions.
3. Count a try that ran out of room as "no answer", not "wrong", and put questions with three
   or more such tries in their own pile. Report the gate both ways.
4. Lower the whole-sheet gate to 3 of 5, and treat "right in substance, wrong by the key's
   format" as a key problem, not a model failure.
5. Top up the draw from the labelled pool (1,615 questions remain) until each kind has at least
   30 middle questions per split.
