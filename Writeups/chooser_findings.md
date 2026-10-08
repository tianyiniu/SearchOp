# How the debate picks its final answer: findings (2 October 2026)

This note records what we learned about the last step of a debate. That step picks one final answer from the answers of the speakers. All work used gpt-oss-20b. Most of it used SuperGPQA run3. Some parts also used SuperGPQA run2 and HLE run1.

## Summary

1. A better chooser gives a small gain only. On the run3 test questions, the best chooser gets about 49%. Run3 gets 46.9% and external self-refine gets 48.1%. The gain over self-refine (about 1 point) is within noise. The goal is 51 to 53%.
2. Low-effort pairwise gives no gain. Its picks are close to chance.
3. Run3's programs have a flaw. When 3 of 4 solvers agree, the program outputs the answer of the last speaker, not the majority answer. This costs about 1.1 points. A ranking by mentions removes the flaw and gets 48.7% with no model calls.
4. In debates where only a minority gave the correct answer, the transcripts rarely show which answer is correct. Reviewers who knew the correct answer could clearly tell only 4 of 44 such debates. A judge has a low ceiling with these transcripts.

## Terms

| Term | Meaning |
|---|---|
| Correct answer | The answer that the dataset gives as correct (the answer key). |
| Solver | One model call that reads the question and commits one answer with its reasoning. |
| Effort | A gpt-oss-20b setting. At high effort, the model reasons for a long time. At low effort, it reasons for a short time. |
| Turn | The unit of cost in the search. A high-effort call costs 3 turns in run3. A low-effort call costs 1 turn. One question can use at most 20 turns. |
| Debate | One program run on one question, in one of the 3 recorded runs. |
| Mention | One speaker in one round that commits an answer. |
| Proposed answer | A different answer that appears at least one time in a debate. |
| Pool | The answers of one set of solvers to one question. We took all pools from recordings. We did not run new solvers. |
| 3H, 2H2L, 4H | Pool types. 3H: 3 high-effort solvers. 2H2L: 2 high-effort and 2 low-effort solvers. 4H: 4 high-effort solvers. |
| Split | A pool or debate with 2 or more proposed answers. Choosers run only on split pools and debates. |
| Chooser | A method that picks the final answer of a pool or debate. |
| Vote | The chooser that picks the answer with the most solvers. If there is a tie, it picks the answer that came first. Report files call it "plurality". |
| Judge | The existing chooser of the search. One high-effort call reads all answers with their counts and picks one. |
| Pairwise | A knockout chooser. The rank-1 answer is the holder. Each other answer challenges the holder, one at a time. The model reads two arguments and picks one. |
| Both orders | The pairwise rule that asks each pair two times, one time in each order. The challenger must win both times. |
| One order | The pairwise rule that asks each pair one time, with the holder first. The challenger wins with one pick. |
| Oracle | The percentage of debates where the correct answer appears at least one time. No chooser can do better. |
| Oracle gap | Oracle minus the actual accuracy. |
| Gain | The accuracy of a chooser minus the accuracy of the method in the comparison, in percentage points. |
| ± | The standard error of a gain, paired by question. A gain smaller than 2 times this number can be noise. |
| Correct flip | The chooser changes a wrong answer to the correct answer. |
| Wrong flip | The chooser changes the correct answer to a wrong answer. |
| Precision | Correct flips divided by all flips. |

## 1. Choosers on recorded pools

We compared choosers on the same recorded pools. We developed them on the train questions. We checked the best ones one time on the test questions.

Test questions, split pools only:

| Pool | Chooser | Accuracy | Gain over the vote | Correct flips | Wrong flips | Precision | Completion tokens per split pool |
|---|---|---:|---:|---:|---:|---:|---:|
| 3H | Vote | 27.0 | - | - | - | - | 0 |
| 3H | Judge | 32.0 | +5.1 ± 2.0 | 22 | 12 | 65% | 13k |
| 3H | Pairwise, both orders | 31.2 | +4.2 ± 1.7 | 18 | 8 | 69% | 37k |
| 2H2L | Vote | 36.5 | - | - | - | - | 0 |
| 2H2L | Judge | 41.3 | +4.7 ± 1.5 | 34 | 13 | 72% | 11k |
| 2H2L | Pairwise, both orders | 41.6 | +5.1 ± 1.2 | 28 | 7 | 80% | 39k |

Test questions, all pools:

| Program | Accuracy | Gain over external self-refine | Gain over run3 |
|---|---:|---:|---:|
| Run3 (recorded) | 46.9 | −1.2 ± 1.4 | - |
| Run2 (recorded) | 48.3 | +0.2 ± 1.5 | +1.4 ± 1.2 |
| External self-refine (recorded) | 48.1 | - | +1.2 ± 1.4 |
| 3H + vote | 47.4 | −0.7 ± 1.3 | +0.6 ± 1.0 |
| 3H + judge | 49.4 | +1.3 ± 1.4 | +2.6 ± 1.0 |
| 3H + pairwise | 49.1 | +1.0 ± 1.3 | +2.2 ± 0.9 |
| 2H2L + judge | 48.8 | +0.7 ± 1.3 | +1.9 ± 1.1 |
| 2H2L + pairwise | 49.0 | +0.9 ± 1.3 | +2.1 ± 1.1 |
| 4H + vote | 48.0 | −0.1 ± 1.3 | +1.1 ± 1.1 |

Findings:

- On test, the judge and pairwise are equal within noise. Pairwise minus judge is −0.3 ± 0.7 on 3H pools and +0.2 ± 0.8 on 2H2L pools.
- The judge is not consistent. On the train 3H pools, the judge is worse than the vote (−1.2 ± 2.4). On test, it is worse than the vote in group 1.
- Pairwise has a gain in every group and on both question sets. It costs about 3 times more tokens than the judge.
- We tested three more choosers on train. None was better than the judge:
  - Blind judge (no counts, no speaker names): +1.6 on both pool types.
  - Resolve (a new solver sees only the proposed options): −7.2 on 3H pools.
  - The `verify` chooser (one call checks each answer): +0.4 on 3H pools and +3.4 on 2H2L pools, but it changes few answers.
- The probe tested pairwise on pairs of one correct and one wrong argument:
  - The model picks the correct answer in about 66% of pairs.
  - When the correct argument has high effort and the wrong argument has low effort, the model is correct in 95%. In the reverse case, it is correct in 18%.
  - When fewer solvers gave the correct answer, the model is near chance (44 to 53%).

## 2. The oracle gap

Test questions:

| Program | Accuracy | Oracle | Oracle gap | Best chooser | Gap recovered |
|---|---:|---:|---:|---:|---:|
| Run3 | 46.9 | 58.1 | 11.2 | - | - |
| Run2 | 48.3 | 49.8 | 1.4 | - | - |
| Self-refine, high effort, in the search code | 48.0 | 51.1 | 3.1 | - | - |
| 3H + vote | 47.4 | 58.0 | 10.6 | 49.4 (judge) | 19% |
| 2H2L + vote | 45.9 | 63.7 | 17.8 | 49.0 (pairwise) | 17% |
| 4H + vote | 48.0 | 61.1 | 13.1 | not measured | - |

The oracle gap has two parts. The numbers are percentages of all debates.

| Pool | Tie: the vote picked a wrong answer from a tie | Minority: one solver had the correct answer, two or more agreed on one wrong answer |
|---|---:|---:|
| 3H | 3.4 | 7.1 |
| 2H2L | 4.8 | 13.0 |
| 4H | 3.9 | 9.2 |

Choosers recover about 25 to 40% of these cases. They also change some correct votes to wrong answers. The net gain is about 2 to 3 points.

## 3. Cheaper pairwise rules

We replayed the recorded pairwise calls under cheaper rules. The accuracies use the recorded calls, so they are not estimates. The token counts of the shorter rules are estimates.

Split pools have few proposed answers: 2.2 to 2.5 on average. So pairwise makes only 1.2 to 1.5 matches per pool. Most of its cost comes from two calls per match and from long high-effort calls.

Test questions, all pools:

| Rule | High-effort calls per split pool (3H / 2H2L) | 3H accuracy | 2H2L accuracy |
|---|---|---:|---:|
| Vote | 0 / 0 | 47.4 | 45.9 |
| All answers, both orders | 2.4 / 2.9 | 49.1 | 49.0 |
| Top 3 answers, both orders | 2.4 / 2.75 | 49.1 | 49.0 |
| Top 2 answers, both orders | 2 / 2 | 49.0 | 48.8 |
| Top 2 answers, one order | 1 / 1 | 49.1 | 48.9 |
| Judge | 1 / 1 | 49.4 | 48.8 |

On train, "top 2 answers, one order" got 52.0% (3H) and 52.5% (2H2L). The judge got 49.8% and 50.3%. On test, the two are equal.

Cautions:

- We chose the "top 2, one order" rule after we saw these results. The pattern is the same on train and test, but a new check is necessary.
- An earlier version of this analysis also had rows for "all answers, one order". Those rows had a replay error in pools with 3 or more answers. This table does not use them.

## 4. Low-effort pairwise on the recorded test debates

Script: `scripts/pairwise_rerank.py`. Run script: `run_pairwise_rerank.sh`.

The script replays every test debate of the routed programs. It ranks the proposed answers:

1. By the number of mentions, more first.
2. Then by the last round in which the answer appears, later first.
3. Then by the first commitment, earlier first.

Then it runs a low-effort pairwise knockout on all proposed answers. It uses both rules.

Accuracy, as a percentage of all test debates:

| Run | Recorded final answer | Rank-1 answer (no model calls) | Pairwise, both orders | Pairwise, one order | Oracle |
|---|---:|---:|---:|---:|---:|
| SuperGPQA run2 | 48.3 | 48.2 | 48.1 | 48.1 | 49.8 |
| SuperGPQA run3 | 46.9 | 48.7 | 48.4 | 46.6 | 58.1 |
| HLE run1 | 12.7 | 12.0 | 12.5 | 12.2 | 16.7 |

Paired differences from the recorded final answer:

- Run3: rank-1 answer +1.8 ± 0.8. Both orders +1.6 ± 0.9. One order −0.3 ± 1.0.
- Run2 and HLE: no rule differs from the recorded final answer by more than its noise.

Why low-effort pairwise fails (run3, matches where exactly one answer is correct):

- It picks the correct answer in 54 to 59% of calls. The high-effort probe got about 66%.
- When the correct answer has fewer mentions, it picks it in 36% of calls. This is worse than chance.
- It picks the second response in 59% of calls. The one-order rule always shows the challenger second. So that rule changes many answers, with 39% precision.

Where the correct answer is, as a percentage of all debates:

| Run | Rank 1 | Rank 2 | Rank 3 | Rank 4 or lower | Not proposed |
|---|---:|---:|---:|---:|---:|
| SuperGPQA run2 | 48.2 | 1.6 | 0 | 0 | 50.2 |
| SuperGPQA run3 | 48.7 | 6.8 | 2.3 | 0.3 | 41.9 |
| HLE run1 | 12.0 | 4.3 | 0.3 | 0 | 83.3 |

Proposed answers per debate: run3 has 1 answer in 65% of debates, 2 in 25%, 3 in 8% and 4 or more in 2%. Run2 has 1 answer in 94%. HLE has 1 answer in 55% and 2 in 36%.

Running pairwise on the top 3 answers or on all answers gives the same result. The difference is at most 0.1 points.

## 5. A flaw in the run3 programs

Run3's programs for groups 0 and 1 start with 4 high-effort solvers. If 3 of them agree, the program stops and outputs the answer of the last speaker. It does not output the majority answer.

Measured on all 900 run3 test debates:

- In 166 debates, 3 of 4 solvers agreed and the program stopped after round 1.
- In 50 of these debates, the last speaker was the lone dissenter. The program output the dissenter's answer.
- The dissenter was correct in 11 of these debates. The majority was correct in 21.
- The net loss is 10 debates, about 1.1 points.

The rank-1 answer of section 4 removes this flaw. This explains most of its gain of +1.8 points on run3.

## 6. A detailed reading of 100 debates

### Method

1. We took a random sample of 100 run3 test questions. Each one had a split debate that proposed the correct answer. 111 questions met this condition. We used one debate per question.
2. In 56 debates, the correct answer had rank 1. In 44 debates, a minority gave it (rank 2 or lower).
3. Ten reviewers (Claude subagents) read 10 debates each, in full. The reviewers knew the correct answer.
4. For each answer, the reviewers recorded the signals below and the type of error. They also rated whether a careful judge without the correct answer could tell which answer is correct.

Caution: the sample files label every high-effort speaker "ran out of room". This label is wrong. It shows that the speaker used 2 calls, which almost every high-effort speaker does. The real signal is the text "(no reply: the reasoning ran out of room)" at the start of a reply.

### Signals

| Signal | Meaning |
|---|---|
| Exact match | The computed value or conclusion of the speaker matches the text of its option exactly, not only "closest". |
| Hedged | The decisive step has words such as "closest", "likely", "approximately" or "assume". |
| Contradicts itself | The result or summary of the speaker does not agree with the letter it commits, or the speaker changes its answer in the reply. |
| Sanity check | The speaker makes a check that passes: units, a limiting case, or substitution of the answer back into the question. |

For each debate, we compared the correct answer with the wrong answer that has the most mentions. We counted only the debates where one side shows the signal and the other side does not. Each cell gives "correct side only / wrong side only".

| Signal | All 100 debates | Correct answer at rank 1 (56) | Correct answer from a minority (44) |
|---|---|---|---|
| Exact match | 23 / 5 | 18 / 3 | 5 / 2 |
| Hedged | 4 / 32 | 1 / 24 | 3 / 8 |
| Contradicts itself | 5 / 28 | 1 / 22 | 4 / 6 |
| Sanity check | 19 / 5 | 17 / 2 | 2 / 3 |

The signals are strong over all 100 debates. But almost all of their strength comes from debates where the majority is already correct. In the 44 debates that a judge must change, the signals are weak.

### Can a judge tell?

The reviewers answered "Could a careful judge tell the correct answer from the transcripts alone?":

| Debates | Yes | Partly | No |
|---|---:|---:|---:|
| Correct answer at rank 1 (56) | 38 | 16 | 2 |
| Correct answer from a minority (44) | 4 | 25 | 15 |

### Types of error in the 136 wrong answers

| Type | Count |
|---|---:|
| Wrong fact or formula | 67 |
| Assumption that the question does not support | 17 |
| Misread question | 12 |
| Correct conclusion, wrong letter | 12 |
| Other (mostly no visible reasoning) | 11 |
| Unit or convention error | 9 |
| Nearest option, no exact match | 4 |
| Calculation error | 4 |

Half of the wrong answers use a wrong fact or formula. A judge that is the same model can find this error only if it knows the fact.

### Checks that need no correct answer

We tested two signals that a script can measure, on all 900 run3 debates:

- Hedging words, found by a text search. In split debates, speakers with these words are correct 29% of the time. Speakers without them are correct 32% of the time. A rule that moves away from a hedged rank-1 answer gets 48.6%. The rank-1 answer alone gets 48.7%. So this signal needs real reading, and the reviewers' "hedged" labels may contain hindsight.
- No visible reply (the reasoning used all of its tokens). In split debates, high-effort speakers with no visible reply are correct 20% of the time. Speakers with a reply are correct 34% of the time. But a ranking that counts these speakers for less gets 48.6%. It almost never changes the rank-1 answer.

### Patterns from several batches

The reviewers found these patterns by reading. We did not measure them, and each one has exceptions.

1. The result of a speaker points to a different option. For example, a speaker computes 8 and commits the option "6.5". Or its value is the text of another option. All such speakers were wrong. Five or more batches found this.
2. A side says that the rival option is "also correct". Then it picks its own option by a tie-break, such as "more specific". The rival option was often the correct answer. There are two exceptions (debates 039 and 041).
3. A speaker adds a value or condition that the question does not give, or does not use data that the question gives. This speaker was usually wrong.
4. Critics usually agree with the majority. They rarely changed a debate to a correct minority answer. One batch was different: there, critics broke tied first rounds correctly in 4 of 4 debates.
5. The speakers of a majority often share one line of reasoning. So their agreement is weaker evidence than it looks.

## 7. Conclusions

- The most certain gain comes from a change to the final read of the programs. Use the vote, or the rank-1 answer, not the answer of the last speaker. On run3, this gives about +1.8 points with no extra calls.
- A judge of these transcripts has a low ceiling. In the debates where a minority has the correct answer, the transcripts mostly do not show which answer is correct. So no judge can probably recover most of the oracle gap.
- The high-effort judge and pairwise give about +2 points over the vote on recorded pools, but cost 1 to 3 high-effort calls. The user stopped work on high-effort judges for now.
- Low-effort pairwise gives no gain. Drop it.

## 8. Open options

1. Change the final read in the program language or in the seeds. This change belongs to the integration step of the plan.
2. A cheap test of an extraction step. A low-effort call reads each argument and answers two questions:
   - Which option does the result of this argument match exactly?
   - Does the argument say that another option is also correct?

   Then the step counts each speaker for the option that its own result supports. This needs about 1,400 low-effort calls on run3. The gain is probably 1 point or less.
3. Medium effort for pairwise is not tested. The script supports it:

   ```bash
   EFFORT=medium OUT=outputs/pairwise_rerank_medium bash run_pairwise_rerank.sh
   ```

## 9. Files

| File | Content |
|---|---|
| `scripts/gap_report.py` | The oracle gap of a test evaluation, from recordings. It makes no model calls. |
| `scripts/chooser_lab.py`, `run_chooser_lab.sh` | Choosers on recorded pools, and the probe. |
| `outputs/chooser_lab/gpt_oss_20b/{train,test}/` | Reports of the chooser lab. |
| `scripts/pairwise_rerank.py`, `run_pairwise_rerank.sh` | The rank-1 answer and low-effort pairwise on recorded test debates. |
| `outputs/pairwise_rerank/summary.md`, `outputs/pairwise_rerank/<run>/report.md` | Reports of section 4. |
| `outputs/judge_patterns/sample/` | The 100 sampled debates, one file per debate, and `index.json`. |
| `outputs/judge_patterns/notes/` | The reviewers' notes, one JSON line per debate. |
| `outputs/judge_patterns/scripts/` | The analysis scripts of sections 2, 3, 5 and 6. |
