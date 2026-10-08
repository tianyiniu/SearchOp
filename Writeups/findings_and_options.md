# SearchOp: what we found, and what to change (5 October 2026)

This note collects what six searches taught us. It then lists changes, from small to large, with what the literature says about each. All numbers come from our recordings unless a source is named.

## Summary

1. No search beat external Self-Refine. All six ended within 1.5 points of it, above or below.
2. The main cause is the model, not the search. On SuperGPQA, Qwen3.5-9B gives the same answer to most questions every time. Voting over several independent thinking-on replies gains nothing. Every program only rearranges calls to that one model, so no program can gain much.
3. The right answer often appears, but mostly as a minority answer. Every method we tried for picking a right minority answer gained 1 to 2 points at most.
4. The search picks programs by luck. A new program and its parent differ on only 8 to 9 questions per group, so each keep-or-drop choice is close to a coin toss.
5. Question groups made from the question text do not predict which program works best.
6. One run takes about 20 hours. We cannot measure a 1-point effect at that speed.
7. Two bugs changed what ran: the turn cap cut programs short, and the dev split was off in one run. Both are fixed.
8. The prompt matters more than the program. Our own speaker prompt is about 3 points worse than the official SuperGPQA prompt. The effect is the same in every group, so per-group prompts do not help.
9. The literature agrees with all of this. For strong reasoning models, multi-agent designs, voting refinements and prompt optimization each give 0 to about 1.5 points on knowledge questions. Larger gains need new information, different models or a trained answer picker.

## Terms

| Term | Meaning |
|---|---|
| Program | A debate recipe: which speakers run, in which order, with which stop rule. |
| Seed | A program we write by hand or ask a model to write. The search starts from the seeds. |
| Child | A new program made by one random edit to an existing program, its parent. |
| Group | A set of questions with similar text descriptions. Each group gets its own program. |
| Routed | Each test question gets the program of its group. |
| Global | One program for all questions. |
| External Self-Refine (ext SR) | Self-Refine run outside our pipeline, with the dataset's official prompt. This is the baseline we must beat. |
| External direct | One answer per question with the official prompt. |
| Executor | Our code that runs a program. Its speakers use our own prompts, not the official one. |
| Thinking on / off | Qwen's reasoning switch. Our code calls these high and low effort. |
| avg@3 | Mean accuracy over 3 independent runs. |
| ± | The standard error of a difference, paired by question. A difference smaller than 2 times this number can be noise. |
| Independent replies | Replies from separate calls that share nothing. Different programs often share replies, so their answers are not independent. |

## 1. Results of all searches

Test questions: 300 SuperGPQA questions (HLE: 200). Differences are in avg@3.

| Run | Model | Searched program | Ext SR | Difference | Hours |
|---|---|---:|---:|---:|---:|
| cluster-pipeline run2 | gpt-oss-20b | 48.3 | 48.1 | +0.2 | - |
| cluster-pipeline run3 | gpt-oss-20b | 46.9 | 48.1 | −1.2 | - |
| v3 HLE run1 | gpt-oss-20b | 12.7 | 11.7 | +1.0 | - |
| global-pipeline run1 | gpt-oss-20b | 49.3 | 48.1 | +1.2 ± 1.3 | 6.8 |
| global-pipeline run1 | Qwen3.5-9B | 59.1 | 58.6 | +0.6 ± 1.2 | 20.1 |
| cluster-pipeline run1 | Qwen3.5-9B | 57.1 | 58.6 | −1.4 ± 1.3 | 22.2 |

In HLE run1, the turn-cap bug (section 5) cut 3 of the 4 group programs short.

## 2. The model leaves little to gain

These numbers use independent replies only.

| Qwen3.5-9B, SuperGPQA | One reply | Most common of 3 | Most common of 5 | Most common of 15 | Right answer in any reply |
|---|---:|---:|---:|---:|---:|
| Thinking on, test | 55.2 | 55.5 | 55.1 | - | 72.7 |
| Thinking on, train | 57.0 | 57.8 | 57.5 | - | 73.7 |
| Thinking off, test | 52.7 | 53.7 | 54.5 | - | 75.0 |
| Thinking off, train | 54.2 | 56.1 | 56.9 | 57.6 | 81.3 |

- With thinking on, voting gains nothing. With thinking off, it gains 2 to 3 points, but only reaches one thinking-on reply.
- On test, over 6 independent thinking-on replies per question:
  - 124 questions: all 6 right; 82 questions: all 6 wrong.
  - 30 questions: 4 or 5 right, so a vote helps.
  - 53 questions: only 1 or 2 right, so a vote hurts.
  - 11 questions: 3 right.
- The two effects cancel. A gain must come from picking a minority answer when it is right.
- External runs show the same. On test, the most common of 3 direct and 3 Self-Refine answers scores 59.0%. External Self-Refine alone scores 58.6%.
- A second model does not add much. All 6 external Qwen runs are wrong on 82 test questions. On those, gpt-oss-20b is right in 4 or more of its 6 runs on only 4 questions, and right at least once on 18.

## 3. Picking a minority answer fails

From the chooser study (chooser_findings.md; gpt-oss-20b):
- On debates with 2 or more answers, a high-effort judge or a pairwise tournament gains 4 to 5 points over the vote.
- Over all questions this is only about 1 point above external Self-Refine. That is within noise.
- Reviewers who knew the right answer could tell the right side in only 4 of 44 debates where it was a minority. The transcripts rarely show which side is right.

## 4. The search cannot see 1-point effects

- The good programs differ by about 1 point. Estimated from 188 programs on train, the true spread is about 1.2 points (standard deviation).
- A child shares its parent's saved replies. The two differ in mark on only 8 to 9 questions per group. So a child's gain over its parent has a noise of about ±2 points (group 0), ±3 (group 1) and ±5 (group 2).
- Children beat their parent 31 to 38% of the time. On average they changed the score by about 0.
- Children chosen on their first run lost 3.3 points on a second run of the same questions. The seeds, which were not chosen, gained 2.5 points.
- Group picks made on 2/3 of train dropped from 65.4% to 58.7% on the other 1/3. One program for all questions dropped from 62.3% to 59.9%.
- v4 chose its final programs on a dev split of 100 questions. They still landed within 1.2 points of Self-Refine.

## 5. Other problems we found

- **Question groups.** Groups made from text do not predict which program wins. Groups made from outcomes are real (they repeat across runs), but neither text nor first-round features predict them. Routing ceilings, scored on held-out runs, were +0.4 to +2.4 points, and no real router came close.
- **Executor prompt.** Our executor's single thinking-on reply scores 55.2% on test and 57.0% on train. External direct, with the official prompt, scores 57.4% and 60.3%. Every program inherits this gap.
- **Bug: turn cap (fixed 5 October).** Edits could make a plan longer than the turn cap. The executor then stopped it early without any warning.
  - The v3 Qwen global program ran as "2 solvers, then a critic", not as written.
  - In HLE run1, 3 of 4 group programs were cut.
  - Both searches now redraw such children, with tests.
- **Bug: dev split off (fixed).** In the v3 Qwen run, a script line forced the dev split off. That run chose its programs on train only.
- **Slow runs.** The v3 Qwen run took 22.2 hours:
  - baselines: 2.0 h;
  - generation 0: 3.9 h;
  - generations 1 to 10: 11.6 h;
  - test: 4.8 h.
  The server makes about 13 million completion tokens per hour.

- **Too few test questions.** These numbers use the variance we measured between external Self-Refine and external direct (80% power, 5% test).

  | Questions needed to detect | 1 point | 2 points | 3 points |
  |---|---:|---:|---:|
  | One run per method | 10,485 | 2,621 | 1,165 |
  | avg@3 per method | 3,953 | 988 | 439 |

  At avg@3, our 300 test questions only detect differences of about 3.6 points or more. So the +3 goal could not be confirmed even if it were met. The 1k paper test split lowers this to about 2 points.

## 6. What the literature says

A literature search (three agents, October 2026) gave the results below. I rechecked the sources marked "checked" against their arXiv pages. The others are from the agents' reading.

### 6.1 Strong models gain little from multi-agent designs

- **The Illusion of Multi-Agent Advantage** (Jwalapuram et al., June 2026; checked).
  - It tests six automated designers (including AFlow, MaAS, ADAS and MAS-Zero) against chain-of-thought self-consistency.
  - Abstract: "automatic MAS consistently underperform CoT-SC despite being up to 10x more expensive."
  - Agent's reading: per-question routers collapse to one choice; 84.9% of GPQA-Diamond questions go to debate.
- **Debate or Vote** (NeurIPS 2025): majority voting explains most of the gain credited to debate.
- **Stop Overvaluing Multi-Agent Debate** and **MASLab** report the same: the gap between multi-agent systems and one agent shrinks as models get stronger.
- **MAS-Zero** with o3-mini on GPQA-Diamond: self-consistency 72.7, debate 77.8, MAS-Zero 76.8. This uses 166 to 198 questions, so it is within noise. The large gains in the automated-design papers (MaAS, AFlow, ScoreFlow) all use weak models such as gpt-4o-mini.
- **Recursive Self-Aggregation** (Venkatraman et al.; checked) tested 1,000 SuperGPQA questions with Qwen3-4B-Instruct:
  - base 41.85, self-refinement 43.5, majority vote 48.2, RSA 47.39;
  - quote: "RSA achieves superior results on all tasks except SuperGPQA, where majority voting is particularly effective due to the multiple-choice answer format."
- Every one of our findings has a published match: no voting gain, debate no better than voting, per-question choice collapsing, and small differences lost in noise.

### 6.2 Picking the right answer

- **Signals from the model itself do not beat the vote on science questions.**
  - **DeepConf** (checked): on GPQA-Diamond, confidence filtering changes accuracy only "within ±1.5 pp". On math it gains up to 14 points.
  - **Reasoning Concentrates Errors** (Althoubi, Sept 2026; checked): "not one beats plain majority voting after correction" across 280 method, dataset and model combinations. Weighted voting agrees with plain voting 98.5% of the time.
  - Self-certainty, entropy-weighted voting and CISC: +0 to +1.5 points.
- **Trained pickers are the only ones with larger gains.** All of these are from the agents' reading.
  - VersaPRM, a trained 8-billion-parameter step scorer, gains +4.1 on MMLU-Pro. It was trained on MMLU-Pro data, with a model that does not think.
  - AggLM, a trained 1.7-billion-parameter combiner, gains +0.8 to +8.4 on math. Most of its gain comes when the top answer has few votes. Math only.
  - Minority Sentinel (checked) is a small classifier on debate logs. It overturns the majority with 81.2% precision. It uses debates between 3 different models.
- **Hariri et al.** (Aug 2026; checked) released 1.4 million sampled attempts, including a SuperGPQA set. The agent read that it covers gpt-oss-20b at high effort, 80 samples per question, with mean accuracy 45.0% and 81.9% right in at least one sample. The Hugging Face page needs a login, so the file contents are not checked.

### 6.3 Faster, more reliable experiments

- **Record once, test offline.** DeepConf and Large Language Monkeys record many samples per question once. They then score any voting, stopping or picking rule by drawing subsets from the record many times. One rule takes seconds to score, not hours.
- **Compare methods on the same recorded replies.** The shared sampling noise then cancels.
- **Small IRT-chosen question sets** (tinyBenchmarks, metabench) estimate a model's score. They cannot separate methods 1 to 2 points apart. Use the full question set.
- **Prefix caching** does not help us, because long replies dominate the time.
- **Small models as stand-ins:** no evidence that method rankings carry over from 4B to 9B or 20B. Some evidence says they do not.

### 6.4 Prompt optimization, including per-cluster prompts

**Our data (free; first-round replies on the same 300 train questions, thinking on):**

| Prompt | Accuracy | Minus our Solver prompt | Group 0 | Group 1 | Group 2 |
|---|---:|---:|---:|---:|---:|
| Official SuperGPQA prompt (external direct) | 60.3 | +3.3 ± 1.2 | +2.8 | +2.9 | +5.6 |
| Verifier persona | 59.0 | +2.0 ± 1.5 | +2.5 | +0.8 | +2.6 |
| Expert persona | 57.2 | +0.2 ± 1.4 | +2.2 | −2.2 | −0.9 |
| Synthesizer persona | 57.2 | +0.2 ± 1.7 | +0.5 | +1.3 | −2.6 |
| Solver persona (ours) | 57.0 | 0 | 0 | 0 | 0 |
| Critic persona | 54.7 | −2.3 ± 1.7 | −3.0 | −2.2 | −0.9 |

- Prompts matter: our Solver prompt costs about 3 points. That is the largest real effect we have measured, larger than any program effect.
- The gap is not answer parsing. On test, no speaker ended without a letter, and only 2 of 19,500 needed a nudge.
- No persona wins in one group and loses in another beyond noise (group standard errors are 1.4 to 5.4 points). One global prompt would capture the effect.
- Mixing prompts adds no variety. Three replies with three different prompts agree on 69% of questions. Three Solver replies agree on 68%.

**Literature:**
- **ReasonLab** (Preet et al., 2026; checked) tested 8 prompting techniques on 10 multiple-choice datasets. With reasoning on, "no technique differs by more than 0.51 pp." Without reasoning, they gain 3.9 to 4.7 points.
- **MAS-PromptBench** (Bai and Shi, June 2026; numbers checked; the agent reads the task model as Qwen3.5-9B with thinking off) applied GEPA and MIPRO, extended to multi-agent systems.
  - GPQA-Diamond: one agent 54.0 → 58.0. Multi-agent setups: 0 to +3.
  - Average on reasoning tasks: +1.4.
  - The gain shrinks as teams grow: +2.4 with 2 agents, then −0.9 with 8 and −2.1 with 10.
- **GEPA** (ICLR 2026) works with a local model that reflects on its own traces (Qwen3-8B). Its math gains are small: AIME-25 27.3 → 32.0 on 30 questions.
- **Per-cluster prompts** gained only with older models that do not think: Mixture-of-Prompts (+11, including few-shot examples) and APS (+3 to +5). The agent found no study for thinking models on knowledge questions.
- **Overfitting is common.** OPRO had a 5 to 20 point train-test gap, and MIPROv2 lost 7.3 on AIME.

**Cost for us:** a prompt change cannot reuse saved replies. To detect a 2-point gain, each candidate prompt needs about 2,000 questions with fresh calls, about 1.5 to 2 GPU-hours with thinking on. Per-cluster evolution multiplies this by clusters × roles × generations, and each cluster has fewer questions, so it would again pick noise.

**Verdict:** per-cluster evolved prompts are unlikely to beat the official prompt by 3 points with a thinking model. One step is cheap and certain: build every speaker's prompt on the official SuperGPQA prompt. That recovers about 3 points for every program. But it only brings us level with the baselines, which already use that prompt.

### 6.5 Effective, but outside our limits

- **Web search** (excluded: security and resource limits). Search-o1 raised QwQ-32B on GPQA-Diamond from 58.1 to 63.6 (checked). Plain retrieval gave only 58.6.
- **Paid frontier models** (excluded: no API budget). This rules out:
  - frontier verifiers and judges, such as Weaver's large verifier set;
  - frontier designer models;
  - our own paid parts: the describer and seed writer (gpt-6-sol) and the HLE judge (gpt-6-luna).
- **Many different open models with a per-question choice** (Skill-MoE, ICML 2026; checked): +8.15 points on average over the best baseline, with 16 models on GPQA, MMLU-Pro, AIME and MedMCQA. Self-MoA warns that mixing models hurts when one model is clearly stronger. Our data show this: gpt-oss-20b adds almost nothing to Qwen3.5-9B on SuperGPQA.

## 7. Options

Any option should start with the same change to how we run experiments:
- Record a bank of independent replies once.
- Score methods offline on the same recorded replies.
- Use 1,000 or more questions.
- Confirm only the final 1 to 3 methods with one live run on the test split.

This replaces the 20-hour live search. An iteration then takes minutes. It also drops the paid describer.

| Option | What changes | Evidence for a gain | Iteration time | Main risk |
|---|---|---|---|---|
| A. Analysis paper | Report why per-question program search does not beat voting for a strong model: bimodal per-question accuracy, noise in selection, groups that do not transfer. Use the 4-dataset baseline tables. | The literature agrees with every finding. This is a strong negative result if it is well powered. | Days. Most data exist. | The Illusion paper (June 2026) covers part of it. We need our own angle, such as the coverage-versus-selection breakdown and the noise analysis. |
| B. Trained answer picker | Train a small local model (for example Qwen3.5-4B) on banked replies to SuperGPQA questions that are not in the test split. It reads the candidate answers and picks one. Per-question choice is then built in. | VersaPRM +4.1 (trained in-domain), AggLM up to +8.4 (math). Our room: the right answer appears in 73 to 80% of questions, against 55% accuracy. | The bank once (about 10 to 20 GPU-hours), then hours per training run. Scoring takes minutes. | Our 44-debate review found the right minority side hard to see. The gains may not carry over from math. |
| C. Several open models, per-question choice | Pick a model, or a combination, for each question from a pool of local open models of similar strength. | Skill-MoE +8.15 on average. Self-MoA: only with models of similar strength. | One bank per model, then minutes. | Our first check is negative: gpt-oss-20b mostly gets right only 4 of the 82 questions Qwen always misses. Hosting many models takes GPU time. |
| E. Evolve prompts per cluster (not recommended) | Search persona prompts per question group, together with the programs. | Our data: prompts move accuracy about 3 points, but the same way in every group. Literature: thinking models gain at most about 0.5 to 1.4 points from prompt changes. | About 1.5 to 2 GPU-hours per candidate prompt (no saved replies can be reused). | Overfitting and noise, as in our program search. |
| D. Save compute instead (rejected) | Thinking off first, and thinking on only when two thinking-off replies disagree. | Tested on our recordings: same accuracy as one thinking-on reply, but only 5% fewer tokens. Qwen's thinking-off replies are long (about 3.7k tokens each). | - | Too small to report. |

**Decision (5 October): A and C are rejected.** The user wants a method paper, not an analysis paper. Other open models would change the baselines. Reviewers would then expect frontier models in the pool. A cheaper model pool always exists, which also weakens any efficiency claim. New work uses one base model, the same one as the baselines.

### Option B in detail: a trained answer picker on the same model

**Room on our test recordings** (6 thinking-on replies per question):
- On 134 of 300 questions, the replies give 2 or more different answers.
- On 41 of those, the vote is right.
- On 53, the right answer is there, but the vote misses it.
- On 40, the right answer is absent.

**What +3 needs:** override the vote correctly on 9 more questions than it does wrongly. That takes about 23 overrides at 70% precision, or about 15 at 80%. Our untrained judges, tested on gpt-oss, overrode with 65 to 80% precision, but too rarely.

**Training data:** 24,529 SuperGPQA questions on disk are in none of our splits.

**Stages, each with a go/no-go check:**
1. **Record a bank of replies with the official prompt and thinking on.**
   - Train bank: about 2,000 unused questions × 6 replies, about 7 GPU-hours.
   - Test bank: the Qwen3.5-9B baseline run on the 1k test already records 15 direct replies per question, for self-consistency. Those replies serve as the test bank at no extra cost.
2. **Offline, minutes:** measure the room on the bank, and score the vote and the untrained model as a judge.
3. **Train a picker** (LoRA on Qwen3.5-9B itself). It reads the question and the candidate replies, then picks one letter. Score it on held-out bank questions.
   - Go only at +2 or more over the vote, with 70% or more precision on overrides.
   - Training needs the server stopped for some hours.
4. **Confirm once** on the 1k test, against all four baselines.

## 8. Next steps

Two checks cost no GPU time and decide between A, B and C:
1. **When the paper baseline runs finish:** compare Qwen3.5-4B, Qwen3.5-9B and gpt-oss-20b on the same test questions.
   - How often does another model get right a question that the strongest model misses? This decides C.
   - On each dataset, how far apart are one reply, the vote and "right in any reply"? This decides where B has room.
2. **Optional:** download the Hariri SuperGPQA bank (needs a Hugging Face login).
   - It has 80 gpt-oss-20b samples per question, which lets us measure how much a perfect picker could gain.
   - If it has token probabilities, we can test confidence signals there with no GPU.

Whatever we choose, every speaker's prompt should start from the official prompt of each dataset.

Open issue: the HLE baseline runs grade answers with gpt-6-luna, which is paid. Under the no-API rule, HLE needs either the cached verdicts or a local judge.
