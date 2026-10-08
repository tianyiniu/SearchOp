# Pipeline to-dos

Reorganised 2026-09-22 after the stage-by-stage walkthrough and the SAT paper (arXiv 2609.22682).
Grouped by pipeline stage so they can be discussed in meeting order. Each item carries a priority
(P0 = do before any new search run, P1 = next run, P2 = later or optional) and the evidence behind it.

## Diagnosis in three lines

- The fitness score is too noisy to select on. A program picked on 25 search questions gains
  +0.3 to +2 points on the other 25 of its own cluster, and 0 on held-out and dev.
- Debate rounds copy rather than correct. Speakers who see earlier answers repeat the leading
  letter 75-90% of the time on gpt-oss and 87-99% on Qwen; fixes and losses cancel.
- Our executor runs speakers with thinking off and a 6,144-token cap. On Qwen that alone costs
  ~7 points (one call: 24.8% in-executor vs 31.6% thinking on). gpt-oss is at chance either way.
  In SAT's terms our whole method sits inside their "linearization" control: one model playing
  every role. Their heterogeneous-roster gain (66.7% vs 56.0% homogeneous) is not available to us.

## Stage 0: the executor (P0, Qwen 35B: ~7 points)

- [ ] P0 Re-run the dev evaluation only, same programs, with thinking ON for every speaker and no
      (or a much larger) reply cap. If routed lands near the 31% of thinking-on direct, the
      deficit is settings, not structure. The Qwen deep-think run's `deep_direct` row is a partial
      version of this test.
- [ ] P0 Decide the executor settings for all future searches from that result. 11% of solver and
      25% of independent replies on Qwen hit the cap before committing.

## Stage 1: describing, clustering, routing

- [ ] P1 Measure describer noise. The route depends on one medium-effort call; 25% of questions
      sit within 0.10 of a second medoid. The free check (the 20 anchors, described twice) is
      contaminated: the anchor and its record are in the prompt, so the pass reproduced the draft
      verbatim. Re-describe 100 dev questions into a separate file, embed, route, count flips
      (commands in the 2026-09-22 chat). Proxy so far: 32/32 "what year" questions got identical
      labels and group; "which statement is correct" questions split 6/2/1 across groups 3/2/0.
- [ ] P2 Consider k=3. On dev the six groups behave as three pairs: 0 and 2 (recall vs category)
      are each other's runner-up on 165 of 183 questions, 1 and 5 (calculation with vs without
      recalling the model) on 65 of 71. Fewer, larger clusters give the search 100 questions each
      at no extra cost and make the routing boundary sharper.
- [ ] P2 Note the difficulty confound in any write-up: group 0 is 77-83% easy, group 5 is 64-69%
      hard, so per-group accuracies are not comparable across groups. Consider difficulty as an
      explicit routing feature rather than a hidden one.
- [ ] P2 Fix the proposal text: 20 anchors (not 10); k swept 2-12 (not 2-30); v3 draws 50 search
      questions weighted to the centre (not the first 50 of a farthest-point order).

## Stage 2a: seeds and grammar

- [ ] P0 Reject seeds that never run or cannot branch. `llm_g5` scored 0% because it has no
      `plan_left -> continue` rule and stops at step 0; `llm_g0` votes over two letters, so the
      critic can never overturn the expert. Add a check that the first decision is not a stop and
      that a vote read has at least three letters; run the model-written seeds through the sanity
      run like the random ones.
- [ ] P1 Strip dead rules (never fired on any search question) before reporting a program;
      random_0 had 6 of 9 dead. Consider stripping them from parents too, so edits hit live rules.
- [ ] P1 Remove redundant conditions: `acts` always equals `step` (every action is one round);
      several boolean conditions restate `n_distinct` and `last_round_agree`. Fewer, more distinct
      edit kinds: choose the opening, choose when to call deep-think, choose visibility, choose
      the readout.
- [ ] P2 Expose visibility as an edit axis: any round may be blind or letters-only. The executor
      has these modes; the grammar only exposes fixed pairings.
- [ ] P2 Blind deep-think move. Deep-think is right 15.3% blind vs 13.0% after other speakers on
      gpt-oss; it matches the leader 80% informed vs 48% blind. Test first with one hand-written
      program (solver, then deep-think) under each visibility on the 300 Qwen dev questions.

## Stage 2b: the search and its fitness signal

- [ ] P0 Paired fitness: score a program as (program right - direct right) on the same questions,
      not raw accuracy. Removes cluster difficulty and question luck. (OrchDebate's credit rule.)
- [ ] P0 Pick, then confirm on other questions: a would-be slot holder is chosen on its 50 search
      questions and must hold up on a second fixed set of 50 from the same cluster before it may
      parent children. This is SAT's "validation probes" and replaces "second replicate on the
      same 50". Report only confirmed scores.
- [ ] P1 No-degradation cap: besides beating the reference on average, the rate of "reference
      right, candidate wrong" must stay under a cap. (Audit: fixes and losses cancel.)
- [ ] P1 Never below the best seed: slot A keeps the best literature protocol unless a searched
      program beats it by more than one paired standard error on the confirmation set. On Qwen,
      3 of 6 final slot-A holders were literally seeds (mad, self_refine).
- [ ] P1 Budget: 10 generations, 100 questions per cluster (same calls as 20 x 50). Generations
      11-20 raised the search score by ~1.5 points and the unselected score by 0.1-0.7 in all
      three runs, at 38-40% of the calls. Half or more of the final holders were in place by
      generation 10.
- [ ] P2 Targeted re-evaluation: extra replicates only for candidates within one standard error
      of a slot (racing). Most calls went to programs never in contention.
- [ ] P2 Count cost in tokens, not turns. Usage is logged per speaker in every recording. A
      deep-think turn is 10-60 ordinary turns; the 5-turn charge lets 16-turn winners look cheap.
      Add a budget condition (`tokens_spent > K`) so a rule can learn to stop on easy questions.
- [ ] P2 Leakage audit of evolved programs is unnecessary for us (no free text is evolved), but
      say so explicitly in the paper; SAT needed a model call for it.

## Stage 3: test time and readout

- [ ] P1 Judge readout (`stop:judge`) that audits the committed replies and picks, as SAT's judge
      does over certificates, only if the Qwen audit shows rounds correcting more than they break.
      Their coverage-to-accuracy gap (87.9% -> 72.8% on knowledge) says the judge is lossy too.
- [ ] P2 Heterogeneity test: run the best programs with two different small models in the roster
      (e.g. Qwen 35B and gpt-oss alternating personas) and measure the repeat-leader rate. SAT's
      homogeneous control loses 10.7 points on math; this is the one lever we have not touched.
- [ ] P2 Portfolio ablation for the paper: run all 6 slot-A programs on every dev question plus a
      judge (SAT's deployment) beside routing, at 6x cost, to show what routing saves.

## Evaluation and reporting (no model calls)

- [ ] P0 Always compare against the external thinking-on baselines; in-executor rows only show
      what the executor costs.
- [ ] P1 Report avg@k as the main metric with the 0/1/2/3-of-3 breakdown; pass@k rewards
      run-to-run variance (gpt-oss: routed tied self-refine on pass@3 while 2.2 points behind on
      avg@3). Report coverage (best-of-programs bound) and accuracy side by side, as SAT does.
- [ ] P1 Whenever a best-of-programs bound is shown, show the K-sample self-consistency bound
      at the same K beside it; a reviewer who has read SAT will ask for it.
- [ ] P1 Add CDWR (win rate on the questions where two methods disagree) for routed vs
      deep_direct and routed vs random cluster.
- [ ] P1 Run the copy-rate audit (repeat-leader rate, fixed vs lost) on every new model before
      drawing conclusions.
- [ ] P2 Per-program dev table for every run (which programs do the work); SAT does not report
      this for its bank and it is a fair criticism of them.

## Reference numbers

| | gpt-oss v3 | gpt-oss deep | Qwen 35B v3 |
|---|---|---|---|
| routed programs, dev avg@3 | 13.4% | 13.0% (rep 1) | 24.8% |
| routed - random group | +1.1 (SE 0.8) | - | -0.7 |
| in-executor direct, dev | 12.2% | 11.3% (rep 1) | 24.8% |
| external self-refine / direct, thinking on | 15.6% / 13.0% | same | 31.3% / 31.6% |
| split-half gain, same cluster | +1.1 | +2.1 | +0.3 |
| repeat-leader rate (solver) | 80% | 77% | 92% |
| dev routes with margin < 0.10 | 24.7% | same | same |
