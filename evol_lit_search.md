# Evolutionary search over debate and search recipes: literature review and candidates

> Written 2026-08-08. The problem that prompted it: our schema-evolution loop
> (`scripts/evolve_debate_mcq.py`), which is allowed to see the answer key, beats a best-of-7
> resampling baseline by only ~4 points at the same number of runs — and the versions that could
> actually be deployed do worse. This document does three things: (a) diagnoses why, using only
> the existing output caches; (b) reviews the relevant literature; (c) proposes ten concrete
> candidate algorithms, ranked.
>
> No experiments were run for this document. Every number is read from files already in
> `outputs/`. Analysis code was read-only.
>
> **Status note (added later).** The companion document `evol_debate_designs.md` records what was
> actually built and run after this review, and which of these candidates survived. Read that for
> current results. The candidate labels C1–C10 below are referenced from there.

**Two terms used throughout.** A *schema* is a debate recipe: which speakers (personas) talk, in
what order, and how the final answer is chosen. The *genome* is the part of the recipe a search is
allowed to edit.

---

## Part 0 — What the current experiment actually is

### 0.1 The debate-side setup

| Component | Where | What it is |
|---|---|---|
| Task | `datasets/supergpqa_filter_strict_full.json` (4030 q) | SuperGPQA multiple-choice questions, pre-filtered to ones Qwen3-14B **fails**. So the base accuracy is ~3.9% by construction. |
| Executor | `debate_mcq.execute_schema` | Qwen3-14B, thinking off, temperature 0.7. Graded by exact letter match. No judge, no documents. |
| Genome | `debate_mcq.py` | `{rounds: [{personas: [...]}], final: "last"｜"vote"｜"synthesizer"}`; personas ∈ {solver, critic, synthesizer}; at most 6 rounds × 4 personas. |
| Search | `evolve_debate_mcq.evolve` | Greedy hill climb with one incumbent. An LLM "architect" (gpt-5.4-mini), which can see whether the answer was right, proposes ONE edit per step. It sees structure only, never facts. At most 6 edits. |
| Fitness | — | **One run per schema.** With `--confirm-runs 1`, a single correct run ends the search as "solved". |
| Control | `scripts/bestofn_mcq.py` | Same executor, minimal schema, 7 independent samples, order-independent pass@k (Chen et al. 2021). pass@k = the chance that at least one of k runs is correct. |

### 0.2 The headline numbers

`outputs/evolve_mcq_strict_summary.json` vs `outputs/bestofn_strict_summary.json`, both n≈4030:

| budget k (runs) | evolution cumulative | best-of-k | gap | evolution marginal | best-of-k marginal |
|---|---|---|---|---|---|
| 1 | 3.90% | 3.92% | −0.02 | 3.90 | 3.92 |
| 2 | 10.05% | 7.03% | +3.02 | **6.15** | **3.11** |
| 3 | 13.33% | 9.62% | +3.71 | 3.28 | 2.59 |
| 4 | 15.56% | 11.84% | +3.72 | 2.23 | 2.22 |
| 5 | 17.92% | 13.79% | +4.13 | 2.36 | 1.95 |
| 6 | 19.60% | 15.52% | +4.08 | 1.68 | **1.73** |
| 7 | **20.97%** | **17.07%** | +3.90 | 1.37 | **1.55** |

### 0.3 The diagnosis — three facts from the trajectory cache

Computed from `outputs/evolve_mcq_strict_cache.jsonl` (n=4029 ok, 845 solved):

**(1) The whole advantage comes from one edit, and that edit is always the same.**
The gap opens at k=2 (+3.02) and never grows much after. By k=6 and k=7, evolution's marginal
return per run falls **below** plain resampling. And the first edit is not really a search:

```
edit action distribution, step 1 (n=3869):   add_round = 99%
top schema shapes after the FIRST edit:
   3446 (89%)  solver; critic  -> last     <-- exactly the `always_critic` arm
    385 (10%)  solver; critic+synthesizer -> last
```

The architect makes essentially the same edit for 89% of questions, regardless of content. The
+3-point advantage over best-of-N is a fixed structural prior ("add a critic"), not per-question
adaptation.

**(2) The test-time arms confirm it — retrieval and synthesis add nothing over that constant.**
From `outputs/retrieve_*_strict_test_summary.json` (n=1000, one run each):

| arm | solve rate |
|---|---|
| `single_pass` | 3.2% |
| `fixed_debate` | 4.1% |
| `self_critique` | 6.3% |
| `retrieve_copy` (copy the nearest solved neighbour's schema) | 7.3% |
| `retrieve_synth` (the method: retrieve top-k, architect synthesizes) | 7.8% |
| **`always_critic`** (the constant `solver;critic`) | **7.9%** |

`retrieve_schema_mcq.py`'s own docstring calls `always_critic` the "decisive control: if this ties
retrieval, similarity/indexing adds nothing." It ties it — and slightly beats it. Similarity over
*question text* carries no signal about *which schema will work*.

**(3) Each extra run yields less as schemas grow.** Hit rate of the k-th run, among questions
that reached step k: 3.90% → 6.41% → 4.06% → 3.22% → 3.73% → 2.88% → 2.52%. Part of this is
survivorship (later steps only see questions that are still unsolved). But combined with the
marginal table above, longer schemas are not buying anything, and the architect flails more as it
goes: by step 6, `set_final` is 15% of actions and `remove_round` is ~0%. **The loop can add but
never backtracks.**

### 0.4 Restating the problem precisely

The loop is not failing because evolution is weak. It is failing because it is barely doing
evolution:

| Ingredient of an evolutionary algorithm | Present? |
|---|---|
| Population | ✗ (one incumbent) |
| Selection under a reliable fitness | ✗ (one yes/no sample at a 3.9% base rate) |
| Variation with diversity pressure | ✗ (99% `add_round`, 89% identical) |
| Archive / elitism / restart | ✗ |
| Crossover / recombination | ✗ |
| Rejection of bad offspring | ✗ (every edit is accepted, good or bad) |
| Sharing knowledge across questions | ✗ (search restarts from `MINIMAL_SCHEMA` per question) |

And crucially: "solved" means one lucky run was correct — which is the definition of best-of-N.
The measurement and the method are entangled. Any method scored this way inherits best-of-N's
curve and can only add whatever its proposals are worth. Here that is ~3 points.

Two different goals are being mixed, and they need different fixes:

- **Upper-bound track** — "can structural search reach schemas that resampling cannot?"
  Correct control: pass@k at the same number of runs. The blocker is the fitness signal, not the
  edit operator.
- **Deployable track** — "does the final schema beat a fixed template at one run?"
  Correct control: `always_critic` at 7.9%. The blocker is that the genome is too small to hold
  anything better than "add a critic", and there is no per-question signal to condition on.

### 0.5 The search side (`scripts/eval_search_step.py`) — an under-used opportunity

The search prong has no evolution loop yet; its `PRESETS` are hand-written. But `SearchConfig` is
already a well-formed genome: ~25 typed knobs (`top_n`, `fetch_k`, `budget`, `max_subqueries`,
`strip_think_subqueries`, `decompose_prompt`, `evidence`, `fetch_chars`, `packing`,
`queries_per_subq`, `max_searches`, `select_k`, `digest`, `follow_links`, `link_rounds`,
`links_per_page`, `link_select_k`, `link_source`, …). And unlike the MCQ prong it has a dense,
cheap score: `recall_surfaced` and `answer_in_evidence_rate` (currently 0.30–0.43 against a ~1.0
ceiling) are continuous numbers, not a 3.9%-base-rate coin flip. This is a much better-conditioned
optimization problem, and exactly the shape that automatic algorithm configuration tools (irace,
Hyperband) were built for. See candidate **C7**.

---

## Part 1 — Literature review

### 1.A LLM-driven evolutionary program and agent search

The direct ancestors of this project. The pattern: an LLM proposes mutations, a programmatic
evaluator scores them, and a population database supplies parents.

- **FunSearch** (Romera-Paredes et al., *Nature* 2024) — LLM plus evaluator evolving programs;
  found new cap-set constructions and bin-packing heuristics. The design choice that matters here:
  the program database is split into **islands** — several semi-isolated populations, with the
  worst islands periodically emptied and reseeded from good ones. That is explicit diversity
  maintenance against exactly the "everything converges to add-a-critic" collapse we have.
- **AlphaEvolve** (Google DeepMind, May 2025) — Gemini plus automated evaluators over a population
  of candidate programs; improved 4×4 matrix multiplication, recovered 0.7% of Borg fleet compute,
  sped up a FlashAttention kernel 23%. Transferable ideas: evolve a *diff* against a full program
  rather than a toy genome; use a **cascade of evaluators** (cheap filter first, expensive
  confirmation later) so budget flows to promising candidates.
- **ShinkaEvolve** (Sakana AI, arXiv 2509.19349) — the most relevant sample-efficiency paper.
  Three ideas, each aimed at a failure mode we have: (1) adaptive parent sampling from the archive
  instead of always using the incumbent; (2) **novelty-based rejection** — refuse to spend an
  evaluation on a candidate too similar to something already tried (our 89%-identical first edits
  are exactly what this kills); (3) a bandit that picks which LLM proposes the next mutation.
  Result: state-of-the-art circle packing in ~150 evaluations, and stronger AIME scaffolds under
  strict query budgets.
- **ADAS / Meta Agent Search** (Hu, Lu, Clune, arXiv 2408.08435) — a meta-agent writes new agents
  as code, seeded from an ever-growing archive of prior discoveries. Lesson: the genome should be
  expressive (code or text), and the archive is the memory that makes later proposals better than
  earlier ones. Our architect sees the last 6 steps of one question's history and nothing else.
- **Darwin Gödel Machine** (Zhang et al., arXiv 2505.22954) — open-ended self-improvement: sample
  a parent from an archive (not the incumbent), modify it, keep it if it is interestingly new.
- **AFlow** (Zhang, Xiang et al., ICLR 2025 Oral, arXiv 2410.10762) — workflow optimization as
  Monte-Carlo tree search (MCTS) over code-represented workflows, with backtracking and reusable
  operators. Beats hand-designed workflows and lets weaker models win on cost. The closest
  published thing to "search over debate schemas" done properly, and its search (MCTS with
  backtracking) is strictly stronger than our greedy chain.
- **MASS** (Zhou et al., arXiv 2502.02533) — the key structural finding: optimize in three
  interleaved stages — per-block prompts, then topology, then a global prompt. Their conclusion:
  prompts and topology matter jointly, and **optimizing topology alone (what we do) is the weakest
  of the three**. A strong prior that our ceiling is set by the genome, not the search.
- **GEPA** (Agrawal et al., arXiv 2507.19457, ICLR 2026 Oral) — reflective prompt evolution:
  sample runs, have an LLM diagnose failures in plain language, propose and test prompt edits.
  The important part: parents are selected from a **Pareto frontier over per-question scores** —
  keep anyone who is best at something — rather than by average score. Reported +6pp average (up
  to +19pp) over a reinforcement-learning baseline with up to 35× fewer rollouts. Pareto selection
  is the direct antidote to "one aggregate number collapses everything onto the same schema."
- **EvoPrompt** (Guo et al., ICLR 2024, arXiv 2309.08532) — genetic algorithms over discrete
  prompts with an LLM as the variation operator. Up to +25% on BBH tasks.
- **Promptbreeder** (Fernando et al., ICML 2024, arXiv 2309.16797) — self-referential: evolves the
  *mutation prompts* alongside the task prompts, and ablating that hurts. Our `MODIFIER_SYSTEM` is
  frozen and hand-written.
- **ELM** (Lehman et al., arXiv 2206.08896) and **Language Model Crossover** (Meyerson et al.,
  arXiv 2302.12170) — foundational: LLMs as mutation *and crossover* operators over text genomes.
  We currently have no crossover at all.

### 1.B Quality-diversity and open-endedness

This body of work exists because greedy objective-driven search converges to a single attractor —
which is empirically what our loop does.

- **MAP-Elites** (Mouret & Clune, 2015) — keep an archive of the best solution in each bin of a
  *behavior space* (bins defined by how a solution behaves, not how good it is). You get diversity
  for free, and the archive shows which regions of the space are good at all. CVT-MAP-Elites
  (Vassiliades et al.) scales the binning to many dimensions.
- **MAP-Elites under noise** — directly relevant, because our fitness is a 3.9%-base-rate coin
  flip. *MAP-Elites for noisy domains by adaptive sampling* (GECCO Companion 2019) and
  **Deep-Grid MAP-Elites** (Flageat & Cully, arXiv 2006.14253) keep several solutions per bin so
  repeated evaluations average out the noise, instead of letting one lucky sample squat in a bin
  forever. That squatting is exactly our "one lucky run counts as solved" pathology, and it has a
  published fix.
- **Novelty Search** (Lehman & Stanley, 2011) — on deceptive landscapes, rewarding behavioral
  novelty beats chasing the objective. Our landscape is deceptive in a specific way: the greedy
  signal always says "add another round," which is locally positive and globally a dead end (see
  the marginal decay at k=6–7).
- **Go-Explore** (Ecoffet et al., *Nature* 2021) — names two failure modes that describe our loop
  exactly: *detachment* (forgetting how to get back to a promising state — we keep no archive;
  each step overwrites the incumbent) and *derailment* (never returning to a schema that looked
  good two edits ago). The fix — return to an archived good state first, then explore — is cheap
  to port.
- **QDAIF** (Bradley et al., ICLR 2024, arXiv 2310.13032) — MAP-Elites where the LLM is both the
  variation operator and the quality/diversity judge. The template for running MAP-Elites over
  LLM-generated text.

### 1.C Optimization under noisy, binary, expensive fitness — the most under-used literature here

Our fitness is a yes/no trial with a ~3.9% success rate, sampled once, at the cost of several LLM
calls. The literature on exactly this is mature and almost entirely absent from the current loop.

- **Jin & Branke (2005)** and Rakshit et al. (2017), the standard surveys, name three families:
  (1) *explicit averaging* — re-evaluate the same candidate several times (our `--confirm-runs`,
  set to 1); (2) *implicit averaging* — use a large population so noise cancels across similar
  individuals; (3) *selection modification* — only prefer A over B when a statistical test says
  so. We currently use none of the three at meaningful strength.
- **SPRT and Fishtest (Stockfish)** — the best practitioner analogue. A sequential probability
  ratio test (SPRT) accepts or rejects a candidate chess-engine patch by playing games *until the
  statistics decide*, rather than fixing the sample size in advance. Fishtest also pairs games
  (the same position played both ways), a large variance reduction. Translation: never accept a
  schema edit because one run was correct; accept it when a sequential paired test over shared
  questions says it beats its parent. Pairing maps onto evaluating both schemas on the same
  questions with the same random seeds.
- **CLOP** (Coulom, ACG 2011) — parameter tuning against win/loss outcomes; fits a local model of
  win rate and discards clearly inferior samples. Built for exactly this problem shape.
- **Racing: F-race → irace** (Birattari; López-Ibáñez et al., 2016) — evaluate a set of candidate
  configurations on a stream of instances and eliminate a configuration as soon as a rank test
  says it is worse. The correct off-the-shelf method for "which schema is best over a distribution
  of questions," and for the `SearchConfig` genome.
- **Successive Halving / Hyperband** (Jamieson & Talwalkar 2016; Li et al. 2018) — give every
  candidate a small budget, keep the top half, double the budget, repeat. Near-optimal allocation
  when evaluations are expensive and noisy — our setting.
- Framing note: choosing the best schema is a *best-arm identification* problem (a bandit
  problem), and our search currently spends its whole budget on distinct arms with **one pull
  each**. Bandit theory says that is the setting in which you learn essentially nothing.

### 1.D Coevolution and game-playing evolution

- **Hillis (1990)** — evolve sorting networks against a coevolving population of test cases,
  scored by their ability to defeat the networks. The arms race both sped up evolution and
  prevented sticking at local optima. Translation: coevolve schemas against a question population,
  up-weighting questions the current elites fail. This also counteracts the "solved by a lucky
  roll" contamination in our train corpus.
- **Blondie24 / Anaconda** (Chellapilla & Fogel, ~1999–2001) — expert-level checkers from
  coevolution using only relative win/loss outcomes. Two lessons: (1) head-to-head comparison of
  two schemas on the same questions is far more informative per LLM call than two independent
  accuracy estimates; (2) selection pressure on a population over many generations extracts
  signal from an extremely noisy per-game outcome.
- Self-play lineage (TD-Gammon → AlphaZero) — the general point: a search procedure plus a
  learned evaluator beats either alone, and the evaluator is trained *from* the search's outcomes.
  Our loop has a search but no learned evaluator; §1.E is how that gets fixed.

### 1.E Tree search, pruning, backtracking, and process supervision in LLM reasoning

Pruning and backtracking in tree-of-thought methods are the same machinery as selection and
rejection in an evolutionary algorithm.

- **Tree of Thoughts** (Yao et al., NeurIPS 2023) — thoughts as tree nodes, search with a
  self-evaluated state score, keep the top k per layer, prune hopeless states, backtrack. The
  mapping is exact: state evaluator = fitness, top-k = selection, pruning = rejection,
  backtracking = restart from an archive. Our loop has the generator and none of the rest.
- **LATS** (Zhou et al., ICML 2024) — MCTS for language agents with LLM value functions and
  self-reflections stored and reused across branches; roughly doubles ReAct on HotPotQA.
  Reflection-as-memory is what our architect lacks across questions.
- **AB-MCTS / "Wider or Deeper?"** (Inoue et al., arXiv 2503.04412; Sakana AI) — the single most
  on-point algorithm for our budget question. At every node it decides, by sampling from Bayesian
  posteriors, whether to go **wider** (draw a fresh sample — i.e. best-of-N) or **deeper** (refine
  an existing one — i.e. edit the schema). Our experiment hand-fixes this choice to "always
  deeper"; our control hand-fixes it to "always wider." Multi-LLM AB-MCTS reached 39.2% on
  ARC-AGI-2, >15pp above any single frontier model.
- **Snell et al., "Scaling LLM Test-Time Compute Optimally"** (ICLR 2025) — the optimal test-time
  strategy depends on question difficulty: some questions want sequential refinement, others want
  parallel sampling. Our setup applies one fixed schedule to all 4030 questions.
- **Large Language Monkeys** (Brown et al., arXiv 2407.21787) — coverage (pass@k) grows
  log-linearly in samples over four orders of magnitude. So *any* method whose metric is "some run
  was correct" will look good, and the interesting quantity is the **selector** that picks the
  answer, not the coverage. (SWE-bench Lite: 15.9% @1 → 56% @250.)
- **Process reward models** — *Let's Verify Step by Step* (Lightman et al.), *Math-Shepherd*, and
  **rStar-Math** (arXiv 2501.04519), which pairs MCTS rollouts with a trained step-level judge and
  took Qwen2.5-Math-7B from 58.8% to 90.0% on MATH. Relevance: our deployable loop stops on
  self-consistency (letter agreement across runs), and our own measurement says that signal is
  worse than useless here — `majority_vote@7 = 2.33%`, *below* pass@1. A verifier that reads the
  debate transcript is the obvious replacement.
- **Adaptive sampling / early stopping** — ESC, ASC, and Difficulty-Adaptive Self-Consistency
  (arXiv 2408.13457): allocate samples by predicted difficulty, with up to 65% cost reduction at
  minimal accuracy loss. This is how to spend a 7-run budget non-uniformly.
- **Limits of self-critique** — *LLMs Cannot Self-Correct Reasoning Yet* (Huang et al., ICLR
  2024): self-correction without external feedback often *hurts*. Our evidence agrees in a weaker
  form: one critic gives a real gain (3.2% → 7.9%), but stacking more critic/synthesizer rounds
  decays the per-run yield below the base rate. This bounds what any purely structural genome can
  deliver.

### 1.F Per-question algorithm selection — the right frame for the routing prong

- **Rice (1976)** — the formal frame: learn a mapping from instance features to the best
  algorithm for that instance.
- **SATzilla** (Xu et al., JAIR 2008) — per-instance solver portfolios built on cheap instance
  features plus models that predict each solver's runtime; won multiple SAT competition medals.
  Critically, its features include **probing**: run a solver briefly and measure its behavior.

  This is the missing ingredient in `retrieve_schema_mcq.py`. Its retrieval embeds *question
  text*, which contains no information about how the executor will behave. SATzilla's answer:
  spend a small probe budget (e.g. two samples of the minimal schema) and use the *observed*
  behavior — answer agreement, answer entropy, response length, refusal — as the features.
  Nothing in our pipeline conditions on executor behavior at selection time, which is why nothing
  beats a constant.

---

## Part 2 — Candidate algorithms

Ranked by (expected gain) × (evidence in the literature) ÷ (implementation cost). Each names the
control it must beat, so the result is interpretable either way.

**Protocol fixes to apply regardless of which candidate runs:**

- **P1 — Redefine "solved."** Stop counting "any single run was correct." Report instead:
  (a) *deployable accuracy* — the final schema executed once (or a majority over m fresh runs);
  and (b) *oracle accuracy* at matched k, overlaid on pass@k. Right now (a) is never reported for
  evolution, and (a) is the number the paper needs.
- **P2 — Always report three controls together**: pass@k at the same number of runs;
  `always_critic` at one run (7.9%); and an oracle over the fixed library (best of {single,
  always_critic, self_critique, fixed_debate} per question) — which caps what any router could
  win.
- **P3 — Paired evaluation.** Compare schemas on the same questions with the same sampling seeds
  (the Fishtest pairing insight). The variance reduction is free accuracy.

---

### C1 — Race-gated hill climbing: accept an edit only when a test says it helps
*(lowest cost, highest certainty; do this first)*

**What.** Keep the current architect and grammar. Change only the acceptance rule: a proposed edit
is accepted only if a sequential paired test says the child beats the parent; otherwise revert and
re-propose. Use SPRT (as in Fishtest) on paired runs, or successive halving over ~4 sibling
proposals per step, or irace-style elimination.

**Why.** §0.3(3) shows the incumbent *degrades* after step ~3 — because every edit is accepted,
good or bad. Chess engine developers faced the same problem (noisy binary outcome, expensive
evaluation, tiny effects) and settled on sequential testing as standard practice.

**Where.** `evolve_debate_mcq.evolve` — replace "run once, keep if correct" with an accept/reject
gate; `--confirm-runs` becomes a real sequential test.

**Budget honesty.** This costs more runs per accepted edit, so plot it against best-of-k at the
true total run count, not at "number of edits."

**Must beat.** pass@k at the same total runs. If race-gated evolution still tracks pass@k, that is
a publishable negative result: structural search over this grammar carries almost no per-question
information.

---

### C2 — Population evolution of schemas, scored over question batches
*(the highest-value reframing)*

**What.** Stop evolving one schema per question from scratch. Evolve a population of ~30–60
schemas whose fitness is solve rate over a shared batch of training questions, using:
- islands with periodic culling and reseeding (FunSearch) for diversity;
- racing / successive halving for fitness — a cheap first pass on 50 questions, survivors get
  200, finalists get 1000;
- an LLM as the mutation operator, plus crossover between two parents (currently absent).

**Why.** This turns a 7-sample noisy hill climb into an algorithm with hundreds of trials per
candidate — a fitness that can actually rank schemas. It also produces a deployable artifact (a
small library of good schemas) rather than 4030 one-off trajectories, and it measures the thing
we actually want to know: is there any schema better than `always_critic` in expectation?

**Why it might work where the current loop cannot.** The per-question loop can never discover a
*reliably* better schema, because it stops at the first lucky roll. A population loop selects for
expected value.

**Must beat.** `always_critic` at one run (7.9%) on `strict_test`. That is the honest bar for a
deployable claim.

---

### C3 — MAP-Elites over schema behavior, with noise-aware cells
*(diversity, plus the substrate for routing)*

**What.** Archive schemas in bins defined by how they behave, keeping the best per bin:
- structural descriptors: number of rounds × number of critic roles × final rule × cost in calls;
- optional instance descriptors: question field (SuperGPQA has ~50) × difficulty.

Use Deep-Grid MAP-Elites or adaptive resampling so a bin's elite must survive re-evaluation — one
lucky roll cannot hold a bin.

**Why.** It directly attacks the 89%-identical-mutation collapse, and the resulting archive *is*
the routing table for the deployable prong. A field × difficulty grid of elites is a far
better-motivated index than text-embedding similarity, which we have shown carries no signal.

**Bonus.** The archive shows whether different question fields actually want different schemas.
If it comes out uniform, that is itself the answer to the paper's central question — worth
reporting either way.

**Must beat.** `retrieve_synth` (7.8%) and `always_critic` (7.9%) at one run.

---

### C4 — Enrich the genome: from topology to topology + text
*(the biggest expected lift on the ceiling; run jointly with C2 or C3)*

**What.** The current grammar has maybe a few hundred meaningfully distinct points and one
dominant gradient. MASS finds topology alone is the weakest axis. Add:

1. **Evolved persona instructions** — `PERSONA_PROMPTS` become part of the genome, edited
   reflectively (GEPA-style: read failed transcripts, diagnose in plain language, propose a prompt
   edit). Still answer-blind — the firewall holds, because we evolve *how to think*, not *what is
   true*.
2. **Role specialization by field** — e.g. a unit-checking critic for engineering, a
   definition-disambiguation critic for law. SuperGPQA's `field` label is free supervision we are
   not using.
3. **Richer aggregation** — confidence-weighted vote, debate-to-consensus with a stopping rule,
   tie-break-by-critic, abstain-and-resample. `FINALS` has three options, and 806 of 845 solved
   schemas just use `"last"`.
4. **Per-round execution parameters** — temperature and samples per round, so structure can say
   "diversify here, converge there."

**Why.** Under the current grammar, "add a critic" may genuinely be the best schema — in which
case no better search will help, and the honest conclusion is that the space was too small.
Expanding it is the only way to tell "our search was weak" from "the space was empty."

**Must beat.** The same bars, plus C2 with the small genome — that comparison separates genome
size from search strength.

---

### C5 — AB-MCTS over the edit tree: let the algorithm choose wider vs deeper
*(resolves evolution-vs-best-of-N by measurement instead of assumption)*

**What.** Build a tree rooted at `MINIMAL_SCHEMA`. At each node, decide by Thompson sampling
(drawing from a posterior over each option's success rate) whether to go **wider** (another run of
the same schema — best-of-N) or **deeper** (an architect edit — evolution). Back up outcomes. Add
pruning (drop branches whose posterior falls below the root's) and backtracking to the archive's
best node.

**Why.** Our two arms are the two degenerate policies of one algorithm; AB-MCTS is the principled
middle. Even if accuracy does not move, the learned wider/deeper ratio per question is a
publishable result.

**Must beat.** pass@k at the same runs, and C1.

---

### C6 — Replace the deployable loop's stopping signal with a transcript verifier
*(the biggest deployability hole)*

**What.** Replace `_consistency()` (letter agreement across repeated runs) with a verifier that
scores the debate transcript — either a prompted LLM critic, or a small process reward model (a
model trained to score reasoning steps) trained on the existing trajectory caches.

**Why.** `majority_vote@7 = 2.33%` is *below* `pass@1 = 3.92%`. On questions the model reliably
gets wrong, consistency measures confident wrongness. Every blind-mode conclusion rests on this
signal, so it must be replaced before blind-mode numbers mean anything.

**Must beat.** The current blind loop, and `always_critic`. Also report the verifier's *selection
accuracy* — given 7 candidate answers, how often does it pick a correct one? That number alone
says how much of the 17.07% oracle gap is recoverable.

*(Correction, from `evol_debate_designs.md` §0.10: the trajectory caches hold letters only, not
transcripts — training a verifier from them is not free; it needs a re-run with transcript
logging.)*

---

### C7 — Automatic configuration (irace / Hyperband) over `SearchConfig` — the search-side prong
*(best signal per dollar; the search side has no optimization loop at all yet)*

**What.** `SearchConfig` in `scripts/eval_search_step.py` is already a ~25-knob genome with a
dense score. Run irace (or Hyperband) over it, with `answer_in_evidence_rate` as the primary
objective and downstream accuracy as confirmation, racing configurations over the 266-question
FRAMES set with early elimination.

**Why this is easier than the MCQ prong.** The signal is dense and continuous (0.30–0.43 against
a ~1.0 ceiling), the base rate is not 4%, and page fetches are free through
`scripts/wiki_backend.py`. Racing was invented for this problem shape.

**Caveat.** `search_info` still hits paid Serper. Racing minimizes evaluations by design, but
budget the run and use `--limit` first, per the onboarding doc.

**Must beat.** The best hand-tuned presets (`decompose_react_union|union`: 42.5% downstream,
0.30 answer-in-evidence; `breadth_link`: 40.4%, 0.43).

---

### C8 — Coevolve questions against schemas, and use head-to-head fitness
*(a cheap add-on to C2/C3 that also fixes a data problem)*

**What.** Two ideas from game-playing evolution:
1. **A "parasite" population of questions** whose sampling weight rises when current elite schemas
   fail them. Elites must then improve on genuinely hard questions instead of farming ones that
   resampling would have solved anyway.
2. **Head-to-head fitness.** Score schemas by win rate against each other on shared question
   batches, not by absolute accuracy. At a 3.9% base rate, a paired comparison on identical
   questions is far more informative per LLM call.

**Why it matters for this corpus.** 157 of 845 "solved" cases (19%) were solved at step 0 — by
the minimal schema on the first try, pure luck. Those questions contaminate the exemplar index
that `retrieve_schema_mcq.py` builds (there is already a `--nontrivial-only` flag; consider making
it the default). Parasite weighting handles this dynamically.

---

### C9 — Novelty rejection on proposed edits
*(near-zero cost; strictly better than the status quo)*

**What.** Before spending a run on a proposed schema, check whether it (or something very close to
it) was already evaluated on a similar question. If so, reject and re-prompt for something
different. ShinkaEvolve reports this as one of three changes that cut evaluations from thousands
to hundreds.

**Why.** 89% of first edits are byte-identical and 99% are the same action type. We are paying
full LLM-call price to rediscover `solver;critic` 3,446 times.

**Cheapest version.** Seed the search *from* `always_critic` instead of `MINIMAL_SCHEMA`, and
forbid the architect from proposing it. This re-baselines the whole experiment: it measures what
evolution adds *on top of* the known-good constant, which is the honest question. **It is a
one-line change and should probably run before anything else in this document.**

---

### C10 — Per-question schema selection with behavioral probe features
*(replaces embedding retrieval, which is demonstrably inert)*

**What.** Build a hardness model per question. Features:
- static: field, difficulty, number of options, token length, presence of numbers/units;
- **probe** (the SATzilla insight): run the minimal schema twice and record letter agreement,
  answer entropy, response length, self-reported confidence.

Train a model predicting each schema's success probability over a small library (single_pass,
always_critic, self_critique, fixed_debate, plus C2/C3 elites), then route each question to its
predicted best schema. Count the probe runs in the budget so the comparison stays fair.

**Why.** Question-text similarity cannot predict which schema wins — the 7.3 / 7.8 / 7.9%
three-way tie shows that cleanly. Observed executor behavior is the feature class that has worked
for 20 years in algorithm selection, and it is the only thing in this list that could produce
genuine per-question adaptation.

**Must beat.** `always_critic` (7.9%), with the oracle-over-fixed-library bound reported
alongside — if that oracle is only ~11%, routing can win at most ~3 points, and you should know
that before spending compute.

---

## Part 3 — Suggested order

| Order | Candidate | Cost | What it settles |
|---|---|---|---|
| 1 | **C9-lite**: seed from `always_critic`, ban re-proposing it | ~1 line | Whether evolution adds anything beyond the known constant. Re-baselines everything. |
| 2 | **P1–P3**: deployable accuracy, paired evaluation, 3 controls | small | Makes every later number interpretable. |
| 3 | **C1**: sequential-test acceptance gate | small–medium | Whether unconditional acceptance was the bug. |
| 4 | **C7**: irace over `SearchConfig` (search side, parallel track) | medium | Highest signal per dollar; the search prong has no optimization at all yet. |
| 5 | **C2 + C9**: island GA over a schema population with racing fitness | medium–large | Whether *any* schema beats `always_critic` in expectation. |
| 6 | **C4**: enrich the genome (persona text, field roles, aggregation) | large | Separates "weak search" from "empty space." The likely source of a real ceiling lift. |
| 7 | **C5**: AB-MCTS wider-vs-deeper | medium | Turns "evolution vs best-of-N" from an assumption into a measurement. |
| 8 | **C6**: transcript verifier replacing self-consistency | medium | Unblocks the deployable (blind) prong entirely. |
| 9 | **C3 / C8 / C10**: MAP-Elites archive, coevolution, probe routing | large | Per-question adaptation, which is the paper's actual thesis. |

---

## Part 4 — Risks and honest expectations

1. **The grammar may already be saturated.** `always_critic` at 7.9% may be the best this space
   holds. Huang et al. (LLMs cannot self-correct intrinsically) and our own per-run decay both
   point that way. If so, C1/C5/C9 produce a clean negative, and the value shifts to C4 (bigger
   space) and C6/C10 (better selection). Plan the narrative for that outcome now.
2. **Racing costs runs.** C1 and C2 spend more calls per accepted change. Plot them against total
   runs, not generations, or the comparison becomes apples-to-oranges again.
3. **The base rate is brutal.** At 3.9%, detecting a true +1pp improvement at 80% power needs on
   the order of thousands of paired trials. This is *the* reason to adopt paired evaluation (P3),
   head-to-head fitness (C8), and sequential tests (C1). They are not optional polish.
4. **The oracle framing can overstate routing headroom.** If evolution's advantage is one constant
   edit, the "oracle upper bound" is misleading. Report the oracle-over-fixed-library number
   early; it may show routing can win very little.
5. **Do not let lucky solves contaminate the exemplar corpus.** 19% of solves happened at step 0.
   Default `--nontrivial-only` in the retrieval index, or weight by confirmed re-runs.

---

## Recommendation

Run C9-lite today: seed the search from `always_critic` and forbid re-proposing it. It is one
line, and it answers the question everything else depends on — whether evolution adds anything
beyond its one constant edit. Adopt P1–P3 at the same time so every number after that is
interpretable. Then C1 (acceptance gate) on the debate side and C7 (irace over `SearchConfig`) on
the search side, in parallel. Hold C2–C6 until those cheap results say the search, the genome, or
both are the problem.

---

## Sources

**LLM-driven evolutionary program/agent search**
- [Mathematical discoveries from program search with large language models (FunSearch), *Nature* 2024](https://www.nature.com/articles/s41586-023-06924-6) · [FunSearch overview](https://en.wikipedia.org/wiki/FunSearch)
- [AlphaEvolve: a Gemini-powered coding agent for designing advanced algorithms — Google DeepMind](https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/) · [AlphaEvolve impact](https://deepmind.google/blog/alphaevolve-impact/) · [overview](https://en.wikipedia.org/wiki/AlphaEvolve)
- [ShinkaEvolve: Towards Open-Ended and Sample-Efficient Program Evolution (arXiv 2509.19349)](https://arxiv.org/abs/2509.19349) · [Sakana AI blog](https://sakana.ai/shinka-evolve/)
- [Automated Design of Agentic Systems / Meta Agent Search (arXiv 2408.08435)](https://arxiv.org/abs/2408.08435) · [project page](https://www.shengranhu.com/ADAS/)
- [Darwin Gödel Machine: Open-Ended Evolution of Self-Improving Agents (arXiv 2505.22954)](https://arxiv.org/abs/2505.22954)
- [AFlow: Automating Agentic Workflow Generation (arXiv 2410.10762, ICLR 2025 Oral)](https://arxiv.org/abs/2410.10762) · [code](https://github.com/FoundationAgents/AFlow)
- [Multi-Agent Design: Optimizing Agents with Better Prompts and Topologies — MASS (arXiv 2502.02533)](https://arxiv.org/abs/2502.02533)
- [GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning (arXiv 2507.19457)](https://arxiv.org/abs/2507.19457) · [code](https://github.com/gepa-ai/gepa)
- [EvoPrompt: Connecting LLMs with Evolutionary Algorithms (arXiv 2309.08532)](https://arxiv.org/abs/2309.08532)
- [Promptbreeder: Self-Referential Self-Improvement via Prompt Evolution (arXiv 2309.16797)](https://arxiv.org/abs/2309.16797)

**Quality-diversity and open-endedness**
- [Illuminating search spaces by mapping elites — MAP-Elites (Mouret & Clune)](https://www.semanticscholar.org/paper/Illuminating-search-spaces-by-mapping-elites-Mouret-Clune/45373921f06a6efebefa6189d2dd80362ab0836e) · [reference implementation](https://github.com/resibots/pymap_elites)
- [MAP-Elites for noisy domains by adaptive sampling (GECCO Companion 2019)](https://dl.acm.org/doi/10.1145/3319619.3321904) · [Fast and stable MAP-Elites in noisy domains using deep grids (arXiv 2006.14253)](https://arxiv.org/pdf/2006.14253)
- [Abandoning Objectives: Evolution Through the Search for Novelty Alone (Lehman & Stanley)](https://dl.acm.org/doi/abs/10.1162/evco_a_00025)
- [First return, then explore — Go-Explore, *Nature* 2021](https://arxiv.org/pdf/2004.12919) · [Go-Explore (arXiv 1901.10995)](https://arxiv.org/abs/1901.10995)
- [Quality-Diversity through AI Feedback (arXiv 2310.13032)](https://arxiv.org/pdf/2310.13032)

**Noisy / expensive fitness, racing, sequential testing**
- [Sequential Probability Ratio Test — Chessprogramming wiki](https://www.chessprogramming.org/Sequential_Probability_Ratio_Test) · [Fishtest mathematics (GSPRT + pentanomial model)](https://official-stockfish.github.io/docs/fishtest-wiki/Fishtest-Mathematics.html) · [Automated Tuning](https://www.chessprogramming.org/Automated_Tuning)
- [CLOP: Confident Local Optimization for Noisy Black-Box Parameter Tuning (Coulom, ACG 2011)](https://www.remi-coulom.fr/CLOP/) · [publisher page](https://link.springer.com/chapter/10.1007/978-3-642-31866-5_13)
- [The irace package: Iterated Racing for Automatic Algorithm Configuration (Op. Res. Perspectives 2016)](https://www.sciencedirect.com/science/article/pii/S2214716015300270) · [docs](https://mlopez-ibanez.github.io/irace/)
- [Non-stochastic Best Arm Identification and Hyperparameter Optimization (Jamieson & Talwalkar)](https://www.semanticscholar.org/paper/Non-stochastic-Best-Arm-Identification-and-Jamieson-Talwalkar/c6c745d7fae9aad4294549d829f7e7415ffb1709) · [Hyperband (JMLR 18)](https://jmlr.org/papers/volume18/16-558/16-558.pdf)
- [Noisy evolutionary optimization algorithms — a comprehensive survey](https://www.sciencedirect.com/science/article/abs/pii/S221065021630308X) · [Recent advances in evolutionary optimization in noisy environments](https://link.springer.com/chapter/10.1007/978-981-10-8642-7_3)

**Coevolution and game-playing evolution**
- [Hillis (1990), Co-evolving parasites improve simulated evolution as an optimization procedure (*Physica D* 42:228–234)](https://www.cse.unr.edu/~sushil/class/gas/papers/DannyHillisCoevolution.pdf)
- [A Comprehensive Survey of Coevolutionary Algorithms Research](https://www.cse.unr.edu/~sushil/pubs/newestPapers/2008/ieeeTecCoEv/coevrepos/corev/)
- [Blondie24: Playing at the Edge of AI (Fogel)](https://en.wikipedia.org/wiki/Blondie24) · [Verifying Anaconda's expert rating by competing against Chinook](https://www.researchgate.net/publication/222647739_Verifying_Anaconda's_expert_rating_by_competing_against_Chinook_Experiments_in_co-evolving_a_neural_checkers_player)

**Tree search, pruning/backtracking, process supervision in reasoning**
- [Tree of Thoughts: Deliberate Problem Solving with Large Language Models (NeurIPS 2023)](https://proceedings.neurips.cc/paper_files/paper/2023/file/271db9922b8d1f4dd7aaef84ed5ac703-Paper-Conference.pdf) · [code](https://github.com/princeton-nlp/tree-of-thought-llm)
- [Language Agent Tree Search (LATS) — arXiv 2310.04406, ICML 2024](https://arxiv.org/abs/2310.04406)
- [Wider or Deeper? Scaling LLM Inference-Time Compute with Adaptive Branching Tree Search — AB-MCTS (arXiv 2503.04412)](https://arxiv.org/abs/2503.04412) · [Sakana AI blog](https://sakana.ai/ab-mcts/)
- [Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters (ICLR 2025)](https://proceedings.iclr.cc/paper_files/paper/2025/file/1b623663fd9b874366f3ce019fdfdd44-Paper-Conference.pdf)
- [Large Language Monkeys: Scaling Inference Compute with Repeated Sampling (arXiv 2407.21787)](https://arxiv.org/abs/2407.21787)
- [rStar-Math: Small LLMs Can Master Math Reasoning with Self-Evolved Deep Thinking (arXiv 2501.04519)](https://arxiv.org/abs/2501.04519)
- [Large Language Models Cannot Self-Correct Reasoning Yet (ICLR 2024, arXiv 2310.01798)](https://arxiv.org/pdf/2310.01798)
- [Make Every Penny Count: Difficulty-Adaptive Self-Consistency (arXiv 2408.13457)](https://arxiv.org/abs/2408.13457)

**Per-instance algorithm selection**
- [SATzilla: Portfolio-based Algorithm Selection for SAT (JAIR 32)](https://jair.org/index.php/jair/article/view/10556) · [SATzilla-07 design and analysis](https://www.cs.ubc.ca/labs/algorithms/Projects/SATzilla/CP07-SATzilla.pdf)

*Cited from background knowledge rather than verified in this session's searches:* Rice (1976)
"The Algorithm Selection Problem"; Lightman et al., *Let's Verify Step by Step* (arXiv 2305.20050);
Math-Shepherd (arXiv 2312.08935); Lehman et al., *Evolution through Large Models* (arXiv 2206.08896);
Meyerson et al., *Language Model Crossover* (arXiv 2302.12170); Jin & Branke (2005) uncertainty
survey; Vassiliades et al., CVT-MAP-Elites; Chen et al. (2021) pass@k estimator.
