# Reading list for the debate-program paper

Compiled 2026-09-20 from the literature searches in this project.

**How far each entry can be trusted.** Only OrchDebate was read in full. Everything else is known
from its abstract or from search summaries, so treat each one-line description as a pointer, not
a verified claim. Entries with no link are ones where I know the paper but did not confirm a URL;
search the title. Two different papers are called "EvoMAS"; both are listed.

**How the list is ordered.** Section 1 is the short list to read first. Sections 2 to 7 are the
full lists by topic. Section 8 is the baseline table. Section 9 is how our work sits among them.

---

## 1. Read these first

| # | Paper | Why it comes first |
|---|---|---|
| 1 | [OrchDebate: Adaptive Orchestration for Multi-Agent Debate via Self-Evolving Skills](https://openreview.net/pdf?id=JyJSJitKjy) (anonymous ACL submission; copy in the repo root) | The closest work. Same framing: a controller reads the debate state and picks the next move. Their controller is an unlearned LLM call every step; ours is a learned if-then program with no controller cost. Their held-out set is about 160 questions, so their 2-4 point gaps are inside the noise, as ours are. |
| 1b | [Self-Organizing Agent Teams Learn to Reason Together](https://arxiv.org/abs/2609.22682) (Pappu, Suzgun, ... Zou; Sep 2026; copy in the repo root as `2609.22682v1.pdf`; read in full) | The strongest published "evolve the debate protocol" result and the paper a reviewer will hold up against our search stage. A designated team member reflects on transcripts and mutates a teamwork strategy (phases, participants, information flow, role prompts); a bank of 10 is frozen and ALL 10 run at test time, with a judge picking. Learned from 15 AIME or 25 GPQA problems; gains come largely from a heterogeneous roster (o3-mini + Claude + DeepSeek: 66.7% vs 56.0% for three copies of o3-mini). No question clustering, no routing, no rule grammar, no cost objective. Borrow: the routing-oracle and coverage-vs-accuracy framing, the leakage audit of evolved artifacts, validation probes before accepting a mutation. |
| 2 | [Single-Agent LLMs Outperform Multi-Agent Systems Under Equal Thinking Token Budgets](https://arxiv.org/abs/2604.02460) | The strongest challenge to the premise: with tokens held equal, one agent thinking longer matches or beats five multi-agent designs, debate included, on three model families. Our gpt-oss results agree with it. |
| 3 | [Stop Overvaluing Multi-Agent Debate](https://arxiv.org/abs/2502.08788) | Short critique of debate evaluation: weak baselines, few benchmarks, inconsistent setups. A checklist of what reviewers will ask. |
| 4 | [Reasoning in Token Economies: Budget-Aware Evaluation of LLM Reasoning Strategies](https://arxiv.org/html/2406.06461) (EMNLP 2024) | Chain-of-thought plus majority vote, scaled to equal compute, often beats debate and reflection. Gives the evaluation protocol for accuracy-against-tokens plots. |
| 5 | [Towards Reliable LLM Evaluation: Correcting the Winner's Curse in Adaptive Benchmarking](https://arxiv.org/abs/2605.05973) | Names our main problem. Once questions are reused inside a search, the winner's score does not estimate fresh-data performance. Proposes a repeated-split reporting protocol (SIREN). Our split-half check is a simple version of the same idea. |
| 6 | [GEPA: Reflective Prompt Evolution](https://arxiv.org/abs/2507.19457) (ICLR 2026 oral) | Keeps any candidate that is best on at least one training instance: the nearest published idea to our per-cluster slots. Read with lexicase selection (section 2D). |
| 7 | [EvoFlow: Evolving Diverse Agentic Workflows On The Fly](https://arxiv.org/abs/2502.07373) | The closest evolutionary method: niching selection over a population of agent workflows, with cost as a second objective. The paper a reviewer will name when they see "evolutionary" and "workflows" in our title. |
| 8 | [MaAS: Multi-agent Architecture Search via Agentic Supernet](https://arxiv.org/abs/2502.04180) and [MasRouter](https://arxiv.org/abs/2502.11133) (ACL 2025) | The query-level competitors: both train a neural controller that picks an agent system per query. Our contrast is a readable program per cluster, matched at test time. |

---

## 2. Evolutionary search (the headline keyword)

### 2A. A language model as the mutation operator: evolving programs

| Paper | What it is | Why it matters to us |
|---|---|---|
| [Evolution through Large Models (ELM)](https://arxiv.org/pdf/2206.08896), Lehman et al., 2022 | The starting point: a code model as the mutation operator in genetic programming, combined with MAP-Elites. | The origin of "LLM as mutator". We moved away from it (random single edits) and should say why: see the mutator-bias note in section 9. |
| [Language Model Crossover (LMX)](https://arxiv.org/abs/2302.12170) | Few-shot prompting as a crossover operator. | The standard citation for LLM-driven recombination. We have no crossover; worth a sentence. |
| [FunSearch](https://www.nature.com/articles/s41586-023-06924-6) (Nature, 2023) | LLM proposes programs, an evaluator scores them, islands keep diversity. Found new results in combinatorics. | The best-known example of the recipe. Its evaluator is exact; ours is noisy, which is the key difference. |
| [AlphaEvolve](https://arxiv.org/abs/2506.13131) (DeepMind, 2025) | FunSearch generalised to whole codebases; MAP-Elites plus islands. | The current flagship. Closed source. |
| [OpenEvolve](https://github.com/codelion/openevolve), [ShinkaEvolve](https://github.com/SakanaAI/ShinkaEvolve), [CodeEvolve](https://arxiv.org/pdf/2510.14150) | Open implementations. ShinkaEvolve adds weighted archive sampling and novelty filters for sample efficiency. | Where to look for archive and island designs that are known to work, and for how they reject near-duplicate children (we do this by behaviour). |
| [LEVI: Stronger Search Architectures Can Substitute for Larger LLMs in Evolutionary Search](https://arxiv.org/html/2605.09764v1) | Argues the search design matters more than the size of the proposing model. | Supports our choice of a cheap, model-free mutation step. |
| [Evolution of Heuristics (EoH)](https://arxiv.org/abs/2401.02051), [ReEvo](https://arxiv.org/abs/2402.01145), [LLaMEA](https://arxiv.org/pdf/2405.20132) | LLM-driven evolution of optimisation heuristics, with reflection between generations. | The "LLM reflects on why a parent failed" idea. We tried guided edits in v2 and they earned nothing measurable. |
| [Eureka](https://arxiv.org/abs/2310.12931) | Evolutionary search over reward functions written as code. | Another exact-evaluator example; useful contrast. |
| LLM-driven multi-island genetic programming for readable rule sets (arXiv 2602.00755) | Evolves interpretable rule sets with an LLM. | The nearest work to "evolving readable if-then rules". |

### 2B. Evolving prompts

| Paper | What it is | Why it matters to us |
|---|---|---|
| [EvoPrompt](https://arxiv.org/abs/2309.08532) (ICLR 2024) | Genetic algorithm and differential evolution over prompts, with the LLM doing crossover and mutation. | The standard citation for evolutionary prompt search. |
| [Promptbreeder](https://arxiv.org/abs/2309.16797) | Evolves task prompts and the mutation prompts together. | Self-referential evolution; we fix the operators instead. |
| [GEPA](https://arxiv.org/abs/2507.19457) | Reflective prompt evolution with per-instance Pareto selection. | See section 1. |
| [Diverse Prompts: Illuminating the Prompt Space with MAP-Elites](https://arxiv.org/abs/2504.14367) | A context-free grammar plus MAP-Elites over prompts. | Closest in form to us: a grammar defines the space and a quality-diversity archive explores it. |
| [DSPy](https://arxiv.org/abs/2310.03714), [MIPRO](https://arxiv.org/abs/2406.11695), [TextGrad](https://arxiv.org/abs/2406.07496) | Non-evolutionary prompt and pipeline optimisers. | What reviewers will name as the alternative to evolution. |

### 2C. Evolving agent systems and workflows

| Paper | What it is | Why it matters to us |
|---|---|---|
| [EvoFlow](https://arxiv.org/abs/2502.07373) | Niching evolutionary search over a population of workflows; accuracy and cost as two objectives; evolves query by query. | See section 1. Differs from us: niches come from tags on workflows, ours from clusters of questions. |
| [EvoAgent](https://arxiv.org/abs/2406.14228) | Grows one agent into a multi-agent system by mutation and crossover of agent settings. | Evolves the agents; we evolve the control flow. |
| [EvoMAS: Evolutionary Generation of Multi-Agent Systems](https://arxiv.org/pdf/2602.06511) (ICML 2026) | Evolves multi-agent systems with one global score per task. | The "one score per task" baseline design that per-cluster search argues against. |
| [EvoMAS: Learning Execution-Time Workflows for Multi-Agent Systems](https://arxiv.org/html/2605.08769v1) | A different paper with the same name: builds the workflow during execution, as a sequential decision problem. | Same goal as our rules (decide the next step from the state), learned differently. |
| [EvoAgentX](https://arxiv.org/abs/2507.03616) (EMNLP 2025 demo) | Open platform bundling TextGrad, AFlow and MIPRO for workflow optimisation. | A possible harness for running AFlow as a baseline. |
| [ADAS: Automated Design of Agentic Systems](https://arxiv.org/abs/2408.08435) | A meta-agent writes new agent designs in code and keeps an archive of them. | Archive-based search without a fitness-proportional selection step. |
| [AFlow](https://arxiv.org/abs/2410.10762) (ICLR 2025) | Monte Carlo tree search over code-written workflows; one workflow per task. | The main train-time baseline (section 8). |

### 2D. Classical ideas we borrow

| Idea | Reference | Where it appears in our method |
|---|---|---|
| Quality-diversity and MAP-Elites | [Mouret and Clune, 2015](https://arxiv.org/abs/1504.04909) | The grid of slots: one cell per cluster per role, each keeping its best program. |
| Multi-task MAP-Elites | [Mouret and Maguire, 2020](https://arxiv.org/abs/2003.04407) | One archive, many tasks, each cell a task: the direct precedent for "one cell per question cluster". |
| Lexicase selection, and lexicase selection of specialists | [arXiv 1905.09372](https://arxiv.org/abs/1905.09372) | Selecting parents that excel on some cases, not on the average. Our slot B (the rank-gap specialist) is a simplified form. |
| Niching and fitness sharing | Goldberg and Richardson, 1987; see EvoFlow for a modern use | Why a single global ranking collapses diversity. |
| Novelty search | Lehman and Stanley, 2011 | Our rule that a child behaving like its parent is redrawn is a behaviour-novelty filter. |
| Multi-objective selection (NSGA-II) | Deb et al., 2002 | Cost as a second objective. We use cost only to break ties; say so. |
| Algorithm portfolios: Hydra | Xu, Hoos and Leyton-Brown, AAAI 2010 | Builds a set of configurations that complement each other. The portfolio view of our per-cluster champions. |
| Instance-specific algorithm configuration: ISAC | [Kadioglu et al., ECAI 2010](https://ai.dmi.unibas.ch/research/reading_group/kadioglu-et-al-ecai2010.pdf) | Cluster the instances, tune one configuration per cluster, send a new instance to the nearest centre. Our whole pipeline in one sentence, from 2010. Cite it. |
| Surveys | [Evolutionary Computation and LLMs: A Survey](https://arxiv.org/html/2505.15741v1); [LLMs for Evolutionary Optimization: A Systematic Survey](https://arxiv.org/pdf/2509.08269) | Where to find anything missing from this list. |

### 2E. Evolution under a noisy score (the part that matters most for us)

Our split-half check showed that a program picked on 25 questions is 0 to 1 point better on the next
25 of the same cluster. This literature is about exactly that.

| Paper | What it is | Why it matters to us |
|---|---|---|
| Jin and Branke, "Evolutionary Optimization in Uncertain Environments: A Survey" (IEEE TEVC, 2005) | The standard survey on noisy fitness. | The vocabulary: resampling, thresholding, robust selection. |
| Rakshit, Konar and Das, "Noisy Evolutionary Optimization Algorithms: A Comprehensive Survey" (2017) | A later, broader survey. | Lists the fixes and what each costs. |
| [Noisy Optimization: Convergence with a Fixed Number of Resamplings](https://arxiv.org/pdf/1404.2553) | Theory: a fixed number of re-evaluations is not enough as the search narrows. | Explains why our 2 replicates stop helping once candidates are within a point of each other. |
| [Adaptive Resampling with Bootstrap for Noisy Multi-Objective Optimization](https://arxiv.org/pdf/2503.21495) | Spend re-evaluations only where the comparison is uncertain. | A principled version of our "confirm a would-be slot holder" step. |
| F-Race (Birattari et al., 2002) and irace (López-Ibáñez et al., 2016) | Racing: drop a candidate as soon as a statistical test says it is worse. | The standard tool in algorithm configuration. Our one-standard-error screen is a one-step race. |
| [Hyperband](https://arxiv.org/abs/1603.06560) and successive halving | Give many candidates a small budget, keep the best fraction, repeat. | The obvious alternative design for our screen. |
| [Correcting the Winner's Curse in Adaptive Benchmarking](https://arxiv.org/abs/2605.05973) | See section 1. | The reporting protocol to follow. |
| ["The Winner's Curse in LLM Pipeline Selection"](https://github.com/juniorcharlie/llm-pipeline-selection) (code and paper) | Best-of-N reporting breaks when the candidates are correlated, and how to correct it. | Our candidates share cached prefixes, so they are strongly correlated. |
| [tinyBenchmarks](https://arxiv.org/abs/2402.14992) and Anchor Points | Small question sets that predict full-benchmark scores. | The other route to cheaper evaluation. Both need results from many runs on the full set, which we ruled out. |

---

## 3. Does debate beat simply spending more compute?

| Paper | One line |
|---|---|
| [Single-Agent LLMs Outperform Multi-Agent Systems Under Equal Thinking Token Budgets](https://arxiv.org/abs/2604.02460) | Section 1. |
| [Stop Overvaluing Multi-Agent Debate](https://arxiv.org/abs/2502.08788) | Section 1. |
| [Budget-Aware Evaluation of LLM Reasoning Strategies](https://arxiv.org/html/2406.06461) | Section 1. |
| [The Cost of Consensus: Isolated Self-Correction Prevails Over Unguided Homogeneous Multi-Agent Debate](https://arxiv.org/html/2605.00914v1) | A thinking-mode Qwen3-8B changes its answer in 2-6% of debate turns, against 20-70% for non-thinking models. One reported debate cost 14.6 times the tokens for 5 points. Bears on whether rounds after a deep-think turn can do anything. |
| [The Illusion of Multi-Agent Advantage](https://arxiv.org/pdf/2606.13003) | Questions multi-agent gains over strong single-call baselines. Skim after the three above. |
| [Multi-Agent Reasoning Improves Compute Efficiency: Pareto-Optimal Test-Time Scaling](https://arxiv.org/pdf/2605.01566) | The opposing view: multi-agent reasoning can sit on the better side of the cost-accuracy curve. Read for balance. |
| [Improving Factuality and Reasoning through Multiagent Debate](https://arxiv.org/abs/2305.14325) (Du et al.) | The debate protocol our `mad` seed implements. |
| [Self-Refine](https://arxiv.org/abs/2303.17651) | The protocol our `self_refine` seed implements. |
| [Self-Consistency](https://arxiv.org/abs/2203.11171) | The vote baseline. |
| [ARMOR-MAD](https://arxiv.org/abs/2606.13197) | Adaptive debate with hand-written control; our `early_exit_agree` seed follows it. |

---

## 4. Choosing when to think hard

This is the favourable framing for the deep-think speaker: the value is in deciding when to spend it.

| Paper | One line |
|---|---|
| [ARES: Adaptive Reasoning Effort Selection for Efficient LLM Agents](https://arxiv.org/html/2603.07915v1) | A small router picks the lowest adequate reasoning level per step; 40-50% fewer tokens with little accuracy loss. |
| [Not All Turns Are Equally Hard: Adaptive Thinking Budgets](https://arxiv.org/html/2604.05164) | Multi-turn reasoning as a compute-allocation problem; a learned per-turn budget policy. |
| [DART: Draft-Agreement Routing for Training-Free Adaptive Thinking Budgets](https://arxiv.org/pdf/2606.23181) | Think longer only when cheap drafts disagree. Almost exactly the rule we hope the search finds ("call deep-think when the solvers disagree"). Read this one. |

---

## 5. Matching a test question to a program: routing and algorithm selection

| Paper | One line |
|---|---|
| [Rethinking Predictive Modeling for LLM Routing: When Simple kNN Beats Complex Learned Routers](https://arxiv.org/pdf/2505.12601) | Nearest neighbours in embedding space match or beat trained routers. Justifies our simple router. |
| [RouterBench](https://arxiv.org/pdf/2403.12031) | The standard routing benchmark and its baseline routers. |
| [The Routing Plateau](https://arxiv.org/pdf/2606.07587) | Routers hit a ceiling set by how separable the queries are. Matches our result that routed equals random-cluster. |
| [LLMRouterBench](https://arxiv.org/html/2601.07206v1) | A larger, later routing benchmark. |
| [LLM Routing with Dueling Feedback](https://arxiv.org/pdf/2510.00841) | Routing learned from pairwise comparisons. |
| [ISAC](https://ai.dmi.unibas.ch/research/reading_group/kadioglu-et-al-ecai2010.pdf) | Section 2D. Nearest cluster centre, with a fallback when the instance is far from all centres. |
| SATzilla (Xu et al.) and [claspfolio 2](https://arxiv.org/pdf/1405.1520) | Per-instance performance prediction for choosing a solver. The learned-predictor route we decided against. |
| [sunny-as2](https://arxiv.org/pdf/2009.03107) and 3S | Nearest-neighbour solver schedules. The precedent for "run two champions when unsure". |
| [Algorithm selection on a meta level](https://link.springer.com/article/10.1007/s10994-022-06161-4) | Choosing among selectors; a survey-like entry point. |

---

## 6. Query-level and task-level agent-system design (the competitors)

| Paper | Level | One line |
|---|---|---|
| [MaAS](https://arxiv.org/abs/2502.04180) | per query | Samples an agent system from a learned "supernet" for each query. Reports AFlow-level accuracy at about one seventh of AFlow's training cost. |
| [MasRouter](https://arxiv.org/abs/2502.11133) | per query | A cascaded trained controller: collaboration mode, roles, and which model each agent uses. |
| [Difficulty-Aware Agentic Orchestration (DAAO)](https://arxiv.org/html/2509.11079v3) | per query | Routes by predicted difficulty. Maps onto our `difficulty` field. |
| [FlowReasoner](https://arxiv.org/pdf/2504.15257), [ScoreFlow](https://arxiv.org/abs/2502.04306), [MAS-GPT](https://arxiv.org/abs/2503.03686) | per query | Fine-tune a model to write a workflow for each query. Need GPU training; cite, do not run. |
| [From Search to Synthesis: Training LLMs as Zero-Shot Workflow Generators](https://arxiv.org/html/2606.30704) | per query | A recent entry in the same line. |
| [Do We Always Need Query-Level Workflows?](https://arxiv.org/html/2601.11147) (ACL Findings 2026) | critique | Argues per-query generation is often unnecessary. Supports our middle ground of one program per cluster. |
| [AFlow](https://arxiv.org/abs/2410.10762), [ADAS](https://arxiv.org/abs/2408.08435), [GPTSwarm](https://arxiv.org/abs/2402.16823) | per task | One searched design for the whole task. |
| [DyLAN](https://arxiv.org/abs/2310.02170), [G-Designer](https://arxiv.org/pdf/2410.11782), [AgentPrune](https://arxiv.org/abs/2410.02506) | per task or query | Optimise who talks to whom. A different question from ours; cite only. |
| [OrchDebate](https://openreview.net/pdf?id=JyJSJitKjy) | per step | Section 1. |
| [SAT, Self-Organizing Agent Teams](https://arxiv.org/abs/2609.22682) | per task, bank of 10 run in parallel | Section 1, entry 1b. Evolves conversation strategies by LLM reflection; no routing. |

---

## 7. Grouping questions by the reasoning they need

| Paper | One line |
|---|---|
| Didolkar et al., "Metacognitive Capabilities of LLMs" (NeurIPS 2024, [arXiv 2405.12205](https://arxiv.org/abs/2405.12205)) | A strong model labels questions by skill, then the labels are clustered. The nearest precedent for our description-then-cluster step. |
| [Mixture-of-Prompts](https://arxiv.org/abs/2407.00256) (ICML 2024) | Cluster the inputs, optimise one prompt per region. |
| LLM task descriptions plus graph clustering for routing (arXiv 2603.19415, Zhang, Karypis et al.) | Descriptions written by a model, clustered, used for routing. |

---

## 8. Baselines for the paper

All run on the same base model, at the same thinking setting as what they are compared with, with
accuracy reported beside tokens.

| Baseline | Train-time part | Test-time part | Status |
|---|---|---|---|
| Direct chain-of-thought, thinking on | none | one call | done for gpt-oss; running for Qwen 9B |
| Self-refine, thinking on | none | generate, critique, refine | done for gpt-oss; running for Qwen 9B |
| Self-consistency, thinking on | none | k calls, majority vote | free: computed from the saved k=3 direct runs |
| Multi-agent debate (Du et al.), thinking on | none | 3 agents, 3 rounds | to do; exists only at short-reply settings (`mad`) |
| One deep-think call (`deep_direct`) | none | one long-thinking call in our executor | built; runs with the deep search |
| LLM orchestrator, as in OrchDebate | none | a model call picks the next move after every round | to build, inside our executor |
| AFlow | tree search on the training questions | one workflow for every question | to do; the key train-time baseline |
| MaAS | trains a supernet controller | samples a system per query | no implementation found in the collaborator's folder |
| MasRouter or DAAO | trains a router | routes per query | optional second query-level method |
| Ours, global champion | evolutionary search | one program for every question | in the dev table |
| Ours, best literature seed per cluster | pick on held-out, no search | route by cluster | to add; shows what evolution adds |
| Ours, random cluster's champion | evolutionary search | random routing | in the dev table; shows what routing adds |

---

## 9. Where our work sits

- **The recipe is old and the application is new.** Cluster the instances, tune one configuration
  per cluster, route by nearest centre: that is ISAC (2010). Multi-task MAP-Elites and lexicase
  selection are the evolutionary precedents for keeping specialists. Nobody we found combines a
  learned if-then debate program, a per-cluster search and a small representative question subset.
- **What separates us from the evolutionary LLM work in 2A and 2B:** their scores are exact or
  nearly so (a program passes its tests or it does not). Ours is a noisy accuracy on 50 questions
  at a model near chance. Section 2E is therefore not background for us; it is the core problem.
- **What separates us from the query-level systems in section 6:** they train a neural controller.
  We produce a readable program with no controller cost at test time.
- **Why our mutations are random and not model-written:** in earlier runs the model as mutator
  kept adding rounds, and in the v2 run guided edits earned nothing measurable. LEVI (2A) supports
  putting the effort into the search design instead.
- **Two papers this could become.** An adaptive-compute paper (programs that learn when to spend a
  deep-think call, judged on accuracy per token; sections 3 and 4), or an evaluation paper
  (workflow search on small question sets mostly selects luck, and a split-half check exposes it;
  section 2E and the winner's-curse papers). The deep-think and Qwen 9B results decide which.
