# Literature review: per-query-type debate programs

Working notes for the related-work section. Collected 2026-09-17 from
abstracts and summaries, not full papers. Check the details against the
PDFs before citing.

Our setting: a debate controller written as an ordered list of if-then rules
(a "program"), found by evolutionary search over cached transcripts. The
planned change is to group training questions by reasoning type, using a
scrubbed reasoning template from a stronger model, and to search for a
program per group instead of one program for all questions.

No paper found does this combination. The pieces exist separately:

| Piece of our pipeline | Closest prior work |
|---|---|
| Group questions, then optimize per group | Mixture-of-Prompts |
| LLM writes a reasoning label per question, labels are clustered | Didolkar et al.; Fine-Grained Latent Task Discovery |
| Selection that keeps specialists instead of one average winner | GEPA; lexicase selection; MAP-Elites and islands (FunSearch, AlphaEvolve) |
| A different multi-agent system per query | MaAS; FlowReasoner; DAAO |
| Evolving multi-agent systems | EvoAgent; EvoMAS; Evolving Interpretable Constitutions |
| Adaptive control of multi-agent debate | OrchDebate; ARMOR-MAD |
| Small representative question subsets | Anchor Points; tinyBenchmarks |

## 1. Group first, then optimize per group

**Mixture-of-Prompts (MoP).** Wang, An, Cheng, Zhou, Hwang, Hsieh. ICML 2024.
arXiv 2407.00256.
Divides the problem space into regions by clustering demonstrations on
semantic similarity, then searches for a separate instruction per region.
Motivated by the same observation as ours: one prompt cannot cover a varied
problem space. Reports an 81% average win rate against single-prompt
optimizers.
*Difference:* they optimize prompts for a single model call; we optimize a
control program for a multi-round debate. Their regions come from text
similarity; ours are meant to come from reasoning shape.

**Metacognitive Capabilities of LLMs.** Didolkar, Goyal, Ke, Guo, Valko,
Lillicrap, Rezende, Bengio, Mozer, Arora. NeurIPS 2024. arXiv 2405.12205.
A strong model (GPT-4) assigns a skill label to each math question. The
labels are clustered into coarser skill families. At test time the model
names the skill for a new question and is shown solved examples from that
skill. Improves GSM8K and MATH for several models.
*Difference:* the closest match to our "scrubbed reasoning template" step.
They use the clusters to pick worked examples; we would use them to pick a
debate program. Their domain is math; ours is 72-field graduate MCQ.

**Scalable Prompt Routing via Fine-Grained Latent Task Discovery.** Zhang,
Adeshina, Guan, Ganesh, Han, Ioannidis, Rangwala, Karypis. 2026. arXiv
2603.19415.
An LLM writes a task description for each prompt. Graph clustering combines
description similarity with which model did best on the prompt, so the
clusters track both wording and outcome. A classifier assigns new prompts to
clusters; a second stage routes among 11 models. Beats the best single model
at under half its cost across 10 benchmarks.
*Difference:* routes between models, not between debate controllers. Their
outcome-aware clustering is an option for us once program outcomes exist.

## 2. Search that keeps specialists

**GEPA: Reflective Prompt Evolution.** Agrawal et al. ICLR 2026 (oral).
arXiv 2507.19457.
Evolutionary prompt optimizer with two ideas we need. (a) Pareto selection
per training instance: every candidate that is best on at least one training
example stays in the pool, and parents are sampled from that pool in
proportion to how many examples they win. Specialists survive by design.
(b) LLM-guided mutation: the model reads execution traces of failures and
proposes the edit, instead of a random edit. Beats GRPO with up to 35x fewer
rollouts.
*Difference:* they evolve prompt text; we evolve a rule program. Their
per-instance selection is a direct answer to our observed collapse, where the
top five evolved programs were all cosmetic variants of one seed.

**Lexicase selection.** Helmuth, Spector and others; specialists analysis in
arXiv 1905.09372.
Parent selection by filtering the population through training cases in random
order. The standard non-LLM way to keep specialists in genetic programming.
Useful as the plain-English explanation of why per-instance selection keeps
diversity, and as a cheaper alternative to GEPA's Pareto set.

**FunSearch / AlphaEvolve.** DeepMind, 2023 and 2025.
LLM-guided program evolution. Diversity comes from islands (separate
sub-populations that occasionally exchange programs) and from MAP-Elites (a
grid where each cell keeps the best program of one kind).
*Relevance:* our question clusters can serve as the MAP-Elites cells: one
population, the best program kept per cluster, parents drawn from any cell.

**Evolving Interpretable Constitutions for Multi-Agent Coordination.** Kumar,
Saito, Niranjani, Yessou, Tan. 2026. arXiv 2602.00755.
LLM-driven genetic programming over islands, evolving readable rule sets that
govern agents in a grid world. Evolved rules beat human-written ones.
*Relevance:* evidence that LLM-guided evolution over readable rules works;
different task.

## 3. A different multi-agent system per query

**MaAS: Multi-agent Architecture Search via Agentic Supernet.** Zhang, Niu et
al. ICML 2025 (oral). arXiv 2502.04180.
A trained controller samples a query-specific agent workflow from a
"supernet" distribution. Uses 6–45% of the inference cost of fixed systems
and beats them by 0.5–11.8 points on six benchmarks.

**FlowReasoner: Reinforcing Query-Level Meta-Agents.** Gao et al. 2025.
arXiv 2504.15257.
A meta-agent trained by distillation from DeepSeek-R1 and then RL builds one
multi-agent system per query.

**DAAO: Difficulty-Aware Agentic Orchestration.** Su et al. WWW 2026. arXiv
2509.11079.
A VAE estimates query difficulty; an operator allocator and an LLM router
build a workflow matched to it.

*Difference from all three:* their per-query decision is a trained neural
network or a generator LLM. Ours is a readable program with no controller
cost at run time, and the per-query variation comes from (a) which cluster
the question falls in and (b) the program's own branching on the debate
state. They show that per-query systems pay off, which supports the
direction.

## 4. Evolving multi-agent systems with one global score

**EvoAgent.** Yuan et al. NAACL 2025. arXiv 2406.14228.
Crossover and mutation over agent configurations, scored by task
performance.

**EvoMAS: Evolutionary Generation of Multi-Agent Systems.** Hu, Zhang, Trager,
Zhang, Yang, Xia, Soatto. ICML 2026. arXiv 2602.06511.
Evolves structured MAS configurations with feedback-conditioned mutation and
an experience memory. +10.5 over EvoAgent on BBEH; 79.1% SWE-Bench-Verified.

*Difference:* both use one fitness per task, the setup that collapsed our
population. Neither evolves a round-by-round debate controller.

## 5. Adaptive debate control

**OrchDebate.** Anonymous ACL submission, OpenReview JyJSJitKjy. See
`11399_OrchDebate_Adaptive_Orch.pdf` in the repo root.
Same framing as ours: a controller reads the debate state and picks the next
action (critic, verifier, diversifier, finalize). The controller is a
zero-shot LLM call at every step and is never learned; only skill prompt
wording evolves.

**ARMOR-MAD.** Niu, Zhang. 2026. arXiv 2606.13197.
Training-free control of heterogeneous debate: skip debate when round-0
answers agree, stop early on convergence, down-weight outlier answers.

*Difference:* nobody in the debate literature learns the controller. Ours is
learned, readable, and free at run time.

## 6. Small representative subsets

**Anchor Points** (Vivek et al. 2023) and **tinyBenchmarks** (Polo et al.
2024, arXiv 2402.14992).
Pick ~100 questions whose correctness patterns stand in for the full
benchmark, by clustering correctness vectors or IRT parameters.
*Difference:* both need outcomes from many runs over the full set, which we
want to avoid. Our substitute is text-only: k-medoids over template
embeddings, taking the medoids as the representative questions.

## Framing for the paper

- Fixed debate protocols and single global controllers both optimize for the
  average question (MoP for prompts; MaAS/DAAO for agent systems make the same
  point).
- Existing per-query systems use trained neural controllers; existing debate
  controllers are unlearned. We learn a readable controller per question type.
- Existing evolutionary MAS methods use one global score, which we show
  collapses the population; we use per-cluster elites (MAP-Elites over
  reasoning types) with per-instance selection (GEPA / lexicase).

## Still to check

- Read GEPA's Pareto selection section for the exact sampling rule.
- Read MoP for how a test query is routed to a region.
- Read the Fine-Grained Latent Task Discovery paper for the clustering details
  (graph construction, number of clusters).
- Look for any 2026 work on evolutionary or QD search over debate protocols.
- "Rethinking the Value of Multi-Agent Workflow: A Strong Single Agent
  Baseline" (arXiv 2601.12307) argues many MAS gains vanish against a strong
  single agent; read it and make sure our baselines answer it.

## Links

- MoP: https://arxiv.org/abs/2407.00256
- Didolkar et al.: https://arxiv.org/abs/2405.12205
- Fine-Grained Latent Task Discovery: https://arxiv.org/abs/2603.19415
- GEPA: https://arxiv.org/abs/2507.19457
- Lexicase specialists: https://arxiv.org/abs/1905.09372
- AlphaEvolve overview: https://www.emergentmind.com/topics/alphaevolve-framework
- Evolving Interpretable Constitutions: https://arxiv.org/abs/2602.00755
- MaAS: https://arxiv.org/abs/2502.04180
- FlowReasoner: https://arxiv.org/abs/2504.15257
- DAAO: https://arxiv.org/abs/2509.11079
- EvoAgent: https://arxiv.org/abs/2406.14228
- EvoMAS: https://arxiv.org/abs/2602.06511
- ARMOR-MAD: https://arxiv.org/abs/2606.13197
- OrchDebate: https://openreview.net/pdf?id=JyJSJitKjy
- tinyBenchmarks: https://arxiv.org/abs/2402.14992
- Workflow-optimization survey: https://arxiv.org/abs/2603.22386
- Strong single-agent baseline: https://arxiv.org/abs/2601.12307
