# Baselines and datasets for the paper

Compiled 2026-10-03 from `literature_review.md`, `literature_reading_list.md` and `outline.txt`.
Three helper agents read 32 of the 33 papers in full text through a tool that summarizes each
page. Check the numbers against the PDFs before citing. OrchDebate was not read: OpenReview
blocked the download, and the copy in the repo root is gone.

SuperGPQA and HLE are our main sets. This file lists what to add to them.

## Decided (2026-10-05)

The user chose this set. It replaces the recommendation at the end of this file.

**Datasets**
- SuperGPQA: the 1,000-question test split (`datasets/supergpqa_2k_test.json`). The 300-question
  split is too small for the paper. The 1,000 hold the old 300 and no question of either train split.
- HLE.
- GPQA-Diamond: 198 questions, test only.
  - Made by `scripts/prepare_gpqa_diamond.py` into `datasets/gpqa_diamond_test.json`.
- MATH level 5: the 1,324 level-5 problems of the MATH test split, split 662/662 by subject.
  - Made by `scripts/prepare_math_l5.py` into `datasets/math_l5_{train,test}.json`.
  - Graded by `math-verify`.

**Baselines**
- Direct CoT, self-consistency, Self-Refine and MAD (Du et al.).
  - All four run from `baselines/run_baselines.py`, one after another.
  - It prints a table with avg@1, vote@3, avg@3, pass@3 and tokens per run.
- MaAS and AFlow: the collaborator runs these, outside this codebase.

**Settings**

| Method | Paper's setting | Ours |
|---|---|---|
| Self-consistency | 40 samples, majority vote | 5 samples per run (the CoT-SC setting of ADAS, AFlow and MaAS), majority vote |
| Self-Refine | up to 4 feedback → refine rounds, full history kept | the same (`--sr-iters 4`). Earlier runs used 2. |
| MAD (Du et al.) | 3 agents, 2 rounds, majority vote of the last round, the same first prompt as the baselines | the same |

Every method uses the dataset's own first prompt:
- SuperGPQA: its zero-shot prompt.
- GPQA: OpenAI simple-evals' prompt.
- MATH: the Qwen / DeepSeek-R1 `\boxed{}` instruction.
- HLE: its own system prompt.

## How I chose

1. The closest papers use it.
2. Our pipeline can run it with little new work. Today it handles multiple choice, and open
   answers graded by the paid judge model (gpt-6-luna). It cannot run code.
3. A 9B thinking model on it is not near 0% or near 100%.

## What the field uses

- **Workflow-search papers** (AFlow, MaAS, DAAO, ScoreFlow, MetaFlow, OneFlow, SCALE) use
  almost the same 6 tasks: GSM8K, MATH, HumanEval, MBPP, HotpotQA and DROP. The test model is
  usually GPT-4o-mini.
- **Debate papers** use mostly multiple-choice science and knowledge sets: MMLU-Pro (5 papers),
  GPQA (4) and MMLU (5). They also use math: MATH (4) and AIME (2).
- **No paper uses SuperGPQA.** Only one uses HLE, with 10 test questions. So our two main sets
  are new to this field. Reviewers will want at least one set they recognize.

## Datasets

### Group 1: run these (multiple choice; fit the pipeline now)

| Dataset | Who uses it | Why | Note |
|---|---|---|---|
| **GPQA-Diamond** (198 questions) | Self-Organizing Agent Teams, Illusion of Multi-Agent Advantage, DART, ADAS, MAS-GPT, Smit et al. (GPQA main) | The best-known hard science set; the closest to SuperGPQA. | Too few to split. Search on the GPQA-Extended questions that are not in Diamond (about 350); test on Diamond. Diamond is never used for search. |
| **MMLU-Pro** (12k questions, 10 options) | Stop Overvaluing MAD, ARMOR-MAD, Self-Organizing Agent Teams, Multi-Agent Compute Efficiency, DART | The most common set in debate papers. Same format as SuperGPQA (A–J), so no code change. | Large, so the same 200 / 100 / 300 split works. |

### Group 2: run one (math, short exact answers)

| Dataset | Who uses it | Note |
|---|---|---|
| **MATH level 5** (the 617-question subset) | AFlow, MaAS, DAAO, EvoFlow, ScoreFlow, ARMOR-MAD | The competitors' own setup, so their numbers can be compared with ours. |
| OlympiadBench (572 questions) | DART | A harder option if level 5 is too easy. |
| AIME 2024 / 2025 (60 questions) | Self-Organizing Agent Teams, EvoMAS, GEPA, DART | Test only. Too small for search. |

Work needed: a rule-based answer checker (for example the `math-verify` library). The current
open-answer grader calls the paid judge model on every answer; do not use it here. A 9B thinking
model may score very high on MATH: check with one short direct run on 50 questions first.

### Group 3: optional (the "search" line in the outline)

- If "search" means multi-hop questions: HotpotQA or DROP (the AFlow set; needs an F1 grader),
  or FRAMES / MuSiQue (used by the equal-token paper).
- If it means web search (GAIA, BrowseComp): this needs tools we do not have.
- **Open question:** which one the outline means.

### Skip

- HumanEval, MBPP: we cannot run code.
- GSM8K, MMLU: too easy for a thinking model.
- GAIA, SWE-Bench and other agent tasks: they need tools.

## Baselines

All on the same base model, with thinking on, and accuracy reported beside tokens.

### A. No training (run all)

| Baseline | Papers that use it | Status |
|---|---|---|
| Direct answer (CoT) | almost all | done |
| Self-consistency (majority vote) | 9 debate + 9 workflow papers | partly free from our recordings; add a version with the same tokens as our program |
| Self-Refine | 11 | done |
| Multi-agent debate (Du et al.) | 9 + 9 | to do at long-reply settings; our `mad` seed is close |
| One agent with the same token budget | 2 (Single-Agent > MAS at equal tokens; Cost of Consensus) | to do; answers the strongest critique |
| Mixture-of-Agents | 3 | optional |

### B. Search once per task (the main competitors)

| Baseline | Notes |
|---|---|
| **AFlow** | The most common search baseline (9 papers). Code: github.com/FoundationAgents/AFlow; `base_url` per model in the config, so it can use our vLLM server. |
| **ADAS** | 6 papers. Already reports GPQA and MMLU, so it suits multiple choice. Code: github.com/ShengranHu/ADAS; needs small edits (model names are fixed in the code). |

Both use a strong model to design the workflow (AFlow used Claude 3.5 Sonnet). For a fair
comparison, use our guide model (gpt-6-sol). That is paid: **agree a budget first.**

### C. Per-query systems (support the query-level selection claim)

| Baseline | Notes |
|---|---|
| **MaAS** | Used by DAAO, Illusion of Multi-Agent Advantage and EvoMAS (execution-time). Code: github.com/bingreeky/MaAS; works with vLLM. |
| DAAO | Built on MaAS. Optional. Code: github.com/AutoAgents-ai/DAAO. |
| MasRouter | Skip. It chooses among several different LLMs; we use one. |
| **DART** | Training-free: think more only when cheap drafts disagree. Very close to the routing rule we tested, so the most direct baseline for our query-level selection. Code: github.com/js-lee-AI/DART; uses Qwen3. |
| kNN router over our programs | The kNN routing paper shows kNN matches learned routers. A cheap, standard router baseline. |

### D. Cite only, do not run

- FlowReasoner, ScoreFlow, MAS-GPT: they need GPU training.
- Self-Organizing Agent Teams: uses several different models.
- OrchDebate: its "a model chooses each step" controller could be built inside our executor later.

### E. Our ablations

- The best seed (no search).
- One program for all questions (no selection).
- Random selection.
- The oracle router: the best program per question, with the answers known.

## Recommendation

The smallest set that would satisfy a reviewer:

- **Datasets:** SuperGPQA, HLE, GPQA-Diamond, MMLU-Pro, MATH level 5.
- **Baselines:** Direct, self-consistency, Self-Refine, Du debate, equal-token single agent,
  AFlow, MaAS, DART, plus our ablations.

Open decisions: the meaning of "search" in the outline, and a guide-model budget for AFlow / ADAS.

## Per-paper reference

Datasets and baselines as reported by each paper (full text unless noted).

### Debate and compute papers

| Paper | Datasets | Baselines | Models | Code |
|---|---|---|---|---|
| Self-Organizing Agent Teams (2609.22682) | AIME 24/25/26, HMMT Feb 26, TheoremQA-physics; GPQA-Diamond (25 train / 100 test), MMLU-Pro (100), BBEH logic (75) | best member, SC (K=10), self-reflection, member vote, debate, MoA, homogeneous team, routing oracle | o3-mini, Claude Sonnet 4, DeepSeek-V3; Gemini-2.5-Flash, Llama-4-Maverick, GPT-4.1 | not stated |
| ARMOR-MAD (2606.13197) | MATH L5, GSM8K, MMLU, MMLU-Pro (200 each) | CoT, SC, Homo-MAD, D-MAD, Hetero vote, Hetero-MAD, MoA | gpt-4o-mini, deepseek-v3, qwen-plus | not stated |
| Du et al. (2305.14325) | arithmetic, GSM8K, MMLU (100 each), chess, biographies | single agent, reflection, majority, debate | gpt-3.5-turbo | project page |
| Stop Overvaluing MAD (2502.08788) | MMLU, MMLU-Pro, CSQA, ARC-C, AGIEval, GSM8K, MATH, HumanEval, MBPP | SA, CoT, SC; SoM, Multi-Persona, EoT, AgentVerse, ChatEval | gpt-4o-mini, claude-3.5-haiku, Llama3.1-8B/70B | anonymous.4open.science/r/MAD-eval-E4C4 |
| Single-Agent > MAS at equal tokens (2604.02460) | FRAMES, MuSiQue (4-hop) | SAS, SAS-L, sequential, subtask-parallel, parallel roles, debate, ensemble | Qwen3-30B-A3B, R1-Distill-70B, Gemini-2.5-Flash/Pro | not stated |
| Cost of Consensus (2605.00914) | GSM-Hard (1,017), MMLU-Hard (100) | base, peer debate, context resetting, self-correction, 10x-token single agent | Qwen2.5-7B, Llama-3.1-8B, Ministral-8B | github.com/sensorlab/llm-debate-dynamics |
| Illusion of Multi-Agent Advantage (2606.13003) | GPQA-Diamond (166 test), HLE-Maths (168), SWE-Bench Lite, BrowseComp-Plus, SMFR | CoT, CoT-SC, DyLAN, MAS-Zero, ADAS, AFlow, MaAS, MAS-Orchestra | GPT-4o, GPT-5, GPT-OSS-120B, Gemini-2.5-Pro | multi-agent-eval.github.io |
| Reasoning in Token Economies (2406.06461) | GSM8K, MATH, TheoremQA, HotpotQA, Game of 24, CSQA (100 each) | CoT-SC, MAD, Reflexion, ToT, Plan-and-Solve, Least-to-Most, PHP, SC² | GPT-3.5/4, Mistral-7B, Llama-2-70B, Mixtral | none |
| Multi-Agent Compute Efficiency (2605.01566) | MMLU-Pro (1,000), BBH | CoT, SC, self-refine, MAD, MoA | Llama 3.1 8B/70B | github.com/Multi-Agent-LLMs/lm-evaluation-harness |
| DART (2606.23181) | MATH-500, OlympiadBench, AIME 24/25, HumanEval, MBPP, ARC-C, MMLU-Pro, GPQA-Diamond | no-think, always-think, majority vote, MLP/GBT routers, supervised, self-verification, oracle | Qwen3 0.6B–32B, DeepSeek-V3.2 | github.com/js-lee-AI/DART |
| Should we be going MAD? (2311.17371) | MedQA, PubMedQA, MMLU clinical, CosmosQA, CIAR, GPQA main (448), chess | single, SC, ensemble refinement, Medprompt, SoM, ChatEval, Multi-Persona, SPP | GPT-3.5 (GPT-4, Mixtral limited) | github.com/instadeepai/DebateLLM |
| OrchDebate (OpenReview JyJSJitKjy) | not verified | not verified | not verified | not verified |

### Workflow-search and per-query papers

| Paper | Datasets | Baselines | Models | Code (vLLM-ready?) |
|---|---|---|---|---|
| AFlow (2410.10762) | HumanEval, MBPP, GSM8K, MATH L5 (617), HotpotQA, DROP (1,000) | IO, CoT, CoT-SC, MultiPersona debate, Self-Refine, MedPrompt, ADAS | executors GPT-4o-mini and others; optimizer Claude-3.5-Sonnet | FoundationAgents/AFlow (yes) |
| MaAS (2502.04180) | HumanEval, MBPP, GSM8K, MATH, MultiArith, GAIA | CoT, ComplexCoT, SC, MultiPersona, LLM-Debate, LLM-Blender, DyLAN, AgentVerse, MacNet, GPTSwarm, AutoAgents, ADAS, AgentSquare, AFlow | gpt-4o-mini | bingreeky/MaAS (yes) |
| MasRouter (2502.11133) | MMLU, GSM8K, MATH (519), HumanEval, MBPP | CoT, SC, graph topologies, LLM-Debate, GPTSwarm, AgentPrune, AFlow, RouteLLM, FrugalGPT, RouterDC | pool of 4 LLMs | yanweiyue/masrouter (probably) |
| DAAO (2509.11079) | HumanEval, MBPP, GSM8K, MATH (617), MMLU, GAIA | CoT, SC, ADAS, AFlow, MaAS, RouteLLM, MasRouter | 4 LLMs | AutoAgents-ai/DAAO (yes) |
| FlowReasoner (2504.15257) | BigCodeBench, HumanEval, MBPP | Self-Refine, LLM-Debate, ADAS, AFlow, MaAS | trained R1-Distill-7B/14B meta-agent | sail-sg/FlowReasoner (yes) |
| ScoreFlow (2502.04306) | the AFlow six | IO, CoT, CoT-SC, MedPrompt, MultiPersona, Self-Refine, ADAS, AFlow | trained Llama-3.1-8B generator; GPT-4o-mini executor | Gen-Verse/ScoreFlow (yes) |
| MAS-GPT (2503.03686) | MATH, GSM8K, GSM-Hard, MMLU, HumanEval(+), GPQA, SciBench, AIME-24 | Single, CoT, SC, LLM-Debate, Self-Refine, SPP, AgentVerse, GPTSwarm, DyLAN | trained Qwen2.5-Coder-32B generator | rui-ye/MAS-GPT (yes) |
| SCALE, "Do We Always Need Query-Level Workflows?" (2601.11147) | the AFlow six | AFlow, AgentPrune, ScoreFlow | Qwen-Plus; Qwen3-8B optimizer | none |
| MetaFlow, "From Search to Synthesis" (2606.30704) | DROP, GSM8K, MBPP, MATH L5; HotpotQA (zero-shot) | IO, CoT, CoT-SC, MedPrompt, MultiPersona, Self-Refine, ADAS, AFlow, ScoreFlow | Qwen3-8B planner | none |
| OneFlow, "Rethinking the Value of MAS Workflow" (2601.12307) | the AFlow six, Shopping-MMLU, TravelPlanner | IO, CoT, CoT-SC, MultiPersona, AFlow | GPT-4o-mini, Claude-3.5-Haiku, Qwen3-8B | none |
| ADAS (2408.08435) | ARC, DROP, MGSM, MMLU, GPQA-Diamond; transfer to GSM8K, GSM-Hard, SVAMP, ASDiv | CoT, CoT-SC, Self-Refine, LLM-Debate, Step-back, Quality-Diversity, Role Assignment | GPT-3.5 agents; GPT-4 meta agent | ShengranHu/ADAS (with small edits) |

### Evolution and routing papers

| Paper | Datasets | Baselines | Code |
|---|---|---|---|
| EvoFlow (2502.07373) | GSM8K, MATH (617), MultiArith, HumanEval, MBPP, ALFWorld | CoT, SC, MultiPersona, LLM-Debate, DyLAN, AgentVerse, GPTSwarm, ADAS, AgentSquare, AFlow and others | "will be available" |
| EvoAgent (2406.14228) | Logic Grid Puzzle, Trivia Creative Writing, Codenames, MMMU, ScienceWorld, TravelPlanner | Direct, CoT, Self-Refine, SPP, AgentVerse, AutoAgents | project page |
| EvoMAS, generation (2602.06511) | BBEH, WorkBench, SWE-Bench Lite/Verified, AIME 24–25 | direct, single agent, peer review, MAD, majority vote, MetaGPT, ChatDev, MAS-GPT, ADAS, EvoAgent and others | amazon-science/EvoMAS |
| EvoMAS, execution-time (2605.08769) | GAIA, HLE (10 eval), DeepResearcher, BrowseComp | GPT-4o(-mini), GPTSwarm, AFlow, G-Designer, MaAS | none |
| GEPA (2507.19457) | HotpotQA, HoVer, IFBench, PUPA, AIME-2025, LiveBench-Math, NPUEval, KernelBench | GRPO, MIPROv2, TextGrad, Trace | gepa-ai/gepa |
| Mixture-of-Prompts (2407.00256) | Instruction Induction, Super-NaturalInstructions, BBH (10 tasks) | APE, InstructZero, OPRO and variants | turningpoint-ai/mixture-of-prompts |
| Fine-Grained Latent Task Discovery (2603.19415) | a pool from NQ, TriviaQA, CSQA, MMLU, ARC-C, OBQA, GSM8K, MATH, HumanEval, MBPP | kNN, MLP, RouteLLM, RouterDC, GraphRouter, IPR | none |
| kNN routing (2505.12601) | RouterBench, AlpacaEval, Open LLM Leaderboard v2, HELM-Lite; vision sets | kNN, linear, MLP, graph and attentive routers, oracle, random | none |
| The Routing Plateau (2606.07587) | RouterBench, R2-Bench, EmbedLLM, SPROUT, two new pools | 21 routers | none |
| ARES (2603.07915) | TAU-Bench, BrowseComp-Plus, WebArena | fixed low/medium/high effort, random, prompted routers | none |
