# Seed programs (12)

Executor: {'executor': 'v3', 'digest': [2000, 2000], 'summary_words': 500, 'window': 32768, 'high_cost': 3, 'eliminator': False, 'prompts': 'f45e78f215', 'turn_cap': 15, 'total_cap': 21, 'last_round_vote': True, 'plain_instruction': True, 'count_read_summaries': True, 'answers': 'open', 'judge': {'model': 'gpt-6-luna', 'effort': 'medium', 'prompt': 'hle1'}, 'model': 'Qwen/Qwen3.5-9B'}, model Qwen/Qwen3.5-9B

| name | source | plan | rules | extra rounds | stop reads | sanity turns |
|---|---|---|---|---|---|---|
| mad | protocol | solver_x3 > solver_x3 > solver_x3 | 1 | - | vote | - |
| early_exit_agree | protocol | solver_x4 | 4 | critic | vote | - |
| expert_first | protocol | expert | 4 | expert_solver, verifier | last_commit | - |
| direct_high | protocol | solver|high | 1 | - | last_commit | - |
| self_refine_high | protocol | solver|high > critic|high > solver|high > critic|high > solver|high | 2 | - | last_commit | - |
| self_consistency_high | protocol | solver_x3|high | 1 | - | vote | - |
| verify_then_decide_high | protocol | solver_x4 | 2 | verifier|high | last_commit | - |
| fresh_on_disagree_high | protocol | solver_x2 | 3 | solver|high|blind | last_commit, vote | - |
| llm_g0_bound_and_scope_check | llm | solver_x2|high | 4 | expert_solver|high|blind, synthesizer, verifier|high | last_commit | - |
| llm_g1_invariant_count_audit | llm | expert_solver > verifier | 2 | synthesizer|high | last_commit | - |
| llm_g2_independent_view_transforms | llm | solver_x3 | 4 | expert|high|blind, synthesizer, verifier | last_commit | - |
| llm_g3_blind_evidence_crosscheck | llm | expert > solver|blind | 4 | expert|high|blind, synthesizer | last_commit, vote | - |

## Programs

### mad (protocol)
```json
{
 "plan": [
  {
   "personas": [
    "solver",
    "solver",
    "solver"
   ]
  },
  {
   "personas": [
    "solver",
    "solver",
    "solver"
   ]
  },
  {
   "personas": [
    "solver",
    "solver",
    "solver"
   ]
  }
 ],
 "rules": [
  {
   "when": [
    "plan_left"
   ],
   "do": "continue"
  }
 ],
 "default": "stop:vote"
}
```

### early_exit_agree (protocol)
```json
{
 "plan": [
  {
   "personas": [
    "solver",
    "solver",
    "solver",
    "solver"
   ]
  }
 ],
 "rules": [
  {
   "when": [
    "plan_left"
   ],
   "do": "continue"
  },
  {
   "when": [
    "step==1",
    "r1_majority>=3"
   ],
   "do": "stop:vote"
  },
  {
   "when": [
    "step==1"
   ],
   "do": "critic"
  },
  {
   "when": [
    "step==2"
   ],
   "do": "critic"
  }
 ],
 "default": "stop:vote"
}
```

### expert_first (protocol)
```json
{
 "plan": [
  {
   "personas": [
    "expert"
   ]
  }
 ],
 "rules": [
  {
   "when": [
    "plan_left"
   ],
   "do": "continue"
  },
  {
   "when": [
    "step==1"
   ],
   "do": "expert_solver"
  },
  {
   "when": [
    "step==2",
    "last_round_agree"
   ],
   "do": "stop:last_commit"
  },
  {
   "when": [
    "step==2"
   ],
   "do": "verifier"
  }
 ],
 "default": "stop:last_commit"
}
```

### direct_high (protocol)
```json
{
 "plan": [
  {
   "personas": [
    "solver"
   ],
   "effort": "high"
  }
 ],
 "rules": [
  {
   "when": [
    "plan_left"
   ],
   "do": "continue"
  }
 ],
 "default": "stop:last_commit"
}
```

### self_refine_high (protocol)
```json
{
 "plan": [
  {
   "personas": [
    "solver"
   ],
   "effort": "high"
  },
  {
   "personas": [
    "critic"
   ],
   "effort": "high"
  },
  {
   "personas": [
    "solver"
   ],
   "effort": "high"
  },
  {
   "personas": [
    "critic"
   ],
   "effort": "high"
  },
  {
   "personas": [
    "solver"
   ],
   "effort": "high"
  }
 ],
 "rules": [
  {
   "when": [
    "last_round:critic",
    "kept_answer"
   ],
   "do": "stop:last_commit"
  },
  {
   "when": [
    "plan_left"
   ],
   "do": "continue"
  }
 ],
 "default": "stop:last_commit"
}
```

### self_consistency_high (protocol)
```json
{
 "plan": [
  {
   "personas": [
    "solver",
    "solver",
    "solver"
   ],
   "effort": "high"
  }
 ],
 "rules": [
  {
   "when": [
    "plan_left"
   ],
   "do": "continue"
  }
 ],
 "default": "stop:vote"
}
```

### verify_then_decide_high (protocol)
```json
{
 "plan": [
  {
   "personas": [
    "solver",
    "solver",
    "solver",
    "solver"
   ]
  }
 ],
 "rules": [
  {
   "when": [
    "plan_left"
   ],
   "do": "continue"
  },
  {
   "when": [
    "step==1"
   ],
   "do": "verifier|high"
  }
 ],
 "default": "stop:last_commit"
}
```

### fresh_on_disagree_high (protocol)
```json
{
 "plan": [
  {
   "personas": [
    "solver",
    "solver"
   ]
  }
 ],
 "rules": [
  {
   "when": [
    "plan_left"
   ],
   "do": "continue"
  },
  {
   "when": [
    "step==1",
    "n_distinct>=2"
   ],
   "do": "solver|high|blind"
  },
  {
   "when": [
    "step==2"
   ],
   "do": "stop:last_commit"
  }
 ],
 "default": "stop:vote"
}
```

### llm_g0_bound_and_scope_check (llm, group 0)
Two independent high-effort derivations test the lower bound and construction; a verifier checks the universal condition when they agree, while disagreement triggers a fresh derivation.
```json
{
 "plan": [
  {
   "personas": [
    "solver",
    "solver"
   ],
   "effort": "high"
  }
 ],
 "rules": [
  {
   "when": [
    "plan_left"
   ],
   "do": "continue"
  },
  {
   "when": [
    "step==1",
    "r1_majority==2"
   ],
   "do": "verifier|high"
  },
  {
   "when": [
    "step==1",
    "n_distinct>=2"
   ],
   "do": "expert_solver|high|blind"
  },
  {
   "when": [
    "last_round:expert+solver"
   ],
   "do": "synthesizer"
  }
 ],
 "default": "stop:last_commit"
}
```

### llm_g1_invariant_count_audit (llm, group 1)
An expert and solver independently select and calculate the invariant, then a verifier checks multiplicities and reporting conventions; only a conflicting result needs synthesis.
```json
{
 "plan": [
  {
   "personas": [
    "expert",
    "solver"
   ]
  },
  {
   "personas": [
    "verifier"
   ]
  }
 ],
 "rules": [
  {
   "when": [
    "plan_left"
   ],
   "do": "continue"
  },
  {
   "when": [
    "step==2",
    "n_distinct>=2"
   ],
   "do": "synthesizer|high"
  }
 ],
 "default": "stop:last_commit"
}
```

### llm_g2_independent_view_transforms (llm, group 2)
Three independent readings reduce symbol misreads; unanimity gets a direction-and-shading check, while disagreement prompts a fresh high-effort transformation and synthesis.
```json
{
 "plan": [
  {
   "personas": [
    "solver",
    "solver",
    "solver"
   ]
  }
 ],
 "rules": [
  {
   "when": [
    "plan_left"
   ],
   "do": "continue"
  },
  {
   "when": [
    "step==1",
    "r1_majority==3"
   ],
   "do": "verifier"
  },
  {
   "when": [
    "step==1"
   ],
   "do": "expert|high|blind"
  },
  {
   "when": [
    "last_round:expert"
   ],
   "do": "synthesizer"
  }
 ],
 "default": "stop:last_commit"
}
```

### llm_g3_blind_evidence_crosscheck (llm, group 3)
An expert recalls the relevant evidence and a blind solver independently compares complete candidates; conflicting choices receive a deeper expert assessment.
```json
{
 "plan": [
  {
   "personas": [
    "expert"
   ]
  },
  {
   "personas": [
    "solver"
   ],
   "sees": "none"
  }
 ],
 "rules": [
  {
   "when": [
    "plan_left"
   ],
   "do": "continue"
  },
  {
   "when": [
    "step==2",
    "n_distinct>=2"
   ],
   "do": "expert|high|blind"
  },
  {
   "when": [
    "last_round:expert",
    "step==3"
   ],
   "do": "synthesizer"
  },
  {
   "when": [
    "last_round:synthesizer"
   ],
   "do": "stop:last_commit"
  }
 ],
 "default": "stop:vote"
}
```
