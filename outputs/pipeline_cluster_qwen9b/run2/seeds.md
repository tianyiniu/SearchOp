# Seed programs (11)

Executor: {'executor': 'v3', 'digest': [2000, 2000], 'summary_words': 500, 'window': 32768, 'high_cost': 3, 'eliminator': False, 'prompts': '032e5eb669', 'turn_cap': 15, 'total_cap': 21, 'last_round_vote': True, 'plain_instruction': True, 'count_read_summaries': True, 'model': 'Qwen/Qwen3.5-9B'}, model Qwen/Qwen3.5-9B

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
| llm_g0_independent_balance_check | llm | solver|high > expert|blind | 2 | critic|high | last_commit | - |
| llm_g1_category_disagreement_expert | llm | expert_solver | 3 | expert|high | last_commit, vote | - |
| llm_g2_blind_attribution_tiebreak | llm | solver_x3 | 4 | expert|blind, verifier | last_commit, vote | - |

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

### llm_g0_independent_balance_check (llm, group 0)
A high-effort solver works through the balance and units, a blind expert checks the setup independently, and a high-effort critic investigates disagreements over signs or scale.
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
    "expert"
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
   "do": "critic|high"
  }
 ],
 "default": "stop:last_commit"
}
```

### llm_g1_category_disagreement_expert (llm, group 1)
An expert and solver independently test the category descriptions; only disagreement triggers a high-effort expert review of definitions, qualifiers, and negations.
```json
{
 "plan": [
  {
   "personas": [
    "expert",
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
   "do": "expert|high"
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

### llm_g2_blind_attribution_tiebreak (llm, group 2)
Three independent solvers handle straightforward recall, while a nonunanimous result prompts a blind expert attribution followed by a verifier's exact-name check.
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
    "r1_majority<3"
   ],
   "do": "expert|blind"
  },
  {
   "when": [
    "step==2"
   ],
   "do": "verifier"
  },
  {
   "when": [
    "step==3"
   ],
   "do": "stop:last_commit"
  }
 ],
 "default": "stop:vote"
}
```
