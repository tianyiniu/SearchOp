# Seed programs (12)

Executor: {'executor': 'v3', 'digest': [2000, 2000], 'summary_words': 500, 'window': 32768, 'high_cost': 3, 'eliminator': False, 'prompts': '6cbb9b0dc4', 'turn_cap': 15, 'total_cap': 21, 'last_round_vote': True, 'plain_instruction': True, 'count_read_summaries': True, 'answers': 'math', 'model': 'Qwen/Qwen3.5-9B'}, model Qwen/Qwen3.5-9B

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
| llm_g0_attainment_check | llm | solver_x2 | 4 | expert|high|blind, verifier | last_commit | - |
| llm_g1_classification_audit | llm | solver_x3 | 3 | critic, expert_solver|blind | last_commit | - |
| llm_g2_independent_rule_check | llm | expert_solver | 2 | expert|high|blind | last_commit | - |
| llm_g3_spatial_crosscheck | llm | expert|high > solver|blind | 2 | verifier|high | last_commit | - |

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

### llm_g0_attainment_check (llm, group 0)
Two independent solvers establish the extremum; a verifier checks attainability when they agree, while a blind high-effort expert resolves disagreements before verification.
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
    "r1_majority==2"
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
   "do": "verifier"
  }
 ],
 "default": "stop:last_commit"
}
```

### llm_g1_classification_audit (llm, group 1)
Three solvers provide independent counts; a critic checks a shared classification, while disagreement triggers a fresh expert-and-solver count without anchoring.
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
   "do": "critic"
  },
  {
   "when": [
    "step==1"
   ],
   "do": "expert_solver|blind"
  }
 ],
 "default": "stop:last_commit"
}
```

### llm_g2_independent_rule_check (llm, group 2)
An expert and solver independently apply the governing rule; only disagreement warrants a blind high-effort expert derivation to settle the requested combination.
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
    "r1_majority==1"
   ],
   "do": "expert|high|blind"
  }
 ],
 "default": "stop:last_commit"
}
```

### llm_g3_spatial_crosscheck (llm, group 3)
A high-effort expert models the geometry, a blind solver independently interprets the arrangement, and a verifier adjudicates conflicting measurements.
```json
{
 "plan": [
  {
   "personas": [
    "expert"
   ],
   "effort": "high"
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
   "do": "verifier|high"
  }
 ],
 "default": "stop:last_commit"
}
```
