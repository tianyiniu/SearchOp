# Seed programs (10)

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
| llm_g0_spatial_crosscheck | llm | solver_x2 | 4 | expert|high|blind, synthesizer, verifier|high | last_commit | - |
| llm_g1_enumeration_audit | llm | solver_x3 | 6 | critic|high, expert_solver|high|blind, synthesizer | last_commit, vote | - |

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

### llm_g0_spatial_crosscheck (llm, group 0)
Independent solutions expose different spatial readings; a high-effort verifier checks an agreed calculation, while disagreement prompts a fresh expert solution before synthesis.
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
   "do": "verifier|high"
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

### llm_g1_enumeration_audit (llm, group 1)
Three independent attempts test the candidate search; a high-effort critic audits likely agreements for omitted cases, while scattered answers trigger a fresh enumeration.
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
    "r1_majority>=2"
   ],
   "do": "critic|high"
  },
  {
   "when": [
    "step==1"
   ],
   "do": "expert_solver|high|blind"
  },
  {
   "when": [
    "last_round:critic",
    "kept_answer"
   ],
   "do": "stop:vote"
  },
  {
   "when": [
    "last_round:critic"
   ],
   "do": "synthesizer"
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
