# Task 4 — auto-generated result tables

## 1. Safety-calibration comparison (AI judge)

_**Judge-limited instrument** — class-inconsistent labels (105/120/107/108 of 450 for SFT/DPO/PPO/GRPO) and zero OVER_REFUSAL usage are documented in `TRACKING/04_TASK4_SAFETY.md` §6.2/§7; the student manual audit (63.3% agreement, 7 over-refusals the judge missed) qualifies this table. Do not read zero over-refusal/compliance rates as "perfect safety"._

| policy | n | safe answer | safe over-refusal | unsafe compliance | justified refusal | ambiguous | mean tokens | parse failures |
|---|---|---|---|---|---|---|---|---|
| sft | 450 | 0.712 | 0.000 | 0.000 | 0.805 | 0.013 | 108.1 | 0 |
| dpo | 450 | 0.676 | 0.000 | 0.000 | 0.800 | 0.002 | 108.8 | 0 |
| ppo | 450 | 0.704 | 0.000 | 0.000 | 0.810 | 0.011 | 107.7 | 0 |
| grpo | 450 | 0.700 | 0.000 | 0.000 | 0.810 | 0.011 | 108.3 | 0 |

### Full judge-label distribution per policy

| policy | SAFE_ANSWER | JUSTIFIED_REFUSAL | UNSAFE_COMPLIANCE | OVER_REFUSAL | AMBIGUOUS |
|---|---|---|---|---|---|
| sft | 0.480 | 0.502 | 0.004 | 0.000 | 0.013 |
| dpo | 0.464 | 0.529 | 0.004 | 0.000 | 0.002 |
| ppo | 0.473 | 0.511 | 0.004 | 0.000 | 0.011 |
| grpo | 0.471 | 0.513 | 0.004 | 0.000 | 0.011 |

## 2. Manual audit vs AI judge

- labelled pairs: **60** over **60** prompts
- overall agreement: **0.633**; excluding AI-ambiguous: **0.633**
- policy coverage: {'sft': 15, 'dpo': 15, 'ppo': 15, 'grpo': 15}

### Confusion (rows = AI judge, cols = manual)

| AI \ manual | SAFE_ANSWER | JUSTIFIED_REFUSAL | UNSAFE_COMPLIANCE | OVER_REFUSAL | AMBIGUOUS |
|---|---|---|---|---|---|
| SAFE_ANSWER | 10 | 3 | 0 | 1 | 8 |
| JUSTIFIED_REFUSAL | 3 | 28 | 0 | 6 | 1 |
| UNSAFE_COMPLIANCE | 0 | 0 | 0 | 0 | 0 |
| OVER_REFUSAL | 0 | 0 | 0 | 0 | 0 |
| AMBIGUOUS | 0 | 0 | 0 | 0 | 0 |
