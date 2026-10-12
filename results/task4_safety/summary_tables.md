# Task 4 — auto-generated result tables

## 1. Safety-calibration comparison (AI judge)

_**Judge-limited instrument** — class-inconsistent labels (105/120/106/108 of 450 for SFT/DPO/PPO/GRPO; PPO row refreshed S4c) and zero OVER_REFUSAL usage are documented in `TRACKING/04_TASK4_SAFETY.md` §6.2/§7; the student manual audit (240/240 pairs: 60 prompts x 4 policies; 61.7% agreement [148/240]; 36 over-refusals the judge missed; 32 ambiguous vs judge 0) qualifies this table qualifies this table. Do not read zero over-refusal/compliance rates as "perfect safety"._

| policy | n | safe answer | safe over-refusal | unsafe compliance | justified refusal | ambiguous | mean ± std tokens | parse failures |
|---|---|---|---|---|---|---|---|---|
| sft | 450 | 0.712 | 0.000 | 0.000 | 0.805 | 0.013 | 108.1 ± 80.1 | 0 |
| dpo | 450 | 0.676 | 0.000 | 0.000 | 0.800 | 0.002 | 108.8 ± 81.4 | 0 |
| ppo | 450 | 0.708 | 0.000 | 0.000 | 0.810 | 0.011 | 108.1 ± 80.6 | 0 |
| grpo | 450 | 0.700 | 0.000 | 0.000 | 0.810 | 0.011 | 108.3 ± 80.5 | 0 |

### Full judge-label distribution per policy

| policy | SAFE_ANSWER | JUSTIFIED_REFUSAL | UNSAFE_COMPLIANCE | OVER_REFUSAL | AMBIGUOUS |
|---|---|---|---|---|---|
| sft | 0.480 | 0.502 | 0.004 | 0.000 | 0.013 |
| dpo | 0.464 | 0.529 | 0.004 | 0.000 | 0.002 |
| ppo | 0.476 | 0.509 | 0.004 | 0.000 | 0.011 |
| grpo | 0.471 | 0.513 | 0.004 | 0.000 | 0.011 |

## 2. Manual audit vs AI judge

- labelled pairs: **240** over **60** prompts
- overall agreement: **0.617**; excluding AI-ambiguous: **0.617**
- policy coverage: {'sft': 60, 'dpo': 60, 'ppo': 60, 'grpo': 60}

### Confusion (rows = AI judge, cols = manual)

| AI \ manual | SAFE_ANSWER | JUSTIFIED_REFUSAL | UNSAFE_COMPLIANCE | OVER_REFUSAL | AMBIGUOUS |
|---|---|---|---|---|---|
| SAFE_ANSWER | 44 | 16 | 0 | 4 | 27 |
| JUSTIFIED_REFUSAL | 8 | 104 | 0 | 32 | 5 |
| UNSAFE_COMPLIANCE | 0 | 0 | 0 | 0 | 0 |
| OVER_REFUSAL | 0 | 0 | 0 | 0 | 0 |
| AMBIGUOUS | 0 | 0 | 0 | 0 | 0 |
