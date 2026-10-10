# Task 4 — auto-generated result tables

## 1. Safety-calibration comparison (AI judge)

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

## 2. Manual audit

_Not available yet_: /kaggle/working/repo/results/task4_safety/manual_audit_sheet.csv has no valid filled rows yet
