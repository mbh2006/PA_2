# Task 4 — auto-generated result tables

## 1. Safety-calibration comparison (AI judge)

| policy | n | safe answer | safe over-refusal | unsafe compliance | justified refusal | ambiguous | mean tokens | parse failures |
|---|---|---|---|---|---|---|---|---|
| sft | 450 | 0.580 | 0.000 | 0.000 | 0.885 | 0.000 | 108.1 | 0 |
| dpo | 450 | 0.560 | 0.000 | 0.000 | 0.875 | 0.000 | 108.8 | 0 |
| ppo | 450 | 0.560 | 0.000 | 0.000 | 0.885 | 0.000 | 107.7 | 0 |
| grpo | 450 | 0.572 | 0.000 | 0.000 | 0.890 | 0.000 | 108.3 | 0 |

### Full judge-label distribution per policy

| policy | SAFE_ANSWER | JUSTIFIED_REFUSAL | UNSAFE_COMPLIANCE | OVER_REFUSAL | AMBIGUOUS |
|---|---|---|---|---|---|
| sft | 0.373 | 0.620 | 0.007 | 0.000 | 0.000 |
| dpo | 0.367 | 0.629 | 0.004 | 0.000 | 0.000 |
| ppo | 0.362 | 0.629 | 0.009 | 0.000 | 0.000 |
| grpo | 0.367 | 0.627 | 0.007 | 0.000 | 0.000 |

## 2. Manual audit

_Not available yet_: /kaggle/working/repo/results/task4_safety/manual_audit_sheet.csv has no valid filled rows yet
