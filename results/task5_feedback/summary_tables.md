# Task 5 — auto-generated result tables

## 1. In-domain (GSM8K) and out-of-domain (SVAMP)

| policy | GSM acc | GSM format | GSM length | pairwise vs SFT (GSM) | SVAMP acc | drop |
|---|---|---|---|---|---|---|
| sft | 0.273 | 0.350 | 274.2 | — | 0.450 | -0.177 |
| rlvr | 0.297 | 0.373 | 273.7 | 0.513 | 0.450 | -0.153 |
| rlaif | 0.297 | 0.377 | 274.5 | 0.537 | 0.450 | -0.153 |

### Verifier–judge agreement (GSM8K)

| challenger | agreement | pairs with verifier disagreement | wins/ties/losses |
|---|---|---|---|
| rlvr | 0.455 | 11 | 117/74/109 |
| rlaif | 0.600 | 5 | 125/72/103 |

## 2. Controlled diagnostic study

- S_reason: verifier **0.000**, judge **0.100**
- S_outcome: verifier **1.000**, judge **0.150**

| mechanism | pair type | n | better | tie | wrong |
|---|---|---|---|---|---|
| verifier | reasoning_only | 20 | 0.000 | 1.000 | 0.000 |
| verifier | outcome_only | 20 | 1.000 | 0.000 | 0.000 |
| verifier | persuasive_filler | 20 | 0.000 | 1.000 | 0.000 |
| verifier | gold_distractor | 20 | 1.000 | 0.000 | 0.000 |
| judge | reasoning_only | 20 | 0.100 | 0.600 | 0.300 |
| judge | outcome_only | 20 | 0.150 | 0.850 | 0.000 |
| judge | persuasive_filler | 20 | 0.100 | 0.350 | 0.550 |
| judge | gold_distractor | 20 | 0.000 | 1.000 | 0.000 |
