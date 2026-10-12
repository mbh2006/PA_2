# Task 5 — auto-generated result tables

## 1. In-domain (GSM8K) and out-of-domain (SVAMP)

| policy | GSM acc | GSM format | GSM length (mean ± std) | pairwise vs SFT (GSM) | SVAMP acc | SVAMP pairwise | SVAMP length (mean ± std) | drop |
|---|---|---|---|---|---|---|---|---|
| sft | 0.273 | 0.350 | 274.2 ± 86.2 | — | 0.450 | — | 164.6 ± 72.0 | -0.177 |
| rlvr | 0.297 | 0.373 | 273.7 ± 84.8 | 0.488 | 0.450 | 0.490 | 164.0 ± 72.1 | -0.153 |
| rlaif | 0.297 | 0.377 | 274.5 ± 85.8 | 0.458 | 0.450 | 0.510 | 164.5 ± 72.3 | -0.153 |

_Judge-output validity (per audit): across the S5r judge cache (554 unique calls) **365 (66%) are multi-choice echoes** ('A, B' style) scored by first mention - not reliable preference expressions; the remaining 189 (34%) are single-answer, of which **164 are genuine TIE and only 25 decisive (20x A, 5x B)**. Note: the eval JSON counters (`judge_multi_token_outputs` = 248 GSM / 97 transfer) count FRESH UNIQUE judge calls, not per-comparison counts; across the actual policy-vs-SFT comparisons the multi-choice-scored fractions are **210/300 (RLVR) and 214/300 (RLAIF) on GSM, and 86/100 for each transfer policy**. Released scores retained for compliance; win-rate interpretation must not rest on them (see `TRACKING/05_TASK5_RLVR_RLAIF.md` SS6.1b/SS7.7-7.8)._

### Verifier–judge agreement (GSM8K)

| challenger | agreement | pairs with verifier disagreement | wins/ties/losses |
|---|---|---|---|
| rlvr | 0.500 | 12 | 108/77/115 |
| rlaif | 0.400 | 5 | 100/75/125 |

_Agreement denominator (per audit): computed ONLY over pairs where the **verifier rewards differ AND the judge returns a decisive A/B (not a tie)** - i.e., the restricted decisive subset of verifier disagreements (GSM totals: RLVR 13 disagreements -> 12 decisive; RLAIF 7 -> 5; transfer: RLVR 4 -> 3, RLAIF 4 -> 4). The JSON field `verifier_judge_agreement` uses this restricted denominator and is NOT the fraction of all verifier disagreements._

## 2. Controlled diagnostic study

- S_reason: verifier **0.000**, judge **0.150**
- S_outcome: verifier **1.000**, judge **0.050**

| mechanism | pair type | n | better | tie | wrong |
|---|---|---|---|---|---|
| verifier | reasoning_only | 20 | 0.000 | 1.000 | 0.000 |
| verifier | outcome_only | 20 | 1.000 | 0.000 | 0.000 |
| verifier | persuasive_filler | 20 | 0.000 | 1.000 | 0.000 |
| verifier | gold_distractor | 20 | 1.000 | 0.000 | 0.000 |
| judge | reasoning_only | 20 | 0.150 | 0.600 | 0.250 |
| judge | outcome_only | 20 | 0.050 | 0.950 | 0.000 |
| judge | persuasive_filler | 20 | 0.050 | 0.400 | 0.550 |
| judge | gold_distractor | 20 | 0.050 | 0.950 | 0.000 |
