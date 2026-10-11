# Task 1 — auto-generated result tables

## 1. Standard + short-run β conditions (budgets differ: standard = 1 epoch, forks = 600 examples)

_Compliance exception (disclosed): subsets are the supplied fixed files **minus prompts whose chat-templated prompt exceeds the released 768-token budget** (standard: 54 dropped → 1,446 used; length-balanced: 58 dropped → 1,442; retained strata 478/482/482; eval drops 10/300 pairs). The released starter encoder mandates filtering; no feasible max_length covers prompts up to 3,563 tokens. All conditions are filtered by the identical rule._

| condition | β | budget | held-out DPO loss | pref. acc | KL (sampled) | RM score | len mean ± std |
|---|---|---|---|---|---|---|---|
| standard | 0.10 | 1 epoch (1446 ex) | 0.673 | 0.638 | -0.00004 | 0.952 | 174.6 ± 100.7 |
| beta_0p03 | 0.03 | 600 ex (short) | 0.691 | 0.634 | -0.00007 | 0.932 | 174.8 ± 102.3 |
| beta_0p1 | 0.10 | 600 ex (short) | 0.686 | 0.641 | -0.00009 | 0.893 | 173.1 ± 98.4 |
| beta_0p3 | 0.30 | 600 ex (short) | 0.677 | 0.628 | -0.00011 | 0.928 | 173.4 ± 102.0 |

## 2. Length-confounding study (standard vs length-balanced, per stratum)

_Training strata after overlong-prompt filtering: preferred_longer 478 / length_matched 482 / rejected_longer 482 (retained ~balanced). Eval rows evaluated per stratum: 78 / 80 / 79._

| stratum | n (std) | acc standard | acc length-balanced |
|---|---|---|---|
| length_matched | 80 | 0.575 | 0.613 |
| preferred_longer | 78 | 0.462 | 0.449 |
| rejected_longer | 79 | 0.658 | 0.772 |
| _overall | 237 | 0.565 | 0.612 |

## 3. Word-limit compliance on the common prompt set

| condition | compliance rate | response tokens (mean ± std) |
|---|---|---|
| standard | 1.000 | 45.5 ± 22.8 |
| length_balanced | 1.000 | 47.0 ± 18.9 |

## 4. Training runs (from train_*.json)

| run | β | examples | steps | final loss | final pref acc | wall-clock s |
|---|---|---|---|---|---|---|
| beta_0p03 | 0.03 | 600 | 38 | 0.690 | 0.625 | 527.0 |
| beta_0p1 | 0.10 | 600 | 38 | 0.683 | 0.625 | 528.8 |
| beta_0p3 | 0.30 | 600 | 38 | 0.673 | 0.625 | 528.9 |
| length_balanced | 0.10 | 1442 | 91 | 0.596 | 1.000 | 982.3 |
| standard | 0.10 | 1446 | 91 | 0.650 | 0.667 | 1282.6 |
