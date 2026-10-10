# Cross-task summary (Task 6 scaffold)

| task | condition | reward signal | held-out reward (RM) | KL | length | extra | compute |
|---|---|---|---|---|---|---|---|
| T1 | T1 SFT baseline | fixed preference pairs (DPO) | 0.875 | 0.00000 | 176.4 | pref-acc 0.000 |  |
| T1 | T1 DPO standard | fixed preference pairs (DPO) | 0.952 | -0.00004 | 174.6 | pref-acc 0.638 | 1283 s, 91 steps |
| T1 | T1 DPO β=0.03 | fixed preference pairs (DPO) | 0.932 | -0.00007 | 174.8 | pref-acc 0.634 | 527 s, 38 steps |
| T1 | T1 DPO β=0.10 | fixed preference pairs (DPO) | 0.893 | -0.00009 | 173.1 | pref-acc 0.641 | 529 s, 38 steps |
| T1 | T1 DPO β=0.30 | fixed preference pairs (DPO) | 0.928 | -0.00011 | 173.4 | pref-acc 0.628 | 529 s, 38 steps |
| T2 | T2 PPO midpoint | learned reward + value critic (PPO) | 1.566 | -0.00004 | 308.9 |  |  |
| T2 | T2 PPO standard (20 upd) | learned reward + value critic (PPO) | 1.470 | -0.00000 | 300.2 |  | 318 s, 20 upd, peak 8.72 GiB |
| T2 | T2 PPO ε=0.20 fork (8 upd) | learned reward + value critic (PPO) | 1.572 | 0.00000 | 303.8 | ε sweep: identical to other ε |  |
| T2 | T2 PPO βKL=0 fork | learned reward + value critic (PPO) | 1.535 | -0.00001 | 316.4 |  |  |
| T2 | T2 PPO βKL=0.10 fork | learned reward + value critic (PPO) | 1.572 | 0.00000 | 303.8 |  |  |
| T2 | T2 PPO βKL=0.20 fork | learned reward + value critic (PPO) | 1.702 | -0.00001 | 298.8 |  |  |
| T3 | T3 GRPO midpoint | group-relative reward, no critic (GRPO) | 1.203 | 0.00000 | 265.4 |  |  |
| T3 | T3 GRPO standard (20 upd, K=4) | group-relative reward, no critic (GRPO) | 1.419 | 0.00019 | 259.9 |  | 399 s, 20 upd, peak 8.59 GiB |
| T3 | T3 GRPO canonical fork (8 upd) | group-relative reward, no critic (GRPO) | 1.424 | 0.00010 | 252.8 |  |  |
| T3 | T3 GRPO Dr.GRPO fork (8 upd) | group-relative reward, no critic (GRPO) | 1.444 | 0.00018 | 273.8 | corr(len,grad) −0.41 vs −0.94 |  |
| T4 | T4 SFT (frozen) | categorical AI judge on XSTest | — | — | 108.1 | safe-answer 0.712, over-refusal 0.000, unsafe-compliance 0.000, justified-refusal 0.805, ambiguous 0.013 |  |
| T4 | T4 DPO (frozen) | categorical AI judge on XSTest | — | — | 108.8 | safe-answer 0.676, over-refusal 0.000, unsafe-compliance 0.000, justified-refusal 0.800, ambiguous 0.002 |  |
| T4 | T4 PPO (frozen) | categorical AI judge on XSTest | — | — | 107.7 | safe-answer 0.704, over-refusal 0.000, unsafe-compliance 0.000, justified-refusal 0.810, ambiguous 0.011 |  |
| T4 | T4 GRPO (frozen) | categorical AI judge on XSTest | — | — | 108.3 | safe-answer 0.700, over-refusal 0.000, unsafe-compliance 0.000, justified-refusal 0.810, ambiguous 0.011 |  |
| T5 | T5 SFT (GSM8K) | — | — | — | 274.2 | exact-acc 0.273, format 0.350 |  |
| T5 | T5 RLVR (GSM8K) | exact verifier | — | — | 273.7 | exact-acc 0.297, format 0.373 |  |
| T5 | T5 RLAIF (GSM8K) | AI pairwise | — | — | 274.5 | exact-acc 0.297, format 0.377 |  |
| T5 | T5 diagnostics (100 responses) | verifier vs pairwise AI judge | — | — | — | S_reason: verifier 0.000 / judge 0.100; S_outcome: verifier 1.000 / judge 0.150 |  |

Notes:
- Reward-scale caveat: T1/T2/T3 all use the same course reward model on 100 generated prompts — comparable within tasks; cross-task comparisons of RM means are indicative only (different prompt pools: DPO eval pairs vs RL prompt pool).
- T2 ε evidence is the cached-batch geometry (clip fractions 11.0%/0.29%/0.01% at ε=0.05/0.2/0.5, salvage batch); on-policy ε forks are identical (clip never activates).
- Paired statistics (vs task baselines, n=100) live in the per-task tracking docs (§7) and generation files.
- T5 diagnostic columns are mechanism-level rates, not RM scores.