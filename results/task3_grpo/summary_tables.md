# Task 3 — auto-generated result tables

## 1. Standard GRPO continuation (20 updates, K=4)

- wall-clock: **397.5 s**, peak VRAM: **8.69 GiB**
- uninformative-group rate over the run: **0.000**

| update | reward | group std | uninf. | KL | policy loss | grad norm | entropy | length |
|---|---|---|---|---|---|---|---|---|
| 1 | 1.827 | 0.473 | no | 0.0002 | -0.2846 | 0.115 | 1.139 | 467 |
| 2 | 2.735 | 0.273 | no | -0.0000 | 0.0001 | 0.172 | 1.189 | 341 |
| 3 | -0.333 | 1.915 | no | 0.0003 | -0.0005 | 0.630 | 0.760 | 88 |
| 4 | 1.735 | 0.488 | no | 0.0004 | -0.0001 | 0.876 | 1.620 | 43 |
| 5 | 0.897 | 1.450 | no | -0.0000 | -0.3301 | 0.074 | 0.651 | 509 |
| 6 | 2.633 | 0.236 | no | 0.0001 | 0.0002 | 0.151 | 1.286 | 274 |
| 7 | -1.953 | 0.270 | no | 0.0005 | 0.0002 | 0.622 | 0.867 | 66 |
| 8 | 0.386 | 0.974 | no | 0.0000 | -0.3897 | 0.068 | 0.426 | 450 |
| 9 | 1.489 | 0.636 | no | 0.0001 | -0.0003 | 0.312 | 1.353 | 105 |
| 10 | 2.034 | 0.160 | no | 0.0003 | 0.0002 | 0.214 | 0.332 | 109 |
| 11 | 2.883 | 0.245 | no | 0.0001 | 0.0001 | 0.102 | 0.221 | 242 |
| 12 | 2.550 | 0.171 | no | 0.0002 | -0.0001 | 0.132 | 1.196 | 398 |
| 13 | 1.367 | 0.861 | no | 0.0000 | 0.0000 | 0.000 | undefined | 512 |
| 14 | 1.856 | 0.438 | no | 0.0002 | -0.4184 | 0.160 | 1.566 | 439 |
| 15 | 0.416 | 1.224 | no | 0.0000 | -0.3467 | 0.084 | 0.747 | 511 |
| 16 | -0.376 | 0.140 | no | -0.0001 | -0.1459 | 0.181 | 2.263 | 317 |
| 17 | 1.572 | 0.648 | no | 0.0001 | 0.0000 | 0.000 | undefined | 512 |
| 18 | 2.048 | 0.152 | no | 0.0003 | 0.0002 | 0.375 | 1.214 | 120 |
| 19 | -4.497 | 1.627 | no | 0.0001 | -0.1485 | 0.100 | 2.329 | 454 |
| 20 | -0.258 | 0.765 | no | -0.0006 | 0.0012 | 0.964 | 0.985 | 20 |

_Column notes: 'uninf.' = reward-variance informativeness (group std <= 1e-6), **not** usable training signal. Updates 13 and 17 have all four completions truncated, so the released truncation mask empties the training batch: zero policy gradient (grad norm 0.000, policy loss 0.000) and **entropy undefined** (NaN = empty mask). The entropy diagnostic covers valid (unmasked) response tokens only and has no support when the mask is empty. Hence 0% uninformative reward groups does not mean every update provides training signal: 2/20 updates provided none._

## 2. Held-out evaluation (standard adapter)

- reward mean: 1.373 (± 1.314)
- KL (sampled): 0.00020
- length: 256.7 (± 192.1)

## 3. Group-size study (equal total generations)

| K | groups | informative rate | mean within-group std | signal variance |
|---|---|---|---|---|
| 2 | 96 | 0.938 | 0.385 | 0.938 |
| 4 | 48 | 0.958 | 0.541 | 0.958 |
| 8 | 24 | 0.958 | 0.619 | 0.958 |

### Per difficulty bin

| difficulty | K=2 | K=4 | K=8 |
|---|---|---|---|
| easy | 1.000 (0.248; var 1.000) | 1.000 (0.410; var 1.000) | 1.000 (0.495; var 1.000) |
| medium | 0.969 (0.412; var 0.969) | 1.000 (0.552; var 1.000) | 1.000 (0.599; var 1.000) |
| hard | 0.844 (0.493; var 0.844) | 0.875 (0.661; var 0.875) | 0.875 (0.763; var 0.875) |

## 4. Canonical vs Dr. GRPO (matched short forks)

| condition | held-out reward | held-out KL | held-out length | corr(length, grad norm) |
|---|---|---|---|---|
| grpo | 1.443 | 0.00016 | 264.2 | -0.936 |
| dr_grpo | 1.443 | 0.00014 | 267.1 | -0.412 |

_Realized generated tokens (per audit): canonical **8,953** vs Dr. GRPO **10,109** (+12.9%). Update counts and caps are equal by the released short-run spec; realized tokens differ because Dr. GRPO's responses are longer - itself part of the effect under study (forcing realized-token equality would mean truncating completions and departing from the released generation settings). Disclosed, not assumed equal._
