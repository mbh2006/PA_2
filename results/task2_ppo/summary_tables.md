# Task 2 — auto-generated result tables

## 1. Standard PPO continuation (20 updates)

- wall-clock: **317.5 s**, peak VRAM: **8.72 GiB**
- mean reward over last 5 updates: **-0.629**

| update | reward | KL | policy loss | value loss | entropy | clip frac | grad norm | length |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.407 | 0.0001 | -0.0141 | 5.1037 | 0.809 | 0.000 | 0.635 | 512 |
| 2 | 2.979 | -0.0002 | -0.3596 | 6.4854 | 0.771 | 0.000 | 0.659 | 319 |
| 3 | -1.771 | -0.0013 | 0.3146 | 2.7234 | 0.321 | 0.000 | 4.442 | 13 |
| 4 | 2.053 | -0.0008 | -0.5851 | 5.8469 | 0.919 | 0.000 | 5.571 | 32 |
| 5 | -0.285 | 0.0001 | 0.0058 | 5.5619 | 0.500 | 0.000 | 0.511 | 512 |
| 6 | 2.699 | -0.0001 | -0.1808 | 2.9550 | 1.047 | 0.000 | 0.458 | 414 |
| 7 | 0.555 | 0.0009 | 0.0490 | 5.2496 | 0.665 | 0.000 | 1.236 | 194 |
| 8 | -1.718 | -0.0001 | 0.0123 | 4.2765 | 0.333 | 0.000 | 0.548 | 512 |
| 9 | 2.316 | 0.0006 | -0.2484 | 3.9286 | 1.383 | 0.000 | 0.880 | 171 |
| 10 | 1.623 | -0.0002 | -0.4774 | 2.9163 | 0.645 | 0.000 | 0.941 | 127 |
| 11 | 2.684 | 0.0000 | -0.3238 | 3.1630 | 0.241 | 0.000 | 0.340 | 217 |
| 12 | -3.328 | 0.0002 | 1.6392 | 8.1781 | 0.718 | 0.000 | 3.997 | 31 |
| 13 | 0.473 | -0.0003 | -0.0361 | 3.4314 | 0.956 | 0.000 | 0.462 | 512 |
| 14 | 1.361 | 0.0001 | -0.0837 | 3.6373 | 1.080 | 0.000 | 0.465 | 512 |
| 15 | -0.083 | 0.0002 | 0.0330 | 4.1702 | 0.529 | 0.000 | 0.436 | 512 |
| 16 | 0.168 | 0.0007 | -0.1676 | 2.3177 | 1.744 | 0.000 | 0.513 | 245 |
| 17 | 0.839 | 0.0001 | -0.0717 | 3.9694 | 0.560 | 0.000 | 0.526 | 512 |
| 18 | 2.316 | -0.0004 | -0.1194 | 2.1641 | 1.012 | 0.000 | 1.248 | 84 |
| 19 | -7.066 | -0.0001 | 0.1968 | 3.1665 | 0.981 | 0.000 | 0.411 | 512 |
| 20 | 0.597 | 0.0012 | -0.6408 | 1.7184 | 0.394 | 0.000 | 2.011 | 15 |

## 2. Held-out evaluation (standard adapter)

- reward mean: 1.470 (± 1.482)
- KL (sampled): -0.00000
- length: 300.2 (± 261.3)
- prompts: 100, truncated: 0.130

## 3. Cached-batch clipping geometry (two supplied batches)

_official and salvage caches contain DIFFERENT rollouts (verified 0/32 match). official: reconstructed ids (+EOS), no returns -> clip/affected only; salvage: exact ids + supplied returns, values from the frozen critic -> surrogate available._

### Batch: official (32 rollouts; source indices in eval pool: 32)

**Adapter: midpoint** — mean |Δlogp| vs stored old: 0.049209

| ε | mean ratio | std ratio | min ratio | max ratio | clip fraction | affected fraction | mean clipped surrogate |
|---|---|---|---|---|---|---|---|
| 0.05 | 1.0120 | 0.5337 | 2.24e-09 | 32.579 | 0.154 | 0.154 | — |
| 0.2 | 1.0120 | 0.5337 | 2.24e-09 | 32.579 | 0.041 | 0.041 | — |
| 0.5 | 1.0120 | 0.5337 | 2.24e-09 | 32.579 | 0.021 | 0.021 | — |

### Batch: salvage (32 rollouts; source indices in eval pool: 32)

**Adapter: midpoint** — mean |Δlogp| vs stored old: 0.019977

| ε | mean ratio | std ratio | min ratio | max ratio | clip fraction | affected fraction | mean clipped surrogate |
|---|---|---|---|---|---|---|---|
| 0.05 | 1.0010 | 0.0353 | 0.688 | 1.555 | 0.110 | 0.110 | 2.5997 |
| 0.2 | 1.0010 | 0.0353 | 0.688 | 1.555 | 0.003 | 0.003 | 2.6076 |
| 0.5 | 1.0010 | 0.0353 | 0.688 | 1.555 | 0.000 | 0.000 | 2.6079 |

## 4. Matched short forks (8 updates)

| ε | held-out reward | held-out KL | held-out length | clip-frac std | grad-norm std | final ratio dev |
|---|---|---|---|---|---|---|
| 0.05 | 1.572 | 0.00000 | 303.8 | 0.000 | 1.910 | 0.000 |
| 0.20 | 1.572 | 0.00000 | 303.8 | 0.000 | 1.910 | 0.000 |
| 0.50 | 1.572 | 0.00000 | 303.8 | 0.000 | 1.910 | 0.000 |

## 5. KL-pressure study (8-update forks)

| β_KL | final reward | final KL | final entropy | final length | held-out reward | held-out KL | held-out length |
|---|---|---|---|---|---|---|---|
| 0.00 | 0.790 | -0.00029 | 0.638 | 278.0 | 1.535 | -0.00001 | 316.4 |
| 0.10 | 0.512 | 0.00024 | 0.681 | 373.3 | 1.572 | 0.00000 | 303.8 |
| 0.20 | 0.588 | 0.00002 | 0.693 | 373.3 | 1.702 | -0.00001 | 298.8 |
