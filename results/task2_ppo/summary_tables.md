# Task 2 — auto-generated result tables

## 1. Standard PPO continuation (20 updates)

- wall-clock: **333.9 s**, peak VRAM: **8.18 GiB**
- mean reward over last 5 updates: **-0.442**

| update | reward | KL | policy loss | value loss | entropy | clip frac | grad norm | length |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.407 | 0.0001 | -0.0130 | 5.0981 | 1.012 | 0.000 | 0.590 | 512 |
| 2 | 2.979 | -0.0006 | -0.3646 | 6.5066 | 0.964 | 0.000 | 0.722 | 319 |
| 3 | -1.771 | -0.0007 | 0.3389 | 2.6712 | 0.566 | 0.000 | 3.196 | 13 |
| 4 | 2.053 | 0.0018 | -0.5840 | 5.5735 | 1.721 | 0.000 | 7.283 | 32 |
| 5 | -0.285 | 0.0002 | 0.0083 | 5.5217 | 0.633 | 0.000 | 0.488 | 512 |
| 6 | 2.936 | -0.0001 | -0.2428 | 3.0331 | 1.356 | 0.000 | 0.463 | 321 |
| 7 | -1.676 | 0.0000 | 1.1548 | 6.8045 | 0.876 | 0.000 | 1.473 | 58 |
| 8 | 1.059 | 0.0002 | -0.1886 | 4.3923 | 0.453 | 0.000 | 0.466 | 281 |
| 9 | 1.010 | -0.0001 | -0.3311 | 2.5660 | 1.385 | 0.000 | 1.033 | 82 |
| 10 | 1.390 | 0.0005 | -0.4265 | 3.6676 | 0.490 | 0.000 | 0.748 | 135 |
| 11 | 2.498 | 0.0003 | -0.3136 | 3.5517 | 0.196 | 0.000 | 0.432 | 214 |
| 12 | 2.732 | -0.0002 | -0.2233 | 3.8727 | 1.073 | 0.000 | 0.651 | 314 |
| 13 | 0.979 | 0.0004 | -0.0505 | 3.6612 | 0.987 | 0.000 | 0.400 | 512 |
| 14 | 1.248 | 0.0001 | -0.1025 | 4.4260 | 1.375 | 0.000 | 0.453 | 512 |
| 15 | -1.750 | -0.0002 | 0.1290 | 3.7857 | 0.804 | 0.000 | 0.379 | 512 |
| 16 | -0.277 | 0.0005 | -0.0128 | 2.4415 | 2.215 | 0.000 | 0.489 | 373 |
| 17 | -0.085 | -0.0000 | -0.0180 | 4.0157 | 0.657 | 0.000 | 0.355 | 512 |
| 18 | 1.973 | -0.0002 | 0.3045 | 1.8554 | 0.958 | 0.000 | 1.997 | 38 |
| 19 | -4.418 | 0.0002 | 0.1621 | 2.8821 | 2.382 | 0.000 | 0.403 | 365 |
| 20 | 0.597 | 0.0012 | -0.5698 | 1.6562 | 0.728 | 0.000 | 1.307 | 15 |

## 2. Held-out evaluation (standard adapter)

- reward mean: 1.569 (± 1.377)
- KL (sampled): 0.00003
- length: 305.1 (± 255.5)
- prompts: 100, truncated: 0.090

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
| 0.05 | 1.0010 | 0.0353 | 0.688 | 1.555 | 0.110 | 0.110 | 2.8092 |
| 0.2 | 1.0010 | 0.0353 | 0.688 | 1.555 | 0.003 | 0.003 | 2.8160 |
| 0.5 | 1.0010 | 0.0353 | 0.688 | 1.555 | 0.000 | 0.000 | 2.8162 |

## 4. Matched short forks (8 updates)

| ε | held-out reward | held-out KL | held-out length | clip-frac std | grad-norm std | final ratio dev |
|---|---|---|---|---|---|---|
| 0.05 | 1.558 | 0.00001 | 312.7 | 0.000 | 2.237 | 0.000 |
| 0.20 | 1.558 | 0.00001 | 312.7 | 0.000 | 2.237 | 0.000 |
| 0.50 | 1.558 | 0.00001 | 312.7 | 0.000 | 2.237 | 0.000 |

## 5. KL-pressure study (8-update forks)

_`final *` columns are training-side (mean of last three updates); `held-out *` columns are from the fixed 100-prompt evaluation protocol (manual requirement: held-out reward, KL, **entropy**, length). Entropy values reported are the **full token-level policy entropy** H_t = −Σ_v π log π over the whole vocabulary (the course-helper sampled-token negative log-probability proxy is retained alongside as `entropy_sampled_response_mean`)._

| β_KL | final reward | final KL | final entropy (train) | final length | held-out reward | held-out KL | **held-out entropy** | held-out length |
|---|---|---|---|---|---|---|---|---|
| 0.00 | 0.588 | 0.00020 | 0.693 | 373.3 | 1.553 | 0.00001 | 1.039 | 304.7 |
| 0.10 | 0.773 | 0.00004 | 0.732 | 220.0 | 1.558 | 0.00001 | 1.030 | 312.7 |
| 0.20 | 1.163 | 0.00003 | 0.807 | 234.7 | 1.518 | 0.00003 | 1.019 | 312.1 |
