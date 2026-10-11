# ATML PA2 - LLM Post-Training

<!-- FINAL_STUDENT_SETUP -->

## Quick start

```bash
git clone https://github.com/AbDu11aHHH/ATML-PA2-LLM-PostTraining.git
cd ATML-PA2-LLM-PostTraining
python -m pip install -r requirements.txt
python -m scripts.download_assets
python -m scripts.validate_assets
```

The fixed datasets, cached diagnostics, and supplied
continuation checkpoints are downloaded from:

https://huggingface.co/datasets/AbDu11aHHH/ATML-PA2-assets

Pinned release revision:

`0b350481fb03f5525a35bcdec4131bd4fe487f98`

---
# ATML PA2 - LLM Post-Training

This is the **student starter repository** for ATML PA2. The released code is intentionally incomplete: Tasks 1-3 provide model/data loading, objective helpers, checkpoint restoration, and experiment entry points, but **you must implement the training loops and ablation orchestration yourself**. Each of Tasks 1-3 also contains one deliberate algorithmic defect in its core objective code; identifying and correcting these defects is part of validating your implementation.

Task 4 supplies the fixed AI safety judge and response-generation utilities, but you must write the evaluation/aggregation code. Task 5 supplies the exact RLVR verifier, the fixed pairwise AI judge used for RLAIF evaluation, and data/model loaders; you must implement the requested evaluation and analysis.

## 1. Clone and install

```bash
git clone https://github.com/COURSE_ORG/ATML-PA2-LLM-PostTraining.git
cd ATML-PA2-LLM-PostTraining
python -m pip install -r requirements.txt
```

## 2. Download the course assets

The large course-created checkpoints and fixed data are distributed as a GitHub Release asset rather than normal Git files. After cloning, run:

```bash
python -m scripts.download_assets
python -m scripts.validate_assets
```

If your instructor provides a direct asset URL separately, use:

```bash
python -m scripts.download_assets --url '<ASSET_URL>'
```

Public base/reward/judge models are downloaded from Hugging Face at runtime and are **not** included in the course asset archive.

The installer also materializes the fixed 100-example Task 5 transfer set from the official SVAMP challenge-set source if it is not already present. The tiny Task 1 word-limit prompt set is tracked directly in this repository.

## 3. Environment check

```bash
python -m scripts.check_environment
```

Run commands from the repository root. The reference environment used to prepare the release pins Transformers 4.57.1, TRL 0.27.2, PEFT 0.17.1, and Tokenizers 0.22.1.

## 4. Supplied course checkpoints

After `download_assets`, these directories should exist:

```text
checkpoints/ppo_midpoint_policy/
checkpoints/ppo_midpoint_value/
checkpoints/grpo_midpoint_policy/
checkpoints/rlvr_policy/
checkpoints/rlaif_policy/
```

PPO and GRPO begin from the supplied continuation checkpoints. RLVR and RLAIF are supplied frozen evaluation policies; students do not retrain them.

The PPO value checkpoint is intentionally released as the exact staff midpoint state, including its imperfect held-out value calibration. Treat critic behavior as an analysis variable rather than assuming a perfect baseline, and start every PPO fork from the identical supplied policy/value state. The default continuation generation cap is 512 tokens for feasibility; frozen evaluation uses the larger cap specified in `configs/ppo.yaml`.

## 5. Task entry points

### Task 1 - DPO

```bash
python -m task1_dpo.train --config configs/dpo.yaml --run-name standard
python -m task1_dpo.evaluate --config configs/dpo.yaml --adapter outputs/task1_dpo/standard --name standard
python -m task1_dpo.ablate_beta --config configs/dpo.yaml
python -m task1_dpo.analyze_length --config configs/dpo.yaml
```

### Task 2 - PPO

```bash
python -m task2_ppo.continue_train --config configs/ppo.yaml --run-name standard
python -m task2_ppo.evaluate --config configs/ppo.yaml --adapter outputs/task2_ppo/standard --name standard
python -m task2_ppo.analyze_clipping --config configs/ppo.yaml
python -m task2_ppo.ablate_kl --config configs/ppo.yaml
```

### Task 3 - GRPO

```bash
python -m task3_grpo.continue_train --config configs/grpo.yaml --run-name standard
python -m task3_grpo.evaluate --config configs/grpo.yaml --adapter outputs/task3_grpo/standard --name standard
python -m task3_grpo.analyze_group_size --config configs/grpo.yaml
python -m task3_grpo.compare_normalization --config configs/grpo.yaml
```

### Task 4 - Safety calibration

The judge loader/parser are supplied. You must implement the requested generation aggregation and evaluation.

```bash
python -m task4_safety.generate_responses --config configs/feedback.yaml
python -m task4_safety.judge_responses --config configs/feedback.yaml
python -m task4_safety.make_audit_sheet --config configs/feedback.yaml
python -m task4_safety.evaluate_safety --config configs/feedback.yaml
```

### Task 5 - RLVR vs RLAIF

The exact verifier and pairwise AI judge are supplied; you implement the evaluation/analysis.

```bash
python -m task5_feedback.evaluate_math --config configs/feedback.yaml --dataset gsm
python -m task5_feedback.score_perturbations --config configs/feedback.yaml
python -m task5_feedback.evaluate_math --config configs/feedback.yaml --dataset transfer
python -m task5_feedback.compare_feedback --config configs/feedback.yaml
```

## 6. Reproducibility rules

- Do not alter course-provided data, cached rollouts, or supplied checkpoints.
- Start every short fork from the **same supplied midpoint checkpoint**.
- Keep prompt IDs, generated-token/update budgets, seed, and evaluation procedure matched across ablations.
- Commit your code, configs, small JSON/CSV logs, and figures. Do not commit downloaded checkpoints, raw course assets, or model caches.
- Record peak VRAM and wall-clock time for the standard PPO and GRPO continuations.

See the assignment manual for the required experiments, metrics, and report questions.

---

# Student implementation notes (PA2 submission by `mbh2006`)

This is the completed submission for ATML PA2. All starter infrastructure is retained;
the task pipelines were implemented on top of it.

## What was implemented

- **Objective validation and fixes** — each of Tasks 1–3 contained one deliberate defect; all three were identified, corrected, and proven with hand-computed unit tests (`tests/test_objectives.py`):
  - `task1_dpo/dpo.py` — DPO logit now uses `policy_margin − ref_margin` (reference term subtracted);
  - `task2_ppo/ppo.py` — clipped surrogate now takes `min(surr1, surr2)` (not `max`);
  - `task3_grpo/grpo.py` — group-relative advantages are standardized **within each prompt group**.
- **Task 1 (DPO)** — training loop with gradient accumulation and frozen-reference log-probs, held-out evaluation (preference accuracy, sampled KL, reward-model score, response length), β ∈ {0.03, 0.10, 0.30} matched short forks, length-balanced condition with per-stratum analysis and word-limit compliance (`task1_dpo/`). **Deterministic prompt-length filtering (disclosed compliance exception):** rows whose chat-templated prompt alone exceeds the released `max_sequence_length` (768 tokens) cannot be trained on and are dropped before subsetting (54/1500 standard-train → 1,446 used; 58/1500 length-balanced → 1,442, retained strata 478/482/482; eval drops 10/300 pairs and 9/246 stratified rows). The released encoder mandates filtering; no feasible max_length covers prompts up to 3,563 tokens; all conditions are filtered by the identical rule and every dropped `source_index` is logged.
- **Task 2 (PPO)** — 20-update continuation from the supplied midpoint, GAE/returns, KL-shaped rewards, clip fraction / gradient-norm / entropy / VRAM / wall-clock instrumentation, cached-batch clip geometry, ε and KL-shaping forks (`task2_ppo/`).
- **Task 3 (GRPO)** — K=4 continuation, per-group advantages, max-length completion masking, equal-generation group-size study on the supplied K=8 cache, canonical vs. Dr. GRPO normalization forks (`task3_grpo/`).
- **Task 4 (safety calibration)** — deterministic generation for SFT / standard DPO / standard PPO / standard GRPO on XSTest, resumable fixed-judge scoring, manual-audit sheet and judge–human agreement analysis (`task4_safety/`).
- **Task 5 (RLVR vs RLAIF)** — in-domain GSM8K evaluation, pairwise AI judging against SFT, controlled 100-response diagnostic scoring (S_reason / S_outcome), out-of-domain SVAMP transfer (`task5_feedback/`).
- **Shared/analysis** — `common/rl_eval.py` (shared RL-prompt evaluation), `analysis/` (tables and figures).

## Reproduce

```bash
# setup (see Quick start above), then per task:
python -m task1_dpo.train --config configs/dpo.yaml --run-name standard
python -m task1_dpo.evaluate --config configs/dpo.yaml --adapter outputs/task1_dpo/standard --name standard
python -m task1_dpo.ablate_beta --config configs/dpo.yaml
python -m task1_dpo.analyze_length --config configs/dpo.yaml

python -m task2_ppo.continue_train --config configs/ppo.yaml --run-name standard
python -m task2_ppo.evaluate --config configs/ppo.yaml --adapter outputs/task2_ppo/standard --name standard
python -m task2_ppo.analyze_clipping --config configs/ppo.yaml        # cached geometry + ε forks
python -m task2_ppo.ablate_kl --config configs/ppo.yaml

python -m task3_grpo.continue_train --config configs/grpo.yaml --run-name standard
python -m task3_grpo.evaluate --config configs/grpo.yaml --adapter outputs/task3_grpo/standard --name standard
python -m task3_grpo.analyze_group_size --config configs/grpo.yaml    # cache-only, no GPU needed
python -m task3_grpo.compare_normalization --config configs/grpo.yaml

python -m task4_safety.generate_responses --config configs/feedback.yaml
python -m task4_safety.judge_responses --config configs/feedback.yaml
python -m task4_safety.make_audit_sheet --config configs/feedback.yaml
python -m task4_safety.evaluate_safety --config configs/feedback.yaml

python -m task5_feedback.evaluate_math --config configs/feedback.yaml --dataset gsm
python -m task5_feedback.score_perturbations --config configs/feedback.yaml
python -m task5_feedback.evaluate_math --config configs/feedback.yaml --dataset transfer
python -m task5_feedback.compare_feedback --config configs/feedback.yaml

python tests/test_objectives.py          # objective validation (3 hand-computed tests)
python tests/smoke_data_pipeline.py      # data pipeline sanity
```

Machine-readable results are written under `results/`; trained adapters under `outputs/`
(both reproducible from the commands above). All conditions use the fixed course splits,
prompt pools, seeds, and decoding settings documented in `configs/`.

## Attribution

- Starter infrastructure: course-provided `ATML-PA2-LLM-PostTraining` repository; all starter
  files retained or adapted as intended by the assignment.
- No external code was copied beyond the course starter. Task loops, evaluation code, analysis
  scripts, and tests for this submission were written for this assignment.
- As permitted by the course AI-usage policy for coding assistance, LLM tools were used during
  development. The objectives and every reported line of code were validated by the included
  tests and are understood by the student.

