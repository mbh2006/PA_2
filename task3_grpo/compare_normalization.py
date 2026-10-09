"""Task 3 length-normalization study: canonical GRPO vs Dr. GRPO short forks.

Both conditions start from the identical supplied midpoint, with identical
prompt order, generation settings, reward function, epsilon, KL-beta and
generated-token budget. Only the sequence normalization changes:
  - canonical GRPO: divide the summed clipped objective by each response's realized length
  - Dr. GRPO: divide by the fixed maximum completion length
"""

from __future__ import annotations

import argparse
import json
import statistics

from common.data import load_yaml, repo_path
from common.metrics import safe_corr
from common.rl_eval import evaluate_adapter_on_rl_eval
from task3_grpo.continue_train import run_grpo


def length_conditioned_stats(history: list[dict]):
    lengths = [h["response_tokens_mean"] for h in history]
    grads = [h["grad_norm_policy"] for h in history]
    return {
        "corr_length_vs_grad_norm": safe_corr(lengths, grads),
        "mean_grad_norm": statistics.fmean(grads) if grads else None,
        "mean_length": statistics.fmean(lengths) if lengths else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--max-eval-prompts", type=int, default=None)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    budget = int(cfg["fork_updates"])
    print("Fork updates:", budget)

    results = {}
    for loss_type in ["grpo", "dr_grpo"]:
        tag = f"norm_{loss_type}"
        out = f"outputs/task3_grpo/{tag}"
        train = run_grpo(args.config, output=out, updates=budget, loss_type=loss_type, run_name=tag)
        ev = evaluate_adapter_on_rl_eval(args.config, out, tag, max_prompts=args.max_eval_prompts)
        results[loss_type] = {
            "tag": tag,
            "train_summary": {k: v for k, v in train.items() if k != "history"},
            "length_conditioned": length_conditioned_stats(train["history"]),
            "heldout_reward": ev["reward_model_score_mean"],
            "heldout_kl": ev["kl_sampled_response_estimator"],
            "heldout_length": ev["response_length_mean"],
            "heldout_length_std": ev["response_length_std"],
        }
        print(loss_type, "->", json.dumps(results[loss_type], indent=2), flush=True)

    results_dir = repo_path(cfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "normalization_study.json").write_text(
        json.dumps({"budget": budget, "results": results}, indent=2), encoding="utf-8"
    )
    print("normalization study complete")


if __name__ == "__main__":
    main()
