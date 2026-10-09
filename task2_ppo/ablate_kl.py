from __future__ import annotations

import argparse
import json
import statistics

from common.data import load_yaml, repo_path
from common.rl_eval import evaluate_adapter_on_rl_eval
from task2_ppo.continue_train import run_ppo


def run_kl_fork(config_path: str, kl_beta: float, budget: int, max_eval_prompts: int | None = None):
    tag = "kl_" + f"{float(kl_beta):g}".replace(".", "p")
    out = f"outputs/task2_ppo/{tag}"
    summary = run_ppo(config_path, output=out, updates=budget, kl_beta=kl_beta, run_name=tag)
    eval_summary = evaluate_adapter_on_rl_eval(config_path, out, tag, max_prompts=max_eval_prompts)
    return {"tag": tag, "kl_beta": kl_beta, "train": summary, "eval": eval_summary}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--max-eval-prompts", type=int, default=None)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    budget = int(cfg["fork_updates"])
    print("KL beta conditions:", cfg["kl_values"])
    print("Fork update budget:", budget)

    results = []
    for kl_beta in cfg["kl_values"]:
        results.append(run_kl_fork(args.config, float(kl_beta), budget, args.max_eval_prompts))

    # Compact comparison table across KL conditions (uses the matched short-fork budget).
    table = []
    for r in results:
        h = r["train"]["history"]
        table.append(
            {
                "tag": r["tag"],
                "kl_beta": r["kl_beta"],
                "final_mean_reward": statistics.fmean([x["effective_reward"] for x in h[-3:]]) if h else None,
                "final_kl": statistics.fmean([x["kl"] for x in h[-3:]]) if h else None,
                "final_entropy": statistics.fmean([x["entropy"] for x in h[-3:]]) if h else None,
                "final_length": statistics.fmean([x["response_tokens"] for x in h[-3:]]) if h else None,
                "heldout_reward": r["eval"]["reward_model_score_mean"],
                "heldout_kl": r["eval"]["kl_sampled_response_estimator"],
                "heldout_length": r["eval"]["response_length_mean"],
            }
        )

    results_dir = repo_path(cfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "kl_study.json").write_text(
        json.dumps({"budget": budget, "conditions": table, "runs": results}, indent=2), encoding="utf-8"
    )
    print(json.dumps(table, indent=2))


if __name__ == "__main__":
    main()
