from __future__ import annotations

import argparse

from common.data import load_yaml
from task1_dpo.evaluate import run_evaluation
from task1_dpo.train import run_training


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--max-eval-prompts", type=int, default=None)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    print("Required beta values:", cfg["betas"])
    print("Short-run examples per condition:", cfg["short_ablation_examples"])

    for beta in cfg["betas"]:
        tag = "beta_" + f"{float(beta):g}".replace(".", "p")
        out = f"outputs/task1_dpo/{tag}"
        # Matched short forks: identical data subset, optimizer, seed, LoRA config; only beta changes.
        run_training(
            args.config,
            tag,
            beta=float(beta),
            max_examples=int(cfg["short_ablation_examples"]),
            output_path=out,
        )
        run_evaluation(args.config, out, tag, max_eval_prompts=args.max_eval_prompts)


if __name__ == "__main__":
    main()
