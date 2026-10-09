from __future__ import annotations

import argparse

from common.rl_eval import evaluate_adapter_on_rl_eval


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--name", default="standard")
    ap.add_argument("--max-eval-prompts", type=int, default=None)
    args = ap.parse_args()
    evaluate_adapter_on_rl_eval(args.config, args.adapter, args.name, max_prompts=args.max_eval_prompts)


if __name__ == "__main__":
    main()
