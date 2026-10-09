from __future__ import annotations

import argparse
import json

import pandas as pd

from common.data import load_yaml, repo_path
from common.generation import batch_generate
from common.models import clear_gpu, load_policy, load_tokenizer


def policy_specs(cfg):
    return {
        "sft": None,
        "dpo": cfg["policies"]["dpo"],
        "ppo": cfg["policies"]["ppo"],
        "grpo": cfg["policies"]["grpo"],
    }


def load_xstest(cfg):
    return pd.read_csv(repo_path(cfg["paths"]["xstest"]))


def generate_for_policy(cfg, policy_name: str, batch_size: int = 4):
    specs = policy_specs(cfg)
    if policy_name not in specs:
        raise KeyError(policy_name)
    adapter = specs[policy_name]
    tokenizer = load_tokenizer(cfg["base_model"])
    model = load_policy(cfg, adapter_path=adapter, trainable=False)
    df = load_xstest(cfg)
    records = []
    for start in range(0, len(df), batch_size):
        chunk = df.iloc[start : start + batch_size]
        prompts = [[{"role": "user", "content": str(x)}] for x in chunk["prompt"].tolist()]
        gen = batch_generate(
            model,
            tokenizer,
            prompts,
            max_prompt_length=256,
            max_new_tokens=int(cfg["safety_max_new_tokens"]),
            temperature=0.0,
            top_p=1.0,
            do_sample=False,
        )
        for (_, row), response, n_tok in zip(chunk.iterrows(), gen["responses"], gen["response_lengths"]):
            records.append(
                {
                    "xstest_id": int(row["xstest_id"]),
                    "policy": policy_name,
                    "prompt": str(row["prompt"]),
                    "benchmark_class": str(row["benchmark_class"]),
                    "type": str(row["type"]),
                    "focus": str(row.get("focus", "")),
                    "response": response,
                    "response_tokens": int(n_tok),
                }
            )
    clear_gpu(model)
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--policies", nargs="*", default=["sft", "dpo", "ppo", "grpo"])
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    print("Policies:", list(policy_specs(cfg)))
    df = load_xstest(cfg)
    print("XSTest rows:", len(df))

    outdir = repo_path(cfg["results_dir"]) / "task4_safety"
    outdir.mkdir(parents=True, exist_ok=True)

    for policy_name in args.policies:
        print(f"=== generating {policy_name} ===", flush=True)
        records = generate_for_policy(cfg, policy_name, batch_size=args.batch_size)
        out = outdir / f"generated_{policy_name}.jsonl"
        with out.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        safe = sum(1 for r in records if r["benchmark_class"] == "SAFE")
        print(f"wrote {len(records)} responses ({safe} safe / {len(records) - safe} unsafe) -> {out}", flush=True)


if __name__ == "__main__":
    main()
