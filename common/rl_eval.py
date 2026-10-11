"""Shared held-out evaluation for tasks 2 (PPO) and 3 (GRPO).

Generates a fixed number of deterministic-per-condition responses on the fixed
RL prompt eval pool, then reports: reward-model score, KL from the reference
policy (sampled-response estimator), response-length statistics, and
truncation/EOS rates. Also saves the generations for qualitative analysis.

Usage from a task evaluation entry point:
    from common.rl_eval import evaluate_adapter_on_rl_eval
    evaluate_adapter_on_rl_eval("configs/ppo.yaml", "outputs/task2_ppo/standard", "standard")
"""

from __future__ import annotations

import argparse
import json
import statistics

import torch

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.generation import batch_generate, response_token_logprobs, score_reward_pairs
from common.logging_utils import set_seed
from common.models import (
    clear_gpu,
    load_policy,
    load_reward_model,
    load_tokenizer,
    reference_mode,
)


@torch.no_grad()
def evaluate_adapter_on_rl_eval(
    config_path: str,
    adapter: str,
    name: str,
    max_prompts: int | None = None,
    batch_size: int = 4,
    save_generations: bool = True,
):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))  # reset eval seed: sampled evaluations become reproducible and prior-state independent
    rows = read_jsonl(cfg["paths"]["rl_prompt_eval"])
    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(cfg, adapter_path=adapter, trainable=False)
    reward_model, reward_tokenizer = load_reward_model(cfg)

    gen_cfg = cfg.get("generation", {})
    max_prompt_length = int(cfg.get("max_prompt_length", 256))
    max_new = int(cfg.get("eval_max_response_length", cfg.get("max_response_length", 512)))
    reward_max_length = int(cfg.get("reward_max_length", 1280))
    if max_prompts is None:
        max_prompts = cfg.get("eval_generation_prompts")

    prompts = []
    for row in rows:
        prompts.append(prompt_messages(row))
        if max_prompts is not None and len(prompts) >= int(max_prompts):
            break

    records = []
    kl_token_sum = 0.0
    kl_token_count = 0
    ent_token_sum = 0.0
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        gen = batch_generate(
            policy,
            tokenizer,
            batch,
            max_prompt_length=max_prompt_length,
            max_new_tokens=max_new,
            temperature=float(gen_cfg.get("temperature", 0.7)),
            top_p=float(gen_cfg.get("top_p", 0.9)),
            do_sample=bool(gen_cfg.get("do_sample", True)),
        )
        pol_logp, _ = response_token_logprobs(
            policy, gen["sequences"], gen["attention_mask"], gen["prompt_width"], gen["response_ids"]
        )
        with reference_mode(policy):
            ref_logp, _ = response_token_logprobs(
                policy, gen["sequences"], gen["attention_mask"], gen["prompt_width"], gen["response_ids"]
            )
        diffs = (pol_logp - ref_logp) * gen["response_mask"]
        per_seq = diffs.sum(-1) / gen["response_mask"].sum(-1).clamp_min(1.0)
        kl_token_sum += float(diffs.sum())
        kl_token_count += int(gen["response_mask"].sum())
        per_seq_ent = ((-pol_logp) * gen["response_mask"]).sum(-1) / gen["response_mask"].sum(-1).clamp_min(1.0)
        ent_token_sum += float(((-pol_logp) * gen["response_mask"]).sum())

        rewards = score_reward_pairs(
            reward_model, reward_tokenizer, batch, gen["responses"], max_length=reward_max_length
        )
        for j in range(len(batch)):
            records.append(
                {
                    "prompt_index": start + j,
                    "prompt": batch[j],
                    "response": gen["responses"][j],
                    "reward_model_score": float(rewards[j]),
                    "kl": float(per_seq[j]),
                    "entropy": float(per_seq_ent[j]),
                    "response_tokens": int(gen["response_lengths"][j]),
                    "terminated_with_eos": bool(gen["terminated_with_eos"][j]),
                    "truncated": bool(gen["truncated"][j]),
                }
            )

    rewards_all = [r["reward_model_score"] for r in records]
    lengths = [r["response_tokens"] for r in records]
    summary = {
        "name": name,
        "adapter": str(repo_path(adapter)),
        "num_prompts": len(records),
        "reward_model_score_mean": statistics.fmean(rewards_all) if rewards_all else None,
        "reward_model_score_std": statistics.pstdev(rewards_all) if len(rewards_all) > 1 else None,
        "kl_sampled_response_estimator": (kl_token_sum / kl_token_count) if kl_token_count else None,
        "kl_per_sequence_mean": statistics.fmean([r["kl"] for r in records]) if records else None,
        "entropy_sampled_response_mean": (ent_token_sum / kl_token_count) if kl_token_count else None,
        "entropy_sampled_response_std": statistics.pstdev([r["entropy"] for r in records]) if len(records) > 1 else None,
        "response_length_mean": statistics.fmean(lengths) if lengths else None,
        "response_length_std": statistics.pstdev(lengths) if len(lengths) > 1 else None,
        "truncated_fraction": (sum(1 for r in records if r["truncated"]) / len(records)) if records else None,
        "eos_fraction": (sum(1 for r in records if r["terminated_with_eos"]) / len(records)) if records else None,
        "max_new_tokens": max_new,
        "decoding": gen_cfg,
    }

    results_dir = repo_path(cfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / f"eval_{name}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if save_generations:
        with (results_dir / f"generations_{name}.jsonl").open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(json.dumps(summary, indent=2))
    clear_gpu(policy, reward_model)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--name", default="standard")
    ap.add_argument("--max-eval-prompts", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args()
    evaluate_adapter_on_rl_eval(
        args.config, args.adapter, args.name, max_prompts=args.max_eval_prompts, batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()
