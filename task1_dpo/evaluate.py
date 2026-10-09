from __future__ import annotations

import argparse
import json
import statistics

import torch

from common.data import (
    encode_prompt_response,
    load_yaml,
    pad_batch,
    preference_responses,
    prompt_messages_from_preference,
    read_jsonl,
    repo_path,
)
from common.generation import (
    batch_generate,
    response_sequence_logprobs,
    response_token_logprobs,
    score_reward_pairs,
)
from common.models import load_policy, load_reward_model, load_tokenizer, reference_mode


def load_evaluation_bundle(config_path: str, adapter: str):
    cfg = load_yaml(config_path)
    return {
        "cfg": cfg,
        "rows": read_jsonl(cfg["paths"]["dpo_standard_eval"]),
        "tokenizer": load_tokenizer(cfg["base_model"]),
        "policy": load_policy(cfg, adapter_path=adapter, trainable=False),
        "reward": load_reward_model(cfg),
    }


@torch.no_grad()
def heldout_pair_metrics(bundle, batch_size: int = 4):
    """Held-out DPO preference accuracy: fraction of pairs with (policy margin - ref margin) > 0."""
    cfg = bundle["cfg"]
    rows = bundle["rows"]
    tokenizer = bundle["tokenizer"]
    policy = bundle["policy"]
    device = next(policy.parameters()).device
    max_len = int(cfg["max_sequence_length"])

    margins, per_pair = [], []
    skipped = 0
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        chosen_list, rejected_list, keep = [], [], []
        for row in chunk:
            prompt = prompt_messages_from_preference(row)
            yc, yr = preference_responses(row)
            try:
                c = encode_prompt_response(tokenizer, prompt, yc, max_len)
                r = encode_prompt_response(tokenizer, prompt, yr, max_len)
            except ValueError:
                skipped += 1
                continue
            chosen_list.append(c)
            rejected_list.append(r)
            keep.append(row)
        if not keep:
            continue
        chosen = {k: v.to(device) for k, v in pad_batch(tokenizer, chosen_list).items()}
        rejected = {k: v.to(device) for k, v in pad_batch(tokenizer, rejected_list).items()}

        pc, _, _ = response_sequence_logprobs(policy, chosen)
        pr, _, _ = response_sequence_logprobs(policy, rejected)
        with reference_mode(policy):
            rc, _, _ = response_sequence_logprobs(policy, chosen)
            rr, _, _ = response_sequence_logprobs(policy, rejected)

        m = (pc - pr) - (rc - rr)
        for j, row in enumerate(keep):
            margins.append(float(m[j]))
            per_pair.append(
                {
                    "source_index": row.get("source_index"),
                    "prompt_id": row.get("prompt_id"),
                    "margin": float(m[j]),
                    "policy_margin": float((pc - pr)[j]),
                    "ref_margin": float((rc - rr)[j]),
                }
            )

    acc = sum(1.0 for v in margins if v > 0) / max(len(margins), 1)
    out = {
        "num_eval_pairs": len(margins),
        "num_skipped_prompt_too_long": skipped,
        "preference_accuracy": acc,
        "mean_margin": statistics.fmean(margins) if margins else None,
    }
    return out, per_pair


@torch.no_grad()
def generation_metrics(bundle, max_prompts: int | None = None, batch_size: int = 4):
    """KL from reference (sampled-response estimator), reward-model score, and length on generated responses."""
    cfg = bundle["cfg"]
    rows = bundle["rows"]
    tokenizer = bundle["tokenizer"]
    policy = bundle["policy"]
    reward_model, reward_tokenizer = bundle["reward"]

    gen_cfg = cfg.get("generation", {})
    max_prompt_length = int(cfg.get("max_prompt_length", 256))
    max_new_tokens = int(cfg.get("max_generation_tokens", 256))

    prompts, seen = [], set()
    for row in rows:
        msgs = prompt_messages_from_preference(row)
        key = json.dumps(msgs, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        prompts.append(msgs)
        if max_prompts is not None and len(prompts) >= max_prompts:
            break

    records = []
    kl_token_weighted_sum = 0.0
    kl_token_count = 0
    for start in range(0, len(prompts), batch_size):
        batch_msgs = prompts[start : start + batch_size]
        gen = batch_generate(
            policy,
            tokenizer,
            batch_msgs,
            max_prompt_length=max_prompt_length,
            max_new_tokens=max_new_tokens,
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
        kl_token_weighted_sum += float(diffs.sum())
        kl_token_count += int(gen["response_mask"].sum())

        rm_scores = score_reward_pairs(
            reward_model,
            reward_tokenizer,
            batch_msgs,
            gen["responses"],
            max_length=int(cfg.get("reward_max_length", 1024)),
        )

        for i in range(len(batch_msgs)):
            records.append(
                {
                    "prompt_index": start + i,
                    "prompt": batch_msgs[i],
                    "response": gen["responses"][i],
                    "response_tokens": int(gen["response_lengths"][i]),
                    "kl": float(per_seq[i]),
                    "reward_model_score": float(rm_scores[i]),
                    "terminated_with_eos": bool(gen["terminated_with_eos"][i]),
                    "truncated": bool(gen["truncated"][i]),
                }
            )

    lengths = [r["response_tokens"] for r in records]
    out = {
        "num_prompts_generated": len(records),
        "kl_sampled_response_estimator": (kl_token_weighted_sum / kl_token_count) if kl_token_count else None,
        "kl_per_sequence_mean": statistics.fmean([r["kl"] for r in records]) if records else None,
        "reward_model_score_mean": statistics.fmean([r["reward_model_score"] for r in records]) if records else None,
        "response_length_mean": statistics.fmean(lengths) if lengths else None,
        "response_length_std": statistics.pstdev(lengths) if len(lengths) > 1 else None,
        "truncated_fraction": (sum(1 for r in records if r["truncated"]) / len(records)) if records else None,
    }
    return out, records


def run_evaluation(
    config_path: str,
    adapter: str,
    name: str,
    max_eval_prompts: int | None = None,
    gen_batch_size: int = 4,
    pair_batch_size: int = 4,
):
    bundle = load_evaluation_bundle(config_path, adapter)
    cfg = bundle["cfg"]
    results_dir = repo_path(cfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    pair_metrics, per_pair = heldout_pair_metrics(bundle, batch_size=pair_batch_size)
    if max_eval_prompts is None:
        max_eval_prompts = cfg.get("eval_generation_prompts")
    gen_metrics, records = generation_metrics(bundle, max_prompts=max_eval_prompts, batch_size=gen_batch_size)

    result = {
        "name": name,
        "adapter": str(repo_path(adapter)),
        "beta": float(cfg.get("beta", 0.0)),
        "seed": int(cfg["seed"]),
        "max_generation_tokens": int(cfg.get("max_generation_tokens", 0)),
        "decoding": cfg.get("generation"),
        "eval_generation_prompts": max_eval_prompts,
        "heldout_pairs": pair_metrics,
        "generation": gen_metrics,
        "per_pair": per_pair,
    }
    (results_dir / f"eval_{name}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (results_dir / f"generations_{name}.jsonl").open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(json.dumps({"name": name, "heldout_pairs": pair_metrics, "generation": gen_metrics}, indent=2))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--name", default="standard")
    ap.add_argument("--max-eval-prompts", type=int, default=None)
    ap.add_argument("--gen-batch-size", type=int, default=4)
    args = ap.parse_args()
    run_evaluation(args.config, args.adapter, args.name, args.max_eval_prompts, args.gen_batch_size)


if __name__ == "__main__":
    main()
