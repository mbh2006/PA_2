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
    prompt_messages,
    prompt_messages_from_preference,
    read_jsonl,
    repo_path,
)
from common.generation import batch_generate, response_sequence_logprobs
from common.metrics import parse_word_limit, word_count
from common.models import load_policy, load_tokenizer, reference_mode
from task1_dpo.train import run_training


def _row_prompt_text(row: dict) -> str:
    messages = row.get("messages")
    if isinstance(messages, list):
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get("role") == "user":
                return str(msg.get("content", ""))
    if isinstance(row.get("prompt"), str):
        return row["prompt"]
    if isinstance(row.get("question"), str):
        return row["question"]
    return ""


@torch.no_grad()
def stratified_margins(cfg: dict, policy, tokenizer, batch_size: int = 4):
    """Preference accuracy per length stratum on the fixed length-stratified held-out set."""
    rows = read_jsonl(cfg["paths"]["dpo_length_eval"])
    device = next(policy.parameters()).device
    max_len = int(cfg["max_sequence_length"])

    buckets: dict[str, list[float]] = {}
    all_values: list[float] = []
    skipped = 0
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        chosen_list, rejected_list, keep = [], [], []
        for row in chunk:
            prompt = prompt_messages_from_preference(row)
            yc, yr = preference_responses(row)
            try:
                chosen_list.append(encode_prompt_response(tokenizer, prompt, yc, max_len))
                rejected_list.append(encode_prompt_response(tokenizer, prompt, yr, max_len))
                keep.append(row)
            except ValueError:
                skipped += 1
        if not keep:
            continue
        chosen = {k: v.to(device) for k, v in pad_batch(tokenizer, chosen_list).items()}
        rejected = {k: v.to(device) for k, v in pad_batch(tokenizer, rejected_list).items()}
        pc, _, _ = response_sequence_logprobs(policy, chosen)
        pr, _, _ = response_sequence_logprobs(policy, rejected)
        with reference_mode(policy):
            rc, _, _ = response_sequence_logprobs(policy, chosen)
            rr, _, _ = response_sequence_logprobs(policy, rejected)
        margins = ((pc - pr) - (rc - rr)).float().cpu().tolist()
        for j, row in enumerate(keep):
            stratum = str(row.get("length_stratum", "unknown"))
            buckets.setdefault(stratum, []).append(margins[j])
            all_values.append(margins[j])

    out = {}
    for stratum, values in sorted(buckets.items()):
        out[stratum] = {
            "num_pairs": len(values),
            "preference_accuracy": sum(1.0 for v in values if v > 0) / max(len(values), 1),
            "mean_margin": statistics.fmean(values) if values else None,
        }
    out["_overall"] = {
        "num_pairs": len(all_values),
        "preference_accuracy": sum(1.0 for v in all_values if v > 0) / max(len(all_values), 1),
        "mean_margin": statistics.fmean(all_values) if all_values else None,
    }
    out["_num_skipped_prompt_too_long"] = skipped
    return out


@torch.no_grad()
def word_limit_metrics(cfg: dict, policy, tokenizer):
    """Response length + word-limit compliance on the common fixed word-limit prompt set."""
    rows = read_jsonl(cfg["paths"]["word_limit_prompts"])
    prompts = [prompt_messages(row) for row in rows]
    texts = [_row_prompt_text(row) for row in rows]
    gen_cfg = cfg.get("generation", {})

    gen = batch_generate(
        policy,
        tokenizer,
        prompts,
        max_prompt_length=int(cfg.get("max_prompt_length", 256)),
        max_new_tokens=int(cfg.get("max_generation_tokens", 256)),
        temperature=float(gen_cfg.get("temperature", 0.7)),
        top_p=float(gen_cfg.get("top_p", 0.9)),
        do_sample=bool(gen_cfg.get("do_sample", True)),
    )

    per_prompt = []
    for i, row in enumerate(rows):
        limit = parse_word_limit(texts[i])
        words = word_count(gen["responses"][i])
        per_prompt.append(
            {
                "prompt": texts[i],
                "word_limit": limit,
                "response_words": words,
                "compliant": (words <= limit) if limit is not None else None,
                "response_tokens": int(gen["response_lengths"][i]),
                "response": gen["responses"][i],
            }
        )

    compliant = [p["compliant"] for p in per_prompt if p["compliant"] is not None]
    return {
        "num_prompts": len(per_prompt),
        "num_with_limit": len(compliant),
        "word_limit_compliance_rate": (sum(1.0 for c in compliant) / len(compliant)) if compliant else None,
        "response_length_mean_tokens": statistics.fmean([p["response_tokens"] for p in per_prompt]) if per_prompt else None,
        "per_prompt": per_prompt,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--skip-train", action="store_true", help="reuse an existing length-balanced adapter")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    balanced = read_jsonl(cfg["paths"]["dpo_length_train"])
    stratified = read_jsonl(cfg["paths"]["dpo_length_eval"])
    print("Length-balanced train rows:", len(balanced))
    print("Length-stratified eval rows:", len(stratified))

    if not args.skip_train:
        run_training(
            args.config,
            "length_balanced",
            dataset_path=cfg["paths"]["dpo_length_train"],
            output_path=cfg["length_output"],
        )

    results_dir = repo_path(cfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = load_tokenizer(cfg["base_model"])

    for tag, adapter in [("standard", cfg["standard_output"]), ("length_balanced", cfg["length_output"])]:
        policy = load_policy(cfg, adapter_path=adapter, trainable=False)
        strat = stratified_margins(cfg, policy, tokenizer)
        word_limit = word_limit_metrics(cfg, policy, tokenizer)
        payload = {
            "condition": tag,
            "adapter": adapter,
            "stratified_eval": strat,
            "word_limit": word_limit,
        }
        (results_dir / f"length_analysis_{tag}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(tag, "->", json.dumps(strat, indent=2))
        del policy
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
