"""Task 2 clipping study.

Part A — fixed cached-batch geometry. The release ships TWO cached PPO rollout
batches that turn out to contain DIFFERENT rollouts (verified 2026-10-09):

  * official `cached/ppo_rollout.pt` — 32 rollouts with old/ref log-probs, critic
    `values` and terminal rewards, but NO token ids. Token ids are reconstructed
    from the fixed eval-prompt text + response text, appending EOS for terminated
    responses (stored log-prob arrays include EOS). `returns` are absent, so only
    clip/affected fractions are reported for this batch (robustness check).
  * `cached/ppo_rollout_salvage_raw.pt` — a different 32-rollout batch WITH exact
    `full_ids`, `response_ids`, `prompt_len` and `returns`. This is the batch used
    for the clipped-surrogate table; advantages = returns − values, where values
    come from the released frozen critic checkpoint.

For every adapter (released midpoint and, when present, the standard continuation)
the cached batch is re-scored and, per epsilon: mean/std/min/max ratio, clip
fraction (== affected-token fraction over valid response tokens), and the mean
clipped surrogate where returns are available. `mean|Δlogp|` between recomputed
and stored old log-probs is reported per batch/adapter so the ratio source and
provenance are auditable.

Part B — matched short forks: identical 8-update continuations from the same
supplied midpoint, changing only clip_epsilon, each followed by the common
held-out evaluation. Stability statistic: std of per-update clip fraction and of
policy gradient norm, plus final ratio deviation.

Run:  python -m task2_ppo.analyze_clipping --config configs/ppo.yaml
      python -m task2_ppo.analyze_clipping --config configs/ppo.yaml --cache-only
"""

from __future__ import annotations

import argparse
import json
import statistics

import torch

from common.data import load_yaml, read_jsonl, repo_path
from common.generation import response_token_logprobs
from common.models import clear_gpu, load_policy, load_tokenizer, load_value_model, token_values
from common.rl_eval import evaluate_adapter_on_rl_eval
from task2_ppo.continue_train import run_ppo


def load_cached_rollouts(path):
    rows = torch.load(repo_path(path), map_location="cpu", weights_only=False)
    if not isinstance(rows, list) or not rows:
        raise ValueError("Expected a non-empty list in the supplied PPO rollout cache")

    normalized = []
    for row in rows:
        row = dict(row)
        if "old_logprobs" not in row and "old_policy_logprobs" in row:
            row["old_logprobs"] = row["old_policy_logprobs"]
        if "ref_logprobs" not in row and "reference_logprobs" in row:
            row["ref_logprobs"] = row["reference_logprobs"]
        normalized.append(row)

    required = {"source_index", "response", "old_logprobs", "ref_logprobs"}
    if not required.issubset(normalized[0]):
        raise ValueError(f"Unexpected PPO cache schema; need at least {sorted(required)}")
    return normalized


def reconstruct_official_sequences(cfg, tokenizer):
    """Official cache: no ids -> re-tokenize prompt+response, append EOS when terminated."""
    rows = load_cached_rollouts(cfg["cached_rollouts"])
    eval_by_index = {r.get("source_index"): r for r in read_jsonl(cfg["paths"]["rl_prompt_eval"])}
    out = []
    missing_prompt = 0
    for row in rows:
        er = eval_by_index.get(row.get("source_index"))
        prompt_text = er.get("prompt") if er else None
        if prompt_text is None:
            missing_prompt += 1
            continue
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        resp_ids = tokenizer(str(row["response"]), add_special_tokens=False)["input_ids"]
        if bool(row.get("terminated_with_eos")):
            resp_ids = resp_ids + [tokenizer.eos_token_id]
        out.append(
            {
                "source_index": row.get("source_index"),
                "full_ids": torch.tensor(list(prompt_ids) + list(resp_ids), dtype=torch.long),
                "prompt_len": len(prompt_ids),
                "response_ids": torch.tensor(resp_ids, dtype=torch.long),
                "old_logprobs": row["old_logprobs"].float(),
                "ref_logprobs": row["ref_logprobs"].float(),
                "values": row.get("values").float() if row.get("values") is not None else None,
                "returns": None,
            }
        )
    print(f"official batch: reconstructed {len(out)} rollouts (missing prompts: {missing_prompt})", flush=True)
    return out


def load_salvage_sequences():
    """Salvage batch: exact ids + returns shipped in the file."""
    rows = torch.load(repo_path("cached/ppo_rollout_salvage_raw.pt"), map_location="cpu", weights_only=False)
    out = []
    for row in rows:
        full = torch.as_tensor(row["full_ids"], dtype=torch.long)
        rid = torch.as_tensor(row["response_ids"], dtype=torch.long)
        out.append(
            {
                "source_index": row.get("source_index"),
                "full_ids": full,
                "prompt_len": int(row["prompt_len"]),
                "response_ids": rid,
                "old_logprobs": row["old_logprobs"].float(),
                "ref_logprobs": row["ref_logprobs"].float(),
                "values": None,
                "returns": torch.as_tensor(row["returns"], dtype=torch.float32),
            }
        )
    print(f"salvage batch: loaded {len(out)} rollouts with exact ids and returns", flush=True)
    return out


@torch.no_grad()
def estimate_values_with_critic(cfg, sequences):
    """Fill missing `values` with the released frozen critic checkpoint."""
    value_model = load_value_model(cfg, cfg["paths"]["ppo_midpoint_value"], train_mode="frozen")
    device = next(value_model.parameters()).device
    for item in sequences:
        if item.get("values") is not None:
            continue
        ids = item["full_ids"].unsqueeze(0).to(device)
        attn = torch.ones_like(ids)
        v = token_values(value_model, ids, attn)[0, item["prompt_len"] - 1 : -1].float().cpu()
        item["values"] = v
    clear_gpu(value_model)
    return sequences


@torch.no_grad()
def rescore_sequences(cfg, sequences, adapter):
    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(cfg, adapter_path=adapter, trainable=False)
    device = next(policy.parameters()).device
    new_logps = []
    for item in sequences:
        ids = item["full_ids"].unsqueeze(0).to(device)
        attn = torch.ones_like(ids)
        rid = item["response_ids"].unsqueeze(0).to(device)
        logp, _ = response_token_logprobs(policy, ids, attn, item["prompt_len"], rid)
        new_logps.append(logp[0].float().cpu())
    clear_gpu(policy)
    return new_logps


def analyze_batch(cfg, sequences, eps_values, adapters, use_surrogate: bool):
    report = {
        "num_rollouts": len(sequences),
        "length_histogram": {},
        "adapter_analyses": {},
        "source_indices_in_eval_pool": None,
    }

    len_counts: dict[tuple[int, int], int] = {}
    for item in sequences:
        pair = (len(item["response_ids"]), len(item["old_logprobs"]))
        len_counts[pair] = len_counts.get(pair, 0) + 1
    report["length_histogram"] = {str(k): v for k, v in len_counts.items()}

    eval_pool = {r.get("source_index") for r in read_jsonl(cfg["paths"]["rl_prompt_eval"])}
    report["source_indices_in_eval_pool"] = sum(1 for it in sequences if it.get("source_index") in eval_pool)

    for tag, adapter in adapters.items():
        if adapter and not repo_path(adapter).exists():
            print(f"skipping adapter {tag}: {adapter} not found", flush=True)
            continue
        new_logps = rescore_sequences(cfg, sequences, adapter)

        diffs = []
        for item, nl in zip(sequences, new_logps):
            m = min(len(nl), len(item["old_logprobs"]))
            if m:
                diffs.append(float((nl[:m] - item["old_logprobs"][:m]).abs().mean()))
        mean_abs_diff = statistics.fmean(diffs) if diffs else None

        eps_table = {}
        for eps in eps_values:
            ratios, affected = [], []
            surr_sum, surr_n = 0.0, 0
            for item, nl in zip(sequences, new_logps):
                m = min(len(nl), len(item["old_logprobs"]))
                if m == 0:
                    continue
                ratio = torch.exp(nl[:m] - item["old_logprobs"][:m])
                ratios.append(ratio)
                affected.append((ratio < (1.0 - eps)) | (ratio > (1.0 + eps)))
                if use_surrogate and item.get("returns") is not None and item.get("values") is not None:
                    mv = min(m, len(item["returns"]), len(item["values"]))
                    if mv:
                        # manual-consistent surrogate: no advantage clamping, token-weighted
                        # aggregation across all valid response tokens (was: clamped, per-rollout mean)
                        adv = item["returns"][:mv] - item["values"][:mv]
                        ratio_ = ratio[:mv]
                        s1 = ratio_ * adv
                        s2 = ratio_.clamp(1.0 - eps, 1.0 + eps) * adv
                        surr_sum += float(torch.minimum(s1, s2).sum())
                        surr_n += mv
            ratio_all = torch.cat(ratios) if ratios else torch.tensor([])
            aff_all = torch.cat(affected) if affected else torch.tensor([], dtype=torch.bool)
            eps_table[str(eps)] = {
                "mean_ratio": float(ratio_all.mean()) if ratio_all.numel() else None,
                "std_ratio": float(ratio_all.std(unbiased=False)) if ratio_all.numel() else None,
                "min_ratio": float(ratio_all.min()) if ratio_all.numel() else None,
                "max_ratio": float(ratio_all.max()) if ratio_all.numel() else None,
                "clip_fraction": float(aff_all.float().mean()) if aff_all.numel() else None,
                "affected_token_fraction": float(aff_all.float().mean()) if aff_all.numel() else None,
                "mean_clipped_surrogate": (surr_sum / surr_n) if surr_n else None,
            }
        report["adapter_analyses"][tag] = {
            "adapter": str(adapter),
            "mean_abs_logp_diff_vs_stored_old": mean_abs_diff,
            "epsilon_table": eps_table,
        }
        print(f"batch analysis vs {tag}: mean|dlogp|={mean_abs_diff if mean_abs_diff is None else round(mean_abs_diff, 6)}", flush=True)
        for e, vals in eps_table.items():
            print(
                f"  eps={e}: clip_frac={vals['clip_fraction']} affected={vals['affected_token_fraction']} "
                f"mean_ratio={vals['mean_ratio']} surrogate={vals['mean_clipped_surrogate']}",
                flush=True,
            )
    return report


def cached_batch_clip_analysis(cfg, eps_values, adapters: dict[str, str]):
    tokenizer = load_tokenizer(cfg["base_model"])
    official = reconstruct_official_sequences(cfg, tokenizer)
    salvage = load_salvage_sequences()
    salvage = estimate_values_with_critic(cfg, salvage)

    report = {
        "note": (
            "official and salvage caches contain DIFFERENT rollouts (verified 0/32 match). "
            "official: reconstructed ids (+EOS), no returns -> clip/affected only; "
            "salvage: exact ids + supplied returns, values from the frozen critic -> surrogate available."
        ),
        "batches": {
            "official": analyze_batch(cfg, official, eps_values, adapters, use_surrogate=False),
            "salvage": analyze_batch(cfg, salvage, eps_values, adapters, use_surrogate=True),
        },
    }
    results_dir = repo_path(cfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "clip_study_cache.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def stability_stats(history):
    clip = [h["clip_fraction"] for h in history]
    pgrad = [h["grad_norm_policy"] for h in history]
    ratio = [h["ratio_mean"] for h in history]
    return {
        "clip_fraction_std": statistics.pstdev(clip) if len(clip) > 1 else 0.0,
        "grad_norm_policy_std": statistics.pstdev(pgrad) if len(pgrad) > 1 else 0.0,
        "ratio_deviation_final": abs(ratio[-1] - 1.0) if ratio else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--cache-only", action="store_true")
    ap.add_argument("--forks-only", action="store_true", help="skip the cached-batch analysis (already saved by a previous session)")
    ap.add_argument("--max-eval-prompts", type=int, default=None)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    eps_values = [float(e) for e in cfg["clip_values"]]

    if args.forks_only:
        cache_path = repo_path(cfg["results_dir"]) / "clip_study_cache.json"
        cache_report = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
        print("forks-only mode: reusing cached-batch analysis from", cache_path, flush=True)
    else:
        adapters = {"midpoint": cfg["paths"]["ppo_midpoint_policy"]}
        standard_adapter = cfg.get("output", "outputs/task2_ppo/standard")
        if repo_path(standard_adapter).exists():
            adapters["standard_final"] = standard_adapter
        else:
            print(f"(standard adapter {standard_adapter} not present yet; cache analysis will use midpoint only)")

        cache_report = cached_batch_clip_analysis(cfg, eps_values, adapters)
        if args.cache_only:
            return

    fork_results = []
    for eps in eps_values:
        tag = "clip_eps_" + f"{eps:g}".replace(".", "p")
        out = f"outputs/task2_ppo/{tag}"
        train = run_ppo(args.config, output=out, updates=int(cfg["fork_updates"]), clip_epsilon=eps, run_name=tag)
        ev = evaluate_adapter_on_rl_eval(args.config, out, tag, max_prompts=args.max_eval_prompts)
        fork_results.append(
            {
                "tag": tag,
                "clip_epsilon": eps,
                "stability": stability_stats(train["history"]),
                "heldout_reward": ev["reward_model_score_mean"],
                "heldout_kl": ev["kl_sampled_response_estimator"],
                "heldout_length": ev["response_length_mean"],
                "train_summary": {k: v for k, v in train.items() if k != "history"},
            }
        )
        print(json.dumps(fork_results[-1], indent=2), flush=True)

    results_dir = repo_path(cfg["results_dir"])
    (results_dir / "clip_study.json").write_text(
        json.dumps({"cache_analysis": cache_report, "forks": fork_results}, indent=2), encoding="utf-8"
    )
    print("clip study complete")


if __name__ == "__main__":
    main()
