"""Task 2 clipping study.

Part A — fixed cached-batch geometry: reconstruct the cached rollouts' token ids
(preferring the shipped salvage cache, falling back to re-tokenization), re-score
them with the released midpoint policy (and the final standard adapter when
available), then report the clipped surrogate and affected-token fraction for
each epsilon in configs/ppo.yaml. Verification statistics (recomputed vs stored
old log-probs) are always reported so the ratio source is auditable.

Part B — matched short forks: identical 8-update continuations from the same
supplied midpoint, changing only clip_epsilon, each followed by the common
held-out evaluation. Stability statistic: standard deviation of the per-update
clip fraction and of the policy gradient norm over the fork, plus final ratio
deviation.

Run:  python -m task2_ppo.analyze_clipping --config configs/ppo.yaml
      python -m task2_ppo.analyze_clipping --config configs/ppo.yaml --cache-only
"""

from __future__ import annotations

import argparse
import json
import math
import statistics

import torch

from common.data import load_yaml, read_jsonl, repo_path
from common.generation import response_token_logprobs
from common.models import clear_gpu, load_policy, load_tokenizer
from common.rl_eval import evaluate_adapter_on_rl_eval
from task2_ppo.continue_train import run_ppo


def load_cached_rollouts(path):
    rows = torch.load(repo_path(path), map_location="cpu", weights_only=False)
    if not isinstance(rows, list) or not rows:
        raise ValueError("Expected a non-empty list in the supplied PPO rollout cache")

    # Instructor iterations used two equivalent names for these fields. Normalize once here so
    # the student analysis code sees one stable interface.
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


def load_salvage(path):
    p = repo_path(path)
    if not p.exists():
        return {}
    rows = torch.load(p, map_location="cpu", weights_only=False)
    out = {}
    for row in rows:
        key = (row.get("source_index"), str(row.get("response", ""))[:64])
        out[key] = row
    return out


def reconstruct_sequences(cfg, cached_rows, salvage):
    """Return list of dicts with full_ids (LongTensor), prompt_len, response_ids, advantage/targets if available."""
    tokenizer = load_tokenizer(cfg["base_model"])
    eval_rows = {row.get("prompt_id"): row for row in read_jsonl(cfg["paths"]["rl_prompt_eval"])}
    eval_by_index = {row.get("source_index"): row for row in eval_rows.values()}

    out = []
    used_salvage = 0
    for row in cached_rows:
        key = (row.get("source_index"), str(row.get("response", ""))[:64])
        s = salvage.get(key)
        full_ids = None
        prompt_len = None
        returns = None
        if s is not None and s.get("full_ids") is not None:
            full_ids = torch.as_tensor(s["full_ids"], dtype=torch.long)
            prompt_len = int(s.get("prompt_len") or s.get("prompt_length") or 0)
            returns = s.get("returns")
            used_salvage += 1
        if full_ids is None or prompt_len in (None, 0):
            er = eval_by_index.get(row.get("source_index"))
            prompt_text = er.get("prompt") if er else None
            if prompt_text is None:
                raise ValueError(f"cannot reconstruct prompt for source_index={row.get('source_index')}")
            prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            resp_ids = tokenizer(str(row["response"]), add_special_tokens=False)["input_ids"]
            full_ids = torch.tensor(list(prompt_ids) + list(resp_ids), dtype=torch.long)
            prompt_len = len(prompt_ids)
        if returns is None:
            returns = s.get("returns") if s else None
        out.append(
            {
                "source_index": row.get("source_index"),
                "prompt_id": row.get("prompt_id"),
                "full_ids": full_ids,
                "prompt_len": prompt_len,
                "response_ids": full_ids[prompt_len:],
                "old_logprobs": row["old_logprobs"].float(),
                "ref_logprobs": row["ref_logprobs"].float(),
                "values": row.get("values").float() if row.get("values") is not None else None,
                "effective_terminal_reward": row.get("effective_terminal_reward"),
                "returns": torch.as_tensor(returns, dtype=torch.float32) if returns is not None else None,
            }
        )
    print(f"reconstructed {len(out)} cached rollouts ({used_salvage} from salvage full_ids)", flush=True)
    return out


@torch.no_grad()
def rescore_sequences(cfg, sequences, adapter):
    """Return list of new logprobs under the given adapter; also report mismatch vs stored old values."""
    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(cfg, adapter_path=adapter, trainable=False)
    device = next(policy.parameters()).device
    new_logps = []
    for item in sequences:
        ids = item["full_ids"].unsqueeze(0).to(device)
        attn = torch.ones_like(ids)
        rid = item["response_ids"].unsqueeze(0).to(device)
        pw = item["prompt_len"]
        logp, _ = response_token_logprobs(policy, ids, attn, pw, rid)
        new_logps.append(logp[0].float().cpu())
    clear_gpu(policy)
    return new_logps


def cached_batch_clip_analysis(cfg, eps_values, adapters: dict[str, str]):
    cached_rows = load_cached_rollouts(cfg["cached_rollouts"])
    salvage = load_salvage("cached/ppo_rollout_salvage_raw.pt")
    sequences = reconstruct_sequences(cfg, cached_rows, salvage)

    # sanity: token lengths vs stored log-prob lengths
    len_counts = {}
    for item in sequences:
        pair = (len(item["response_ids"]), len(item["old_logprobs"]))
        len_counts[pair] = len_counts.get(pair, 0) + 1
    print("(response_len, stored_logp_len) histogram:", len_counts, flush=True)

    report = {"num_rollouts": len(sequences), "adapter_analyses": {}, "length_histogram": {str(k): v for k, v in len_counts.items()}}

    for tag, adapter in adapters.items():
        if adapter and not repo_path(adapter).exists():
            print(f"skipping adapter {tag}: {adapter} not found", flush=True)
            continue
        new_logps = rescore_sequences(cfg, sequences, adapter)

        # verification: recomputed vs stored old log-probs
        diffs = []
        for item, nl in zip(sequences, new_logps):
            m = min(len(nl), len(item["old_logprobs"]))
            diffs.append(float((nl[:m] - item["old_logprobs"][:m]).abs().mean()))
        mean_abs_diff = statistics.fmean(diffs) if diffs else None

        eps_table = {}
        for eps in eps_values:
            ratios, affected, surrogates = [], [], []
            for item, nl in zip(sequences, new_logps):
                m = min(len(nl), len(item["old_logprobs"]))
                if m == 0:
                    continue
                ratio = torch.exp(nl[:m] - item["old_logprobs"][:m])
                if item["returns"] is not None and item["values"] is not None:
                    mv = min(m, len(item["returns"]), len(item["values"]))
                    adv = (item["returns"][:mv] - item["values"][:mv]).clamp(-10.0, 10.0)
                    ratio_ = ratio[:mv]
                    s1 = ratio_ * adv
                    s2 = ratio_.clamp(1.0 - eps, 1.0 + eps) * adv
                    surrogates.append(float(torch.minimum(s1, s2).mean()))
                ratios.append(ratio)
                affected.append((ratio < (1.0 - eps)) | (ratio > (1.0 + eps)))
            ratio_all = torch.cat(ratios) if ratios else torch.tensor([])
            aff_all = torch.cat(affected) if affected else torch.tensor([], dtype=torch.bool)
            eps_table[str(eps)] = {
                "mean_ratio": float(ratio_all.mean()) if ratio_all.numel() else None,
                "std_ratio": float(ratio_all.std(unbiased=False)) if ratio_all.numel() else None,
                "min_ratio": float(ratio_all.min()) if ratio_all.numel() else None,
                "max_ratio": float(ratio_all.max()) if ratio_all.numel() else None,
                "clip_fraction": float(aff_all.float().mean()) if aff_all.numel() else None,
                "affected_token_fraction": float(aff_all.float().mean()) if aff_all.numel() else None,
                "mean_clipped_surrogate": statistics.fmean(surrogates) if surrogates else None,
            }
        report["adapter_analyses"][tag] = {
            "adapter": str(adapter),
            "mean_abs_logp_diff_vs_stored_old": mean_abs_diff,
            "epsilon_table": eps_table,
        }
        print(f"cache analysis vs {tag}: mean|dlogp|={mean_abs_diff:.6f}", flush=True)
        for e, vals in eps_table.items():
            print(f"  eps={e}: clip_frac={vals['clip_fraction']:.4f} affected={vals['affected_token_fraction']:.4f} "
                  f"mean_ratio={vals['mean_ratio']:.4f} surrogate={vals['mean_clipped_surrogate']}", flush=True)

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
