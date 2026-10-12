"""Direct length-conditioned gradient diagnostic (Task 3, "S3c").

This script reproduces the analysis executed as the Kaggle kernel
`mbh2006/atml-pa2-s3c-grad-diag` (stage folder `kaggle/session3c_grad_diag/`,
outside this repository). For every SAVED completion of each normalization
fork it computes

    g_k = || grad_theta  log pi_theta(y_k | x) ||_2      (over trainable LoRA params)

on the fixed saved batch (no training, no generation), the per-completion
objective weights implied by each normalization

    canonical GRPO contribution ~ |A_k| * g_k / T_k
    Dr. GRPO       contribution ~ |A_k| * g_k

and reports per-fork log-log scaling (alpha in ||g|| ~ T^alpha), Pearson
correlations, and length-quartile binned means. Output:
`results/task3_grpo/grad_length_diag.json` (the file the report cites).

Requirements: GPU + the saved fork adapters and rollouts, e.g. fetched from
the S3b kernel output or reproduced by `python -m task3_grpo.compare_normalization`
(adapter + rollouts paths are read from the config/outputs layout below).

Usage:
    python -m task3_grpo.grad_length_diag --config configs/grpo.yaml
"""

from __future__ import annotations

import argparse
import math
import sys

import numpy as np

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.models import load_policy, load_tokenizer, trainable_parameters


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    denom = math.sqrt(sxx * syy)
    return sxy / denom if denom > 0 else None


def loglog_slope(xs, ys):
    lx = [math.log(x) for x in xs]
    ly = [math.log(y) for y in ys]
    n = len(lx)
    mx = sum(lx) / n
    my = sum(ly) / n
    sxx = sum((x - mx) ** 2 for x in lx)
    if sxx <= 0:
        return None, None
    slope = sum((x - mx) * (y - my) for x, y in zip(lx, ly)) / sxx
    return slope, my - slope * mx


def quartile_bins(recs, nbins=4):
    recs = sorted(recs, key=lambda r: r["tokens"])
    n = len(recs)
    edges = [round(i * n / nbins) for i in range(nbins + 1)]
    out = []
    for b in range(nbins):
        chunk = recs[edges[b]:edges[b + 1]]
        if not chunk:
            continue
        out.append({
            "bin": b + 1,
            "n": len(chunk),
            "tokens_mean": sum(r["tokens"] for r in chunk) / len(chunk),
            "grad_norm_mean": sum(r["grad_norm"] for r in chunk) / len(chunk),
            "canonical_contrib_mean": sum(r["grad_norm"] / r["tokens"] for r in chunk) / len(chunk),
            "dr_contrib_mean": sum(r["grad_norm"] for r in chunk) / len(chunk),
            "adv_weighted_canonical_mean": sum(r["adv_weighted_grad"] / r["tokens"] for r in chunk) / len(chunk),
            "adv_weighted_dr_mean": sum(r["adv_weighted_grad"] for r in chunk) / len(chunk),
        })
    return out


def run_diagnostic(config_path: str):
    import json

    import torch

    cfg = load_yaml(config_path)
    tok = load_tokenizer(cfg["base_model"])
    prompt_rows = read_jsonl(cfg["paths"]["rl_prompt_train"])
    out_dir = repo_path(cfg["outputs_dir"]) / "task3_grpo"
    results_dir = repo_path(cfg["results_dir"]) / "task3_grpo"
    results_dir.mkdir(parents=True, exist_ok=True)

    report = {}
    for tag, loss_tag in [("norm_grpo", "grpo"), ("norm_dr_grpo", "dr_grpo")]:
        policy = load_policy(cfg, adapter_path=str(out_dir / tag), trainable=True)
        params = trainable_parameters(policy)
        rows = read_jsonl(out_dir / tag / f"rollouts_{loss_tag}.jsonl")
        recs = []
        for row in rows:
            i = int(row["update"]) - 1
            messages = prompt_messages(prompt_rows[i % len(prompt_rows)])
            rewards = np.asarray(row["rewards"], dtype=float)
            std = float(rewards.std())
            adv = (rewards - rewards.mean()) / (std + 1e-6) if std > 1e-6 else np.zeros_like(rewards)
            prompt_ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
            pw = len(prompt_ids)
            for j, resp in enumerate(row["responses"]):
                resp_ids = tok(resp, add_special_tokens=False)["input_ids"]
                if len(resp_ids) == 0:
                    continue
                ids = torch.tensor([prompt_ids + resp_ids], device="cuda")
                attn = torch.ones_like(ids)
                out = policy(input_ids=ids, attention_mask=attn, use_cache=False)
                logits = out.logits[:, pw - 1:-1, :].float()
                logp = torch.log_softmax(logits, dim=-1)
                tgt = torch.tensor([resp_ids], device="cuda")
                seq_logp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).sum()
                policy.zero_grad(set_to_none=True)
                seq_logp.backward()
                g2 = 0.0
                for p in params:
                    if p.grad is not None:
                        g2 += float(p.grad.detach().float().pow(2).sum())
                gnorm = math.sqrt(g2)
                policy.zero_grad(set_to_none=True)
                recs.append({
                    "update": int(row["update"]), "k": j, "tokens": len(resp_ids),
                    "truncated": bool(row["truncated"][j]), "reward": float(rewards[j]),
                    "advantage": float(adv[j]), "grad_norm": gnorm,
                    "adv_weighted_grad": abs(float(adv[j])) * gnorm,
                })
            print(f"[{tag}] update {row['update']} done", flush=True)

        sub = [r for r in recs if not r["truncated"]]
        stats = {"tag": tag, "n": len(recs), "n_nontruncated": len(sub)}
        if len(sub) >= 3:
            stats["pearson_logT_logG"] = pearson([math.log(r["tokens"]) for r in sub], [math.log(r["grad_norm"]) for r in sub])
            slope, intercept = loglog_slope([r["tokens"] for r in sub], [r["grad_norm"] for r in sub])
            stats["loglog_slope_alpha"] = slope
            stats["loglog_intercept"] = intercept
            stats["pearson_T_canonical_contrib"] = pearson([r["tokens"] for r in sub], [r["grad_norm"] / r["tokens"] for r in sub])
            stats["pearson_T_dr_contrib"] = pearson([r["tokens"] for r in sub], [r["grad_norm"] for r in sub])
            stats["pearson_T_adv_canonical_contrib"] = pearson([r["tokens"] for r in sub], [r["adv_weighted_grad"] / r["tokens"] for r in sub])
            stats["pearson_T_adv_dr_contrib"] = pearson([r["tokens"] for r in sub], [r["adv_weighted_grad"] for r in sub])
            stats["quartile_bins"] = quartile_bins(sub)
        report[tag] = {"stats": stats, "completions": recs}
        del policy
        torch.cuda.empty_cache()

    out_path = results_dir / "grad_length_diag.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("saved ->", out_path, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    args = ap.parse_args()
    run_diagnostic(args.config)


if __name__ == "__main__":
    main()
