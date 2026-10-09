"""Task 3 group-size study (no policy training required).

Uses the supplied K=8 completion/reward cache. For K in {2, 4, 8}, keep the total
number of cached completions fixed: each prompt's 8 completions are partitioned
into (8/K) disjoint consecutive groups, so every completion appears exactly once
and the total generation budget is identical across K. All quantities are
computed with a single documented binning rule: prompt-difficulty bins are
terciles of per-prompt mean reward over the same fixed cache (easy = top tercile).

Metrics per K: informative-group rate (group reward std > 1e-6), mean within-group
reward std, pooled variance of the group-relative signal, and the same quantities
per difficulty bin.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict

import numpy as np

from common.data import load_yaml, read_jsonl, repo_path

GROUP_STD_TOL = 1e-6


def load_k8_cache(path):
    rows = read_jsonl(path)
    by_prompt = defaultdict(list)
    for row in rows:
        by_prompt[str(row["source_index"])].append(row)
    bad = {pid: len(group) for pid, group in by_prompt.items() if len(group) < 8}
    if bad:
        raise ValueError(f"Expected at least K=8 cached completions per prompt; short groups: {bad}")
    for group in by_prompt.values():
        group.sort(key=lambda x: int(x.get("generation_index", 0)))
    return by_prompt


def regroup_equal_generation_budget(by_prompt, k: int):
    """Partition each prompt's 8 completions into disjoint consecutive groups of size k.

    Returns list of (prompt_id, [rewards]) groups. Total completions stays constant across K.
    """
    if 8 % k != 0 or k <= 0:
        raise ValueError(f"K={k} must divide the cached K=8 completions exactly")
    groups = []
    for pid, rows in by_prompt.items():
        rows = rows[:8]
        for start in range(0, 8, k):
            rewards = [float(r["reward"]) for r in rows[start : start + k]]
            groups.append((pid, rewards))
    return groups


def group_stats(groups):
    informative, stds, all_adv = [], [], []
    for pid, rewards in groups:
        if len(rewards) < 2:
            std = 0.0
        else:
            std = float(np.std(rewards))
        informative.append(std > GROUP_STD_TOL)
        stds.append(std)
        if len(rewards) > 1 and std > GROUP_STD_TOL:
            mean = float(np.mean(rewards))
            all_adv.extend([(r - mean) / std for r in rewards])
        elif len(rewards) > 1:
            all_adv.extend([0.0] * len(rewards))
    return {
        "num_groups": len(groups),
        "informative_group_rate": float(np.mean(informative)) if informative else None,
        "mean_within_group_reward_std": float(np.mean(stds)) if stds else None,
        "variance_of_group_relative_signal": float(np.var(all_adv)) if all_adv else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    by_prompt = load_k8_cache(cfg["group_cache"])
    print("Cached prompts:", len(by_prompt))
    print("Group sizes to analyze:", cfg["group_sizes"])

    first = next(iter(by_prompt.values()))
    print("Cache row keys:", sorted(first[0].keys()))

    # One documented binning rule: terciles of per-prompt mean reward over the fixed cache.
    prompt_mean_reward = {pid: float(np.mean([float(r["reward"]) for r in rows])) for pid, rows in by_prompt.items()}
    values = sorted(prompt_mean_reward.values())
    t1, t2 = np.quantile(values, [1 / 3, 2 / 3])
    difficulty = {}
    for pid, mean_reward in prompt_mean_reward.items():
        if mean_reward >= t2:
            difficulty[pid] = "easy"
        elif mean_reward >= t1:
            difficulty[pid] = "medium"
        else:
            difficulty[pid] = "hard"
    difficulty_shares = {
        name: sum(1 for v in difficulty.values() if v == name) for name in ["easy", "medium", "hard"]
    }

    study = {
        "cache_prompts": len(by_prompt),
        "partition_rule": "each prompt's 8 cached completions split into disjoint consecutive groups of size K; total completions fixed",
        "binning_rule": "terciles of per-prompt mean reward over the fixed K=8 cache (easy = top tercile)",
        "bin_thresholds": {"t1": float(t1), "t2": float(t2)},
        "bin_prompt_counts": difficulty_shares,
        "results": {},
    }

    for k in cfg["group_sizes"]:
        k = int(k)
        groups = regroup_equal_generation_budget(by_prompt, k)
        overall = group_stats(groups)
        per_bin = {}
        for name in ["easy", "medium", "hard"]:
            bin_groups = [(pid, rewards) for pid, rewards in groups if difficulty[pid] == name]
            per_bin[name] = group_stats(bin_groups) if bin_groups else None
        study["results"][f"K={k}"] = {"overall": overall, "per_difficulty": per_bin}
        print(f"K={k}: overall={overall}")
        for name, stats in per_bin.items():
            if stats:
                print(f"   {name:6s}: rate={stats['informative_group_rate']:.3f} "
                      f"std={stats['mean_within_group_reward_std']:.3f} "
                      f"signal_var={stats['variance_of_group_relative_signal']:.3f} (n={stats['num_groups']})")

    results_dir = repo_path(cfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "group_size_study.json").write_text(json.dumps(study, indent=2), encoding="utf-8")
    print("saved ->", results_dir / "group_size_study.json")


if __name__ == "__main__":
    main()
