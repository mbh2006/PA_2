"""Task 2 (PPO) analysis: required tables + figures from saved results.

Outputs:
  results/task2_ppo/summary_tables.md
  results/figures/task2/ppo_trajectories.{png,pdf}
  results/figures/task2/ppo_clip_study.{png,pdf}
  results/figures/task2/ppo_kl_study.{png,pdf}

Run from repo root (CPU only):
  python analysis/task2_plots.py
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from common.data import load_yaml, repo_path  # noqa: E402


def load_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(x, nd=3):
    if x is None:
        return "—"
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, (int, float)):
        return f"{x:.{nd}f}" if isinstance(x, float) else str(x)
    return str(x)


def save(fig, outbase: Path):
    fig.tight_layout()
    fig.savefig(outbase.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(outbase.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def fig_trajectories(ppo: dict, outbase: Path):
    hist = (ppo or {}).get("history", [])
    if not hist:
        return False
    steps = [h["update"] for h in hist]
    panels = [
        ("mean learned reward", [h["effective_reward"] for h in hist]),
        ("KL from reference", [h["kl"] for h in hist]),
        ("policy loss", [h["policy_loss"] for h in hist]),
        ("value loss", [h["value_loss"] for h in hist]),
        ("entropy (sampled)", [h["entropy"] for h in hist]),
        ("clip fraction", [h["clip_fraction"] for h in hist]),
        ("policy grad norm", [h["grad_norm_policy"] for h in hist]),
        ("response length (tokens)", [h["response_tokens"] for h in hist]),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(18, 7))
    for ax, (title, ys) in zip(axes.flat, panels):
        ax.plot(steps, ys, marker="o", ms=3)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("update")
        ax.grid(alpha=0.3)
    fig.suptitle("Standard PPO continuation from the supplied midpoint (20 updates)")
    save(fig, outbase)
    return True


def fig_clip_study(cache: dict, study: dict, outbase: Path):
    batches = (cache or {}).get("batches", {})
    forks = (study or {}).get("forks", [])
    if not batches and not forks:
        return False

    ncols = (1 if batches else 0) + (1 if forks else 0)
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 4))
    axes = [axes] if ncols == 1 else list(axes.flat)
    idx = 0

    if batches:
        ax = axes[idx]
        idx += 1
        for bname, brep in batches.items():
            for tag, analysis in brep.get("adapter_analyses", {}).items():
                eps = sorted(float(e) for e in analysis["epsilon_table"])
                clip = [analysis["epsilon_table"][f"{e:g}"]["clip_fraction"] for e in eps]
                style = "o-" if tag == "midpoint" else "s--"
                ax.plot(eps, clip, style, label=f"{bname}: {tag}")
        ax.set_xlabel("clip ε")
        ax.set_ylabel("clip fraction")
        ax.set_title("Cached-batch clipping geometry")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    if forks:
        ax = axes[idx]
        eps = [f["clip_epsilon"] for f in forks]
        reward = [f["heldout_reward"] for f in forks]
        kl = [f["heldout_kl"] for f in forks]
        length = [f["heldout_length"] for f in forks]
        ax.plot(eps, reward, "o-", label="held-out reward")
        ax.plot(eps, kl, "s-", label="held-out KL")
        ax2 = ax.twinx()
        ax2.plot(eps, length, "^--", color="tab:green", label="held-out length")
        ax.set_xlabel("clip ε")
        ax.set_title("Matched short forks (8 updates)")
        ax.legend(fontsize=7, loc="upper left")
        ax2.legend(fontsize=7, loc="upper right")
        ax.grid(alpha=0.3)

    fig.suptitle("PPO clipping study (ε sweep)", y=1.02)
    save(fig, outbase)
    return True


def fig_kl_study(study: dict, outbase: Path, kl_evals: dict | None = None):
    kl_evals = kl_evals or {}
    conds = (study or {}).get("conditions", [])
    if not conds:
        return False
    betas = [c["kl_beta"] for c in conds]
    ho_ent = [kl_evals.get(c.get("tag", ""), {}).get("entropy_sampled_response_mean") for c in conds]
    use_ho = all(v is not None for v in ho_ent)
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.6))
    panels = [
        ("held-out reward", [c["heldout_reward"] for c in conds]),
        ("held-out KL", [c["heldout_kl"] for c in conds]),
        ("held-out entropy" if use_ho else "training entropy", ho_ent if use_ho else [c["final_entropy"] for c in conds]),
        ("held-out length", [c["heldout_length"] for c in conds]),
    ]
    for ax, (title, ys) in zip(axes, panels):
        ax.plot(betas, ys, "o-")
        ax.set_xlabel("KL β")
        ax.set_title(title)
        ax.grid(alpha=0.3)
    fig.suptitle("PPO KL-pressure study (matched 8-update forks)", y=1.03)
    save(fig, outbase)
    return True


def tables_md(ppo: dict, cache: dict, study: dict, kl_study: dict, eval_std: dict, kl_evals: dict | None = None):
    kl_evals = kl_evals or {}
    lines = ["# Task 2 — auto-generated result tables", ""]

    if ppo:
        h = ppo.get("history", [])
        lines += [
            "## 1. Standard PPO continuation (20 updates)",
            "",
            f"- wall-clock: **{fmt(ppo.get('wall_clock_seconds'), 1)} s**, peak VRAM: **{fmt(ppo.get('peak_vram_gib'), 2)} GiB**",
            f"- mean reward over last 5 updates: **{fmt(ppo.get('mean_reward_last5'))}**",
            "",
            "| update | reward | KL | policy loss | value loss | entropy | clip frac | grad norm | length |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for r in h:
            lines.append(
                f"| {r['update']} | {fmt(r['effective_reward'])} | {fmt(r['kl'], 4)} | {fmt(r['policy_loss'], 4)} | "
                f"{fmt(r['value_loss'], 4)} | {fmt(r['entropy'], 3)} | {fmt(r['clip_fraction'])} | "
                f"{fmt(r['grad_norm_policy'], 3)} | {r['response_tokens']} |"
            )
        lines.append("")
    if eval_std:
        lines += [
            "## 2. Held-out evaluation (standard adapter)",
            "",
            f"- reward mean: {fmt(eval_std.get('reward_model_score_mean'))} (± {fmt(eval_std.get('reward_model_score_std'))})",
            f"- KL (sampled): {fmt(eval_std.get('kl_sampled_response_estimator'), 5)}",
            f"- length: {fmt(eval_std.get('response_length_mean'), 1)} (± {fmt(eval_std.get('response_length_std'), 1)})",
            f"- prompts: {eval_std.get('num_prompts')}, truncated: {fmt(eval_std.get('truncated_fraction'))}",
            "",
        ]

    batches = (cache or {}).get("batches", {})
    if batches:
        lines += ["## 3. Cached-batch clipping geometry (two supplied batches)", ""]
        lines.append(f"_{cache.get('note', '')}_")
        lines.append("")
        for bname, brep in batches.items():
            lines.append(f"### Batch: {bname} ({brep.get('num_rollouts')} rollouts; source indices in eval pool: {brep.get('source_indices_in_eval_pool')})")
            lines.append("")
            for tag, analysis in brep.get("adapter_analyses", {}).items():
                lines.append(
                    f"**Adapter: {tag}** — mean |Δlogp| vs stored old: {fmt(analysis.get('mean_abs_logp_diff_vs_stored_old'), 6)}"
                )
                lines.append("")
                lines.append("| ε | mean ratio | std ratio | min ratio | max ratio | clip fraction | affected fraction | mean clipped surrogate |")
                lines.append("|---|---|---|---|---|---|---|---|")
                for e, vals in analysis["epsilon_table"].items():
                    lines.append(
                        f"| {e} | {fmt(vals['mean_ratio'], 4)} | {fmt(vals['std_ratio'], 4)} | "
                        f"{vals['min_ratio']:.3g} | {fmt(vals['max_ratio'], 3)} | {fmt(vals['clip_fraction'])} | "
                        f"{fmt(vals['affected_token_fraction'])} | {fmt(vals['mean_clipped_surrogate'], 4)} |"
                    )
                lines.append("")
    if (study or {}).get("forks"):
        lines += ["## 4. Matched short forks (8 updates)", ""]
        lines.append("| ε | held-out reward | held-out KL | held-out length | clip-frac std | grad-norm std | final ratio dev |")
        lines.append("|---|---|---|---|---|---|---|")
        for f in study["forks"]:
            s = f.get("stability", {})
            lines.append(
                f"| {fmt(f['clip_epsilon'], 2)} | {fmt(f['heldout_reward'])} | {fmt(f['heldout_kl'], 5)} | "
                f"{fmt(f['heldout_length'], 1)} | {fmt(s.get('clip_fraction_std'))} | "
                f"{fmt(s.get('grad_norm_policy_std'))} | {fmt(s.get('ratio_deviation_final'))} |"
            )
        lines.append("")
    if kl_study and kl_study.get("conditions"):
        lines += ["## 5. KL-pressure study (8-update forks)", ""]
        lines.append("_`final *` columns are training-side (mean of last three updates); `held-out *` columns are from the fixed 100-prompt evaluation protocol (manual requirement: held-out reward, KL, **entropy**, length). Entropy values reported are the **full token-level policy entropy** H_t = −Σ_v π log π over the whole vocabulary (the course-helper sampled-token negative log-probability proxy is retained alongside as `entropy_sampled_response_mean`)._")
        lines.append("")
        lines.append("| β_KL | final reward | final KL | final entropy (train) | final length | held-out reward | held-out KL | **held-out entropy** | held-out length |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for c in kl_study["conditions"]:
            ev = kl_evals.get(c.get("tag", ""), {})
            ho_ent = ev.get("entropy_full_vocab_mean", ev.get("entropy_sampled_response_mean"))
            lines.append(
                f"| {fmt(c['kl_beta'], 2)} | {fmt(c['final_mean_reward'])} | {fmt(c['final_kl'], 5)} | "
                f"{fmt(c['final_entropy'], 3)} | {fmt(c['final_length'], 1)} | {fmt(c['heldout_reward'])} | "
                f"{fmt(c['heldout_kl'], 5)} | {fmt(ho_ent, 3)} | {fmt(c['heldout_length'], 1)} |"
            )
        lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    results_dir = repo_path(cfg["results_dir"])
    fig_dir = repo_path("results") / "figures" / "task2"
    fig_dir.mkdir(parents=True, exist_ok=True)

    ppo = load_json(results_dir / "ppo_standard.json")
    cache = load_json(results_dir / "clip_study_cache.json")
    study = load_json(results_dir / "clip_study.json")
    kl_study = load_json(results_dir / "kl_study.json")
    eval_std = load_json(results_dir / "eval_standard.json")
    kl_evals = {}
    for path in sorted(results_dir.glob("eval_kl_*.json")):
        payload = load_json(path)
        if payload:
            kl_evals[path.stem[len("eval_"):]] = payload  # keys: kl_0, kl_0p1, kl_0p2

    md = tables_md(ppo, cache, study, kl_study, eval_std, kl_evals)
    (results_dir / "summary_tables.md").write_text(md, encoding="utf-8")
    print(md)

    made = []
    if fig_trajectories(ppo, fig_dir / "ppo_trajectories"):
        made.append("ppo_trajectories")
    if fig_clip_study(cache, study, fig_dir / "ppo_clip_study"):
        made.append("ppo_clip_study")
    if fig_kl_study(kl_study, fig_dir / "ppo_kl_study", kl_evals):
        made.append("ppo_kl_study")
    print("figures written:", made)


if __name__ == "__main__":
    main()
