"""Task 3 (GRPO) analysis: required tables + figures from saved results.

Outputs:
  results/task3_grpo/summary_tables.md
  results/figures/task3/grpo_trajectories.{png,pdf}
  results/figures/task3/grpo_group_size.{png,pdf}
  results/figures/task3/grpo_normalization.{png,pdf}

Run from repo root (CPU only):
  python analysis/task3_plots.py
"""

from __future__ import annotations

import argparse
import json
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


def fig_trajectories(grpo: dict, outbase: Path):
    hist = (grpo or {}).get("history", [])
    if not hist:
        return False
    steps = [h["update"] for h in hist]
    panels = [
        ("mean reward", [h["mean_reward"] for h in hist]),
        ("within-group reward std", [h["group_reward_std"] for h in hist]),
        ("KL from reference", [h["kl"] for h in hist]),
        ("policy loss", [h["policy_loss"] for h in hist]),
        ("policy grad norm", [h["grad_norm_policy"] for h in hist]),
        ("entropy (sampled)", [h["entropy"] for h in hist]),
        ("clip fraction", [h["clip_fraction"] for h in hist]),
        ("mean response length", [h["response_tokens_mean"] for h in hist]),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(18, 7))
    for ax, (title, ys) in zip(axes.flat, panels):
        ax.plot(steps, ys, marker="o", ms=3)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("update")
        ax.grid(alpha=0.3)
    fig.suptitle(f"Standard GRPO continuation (K={(grpo or {}).get('num_generations', '?')}, 20 updates)")
    save(fig, outbase)
    return True


def fig_group_size(study: dict, outbase: Path):
    results = (study or {}).get("results", {})
    if not results:
        return False
    ks = sorted(int(k.split("=")[1]) for k in results)
    overall_rate = [results[f"K={k}"]["overall"]["informative_group_rate"] for k in ks]
    overall_std = [results[f"K={k}"]["overall"]["mean_within_group_reward_std"] for k in ks]
    overall_var = [results[f"K={k}"]["overall"]["variance_of_group_relative_signal"] for k in ks]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(ks, overall_rate, "o-", label="informative-group rate")
    ax1.plot(ks, overall_var, "s--", label="variance of group-relative signal")
    ax1.set_xlabel("group size K")
    ax1.set_ylim(0, 1.05)
    ax1.set_title("Equal-generation group-size study (overall)")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    colors = {"easy": "tab:green", "medium": "tab:orange", "hard": "tab:red"}
    for name in ["easy", "medium", "hard"]:
        rate = [results[f"K={k}"]["per_difficulty"][name]["informative_group_rate"] for k in ks]
        ax2.plot(ks, rate, "o-", color=colors[name], label=name)
    ax2.set_xlabel("group size K")
    ax2.set_ylim(0, 1.05)
    ax2.set_title("Informative-group rate by prompt difficulty")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    save(fig, outbase)
    return True


def fig_normalization(study: dict, outbase: Path):
    results = (study or {}).get("results", {})
    if "grpo" not in results or "dr_grpo" not in results:
        return False
    tags = ["grpo", "dr_grpo"]
    labels = ["canonical GRPO", "Dr. GRPO"]
    metrics = [
        ("held-out reward", [results[t]["heldout_reward"] for t in tags]),
        ("held-out KL", [results[t]["heldout_kl"] for t in tags]),
        ("held-out length", [results[t]["heldout_length"] for t in tags]),
        ("corr(length, grad norm)", [results[t]["length_conditioned"]["corr_length_vs_grad_norm"] for t in tags]),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.6))
    for ax, (title, ys) in zip(axes, metrics):
        ax.bar(labels, ys, color=["tab:blue", "tab:orange"])
        ax.set_title(title, fontsize=10)
        ax.tick_params(axis="x", labelrotation=10)
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle("Length-normalization study (matched short continuations)", y=1.03)
    save(fig, outbase)
    return True


def tables_md(grpo: dict, study: dict, norm: dict, eval_std: dict, diag: dict | None = None):
    lines = ["# Task 3 — auto-generated result tables", ""]
    if grpo:
        h = grpo.get("history", [])
        lines += [
            "## 1. Standard GRPO continuation (20 updates, K=4)",
            "",
            f"- wall-clock: **{fmt(grpo.get('wall_clock_seconds'), 1)} s**, peak VRAM: **{fmt(grpo.get('peak_vram_gib'), 2)} GiB**",
            f"- uninformative-group rate over the run: **{fmt(grpo.get('uninformative_group_rate'))}**",
            "",
            "| update | reward | group std | uninf. | KL | policy loss | grad norm | entropy | length |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for r in h:
            ent = "undefined" if (r["entropy"] != r["entropy"]) else fmt(r["entropy"], 3)  # NaN = empty training mask
            lines.append(
                f"| {r['update']} | {fmt(r['mean_reward'])} | {fmt(r['group_reward_std'])} | "
                f"{'yes' if r['uninformative_group'] else 'no'} | {fmt(r['kl'], 4)} | {fmt(r['policy_loss'], 4)} | "
                f"{fmt(r['grad_norm_policy'], 3)} | {ent} | {fmt(r['response_tokens_mean'], 0)} |"
            )
        lines.append("")
        lines.append("_Column notes: 'uninf.' = reward-variance informativeness (group std <= 1e-6), **not** usable training signal. Updates 13 and 17 have all four completions truncated, so the released truncation mask empties the training batch: zero policy gradient (grad norm 0.000, policy loss 0.000) and **entropy undefined** (NaN = empty mask). The entropy diagnostic covers valid (unmasked) response tokens only and has no support when the mask is empty. Hence 0% uninformative reward groups does not mean every update provides training signal: 2/20 updates provided none._")
        lines.append("")
    if eval_std:
        lines += [
            "## 2. Held-out evaluation (standard adapter)",
            "",
            f"- reward mean: {fmt(eval_std.get('reward_model_score_mean'))} (± {fmt(eval_std.get('reward_model_score_std'))})",
            f"- KL (sampled): {fmt(eval_std.get('kl_sampled_response_estimator'), 5)}",
            f"- length: {fmt(eval_std.get('response_length_mean'), 1)} (± {fmt(eval_std.get('response_length_std'), 1)})",
            "",
        ]
    if (study or {}).get("results"):
        lines += ["## 3. Group-size study (equal total generations)", ""]
        rows = study["results"]
        ks = sorted(int(k.split("=")[1]) for k in rows)
        lines.append("| K | groups | informative rate | mean within-group std | signal variance |")
        lines.append("|---|---|---|---|---|")
        for k in ks:
            o = rows[f"K={k}"]["overall"]
            lines.append(
                f"| {k} | {o['num_groups']} | {fmt(o['informative_group_rate'])} | "
                f"{fmt(o['mean_within_group_reward_std'])} | {fmt(o['variance_of_group_relative_signal'])} |"
            )
        lines.append("")
        lines.append("### Per difficulty bin")
        lines.append("")
        lines.append("| difficulty | " + " | ".join(f"K={k}" for k in ks) + " |")
        lines.append("|---" * (len(ks) + 1) + "|")
        for name in ["easy", "medium", "hard"]:
            cells = []
            for k in ks:
                b = rows[f"K={k}"]["per_difficulty"][name]
                cells.append(
                    f"{fmt(b['informative_group_rate'])} ({fmt(b['mean_within_group_reward_std'])}; "
                    f"var {fmt(b['variance_of_group_relative_signal'])})"
                )
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
        lines.append("")
    if (norm or {}).get("results"):
        lines += ["## 4. Canonical vs Dr. GRPO (matched short forks)", ""]
        lines.append("| condition | held-out reward | held-out KL | held-out length | corr(length, grad norm) |")
        lines.append("|---|---|---|---|---|")
        for key, r in norm["results"].items():
            lc = r.get("length_conditioned", {})
            lines.append(
                f"| {key} | {fmt(r['heldout_reward'])} | {fmt(r['heldout_kl'], 5)} | {fmt(r['heldout_length'], 1)} | "
                f"{fmt(lc.get('corr_length_vs_grad_norm'))} |"
            )
        lines.append("")
        lines.append("_Realized generated tokens (per audit): canonical **8,953** vs Dr. GRPO **10,109** (+12.9%). Update counts and caps are equal by the released short-run spec; realized tokens differ because Dr. GRPO's responses are longer - itself part of the effect under study (forcing realized-token equality would mean truncating completions and departing from the released generation settings). Disclosed, not assumed equal._")
        lines.append("")
    if diag:
        lines += ["### 4b. Direct length-conditioned gradient diagnostic (fixed saved batch, S3c)", ""]
        lines.append("_Per-completion gradient norms of log pi(y|x) over LoRA params measured on each fork's SAVED rollouts (no training, no generation). alpha = log-log slope of ‖g‖ vs completion length T; canonical contributions scale with ‖g‖/T (sequence mean), Dr. GRPO with ‖g‖ (constant). Non-truncated completions only (training-relevant). n per fork is small - treat as a fixed-batch diagnostic, not a law._")
        lines.append("")
        lines.append("| fork | n (non-trunc) | alpha (‖g‖ ~ T^alpha) | r(log T, log ‖g‖) | r(T, ‖g‖) [Dr.] | r(T, ‖g‖/T) [canonical] |")
        lines.append("|---|---|---|---|---|---|")
        for tag, d in diag.items():
            s = d.get("stats", {})
            lines.append(
                f"| {tag} | {s.get('n_nontruncated')} | {fmt(s.get('loglog_slope_alpha'), 2)} | "
                f"{fmt(s.get('pearson_logT_logG'), 2)} | {fmt(s.get('pearson_T_dr_contrib'), 2)} | "
                f"{fmt(s.get('pearson_T_canonical_contrib'), 2)} |"
            )
        lines.append("")
        lines.append("| fork | length quartile | tokens (mean) | ‖g‖ mean | ‖g‖/T mean (canonical) | |A|*‖g‖ mean (Dr.) |")
        lines.append("|---|---|---|---|---|---|")
        for tag, d in diag.items():
            for b in d.get("stats", {}).get("quartile_bins", []):
                lines.append(
                    f"| {tag} | Q{b['bin']} | {fmt(b['tokens_mean'], 0)} | {fmt(b['grad_norm_mean'])} | "
                    f"{fmt(b['canonical_contrib_mean'], 5)} | {fmt(b['adv_weighted_dr_mean'])} |"
                )
        lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    results_dir = repo_path(cfg["results_dir"])
    fig_dir = repo_path("results") / "figures" / "task3"
    fig_dir.mkdir(parents=True, exist_ok=True)

    grpo = load_json(results_dir / "grpo_standard.json")
    study = load_json(results_dir / "group_size_study.json")
    norm = load_json(results_dir / "normalization_study.json")
    eval_std = load_json(results_dir / "eval_standard.json")
    diag = load_json(results_dir / "grad_length_diag.json")

    md = tables_md(grpo, study, norm, eval_std, diag)
    (results_dir / "summary_tables.md").write_text(md, encoding="utf-8")
    print(md)

    made = []
    if fig_trajectories(grpo, fig_dir / "grpo_trajectories"):
        made.append("grpo_trajectories")
    if fig_group_size(study, fig_dir / "grpo_group_size"):
        made.append("grpo_group_size")
    if fig_normalization(norm, fig_dir / "grpo_normalization"):
        made.append("grpo_normalization")
    print("figures written:", made)


if __name__ == "__main__":
    main()
