"""Task 5 (RLVR vs RLAIF) analysis: required tables + figures.

Outputs:
  results/task5_feedback/summary_tables.md
  results/figures/task5/task5_accuracy.{png,pdf}
  results/figures/task5/task5_diagnostics.{png,pdf}

Run from repo root (CPU only):
  python analysis/task5_plots.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from common.data import load_yaml, repo_path  # noqa: E402

PAIR_ORDER = ["reasoning_only", "outcome_only", "persuasive_filler", "gold_distractor"]


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


def fig_accuracy(gsm: dict, transfer: dict | None, outbase: Path):
    if not gsm:
        return False
    policies = list(gsm["policies"].keys())
    x = np.arange(len(policies))
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))

    ax = axes[0]
    ax.bar(x - 0.18, [gsm["policies"][p]["exact_accuracy"] for p in policies], 0.36, label="GSM8K")
    if transfer:
        ax.bar(x + 0.18, [transfer["policies"][p]["exact_accuracy"] for p in policies], 0.36, label="SVAMP")
    ax.set_xticks(x, policies)
    ax.set_title("Exact-answer accuracy")
    ax.set_ylim(0, max(1.0, ax.get_ylim()[1]))
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.bar(x, [gsm["policies"][p]["format_compliance"] for p in policies])
    ax.set_xticks(x, policies)
    ax.set_title("Format compliance (GSM8K)")

    ax = axes[2]
    ax.bar(x - 0.18, [gsm["policies"][p]["response_length_mean"] for p in policies], 0.36, label="GSM8K")
    if transfer:
        ax.bar(x + 0.18, [transfer["policies"][p]["response_length_mean"] for p in policies], 0.36, label="SVAMP")
    ax.set_xticks(x, policies)
    ax.set_title("Mean response length")
    ax.legend(fontsize=8)

    for ax in axes:
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle("Task 5: SFT vs RLVR vs RLAIF (deterministic decoding)", y=1.03)
    save(fig, outbase)
    return True


def fig_diagnostics(diag: dict, outbase: Path):
    table = (diag or {}).get("table", {})
    if not table:
        return False
    kinds = [k for k in PAIR_ORDER if k in table.get("judge", {})]
    x = np.arange(len(kinds))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, mech in zip(axes, ["verifier", "judge"]):
        ax.bar(x - 0.25, [table[mech][k]["better_rate"] for k in kinds], 0.25, label="better")
        ax.bar(x, [table[mech][k]["tie_rate"] for k in kinds], 0.25, label="tie")
        ax.bar(x + 0.25, [table[mech][k]["wrong_rate"] for k in kinds], 0.25, label="wrong")
        ax.set_xticks(x, [k.replace("_", "\n") for k in kinds], fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{mech} preferences on controlled pairs")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle("Controlled reward diagnostics (20 problems × 4 pair types)", y=1.03)
    save(fig, outbase)
    return True


def tables_md(gsm: dict, transfer: dict, diag: dict, comparison: dict):
    lines = ["# Task 5 — auto-generated result tables", ""]
    if gsm:
        lines += ["## 1. In-domain (GSM8K) and out-of-domain (SVAMP)", ""]
        lines.append("| policy | GSM acc | GSM format | GSM length | pairwise vs SFT (GSM) | SVAMP acc | drop |")
        lines.append("|---|---|---|---|---|---|---|")
        for p, s in gsm["policies"].items():
            pw = gsm["pairwise_vs_sft"].get(p, {})
            t = (transfer or {}).get("policies", {}).get(p)
            acc_t = t["exact_accuracy"] if t else None
            drop = (s["exact_accuracy"] - acc_t) if t else None
            lines.append(
                f"| {p} | {fmt(s['exact_accuracy'])} | {fmt(s['format_compliance'])} | {fmt(s['response_length_mean'],1)} | "
                f"{fmt(pw.get('policy_win_rate_vs_sft'))} | {fmt(acc_t)} | {fmt(drop)} |"
            )
        lines.append("")
        lines += ["### Verifier–judge agreement (GSM8K)", ""]
        lines.append("| challenger | agreement | pairs with verifier disagreement | wins/ties/losses |")
        lines.append("|---|---|---|---|")
        for p, pw in gsm.get("pairwise_vs_sft", {}).items():
            lines.append(
                f"| {p} | {fmt(pw.get('verifier_judge_agreement'))} | {pw.get('num_pairs_with_verifier_disagreement')} | "
                f"{pw.get('wins')}/{pw.get('ties')}/{pw.get('losses')} |"
            )
        lines.append("")
    if diag:
        lines += [
            "## 2. Controlled diagnostic study",
            "",
            f"- S_reason: verifier **{fmt(diag['S_reason']['verifier'])}**, judge **{fmt(diag['S_reason']['judge'])}**",
            f"- S_outcome: verifier **{fmt(diag['S_outcome']['verifier'])}**, judge **{fmt(diag['S_outcome']['judge'])}**",
            "",
            "| mechanism | pair type | n | better | tie | wrong |",
            "|---|---|---|---|---|---|",
        ]
        for mech in ["verifier", "judge"]:
            for k in PAIR_ORDER:
                row = diag["table"][mech][k]
                lines.append(
                    f"| {mech} | {k} | {row['n']} | {fmt(row['better_rate'])} | {fmt(row['tie_rate'])} | {fmt(row['wrong_rate'])} |"
                )
        lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    t5_dir = repo_path(cfg["results_dir"]) / "task5_feedback"
    fig_dir = repo_path("results") / "figures" / "task5"
    fig_dir.mkdir(parents=True, exist_ok=True)

    gsm = load_json(t5_dir / "eval_gsm.json")
    transfer = load_json(t5_dir / "eval_transfer.json")
    diag = load_json(t5_dir / "diagnostics_scored.json")
    comparison = load_json(t5_dir / "comparison.json")

    md = tables_md(gsm, transfer, diag, comparison)
    (t5_dir / "summary_tables.md").write_text(md, encoding="utf-8")
    print(md)

    made = []
    if fig_accuracy(gsm, transfer, fig_dir / "task5_accuracy"):
        made.append("task5_accuracy")
    if fig_diagnostics(diag, fig_dir / "task5_diagnostics"):
        made.append("task5_diagnostics")
    print("figures written:", made)


if __name__ == "__main__":
    main()
