"""Task 4 (safety) analysis: required tables + figures from safety_eval.json.

Outputs:
  results/task4_safety/summary_tables.md
  results/figures/task4/safety_calibration.{png,pdf}
  results/figures/task4/safety_categories.{png,pdf}
  results/figures/task4/safety_confusion.{png,pdf}   (when manual audit labels exist)

Run from repo root (CPU only):
  python analysis/task4_plots.py
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

LABELS = ["SAFE_ANSWER", "JUSTIFIED_REFUSAL", "UNSAFE_COMPLIANCE", "OVER_REFUSAL", "AMBIGUOUS"]
POLICIES = ["sft", "dpo", "ppo", "grpo"]


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


def fig_calibration(evald: dict, outbase: Path):
    aggs = evald.get("aggregates", {})
    policies = [p for p in POLICIES if p in aggs]
    if not policies:
        return False
    token_stats = evald.get("_token_stats", {})
    fig, axes = plt.subplots(1, 4, figsize=(18, 3.8))
    x = np.arange(len(policies))

    ax = axes[0]
    ax.bar(x - 0.18, [aggs[p]["safe_answer_rate"] for p in policies], 0.36, label="safe answer")
    ax.bar(x + 0.18, [aggs[p]["safe_over_refusal_rate"] for p in policies], 0.36, label="over-refusal")
    ax.set_xticks(x, policies)
    ax.set_title("Safe prompts")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.bar(x - 0.18, [aggs[p]["unsafe_compliance_rate"] for p in policies], 0.36, label="unsafe compliance")
    ax.bar(x + 0.18, [aggs[p]["unsafe_justified_refusal_rate"] for p in policies], 0.36, label="justified refusal")
    ax.set_xticks(x, policies)
    ax.set_title("Unsafe prompts")
    ax.legend(fontsize=8)

    ax = axes[2]
    ax.bar(x, [aggs[p]["ambiguous_rate"] for p in policies], 0.36, color="tab:purple")
    ax.set_xticks(x, policies)
    ax.set_ylim(0, max(0.05, 1.3 * max(aggs[p]["ambiguous_rate"] or 0 for p in policies)))
    ax.set_title("Ambiguous-judge rate (own axis)")
    ax.legend(fontsize=8)

    ax = axes[3]
    means = [aggs[p]["mean_response_tokens"] for p in policies]
    stds = [token_stats.get(p, (None, None))[1] if token_stats.get(p) else 0.0 for p in policies]
    ax.bar(x, means, 0.36, color="tab:gray", yerr=[s or 0.0 for s in stds], capsize=4)
    ax.set_xticks(x, policies)
    ax.set_title("Mean response tokens ± std (own axis)")

    for ax in axes:
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle("Safety calibration across the four frozen policies (XSTest) — judge-limited instrument; see caveat", y=1.03)
    save(fig, outbase)
    return True


def fig_categories(evald: dict, outbase: Path):
    cats = evald.get("category_level", {})
    policies = [p for p in POLICIES if cats.get(p)]
    if not policies:
        return False
    types = sorted({t for p in policies for t in cats[p]})
    fig, axes = plt.subplots(2, 2, figsize=(17, 9))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for ax, policy in zip(axes.flat, policies):
        bottom = np.zeros(len(types))
        for li, label in enumerate(LABELS):
            vals = np.array([cats[policy][t].get(label, 0) or 0 for t in types])
            ax.bar(range(len(types)), vals, bottom=bottom, label=label, color=colors[li])
            bottom += vals
        ax.set_xticks(range(len(types)), types, rotation=45, ha="right", fontsize=7)
        ax.set_title(policy, fontsize=10)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3, axis="y")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=8, frameon=False)
    fig.suptitle("Judge-label distribution across XSTest categories", y=1.0)
    fig.tight_layout(rect=(0, 0.04, 1, 0.98))
    fig.savefig(outbase.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(outbase.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return True


def fig_confusion(evald: dict, outbase: Path):
    audit = evald.get("manual_audit", {})
    if not audit.get("available") or not audit.get("confusion_ai_vs_manual"):
        return False
    matrix = np.array([[audit["confusion_ai_vs_manual"][ai].get(man, 0) for man in LABELS] for ai in LABELS])
    fig, ax = plt.subplots(figsize=(7.5, 6))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(LABELS)), LABELS, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(LABELS)), LABELS, fontsize=8)
    ax.set_xlabel("manual label")
    ax.set_ylabel("AI judge label")
    ax.set_title(f"Judge vs manual (agreement {fmt(audit.get('overall_agreement'))}, n={audit.get('num_labelled_pairs')})")
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                    color="white" if matrix[i, j] > matrix.max() * 0.6 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.8)
    save(fig, outbase)
    return True


def tables_md(evald: dict):
    lines = ["# Task 4 — auto-generated result tables", ""]
    aggs = evald.get("aggregates", {})
    token_stats = evald.get("_token_stats", {})
    policies = [p for p in POLICIES if p in aggs]
    if policies:
        lines += [
            "## 1. Safety-calibration comparison (AI judge)",
            "",
            "_**Judge-limited instrument** — class-inconsistent labels (105/120/107/108 of 450 for SFT/DPO/PPO/GRPO) and zero OVER_REFUSAL usage are documented in `TRACKING/04_TASK4_SAFETY.md` §6.2/§7; the student manual audit (63.3% agreement, 7 over-refusals the judge missed) qualifies this table. Do not read zero over-refusal/compliance rates as \"perfect safety\"._",
            "",
            "| policy | n | safe answer | safe over-refusal | unsafe compliance | justified refusal | ambiguous | mean ± std tokens | parse failures |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for p in policies:
            a = aggs[p]
            mean_t, std_t = token_stats.get(p, (a["mean_response_tokens"], None))
            lines.append(
                f"| {p} | {a['num_prompts']} | {fmt(a['safe_answer_rate'])} | {fmt(a['safe_over_refusal_rate'])} | "
                f"{fmt(a['unsafe_compliance_rate'])} | {fmt(a['unsafe_justified_refusal_rate'])} | "
                f"{fmt(a['ambiguous_rate'])} | {fmt(mean_t, 1)} ± {fmt(std_t, 1)} | {a.get('parse_failure_count', 0)} |"
            )
        lines.append("")
        lines.append("### Full judge-label distribution per policy")
        lines.append("")
        lines.append("| policy | " + " | ".join(LABELS) + " |")
        lines.append("|---" * (len(LABELS) + 1) + "|")
        for p in policies:
            dist = aggs[p]["judge_label_distribution"]
            lines.append(f"| {p} | " + " | ".join(fmt(dist.get(l)) for l in LABELS) + " |")
        lines.append("")

    audit = evald.get("manual_audit", {})
    if audit.get("available"):
        lines += [
            "## 2. Manual audit vs AI judge",
            "",
            f"- labelled pairs: **{audit.get('num_labelled_pairs')}** over **{audit.get('num_distinct_prompts')}** prompts",
            f"- overall agreement: **{fmt(audit.get('overall_agreement'))}**; excluding AI-ambiguous: **{fmt(audit.get('agreement_excluding_ai_ambiguous'))}**",
            f"- policy coverage: {audit.get('policy_coverage')}",
            "",
            "### Confusion (rows = AI judge, cols = manual)",
            "",
            "| AI \\ manual | " + " | ".join(LABELS) + " |",
            "|---" * (len(LABELS) + 1) + "|",
        ]
        for ai in LABELS:
            row = audit["confusion_ai_vs_manual"].get(ai, {})
            lines.append(f"| {ai} | " + " | ".join(str(row.get(man, 0)) for man in LABELS) + " |")
        lines.append("")
    else:
        lines += ["## 2. Manual audit", "", f"_Not available yet_: {audit.get('note', 'labels pending')}", ""]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    safety_dir = repo_path(cfg["results_dir"]) / "task4_safety"
    fig_dir = repo_path("results") / "figures" / "task4"
    fig_dir.mkdir(parents=True, exist_ok=True)

    evald = load_json(safety_dir / "safety_eval.json")
    if evald is None:
        print("safety_eval.json not found yet; run task4_safety.evaluate_safety first.")
        return
    token_stats = {}
    for pol in POLICIES:
        path = safety_dir / f"judged_{pol}.jsonl"
        if path.exists():
            rows = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
            toks = [r.get("response_tokens", 0) for r in rows]
            if toks:
                token_stats[pol] = (float(np.mean(toks)), float(np.std(toks)))
    evald["_token_stats"] = token_stats
    md = tables_md(evald)
    (safety_dir / "summary_tables.md").write_text(md, encoding="utf-8")
    print(md)

    made = []
    if fig_calibration(evald, fig_dir / "safety_calibration"):
        made.append("safety_calibration")
    if fig_categories(evald, fig_dir / "safety_categories"):
        made.append("safety_categories")
    if fig_confusion(evald, fig_dir / "safety_confusion"):
        made.append("safety_confusion")
    print("figures written:", made)


if __name__ == "__main__":
    main()
