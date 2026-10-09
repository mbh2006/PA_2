"""Task 1 (DPO) analysis: required tables + figures from saved results.

Outputs (all under the repo):
  results/task1_dpo/summary_tables.md      <- report-ready markdown tables
  results/figures/task1/beta_sweep.{png,pdf}
  results/figures/task1/length_strata.{png,pdf}
  results/figures/task1/standard_train_curve.{png,pdf}

Run from repo root (CPU-only, no GPU needed):
  python analysis/task1_plots.py
"""

from __future__ import annotations

import argparse
import json
import math
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


def heldout_loss_from_margins(margins: list[float], beta: float):
    if not margins:
        return None
    vals = [-math.log(1.0 / (1.0 + math.exp(-beta * m))) for m in margins]  # -log sigmoid(beta*m)
    return sum(vals) / len(vals)


def collect_conditions(results_dir: Path):
    """Return dict condition -> eval payload for every eval_*.json found."""
    out = {}
    for path in sorted(results_dir.glob("eval_*.json")):
        name = path.stem[len("eval_"):]
        payload = load_json(path)
        if payload:
            out[name] = payload
    return out


def condition_row(name, payload, beta_override=None):
    pairs = payload.get("heldout_pairs", {})
    gen = payload.get("generation", {})
    beta = beta_override if beta_override is not None else payload.get("beta")
    margins = [p["margin"] for p in payload.get("per_pair", []) if "margin" in p]
    return {
        "name": name,
        "beta": beta,
        "heldout_loss": heldout_loss_from_margins(margins, beta) if beta is not None else None,
        "pref_acc": pairs.get("preference_accuracy"),
        "kl": gen.get("kl_sampled_response_estimator"),
        "rm": gen.get("reward_model_score_mean"),
        "len_mean": gen.get("response_length_mean"),
        "len_std": gen.get("response_length_std"),
        "n_pairs": pairs.get("num_eval_pairs"),
        "n_gen": gen.get("num_prompts_generated"),
    }


def fmt(x, nd=3):
    if x is None:
        return "—"
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, (int, float)):
        return f"{x:.{nd}f}" if isinstance(x, float) else str(x)
    return str(x)


def tables_markdown(conditions: dict, length_payloads: dict, train_summaries: dict):
    lines = ["# Task 1 — auto-generated result tables", ""]
    # Correct per-condition beta comes from the training metadata, never the config default.
    beta_map = {name: t.get("beta") for name, t in train_summaries.items()}

    lines.append("## 1. Standard + short-run β conditions (budgets differ: standard = 1 epoch, forks = 600 examples)")
    lines.append("")
    lines.append("| condition | β | budget | held-out DPO loss | pref. acc | KL (sampled) | RM score | len mean ± std |")
    lines.append("|---|---|---|---|---|---|---|---|")
    order = ["standard"] + [f"beta_{b}" for b in ["0p03", "0p1", "0p3"]]
    for name in order:
        if name not in conditions:
            continue
        r = condition_row(name, conditions[name], beta_override=beta_map.get(name))
        budget = "1 epoch (1500 ex)" if name == "standard" else "600 ex (short)"
        ln = "—" if r["len_mean"] is None else f"{fmt(r['len_mean'],1)} ± {fmt(r['len_std'],1)}"
        lines.append(
            f"| {name} | {fmt(r['beta'],2)} | {budget} | {fmt(r['heldout_loss'])} | {fmt(r['pref_acc'])} | "
            f"{fmt(r['kl'],5)} | {fmt(r['rm'])} | {ln} |"
        )
    lines.append("")

    if length_payloads:
        lines.append("## 2. Length-confounding study (standard vs length-balanced, per stratum)")
        lines.append("")
        std = length_payloads.get("standard", {})
        bal = length_payloads.get("length_balanced", {})
        std_strata = std.get("stratified_eval", {})
        bal_strata = bal.get("stratified_eval", {})
        strata = [k for k in sorted(set(std_strata) | set(bal_strata)) if not k.startswith("_")]
        lines.append("| stratum | n (std) | acc standard | acc length-balanced |")
        lines.append("|---|---|---|---|")
        for s in strata + ["_overall"]:
            a = std_strata.get(s, {})
            b = bal_strata.get(s, {})
            lines.append(
                f"| {s} | {fmt(a.get('num_pairs'))} | {fmt(a.get('preference_accuracy'))} | "
                f"{fmt(b.get('preference_accuracy'))} |"
            )
        lines.append("")
        lines.append("## 3. Word-limit compliance on the common prompt set")
        lines.append("")
        lines.append("| condition | compliance rate | mean response tokens |")
        lines.append("|---|---|---|")
        for tag, payload in length_payloads.items():
            wl = payload.get("word_limit", {})
            lines.append(
                f"| {tag} | {fmt(wl.get('word_limit_compliance_rate'))} | {fmt(wl.get('response_length_mean_tokens'),1)} |"
            )
        lines.append("")

    if train_summaries:
        lines.append("## 4. Training runs (from train_*.json)")
        lines.append("")
        lines.append("| run | β | examples | steps | final loss | final pref acc | wall-clock s |")
        lines.append("|---|---|---|---|---|---|---|")
        for name, t in sorted(train_summaries.items()):
            lines.append(
                f"| {name} | {fmt(t.get('beta'),2)} | {t.get('num_examples')} | {t.get('optimizer_steps')} | "
                f"{fmt(t.get('final_mean_loss'))} | {fmt(t.get('final_mean_preference_accuracy'))} | "
                f"{fmt(t.get('wall_clock_seconds'),1)} |"
            )
        lines.append("")
    return "\n".join(lines)


def fig_beta_sweep(conditions, outbase, beta_map=None):
    beta_map = beta_map or {}
    betas, accs, kls, rms, lens = [], [], [], [], []
    for name in ["beta_0p03", "beta_0p1", "beta_0p3"]:
        if name not in conditions:
            continue
        r = condition_row(name, conditions[name], beta_override=beta_map.get(name))
        betas.append(r["beta"])
        accs.append(r["pref_acc"])
        kls.append(r["kl"])
        rms.append(r["rm"])
        lens.append(r["len_mean"])
    if not betas:
        return False

    std = condition_row("standard", conditions["standard"]) if "standard" in conditions else None
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.6))
    panels = [
        ("held-out preference accuracy", accs, std["pref_acc"] if std else None),
        ("KL from reference (sampled)", kls, std["kl"] if std else None),
        ("reward-model score", rms, std["rm"] if std else None),
        ("response length (tokens)", lens, std["len_mean"] if std else None),
    ]
    for ax, (title, ys, ref) in zip(axes, panels):
        ax.plot(betas, ys, "o-", label="short-run forks (600 ex)")
        if ref is not None:
            ax.axhline(ref, color="gray", ls="--", lw=1, label="standard (1 epoch, β=0.10)")
        ax.set_xscale("log")
        ax.set_xlabel("β")
        ax.set_title(title)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.suptitle("DPO β sweep (matched short forks) vs standard run", y=1.02)
    fig.tight_layout()
    fig.savefig(outbase.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(outbase.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return True


def fig_length_strata(length_payloads, outbase):
    std = length_payloads.get("standard", {}).get("stratified_eval", {})
    bal = length_payloads.get("length_balanced", {}).get("stratified_eval", {})
    strata = [s for s in ["preferred_longer", "length_matched", "rejected_longer", "_overall"] if s in std or s in bal]
    if not strata:
        return False
    labels = [s.replace("_", " ") for s in strata]
    std_acc = [std.get(s, {}).get("preference_accuracy", float("nan")) for s in strata]
    bal_acc = [bal.get(s, {}).get("preference_accuracy", float("nan")) for s in strata]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))
    x = range(len(strata))
    ax1.bar([i - 0.18 for i in x], std_acc, width=0.36, label="standard DPO")
    ax1.bar([i + 0.18 for i in x], bal_acc, width=0.36, label="length-balanced DPO")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(labels, fontsize=8)
    ax1.set_ylabel("preference accuracy")
    ax1.set_title("Held-out accuracy by length stratum")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3, axis="y")

    conds = [c for c in ["standard", "length_balanced"] if c in length_payloads]
    rates = [length_payloads[c].get("word_limit", {}).get("word_limit_compliance_rate") for c in conds]
    toks = [length_payloads[c].get("word_limit", {}).get("response_length_mean_tokens") for c in conds]
    ax2.bar(conds, rates, color=["tab:blue", "tab:orange"])
    for i, (r, t) in enumerate(zip(rates, toks)):
        if r is not None:
            ax2.text(i, r + 0.02, f"{r:.2f}\n({t:.0f} tok)", ha="center", fontsize=8)
    ax2.set_ylim(0, 1.15)
    ax2.set_ylabel("word-limit compliance")
    ax2.set_title("Word-limit compliance (common prompt set)")
    ax2.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(outbase.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(outbase.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return True


def fig_train_curve(train, outbase):
    hist = (train or {}).get("train_history", [])
    if not hist:
        return False
    steps = [h["step"] for h in hist]
    loss = [h["mean_loss"] for h in hist]
    acc = [h["mean_preference_accuracy"] for h in hist]

    fig, ax1 = plt.subplots(figsize=(7, 3.6))
    ax1.plot(steps, loss, color="tab:red", label="loss")
    ax1.set_xlabel("optimizer step")
    ax1.set_ylabel("DPO loss", color="tab:red")
    ax1.tick_params(axis="y", labelcolor="tab:red")
    ax1.grid(alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(steps, acc, color="tab:blue", label="preference accuracy")
    ax2.set_ylabel("train preference accuracy", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")
    fig.suptitle("Standard DPO training (1 epoch)")
    fig.tight_layout()
    fig.savefig(outbase.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(outbase.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    results_dir = repo_path(cfg["results_dir"])
    fig_dir = repo_path("results") / "figures" / "task1"
    fig_dir.mkdir(parents=True, exist_ok=True)

    conditions = collect_conditions(results_dir)
    print("conditions found:", sorted(conditions))

    length_payloads = {}
    for tag in ["standard", "length_balanced"]:
        payload = load_json(results_dir / f"length_analysis_{tag}.json")
        if payload:
            length_payloads[tag] = payload

    train_summaries = {}
    for path in sorted(results_dir.glob("train_*.json")):
        payload = load_json(path)
        if payload:
            train_summaries[path.stem[len("train_"):]] = payload

    beta_map = {name: t.get("beta") for name, t in train_summaries.items()}

    md = tables_markdown(conditions, length_payloads, train_summaries)
    (results_dir / "summary_tables.md").write_text(md, encoding="utf-8")
    print(md)

    made = []
    if fig_beta_sweep(conditions, fig_dir / "beta_sweep", beta_map=beta_map):
        made.append("beta_sweep")
    if fig_length_strata(length_payloads, fig_dir / "length_strata"):
        made.append("length_strata")
    std_train = train_summaries.get("standard") or train_summaries.get("smoke")
    if fig_train_curve(std_train, fig_dir / "standard_train_curve"):
        made.append("standard_train_curve (from '%s')" % ("standard" if "standard" in train_summaries else "smoke"))
    print("figures written:", made)


if __name__ == "__main__":
    main()
