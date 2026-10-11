"""Task 6 synthesis: assemble one cross-task summary from saved results.

Reads whatever task results exist and writes:
  results/cross_task/summary.md   (markdown table + notes)

Run from repo root (CPU only):
  python analysis/task6_cross_task.py
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from common.data import load_yaml, repo_path  # noqa: E402


def load(path):
    p = pathlib.Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def fmt(x, nd=3):
    if x is None:
        return "—"
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, (int, float)):
        return f"{x:.{nd}f}" if isinstance(x, float) else str(x)
    return str(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dpo-config", default="configs/dpo.yaml")
    ap.add_argument("--ppo-config", default="configs/ppo.yaml")
    ap.add_argument("--grpo-config", default="configs/grpo.yaml")
    ap.add_argument("--feedback-config", default="configs/feedback.yaml")
    args = ap.parse_args()
    base = repo_path("results")
    outdir = base / "cross_task"
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []

    # ---- Task 1: DPO conditions ----
    t1 = base / "task1_dpo"
    for name, label in [("sft_baseline", "T1 SFT baseline"), ("standard", "T1 DPO standard"),
                        ("beta_0p03", "T1 DPO β=0.03"), ("beta_0p1", "T1 DPO β=0.10"),
                        ("beta_0p3", "T1 DPO β=0.30"), ("length_balanced", "T1 DPO length-balanced")]:
        ev = load(t1 / f"eval_{name}.json")
        if not ev:
            continue
        g = ev.get("generation", {})
        rows.append({
            "task": "T1", "condition": label,
            "signal": "fixed preference pairs (DPO)",
            "budget": f"{ev.get('eval_generation_prompts', '?')} gen prompts",
            "reward": g.get("reward_model_score_mean"),
            "kl": g.get("kl_sampled_response_estimator"),
            "length": g.get("response_length_mean"),
            "extra": f"pref-acc {fmt(ev.get('heldout_pairs', {}).get('preference_accuracy'))}",
            "compute": "",
        })
    for name, tag in [("standard", "T1 DPO standard"), ("beta_0p03", "T1 DPO β=0.03"),
                      ("beta_0p1", "T1 DPO β=0.10"), ("beta_0p3", "T1 DPO β=0.30"),
                      ("length_balanced", "T1 DPO length-balanced")]:
        tr = load(t1 / f"train_{name}.json")
        if tr:
            for r in rows:
                if r["condition"] == tag:
                    r["compute"] = f"{fmt(tr.get('wall_clock_seconds', 0), 0)} s, {tr.get('optimizer_steps', '?')} steps"

    # ---- Task 2: PPO ----
    t2 = base / "task2_ppo"
    for name, label, extra in [
        ("midpoint_baseline", "T2 PPO midpoint", ""),
        ("standard", "T2 PPO standard (20 upd)", ""),
        ("clip_eps_0p2", "T2 PPO ε=0.20 fork (8 upd)", "ε sweep: identical to other ε"),
        ("kl_0", "T2 PPO βKL=0 fork", ""),
        ("kl_0p1", "T2 PPO βKL=0.10 fork", ""),
        ("kl_0p2", "T2 PPO βKL=0.20 fork", ""),
    ]:
        ev = load(t2 / f"eval_{name}.json")
        if not ev:
            continue
        rows.append({
            "task": "T2", "condition": label, "signal": "learned reward + value critic (PPO)",
            "budget": "100 gen prompts", "reward": ev.get("reward_model_score_mean"),
            "kl": ev.get("kl_sampled_response_estimator"), "length": ev.get("response_length_mean"),
            "extra": extra, "compute": "",
        })
    tr = load(t2 / "ppo_standard.json")
    if tr:
        for r in rows:
            if r["condition"].startswith("T2 PPO standard"):
                r["compute"] = f"{fmt(tr.get('wall_clock_seconds', 0), 0)} s, {tr.get('updates')} upd, peak {fmt(tr.get('peak_vram_gib'), 2)} GiB"

    # ---- Task 3: GRPO ----
    t3 = base / "task3_grpo"
    for name, label, extra in [
        ("midpoint_baseline", "T3 GRPO midpoint", ""),
        ("standard", "T3 GRPO standard (20 upd, K=4)", ""),
        ("norm_grpo", "T3 GRPO canonical fork (8 upd)", ""),
        ("norm_dr_grpo", "T3 GRPO Dr.GRPO fork (8 upd)", "corr(len,grad) −0.41 vs −0.94"),
    ]:
        ev = load(t3 / f"eval_{name}.json")
        if not ev:
            continue
        rows.append({
            "task": "T3", "condition": label, "signal": "group-relative reward, no critic (GRPO)",
            "budget": "100 gen prompts", "reward": ev.get("reward_model_score_mean"),
            "kl": ev.get("kl_sampled_response_estimator"), "length": ev.get("response_length_mean"),
            "extra": extra, "compute": "",
        })
    tr = load(t3 / "grpo_standard.json")
    if tr:
        for r in rows:
            if r["condition"].startswith("T3 GRPO standard"):
                r["compute"] = f"{fmt(tr.get('wall_clock_seconds', 0), 0)} s, {tr.get('updates')} upd, peak {fmt(tr.get('peak_vram_gib'), 2)} GiB"

    # ---- Task 4: safety (if available) ----
    t4 = load(base / "task4_safety" / "safety_eval.json")
    if t4:
        for pol, a in t4.get("aggregates", {}).items():
            rows.append({
                "task": "T4", "condition": f"T4 {pol.upper()} (frozen)",
                "signal": "categorical AI judge on XSTest", "budget": "450 XSTest prompts",
                "reward": None, "kl": None, "length": a.get("mean_response_tokens"),
                "extra": (f"safe-answer {fmt(a.get('safe_answer_rate'))}, over-refusal {fmt(a.get('safe_over_refusal_rate'))}, "
                          f"unsafe-compliance {fmt(a.get('unsafe_compliance_rate'))}, justified-refusal {fmt(a.get('unsafe_justified_refusal_rate'))}, "
                          f"ambiguous {fmt(a.get('ambiguous_rate'))}"),
                "compute": "",
            })

    # ---- Task 5: feedback source (if available) ----
    t5 = base / "task5_feedback"
    gsm = load(t5 / "eval_gsm.json")
    if gsm:
        for pol, s in gsm.get("policies", {}).items():
            rows.append({
                "task": "T5", "condition": f"T5 {pol.upper()} (GSM8K)",
                "signal": "exact verifier" if pol == "rlvr" else ("AI pairwise" if pol == "rlaif" else "—"),
                "budget": "300 GSM8K prompts",
                "reward": None, "kl": None, "length": s.get("response_length_mean"),
                "extra": f"exact-acc {fmt(s.get('exact_accuracy'))}, format {fmt(s.get('format_compliance'))}",
                "compute": "",
            })
    diag = load(t5 / "diagnostics_scored.json")
    if diag:
        rows.append({
            "task": "T5", "condition": "T5 diagnostics (100 responses)",
            "signal": "verifier vs pairwise AI judge", "budget": "5 pair types",
            "reward": None, "kl": None, "length": None,
            "extra": (f"S_reason: verifier {fmt(diag['S_reason']['verifier'])} / judge {fmt(diag['S_reason']['judge'])}; "
                      f"S_outcome: verifier {fmt(diag['S_outcome']['verifier'])} / judge {fmt(diag['S_outcome']['judge'])}"),
            "compute": "",
        })

    # ---- markdown ----
    lines = ["# Cross-task summary (Task 6 scaffold)", ""]
    lines.append("| task | condition | reward signal | held-out reward (RM) | KL | length | extra | compute |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in rows:
        lines.append(
            f"| {r['task']} | {r['condition']} | {r['signal']} | {fmt(r['reward'])} | {fmt(r['kl'], 5)} | "
            f"{fmt(r['length'], 1)} | {r['extra']} | {r['compute']} |"
        )
    lines += [
        "",
        "Notes:",
        "- Reward-scale caveat: T1/T2/T3 all use the same course reward model on 100 generated prompts — comparable within tasks; cross-task comparisons of RM means are indicative only (different prompt pools: DPO eval pairs vs RL prompt pool).",
        "- **Response-length caps differ by task (T1/T4: 256 tokens; T2: 768; T3/T5: 512) and lengths are censored at those caps** — never compare lengths across tasks without citing this; within-task comparisons use one fixed cap.",
        "- T2 ε evidence is the cached-batch geometry (clip fractions 11.0%/0.29%/0.01% at ε=0.05/0.2/0.5, salvage batch); on-policy ε forks are identical (clip never activates).",
        "- Paired statistics (vs task baselines, n=100) live in the per-task tracking docs (§7) and generation files.",
        "- T5 diagnostic columns are mechanism-level rates, not RM scores.",
    ]
    (outdir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print("\nsaved ->", outdir / "summary.md")


if __name__ == "__main__":
    main()
