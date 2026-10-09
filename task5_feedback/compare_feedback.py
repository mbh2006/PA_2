"""Task 5 final synthesis: combine in-domain, controlled-diagnostic, and transfer results
into the RLVR-vs-RLAIF comparison (feedback coverage, noise, exploitability, cost)."""

from __future__ import annotations

import argparse
import json

from common.data import load_yaml, repo_path


def load_json(path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    results_dir = repo_path(cfg["results_dir"]) / "task5_feedback"

    gsm = load_json(results_dir / "eval_gsm.json")
    transfer = load_json(results_dir / "eval_transfer.json")
    diagnostics = load_json(results_dir / "diagnostics_scored.json")

    missing = [name for name, obj in [("gsm", gsm), ("transfer", transfer), ("diagnostics", diagnostics)] if obj is None]
    if missing:
        raise SystemExit(f"Missing result files for: {missing}. Run the evaluation stages first.")

    comparison = {"in_domain": {}, "transfer": {}, "diagnostics": diagnostics["table"], "sensitivities": {
        "S_reason": diagnostics["S_reason"], "S_outcome": diagnostics["S_outcome"]}}

    for policy, summary in gsm["policies"].items():
        row = {
            "exact_accuracy": summary["exact_accuracy"],
            "format_compliance": summary["format_compliance"],
            "response_length_mean": summary["response_length_mean"],
        }
        if policy != "sft":
            row["pairwise_win_rate_vs_sft"] = gsm["pairwise_vs_sft"][policy]["policy_win_rate_vs_sft"]
            row["verifier_judge_agreement"] = gsm["pairwise_vs_sft"][policy]["verifier_judge_agreement"]
        t = transfer["policies"].get(policy)
        if t:
            row["transfer_accuracy"] = t["exact_accuracy"]
            row["transfer_length_mean"] = t["response_length_mean"]
            row["accuracy_drop"] = summary["exact_accuracy"] - t["exact_accuracy"]
            if policy != "sft":
                row["transfer_pairwise_win_rate_vs_sft"] = transfer["pairwise_vs_sft"][policy]["policy_win_rate_vs_sft"]
        comparison["in_domain"][policy] = row

    comparison["feedback_source_notes"] = {
        "verifier": {
            "coverage": "exact final-answer correctness only; binary, ignores reasoning quality and style",
            "noise": "deterministic (reproducible), but only as correct as the parser specification",
            "exploitability": "rewards lucky final answers and ignores corrupted reasoning (S_reason is definitionally tied for same-reward responses)",
            "cost": "negligible",
        },
        "ai_judge": {
            "coverage": "pairwise quality incl. reasoning, consistency, relevance",
            "noise": "stochastic judge preferences; inherits style/verbosity/persuasion biases",
            "exploitability": "can be gamed by persuasive filler or confidence; cache makes repeated use deterministic",
            "cost": "one 3B judge forward per comparison (cached)",
        },
    }

    (results_dir / "comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
