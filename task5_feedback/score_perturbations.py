"""Task 5 controlled reward diagnostic study.

Scores the supplied 100-response diagnostic set with both feedback mechanisms:
  - RLVR: the exact `#### <number>` verifier
  - RLAIF: the fixed pairwise AI judge

For each of four controlled pair types, report better-response / tie / wrong-
preference rates. Reasoning sensitivity S_reason = Pr[R(clean) > R(corrupt
reasoning)] on same-final-answer pairs; outcome sensitivity S_outcome =
Pr[R(correct final) > R(wrong final)] on outcome-changing pairs. The filler and
distractor pairs probe style susceptibility and gold-number robustness.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict

from common.data import load_yaml, read_jsonl, repo_path
from task5_feedback.rlaif import PairwiseAIJudge
from task5_feedback.rlvr import exact_reward

EXPECTED_VARIANTS = {
    "clean_correct",
    "corrupt_reasoning_correct_final",
    "good_reasoning_wrong_final",
    "persuasive_filler_correct",
    "gold_distractor_wrong_final",
}

# (pair kind, diagnostically better variant, worse variant)
CONTROLLED_PAIRS = [
    ("reasoning_only", "clean_correct", "corrupt_reasoning_correct_final"),
    ("outcome_only", "clean_correct", "good_reasoning_wrong_final"),
    ("persuasive_filler", "clean_correct", "persuasive_filler_correct"),
    ("gold_distractor", "clean_correct", "gold_distractor_wrong_final"),
]


def load_diagnostic_groups(path):
    rows = read_jsonl(path)
    by_problem = defaultdict(dict)
    for row in rows:
        by_problem[str(row["problem_id"])][row["variant_type"]] = row
    for pid, variants in by_problem.items():
        missing = EXPECTED_VARIANTS - set(variants)
        if missing:
            raise ValueError(f"Problem {pid} missing variants: {sorted(missing)}")
    return by_problem


def verifier_preference(better_row, worse_row):
    rb = exact_reward(better_row["response"], better_row["gold_final"])
    rw = exact_reward(worse_row["response"], worse_row["gold_final"])
    if rb > rw:
        return "better"
    if rb < rw:
        return "wrong"
    return "tie"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    groups = load_diagnostic_groups(cfg["paths"]["task5_diagnostics"])
    print("Diagnostic problems:", len(groups))
    print("Variants/problem:", sorted(EXPECTED_VARIANTS))

    results_dir = repo_path(cfg["results_dir"]) / "task5_feedback"
    results_dir.mkdir(parents=True, exist_ok=True)
    judge = PairwiseAIJudge(cfg, results_dir / "judge_cache.json")

    counts = {"verifier": defaultdict(Counter), "judge": defaultdict(Counter)}
    per_problem = defaultdict(dict)
    for pid, variants in sorted(groups.items(), key=lambda kv: int(kv[0]) if str(kv[0]).isdigit() else kv[0]):
        question = variants["clean_correct"]["question"]
        for kind, better_key, worse_key in CONTROLLED_PAIRS:
            better, worse = variants[better_key], variants[worse_key]

            v_out = verifier_preference(better, worse)
            counts["verifier"][kind][v_out] += 1

            pref = judge.compare(question, better["response"], worse["response"])
            j_out = {"A": "better", "TIE": "tie", "B": "wrong"}[pref]
            counts["judge"][kind][j_out] += 1

            per_problem[pid][kind] = {"verifier": v_out, "judge": j_out}

    def rates(counter: Counter):
        n = sum(counter.values())
        return {
            "n": n,
            "better_rate": counter.get("better", 0) / n if n else None,
            "tie_rate": counter.get("tie", 0) / n if n else None,
            "wrong_rate": counter.get("wrong", 0) / n if n else None,
        }

    table = {
        mech: {kind: rates(counts[mech][kind]) for kind, _, _ in CONTROLLED_PAIRS}
        for mech in ["verifier", "judge"]
    }

    report = {
        "num_problems": len(groups),
        "pair_types": [k for k, _, _ in CONTROLLED_PAIRS],
        "table": table,
        "S_reason": {
            "verifier": table["verifier"]["reasoning_only"]["better_rate"],
            "judge": table["judge"]["reasoning_only"]["better_rate"],
        },
        "S_outcome": {
            "verifier": table["verifier"]["outcome_only"]["better_rate"],
            "judge": table["judge"]["outcome_only"]["better_rate"],
        },
        "per_problem": {pid: dict(v) for pid, v in per_problem.items()},
    }
    (results_dir / "diagnostics_scored.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"table": table, "S_reason": report["S_reason"], "S_outcome": report["S_outcome"]}, indent=2))


if __name__ == "__main__":
    main()
