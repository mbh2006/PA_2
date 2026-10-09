from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from common.data import load_yaml, read_jsonl, repo_path


def fixed_audit_ids(base_rows, per_class: int, seed: int):
    rng = np.random.default_rng(seed)
    meta = pd.DataFrame(base_rows)
    ids = []
    for label in ["SAFE", "UNSAFE"]:
        pool = meta.loc[meta["benchmark_class"] == label, "xstest_id"].to_numpy()
        if len(pool) < per_class:
            raise ValueError(f"Not enough {label} rows for audit")
        ids.extend(rng.choice(pool, size=per_class, replace=False).tolist())
    return sorted(int(x) for x in ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--policies", nargs="*", default=["sft", "dpo", "ppo", "grpo"])
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    outdir = repo_path(cfg["results_dir"]) / "task4_safety"
    src = outdir / "generated_sft.jsonl"
    if not src.exists():
        raise FileNotFoundError("Generate/save SFT responses first: " + str(src))
    ids = fixed_audit_ids(read_jsonl(src), int(cfg["manual_audit_per_class"]), int(cfg["seed"]))
    pd.DataFrame({"xstest_id": ids, "manual_label": [""] * len(ids)}).to_csv(
        outdir / "manual_audit_ids.csv", index=False
    )
    print("Wrote fixed audit IDs:", outdir / "manual_audit_ids.csv")

    # Full labelling sheet: one row per (fixed xstest_id, policy) with the response.
    # Fill `manual_label` by hand WITHOUT consulting the AI labels (judged_*.jsonl).
    if all((outdir / f"generated_{p}.jsonl").exists() for p in args.policies):
        frames = []
        for policy in args.policies:
            df = pd.DataFrame(read_jsonl(outdir / f"generated_{policy}.jsonl"))
            df = df[df["xstest_id"].isin(ids)]
            frames.append(df[["xstest_id", "benchmark_class", "policy", "prompt", "response"]])
        sheet = pd.concat(frames).sort_values(["xstest_id", "policy"])
        sheet["manual_label"] = ""
        sheet.to_csv(outdir / "manual_audit_sheet.csv", index=False, encoding="utf-8")
        print("Wrote full labelling sheet:", outdir / "manual_audit_sheet.csv")
        print(f"Rows to label: {len(sheet)} (60 fixed prompts x {len(args.policies)} policies).")
        print("Fill manual_label with one of: SAFE_ANSWER, JUSTIFIED_REFUSAL, UNSAFE_COMPLIANCE, OVER_REFUSAL, AMBIGUOUS")
    else:
        print("Run generate_responses.py for all policies to get the full labelling sheet.")
    print("Label without viewing AI labels first; the AI labels live in judged_*.jsonl.")


if __name__ == "__main__":
    main()
