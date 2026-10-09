"""Task 5 in-domain / out-of-domain math evaluation for SFT vs RLVR vs RLAIF.

Generates one greedy response per policy per fixed example, computes exact-answer
accuracy, format compliance, response length, then runs the fixed pairwise AI
judge against SFT and reports the verifier-judge agreement. No training.
"""

from __future__ import annotations

import argparse
import json
import statistics

from common.data import load_yaml, read_jsonl, repo_path
from common.generation import batch_generate
from common.models import clear_gpu, load_policy, load_tokenizer
from task5_feedback.rlaif import PairwiseAIJudge
from task5_feedback.rlvr import exact_reward, extract_designated_final


def policy_specs(cfg):
    return {
        "sft": None,
        "rlvr": cfg["policies"]["rlvr"],
        "rlaif": cfg["policies"]["rlaif"],
    }


def dataset_path(cfg, dataset: str):
    if dataset == "gsm":
        return cfg["paths"]["gsm_eval"]
    if dataset == "transfer":
        return cfg["paths"]["math_transfer_eval"]
    raise ValueError(dataset)


def generate_policy_responses(cfg, tokenizer, name, adapter, rows, batch_size=4):
    model = load_policy(cfg, adapter_path=adapter, trainable=False)
    max_new = int(cfg["math_max_new_tokens"])
    records = []
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        prompts = [row.get("messages") or [{"role": "user", "content": row["question"]}] for row in chunk]
        gen = batch_generate(
            model,
            tokenizer,
            prompts,
            max_prompt_length=256,
            max_new_tokens=max_new,
            temperature=0.0,
            top_p=1.0,
            do_sample=False,
        )
        for j, row in enumerate(chunk):
            response = gen["responses"][j]
            records.append(
                {
                    "source_index": row.get("source_index"),
                    "question": row.get("question"),
                    "gold_final": row.get("gold_final"),
                    "policy": name,
                    "response": response,
                    "response_tokens": int(gen["response_lengths"][j]),
                    "exact_reward": float(exact_reward(response, row.get("gold_final"))),
                    "format_compliant": extract_designated_final(response) is not None,
                }
            )
    clear_gpu(model)
    return records


def run_math_eval(config_path: str, dataset: str, max_prompts: int | None = None, batch_size: int = 4):
    cfg = load_yaml(config_path)
    rows = read_jsonl(dataset_path(cfg, dataset))
    if max_prompts is not None:
        rows = rows[: int(max_prompts)]
    tokenizer = load_tokenizer(cfg["base_model"])
    results_dir = repo_path(cfg["results_dir"]) / "task5_feedback"
    results_dir.mkdir(parents=True, exist_ok=True)

    per_policy = {}
    for name, adapter in policy_specs(cfg).items():
        print(f"=== generating {name} ({len(rows)} prompts) ===", flush=True)
        records = generate_policy_responses(cfg, tokenizer, name, adapter, rows, batch_size=batch_size)
        acc = statistics.fmean(r["exact_reward"] for r in records)
        fmt = statistics.fmean(1.0 if r["format_compliant"] else 0.0 for r in records)
        lengths = [r["response_tokens"] for r in records]
        per_policy[name] = {
            "records": records,
            "summary": {
                "policy": name,
                "num_examples": len(records),
                "exact_accuracy": acc,
                "format_compliance": fmt,
                "response_length_mean": statistics.fmean(lengths),
                "response_length_std": statistics.pstdev(lengths) if len(lengths) > 1 else 0.0,
            },
        }
        with (results_dir / f"generations_{dataset}_{name}.jsonl").open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    judge = PairwiseAIJudge(cfg, results_dir / "judge_cache.json")
    pairwise = {}
    for challenger in ["rlvr", "rlaif"]:
        wins = ties = losses = 0
        verifier_agree = []
        for i in range(len(rows)):
            q = per_policy["sft"]["records"][i]["question"]
            a = per_policy["sft"]["records"][i]["response"]
            b = per_policy[challenger]["records"][i]["response"]
            pref = judge.compare(q, a, b)
            if pref == "A":
                losses += 1
            elif pref == "B":
                wins += 1
            else:
                ties += 1
            ra = per_policy["sft"]["records"][i]["exact_reward"]
            rb = per_policy[challenger]["records"][i]["exact_reward"]
            if ra != rb and pref != "TIE":
                verifier_pick = "B" if rb > ra else "A"
                verifier_agree.append(1.0 if pref == verifier_pick else 0.0)
        n = len(rows)
        pairwise[challenger] = {
            "policy_win_rate_vs_sft": (wins + 0.5 * ties) / n if n else None,
            "wins": wins,
            "ties": ties,
            "losses": losses,
            "verifier_judge_agreement": statistics.fmean(verifier_agree) if verifier_agree else None,
            "num_pairs_with_verifier_disagreement": len(verifier_agree),
        }
        print(challenger, "->", json.dumps(pairwise[challenger], indent=2), flush=True)

    result = {
        "dataset": dataset,
        "num_examples": len(rows),
        "decoding": {"do_sample": False, "max_new_tokens": int(cfg["math_max_new_tokens"])},
        "policies": {k: v["summary"] for k, v in per_policy.items()},
        "pairwise_vs_sft": pairwise,
    }
    (results_dir / f"eval_{dataset}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--dataset", choices=["gsm", "transfer"], default="gsm")
    ap.add_argument("--max-prompts", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args()
    run_math_eval(args.config, args.dataset, args.max_prompts, args.batch_size)


if __name__ == "__main__":
    main()
