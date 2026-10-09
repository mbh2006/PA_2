from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from common.data import (
    encode_prompt_response,
    load_yaml,
    pad_batch,
    preference_responses,
    prompt_messages_from_preference,
    read_jsonl,
    repo_path,
)
from common.generation import response_sequence_logprobs
from common.logging_utils import set_seed
from common.models import load_policy, load_tokenizer, reference_mode, trainable_parameters
from task1_dpo.dpo import dpo_loss


def make_collate(tokenizer, max_length):
    def collate(rows):
        chosen, rejected = [], []
        for row in rows:
            prompt = prompt_messages_from_preference(row)
            yc, yr = preference_responses(row)
            chosen.append(encode_prompt_response(tokenizer, prompt, yc, max_length))
            rejected.append(encode_prompt_response(tokenizer, prompt, yr, max_length))
        return pad_batch(tokenizer, chosen), pad_batch(tokenizer, rejected)
    return collate


def filter_overlong_prompts(rows: list[dict], tokenizer, max_sequence_length: int):
    """Drop rows whose chat-templated prompt alone does not fit the sequence budget.

    `encode_prompt_response` keeps the prompt intact and refuses prompts that cannot
    fit; such rows are unusable for DPO at the released `max_sequence_length` (768).
    Filtering is deterministic and every dropped row is logged in the run summary.
    """
    kept, dropped = [], []
    for row in rows:
        messages = prompt_messages_from_preference(row)
        n_tokens = len(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True))
        if n_tokens >= int(max_sequence_length):
            dropped.append(
                {
                    "source_index": row.get("source_index"),
                    "prompt_id": row.get("prompt_id"),
                    "prompt_tokens": n_tokens,
                }
            )
        else:
            kept.append(row)
    return kept, dropped


def prepare_dpo_run(config_path: str, dataset_path: str | None = None, beta: float | None = None, max_examples: int | None = None):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))
    path = dataset_path or cfg["paths"]["dpo_standard_train"]
    rows = read_jsonl(path)

    tokenizer = load_tokenizer(cfg["base_model"])
    rows, dropped_overlong = filter_overlong_prompts(rows, tokenizer, int(cfg["max_sequence_length"]))
    if dropped_overlong:
        print(
            f"[prepare_dpo_run] filtered {len(dropped_overlong)} overlong-prompt rows "
            f"(prompt >= {cfg['max_sequence_length']} tokens); kept {len(rows)}. "
            f"first dropped: {dropped_overlong[:5]}",
            flush=True,
        )
    if max_examples is not None:
        rows = rows[: int(max_examples)]

    model = load_policy(cfg, trainable=True, fresh_lora=True)
    loader = DataLoader(
        rows,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        collate_fn=make_collate(tokenizer, int(cfg["max_sequence_length"])),
    )
    optimizer = AdamW(
        trainable_parameters(model),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
    )
    return {
        "cfg": cfg,
        "rows": rows,
        "dropped_overlong": dropped_overlong,
        "tokenizer": tokenizer,
        "model": model,
        "loader": loader,
        "optimizer": optimizer,
        "beta": float(cfg["beta"] if beta is None else beta),
    }


def run_training(config_path: str, run_name: str, dataset_path: str | None = None, output_path: str | None = None, beta: float | None = None, max_examples: int | None = None):
    """Full DPO training loop: policy/reference log-probs, grad accumulation, logging, checkpointing."""
    bundle = prepare_dpo_run(config_path, dataset_path, beta, max_examples)
    cfg = bundle["cfg"]
    model = bundle["model"]
    tokenizer = bundle["tokenizer"]
    loader = bundle["loader"]
    optimizer = bundle["optimizer"]
    beta_value = float(bundle["beta"])

    output = repo_path(output_path or cfg["standard_output"])
    output.mkdir(parents=True, exist_ok=True)

    device = next(model.parameters()).device
    grad_accum = max(1, int(cfg.get("grad_accum_steps", 1)))
    max_grad_norm = float(cfg.get("max_grad_norm", 1.0))
    epochs = max(1, int(cfg.get("epochs", 1)))

    def move(batch):
        return {k: v.to(device) for k, v in batch.items()}

    history: list[dict] = []
    t0 = time.time()

    def optimizer_step(step: int, window: int, loss_sum: float, acc_sum: float, epoch: int):
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters(model), max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        record = {
            "step": step,
            "epoch": epoch,
            "mean_loss": loss_sum / max(window, 1),
            "mean_preference_accuracy": acc_sum / max(window, 1),
            "grad_norm": float(grad_norm),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "elapsed_seconds": round(time.time() - t0, 2),
        }
        history.append(record)
        print(
            f"[{run_name}] step {step:4d} | loss {record['mean_loss']:.4f} | "
            f"pref-acc {record['mean_preference_accuracy']:.3f} | grad-norm {record['grad_norm']:.3f}",
            flush=True,
        )
        return record

    model.train()
    optimizer.zero_grad(set_to_none=True)
    step = 0
    micro = 0
    window = 0
    loss_sum = 0.0
    acc_sum = 0.0
    last_record = None

    for epoch in range(1, epochs + 1):
        for chosen, rejected in loader:
            chosen, rejected = move(chosen), move(rejected)

            policy_chosen_logp, _, _ = response_sequence_logprobs(model, chosen)
            policy_rejected_logp, _, _ = response_sequence_logprobs(model, rejected)

            with torch.no_grad(), reference_mode(model):
                ref_chosen_logp, _, _ = response_sequence_logprobs(model, chosen)
                ref_rejected_logp, _, _ = response_sequence_logprobs(model, rejected)

            loss, diag = dpo_loss(
                policy_chosen_logp,
                policy_rejected_logp,
                ref_chosen_logp,
                ref_rejected_logp,
                beta_value,
            )
            (loss / grad_accum).backward()

            micro += 1
            window += 1
            loss_sum += float(loss.detach())
            acc_sum += float(diag["preference_accuracy"])

            if micro % grad_accum == 0:
                step += 1
                last_record = optimizer_step(step, window, loss_sum, acc_sum, epoch)
                window = 0
                loss_sum = 0.0
                acc_sum = 0.0

    if window > 0:
        step += 1
        last_record = optimizer_step(step, window, loss_sum, acc_sum, epochs)

    model.save_pretrained(str(output))

    summary = {
        "run_name": run_name,
        "beta": beta_value,
        "epochs": epochs,
        "num_examples": len(bundle["rows"]),
        "num_dropped_overlong_prompts": len(bundle.get("dropped_overlong", [])),
        "dropped_overlong_prompts": bundle.get("dropped_overlong", [])[:200],
        "batch_size": int(cfg.get("batch_size", 2)),
        "grad_accum_steps": grad_accum,
        "optimizer_steps": step,
        "learning_rate": float(cfg.get("learning_rate", 0.0)),
        "max_sequence_length": int(cfg.get("max_sequence_length", 0)),
        "seed": int(cfg["seed"]),
        "output": str(output),
        "wall_clock_seconds": round(time.time() - t0, 1),
        "final_mean_loss": last_record["mean_loss"] if last_record else None,
        "final_mean_preference_accuracy": last_record["mean_preference_accuracy"] if last_record else None,
        "train_history": history,
    }
    results_dir = repo_path(cfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / f"train_{run_name}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "run_meta.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[{run_name}] done: {step} optimizer steps, {summary['wall_clock_seconds']}s -> {output}", flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--run-name", default="standard")
    ap.add_argument("--dataset")
    ap.add_argument("--output")
    ap.add_argument("--beta", type=float)
    ap.add_argument("--max-examples", type=int)
    args = ap.parse_args()
    run_training(args.config, args.run_name, args.dataset, args.output, args.beta, args.max_examples)


if __name__ == "__main__":
    main()
