from __future__ import annotations

import argparse
import json
import statistics
import time

import torch
from torch.optim import AdamW

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.generation import batch_generate, response_token_logprobs, score_reward_pairs
from common.logging_utils import set_seed
from common.metrics import full_vocab_entropy_sums, masked_mean, sampled_kl, sample_entropy
from common.models import load_policy, load_reward_model, load_tokenizer, reference_mode, trainable_parameters
from task3_grpo.grpo import group_relative_advantages, grpo_policy_loss, mask_truncated_sequences

GROUP_STD_TOL = 1e-6


def prepare_grpo_continuation(config_path: str):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))
    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(
        cfg,
        adapter_path=cfg["paths"]["grpo_midpoint_policy"],
        trainable=True,
    )
    reward_model, reward_tokenizer = load_reward_model(cfg)
    prompts = read_jsonl(cfg["paths"]["rl_prompt_train"])
    optimizer = AdamW(trainable_parameters(policy), lr=float(cfg["learning_rate"]))
    return {
        "cfg": cfg,
        "tokenizer": tokenizer,
        "policy": policy,
        "reward_model": reward_model,
        "reward_tokenizer": reward_tokenizer,
        "prompt_rows": prompts,
        "optimizer": optimizer,
    }


def run_grpo(config_path: str, output: str | None = None, updates: int | None = None, loss_type: str = "grpo", run_name: str = "standard"):
    """GRPO continuation from the supplied midpoint: K completions per prompt, group-relative advantages, clipped objective."""
    bundle = prepare_grpo_continuation(config_path)
    cfg = bundle["cfg"]
    policy = bundle["policy"]
    tokenizer = bundle["tokenizer"]
    reward_model = bundle["reward_model"]
    reward_tokenizer = bundle["reward_tokenizer"]
    prompt_rows = bundle["prompt_rows"]
    optimizer = bundle["optimizer"]

    n_updates = int(updates if updates is not None else cfg["updates"])
    prompts_per_update = int(cfg.get("prompts_per_update", 1))
    k_gen = int(cfg["num_generations"])
    policy_epochs = int(cfg.get("policy_epochs", 1))
    eps = float(cfg["clip_epsilon"])
    beta = float(cfg["kl_beta"])
    max_grad_norm = float(cfg.get("max_grad_norm", 1.0))
    max_prompt_length = int(cfg["max_prompt_length"])
    max_completion_length = int(cfg["max_completion_length"])
    mask_truncated = bool(cfg.get("mask_truncated_completions", True))
    reward_max_length = int(cfg.get("reward_max_length", 1280))
    gen_cfg = cfg.get("generation", {})

    device = next(policy.parameters()).device
    out = repo_path(output or cfg["output"])
    out.mkdir(parents=True, exist_ok=True)
    results_dir = repo_path(cfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    rollouts_path = out / f"rollouts_{loss_type}.jsonl"

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    history: list[dict] = []
    t0 = time.time()
    policy.train()

    with rollouts_path.open("w", encoding="utf-8") as rollouts_file:
        for i in range(n_updates):
            t_upd = time.time()
            row = prompt_rows[(i * prompts_per_update) % len(prompt_rows)]
            messages = prompt_messages(row)

            gen = batch_generate(
                policy,
                tokenizer,
                [messages] * k_gen,
                max_prompt_length=max_prompt_length,
                max_new_tokens=max_completion_length,
                temperature=float(gen_cfg.get("temperature", 0.7)),
                top_p=float(gen_cfg.get("top_p", 0.9)),
                do_sample=bool(gen_cfg.get("do_sample", True)),
            )
            # batch_generate returns tensors created inside torch.inference_mode();
            # clone them into normal tensors before any autograd-tracked use.
            seq = gen["sequences"].clone()
            attn = gen["attention_mask"].clone()
            pw = gen["prompt_width"]
            rid = gen["response_ids"].clone()
            rmask = gen["response_mask"].clone().to(device)
            token_mask = mask_truncated_sequences(rmask, gen["truncated"]) if mask_truncated else rmask

            with torch.no_grad():
                old_logp, _ = response_token_logprobs(policy, seq, attn, pw, rid)
                with reference_mode(policy):
                    ref_logp, _ = response_token_logprobs(policy, seq, attn, pw, rid)

            rewards_t = score_reward_pairs(
                reward_model, reward_tokenizer, [messages] * k_gen, gen["responses"], max_length=reward_max_length
            ).to(device)
            group_ids = torch.zeros(k_gen, dtype=torch.long, device=device)
            seq_adv = group_relative_advantages(rewards_t, group_ids)

            group_std = float(rewards_t.std(unbiased=False)) if k_gen > 1 else 0.0
            uninformative = group_std <= GROUP_STD_TOL

            pol_loss_v = None
            grad_norm_v = None
            entropy_full_v = None
            diag = {}
            nonfinite_skips = 0
            for _ in range(policy_epochs):
                new_logp, new_logits = response_token_logprobs(policy, seq, attn, pw, rid)
                loss, diag = grpo_policy_loss(
                    new_logp,
                    old_logp,
                    seq_adv,
                    token_mask,
                    ref_logp,
                    eps=eps,
                    beta=beta,
                    loss_type=loss_type,
                    max_completion_length=max_completion_length,
                )
                if not bool(torch.isfinite(loss).item()):
                    nonfinite_skips += 1
                    print(f"[{run_name}] update {i + 1}: non-finite loss; skipping this epoch", flush=True)
                    optimizer.zero_grad(set_to_none=True)
                    continue
                loss.backward()
                grad_norm_v = torch.nn.utils.clip_grad_norm_(trainable_parameters(policy), max_grad_norm)
                if not bool(torch.isfinite(grad_norm_v).item()):
                    nonfinite_skips += 1
                    print(f"[{run_name}] update {i + 1}: non-finite grad norm; skipping step", flush=True)
                    optimizer.zero_grad(set_to_none=True)
                    continue
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                pol_loss_v = float(loss.detach())
                # manual metric: full token-level policy entropy over the entire vocabulary
                with torch.no_grad():
                    fsum, fcount = full_vocab_entropy_sums(new_logits.detach(), token_mask)
                entropy_full_v = (fsum / fcount) if fcount else float("nan")

            if pol_loss_v is None:
                pol_loss_v = float("nan")
            if grad_norm_v is None:
                grad_norm_v = float("nan")
            if entropy_full_v is None:
                entropy_full_v = float("nan")
            kl_value = float(sampled_kl(old_logp, ref_logp, rmask))
            record = {
                "update": i + 1,
                "source_index": row.get("source_index"),
                "prompt_id": row.get("prompt_id"),
                "mean_reward": float(rewards_t.mean()),
                "group_reward_std": group_std,
                "uninformative_group": bool(uninformative),
                "kl": kl_value,
                "policy_loss": pol_loss_v,
                "grad_norm_policy": float(grad_norm_v),
                "entropy": entropy_full_v,
                "entropy_proxy": float(diag.get("sample_entropy", torch.tensor(float("nan")))),
                "clip_fraction": float(diag.get("clip_fraction", float("nan"))),
                "ratio_mean": float(diag.get("ratio_mean", float("nan"))),
                "response_tokens_mean": float(rmask.sum(-1).float().mean()),
                "truncated_count": int(sum(gen["truncated"])),
                "loss_type": loss_type,
                "elapsed_seconds": round(time.time() - t_upd, 1),
                "nonfinite_skips": nonfinite_skips,
            }
            history.append(record)
            rollouts_file.write(
                json.dumps(
                    {
                        "update": i + 1,
                        "source_index": row.get("source_index"),
                        "prompt_id": row.get("prompt_id"),
                        "responses": gen["responses"],
                        "rewards": [float(x) for x in rewards_t.tolist()],
                        "truncated": [bool(t) for t in gen["truncated"]],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            rollouts_file.flush()

            print(
                f"[{run_name}] upd {i + 1:3d}/{n_updates} | reward {record['mean_reward']:+.3f} | gstd {group_std:.3f} "
                f"| KL {kl_value:+.4f} | loss {pol_loss_v:+.4f} | grad {record['grad_norm_policy']:.3f} "
                f"| ent {record['entropy']:+.3f} | len {record['response_tokens_mean']:.0f} | {record['elapsed_seconds']}s",
                flush=True,
            )

    policy.save_pretrained(str(out))
    peak_vram = round(torch.cuda.max_memory_allocated() / 2**30, 2) if torch.cuda.is_available() else None
    summary = {
        "run_name": run_name,
        "loss_type": loss_type,
        "updates": n_updates,
        "num_generations": k_gen,
        "clip_epsilon": eps,
        "kl_beta": beta,
        "mask_truncated_completions": mask_truncated,
        "max_completion_length": max_completion_length,
        "seed": int(cfg["seed"]),
        "output": str(out),
        "wall_clock_seconds": round(time.time() - t0, 1),
        "peak_vram_gib": peak_vram,
        "mean_reward_last5": statistics.fmean([r["mean_reward"] for r in history[-5:]]) if history else None,
        "uninformative_group_rate": (sum(1 for r in history if r["uninformative_group"]) / len(history)) if history else None,
        "history": history,
    }
    (results_dir / f"grpo_{run_name}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[{run_name}] done: {n_updates} updates, {summary['wall_clock_seconds']}s, peak VRAM {peak_vram} GiB -> {out}", flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--output")
    ap.add_argument("--updates", type=int)
    ap.add_argument("--loss-type", choices=["grpo", "dr_grpo"], default="grpo")
    ap.add_argument("--run-name", default="standard")
    args = ap.parse_args()
    run_grpo(args.config, args.output, args.updates, args.loss_type, args.run_name)


if __name__ == "__main__":
    main()
