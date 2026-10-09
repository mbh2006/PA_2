from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.optim import AdamW

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.generation import batch_generate, response_token_logprobs, score_reward_pairs
from common.logging_utils import set_seed
from common.metrics import masked_mean, sampled_kl, sample_entropy
from common.models import (
    load_policy,
    load_reward_model,
    load_tokenizer,
    load_value_model,
    reference_mode,
    token_values,
    trainable_parameters,
    value_parameter_groups,
)
from task2_ppo.ppo import compute_gae, ppo_policy_loss, shaped_rewards, value_mse_loss


def prepare_ppo_continuation(config_path: str):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))

    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(
        cfg,
        adapter_path=cfg["paths"]["ppo_midpoint_policy"],
        trainable=True,
    )
    value_model = load_value_model(
        cfg,
        cfg["paths"]["ppo_midpoint_value"],
        train_mode=cfg.get("value_train_mode", "head_only"),
    )
    reward_model, reward_tokenizer = load_reward_model(cfg)
    prompts = read_jsonl(cfg["paths"]["rl_prompt_train"])

    policy_optimizer = AdamW(
        trainable_parameters(policy),
        lr=float(cfg["policy_learning_rate"]),
    )
    value_optimizer = AdamW(
        value_parameter_groups(
            value_model,
            lora_lr=float(cfg["value_lora_learning_rate"]),
            head_lr=float(cfg["value_head_learning_rate"]),
        ),
        weight_decay=0.0,
    )

    return {
        "cfg": cfg,
        "tokenizer": tokenizer,
        "policy": policy,
        "value_model": value_model,
        "reward_model": reward_model,
        "reward_tokenizer": reward_tokenizer,
        "prompt_rows": prompts,
        "policy_optimizer": policy_optimizer,
        "value_optimizer": value_optimizer,
    }


def run_ppo(config_path: str, output: str | None = None, updates: int | None = None, clip_epsilon: float | None = None, kl_beta: float | None = None, run_name: str = "standard"):
    """PPO continuation from the supplied midpoint: rollouts -> GAE -> clipped update, with full logging."""
    bundle = prepare_ppo_continuation(config_path)
    cfg = bundle["cfg"]
    policy = bundle["policy"]
    value_model = bundle["value_model"]
    tokenizer = bundle["tokenizer"]
    reward_model = bundle["reward_model"]
    reward_tokenizer = bundle["reward_tokenizer"]
    prompt_rows = bundle["prompt_rows"]
    policy_optimizer = bundle["policy_optimizer"]
    value_optimizer = bundle["value_optimizer"]

    n_updates = int(updates if updates is not None else cfg["updates"])
    eps = float(clip_epsilon if clip_epsilon is not None else cfg["clip_epsilon"])
    klib = float(kl_beta if kl_beta is not None else cfg["kl_beta"])
    prompts_per_update = int(cfg.get("prompts_per_update", 1))
    ppo_epochs = int(cfg.get("ppo_epochs", 2))
    gamma = float(cfg.get("gamma", 1.0))
    lam = float(cfg.get("gae_lambda", 0.95))
    value_coef = float(cfg.get("value_coef", 0.5))
    max_grad_norm = float(cfg.get("max_grad_norm", 1.0))
    missing_eos_penalty = float(cfg.get("missing_eos_penalty", 0.0))
    max_prompt_length = int(cfg["max_prompt_length"])
    max_response_length = int(cfg["max_response_length"])
    reward_max_length = int(cfg.get("reward_max_length", 1280))
    gen_cfg = cfg.get("generation", {})

    device = next(policy.parameters()).device
    out = repo_path(output or cfg["output"])
    out.mkdir(parents=True, exist_ok=True)
    results_dir = repo_path(cfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    rollouts_path = out / "rollouts.jsonl"

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    history: list[dict] = []
    t0 = time.time()
    policy.train()
    value_model.train()

    with rollouts_path.open("w", encoding="utf-8") as rollouts_file:
        for i in range(n_updates):
            t_upd = time.time()
            row = prompt_rows[(i * prompts_per_update) % len(prompt_rows)]
            messages = prompt_messages(row)

            gen = batch_generate(
                policy,
                tokenizer,
                [messages],
                max_prompt_length=max_prompt_length,
                max_new_tokens=max_response_length,
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

            with torch.no_grad():
                old_logp, _ = response_token_logprobs(policy, seq, attn, pw, rid)
                with reference_mode(policy):
                    ref_logp, _ = response_token_logprobs(policy, seq, attn, pw, rid)
                values = token_values(value_model, seq, attn)[:, pw:]

            raw_reward = float(
                score_reward_pairs(
                    reward_model, reward_tokenizer, [messages], [gen["responses"][0]], max_length=reward_max_length
                )[0]
            )
            terminated = bool(gen["terminated_with_eos"][0])
            effective_reward = raw_reward - (missing_eos_penalty if not terminated else 0.0)
            task_reward = torch.tensor([effective_reward], device=device)

            rewards = shaped_rewards(task_reward, old_logp.detach(), ref_logp, rmask, klib)
            advantages, returns = compute_gae(rewards, values, rmask, gamma=gamma, lam=lam)

            pol_loss_v = val_loss_v = clip_frac_v = ratio_v = entropy_v = None
            p_grad = v_grad = None
            for _ in range(ppo_epochs):
                new_logp, _ = response_token_logprobs(policy, seq, attn, pw, rid)
                pol_loss, ratio, clip_frac = ppo_policy_loss(new_logp, old_logp, advantages, rmask, eps=eps)
                v_pred = token_values(value_model, seq, attn)[:, pw:]
                val_loss = value_mse_loss(v_pred, returns, rmask)
                loss = pol_loss + value_coef * val_loss
                loss.backward()

                p_grad = torch.nn.utils.clip_grad_norm_(trainable_parameters(policy), max_grad_norm)
                v_grad = torch.nn.utils.clip_grad_norm_(trainable_parameters(value_model), max_grad_norm)
                policy_optimizer.step()
                value_optimizer.step()
                policy_optimizer.zero_grad(set_to_none=True)
                value_optimizer.zero_grad(set_to_none=True)

                pol_loss_v = float(pol_loss.detach())
                val_loss_v = float(val_loss.detach())
                clip_frac_v = float(clip_frac)
                ratio_v = float(masked_mean(ratio, rmask))
                entropy_v = float(sample_entropy(new_logp.detach(), rmask))

            kl_value = float(sampled_kl(old_logp, ref_logp, rmask))
            record = {
                "update": i + 1,
                "source_index": row.get("source_index"),
                "prompt_id": row.get("prompt_id"),
                "raw_reward": raw_reward,
                "effective_reward": effective_reward,
                "kl": kl_value,
                "policy_loss": pol_loss_v,
                "value_loss": val_loss_v,
                "entropy": entropy_v,
                "clip_fraction": clip_frac_v,
                "ratio_mean": ratio_v,
                "grad_norm_policy": float(p_grad),
                "grad_norm_value": float(v_grad),
                "response_tokens": int(rmask.sum().item()),
                "terminated_with_eos": terminated,
                "truncated": bool(gen["truncated"][0]),
                "elapsed_seconds": round(time.time() - t_upd, 1),
            }
            history.append(record)
            rollouts_file.write(
                json.dumps(
                    {
                        "update": i + 1,
                        "source_index": row.get("source_index"),
                        "prompt_id": row.get("prompt_id"),
                        "response": gen["responses"][0],
                        "raw_reward": raw_reward,
                        "effective_reward": effective_reward,
                        "kl": kl_value,
                        "response_tokens": record["response_tokens"],
                        "terminated_with_eos": terminated,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            rollouts_file.flush()

            print(
                f"[{run_name}] upd {i + 1:3d}/{n_updates} | reward {effective_reward:+.3f} | KL {kl_value:+.4f} "
                f"| pi-loss {pol_loss_v:+.4f} | v-loss {val_loss_v:.4f} | clip {clip_frac_v:.3f} "
                f"| ent {entropy_v:+.3f} | len {record['response_tokens']:3d} | {record['elapsed_seconds']}s",
                flush=True,
            )

    policy.save_pretrained(str(out))
    value_model.save_pretrained(str(out / "value_adapter"))

    peak_vram = None
    if torch.cuda.is_available():
        peak_vram = round(torch.cuda.max_memory_allocated() / 2**30, 2)

    summary = {
        "run_name": run_name,
        "updates": n_updates,
        "clip_epsilon": eps,
        "kl_beta": klib,
        "prompts_per_update": prompts_per_update,
        "ppo_epochs": ppo_epochs,
        "gamma": gamma,
        "gae_lambda": lam,
        "value_coef": value_coef,
        "missing_eos_penalty": missing_eos_penalty,
        "max_response_length": max_response_length,
        "seed": int(cfg["seed"]),
        "output": str(out),
        "wall_clock_seconds": round(time.time() - t0, 1),
        "peak_vram_gib": peak_vram,
        "mean_reward_last5": float(sum(r["effective_reward"] for r in history[-5:]) / max(len(history[-5:]), 1)),
        "history": history,
    }
    (results_dir / f"ppo_{run_name}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[{run_name}] done: {n_updates} updates, {summary['wall_clock_seconds']}s, peak VRAM {peak_vram} GiB -> {out}", flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--output")
    ap.add_argument("--updates", type=int)
    ap.add_argument("--clip-epsilon", type=float)
    ap.add_argument("--kl-beta", type=float)
    ap.add_argument("--run-name", default="standard")
    args = ap.parse_args()
    run_ppo(args.config, args.output, args.updates, args.clip_epsilon, args.kl_beta, args.run_name)


if __name__ == "__main__":
    main()
