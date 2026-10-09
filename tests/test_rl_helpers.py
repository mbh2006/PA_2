"""Hand-computed unit tests for the PPO / GRPO helper math.

No GPU or models needed — pure tensor checks against values computed by hand.

Run:
    D:\\MyTools\\Anaconda\\envs\\atml-pa0\\python.exe tests/test_rl_helpers.py
"""

from __future__ import annotations

import math
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from task2_ppo.ppo import (  # noqa: E402
    compute_gae,
    normalize_advantages,
    shaped_rewards,
    value_mse_loss,
)
from task3_grpo.grpo import grpo_policy_loss, mask_truncated_sequences  # noqa: E402


def test_gae_hand_computed():
    rewards = torch.tensor([[1.0, 2.0]])
    values = torch.tensor([[0.5, 0.7]])
    mask = torch.tensor([[1.0, 1.0]])

    adv, ret = compute_gae(rewards, values, mask, gamma=1.0, lam=1.0)

    # Hand computation (gamma=lam=1, final bootstrap zero):
    #   t=1: delta = 2.0 + 0 - 0.7 = 1.3  -> adv[1] = 1.3
    #   t=0: delta = 1.0 + 0.7 - 0.5 = 1.2 ; adv[0] = 1.2 + 1.3 = 2.5
    #   returns = advantages + values = [3.0, 2.0]
    assert torch.allclose(adv, torch.tensor([[2.5, 1.3]]), atol=1e-6), adv
    assert torch.allclose(ret, torch.tensor([[3.0, 2.0]]), atol=1e-6), ret

    # A padding position must not propagate: mask [1, 0]
    #   last valid position is t=0 and bootstraps at zero:
    #   delta = r0 + 0 - v0 = 1.0 - 0.5 = 0.5
    adv2, ret2 = compute_gae(rewards, values, torch.tensor([[1.0, 0.0]]), gamma=1.0, lam=1.0)
    assert torch.allclose(adv2, torch.tensor([[0.5, 0.0]]), atol=1e-6), adv2
    print("PASS  test_gae_hand_computed")


def test_shaped_rewards_hand_computed():
    task_reward = torch.tensor([1.0])
    policy_logp = torch.tensor([[-0.1, -0.2]])
    ref_logp = torch.tensor([[-0.3, -0.4]])
    mask = torch.tensor([[1.0, 1.0]])
    rewards = shaped_rewards(task_reward, policy_logp, ref_logp, mask, beta_kl=0.1)

    # KL shaping: -0.1 * (0.2, 0.2) = (-0.02, -0.02); task reward added at last valid token
    expected = torch.tensor([[-0.02, 0.98]])
    assert torch.allclose(rewards, expected, atol=1e-6), rewards
    print("PASS  test_shaped_rewards_hand_computed")


def test_value_loss_and_advantage_normalization():
    pred = torch.tensor([[1.0, 2.0]])
    returns = torch.tensor([[2.0, 4.0]])
    mask = torch.tensor([[1.0, 1.0]])
    # MSE = ((1-2)^2 + (2-4)^2)/2 = (1 + 4)/2 = 2.5
    loss = value_mse_loss(pred, returns, mask)
    assert math.isclose(float(loss), 2.5, abs_tol=1e-6), loss

    adv = torch.tensor([[1.0, 3.0, 99.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0]])
    norm = normalize_advantages(adv, mask)
    # masked mean = 2, std = 1 -> [-1, +1]; padded position stays 0
    assert torch.allclose(norm, torch.tensor([[-1.0, 1.0, 0.0]]), atol=1e-6), norm
    print("PASS  test_value_loss_and_advantage_normalization")


def test_grpo_loss_denominators_and_clipping():
    # ratio = 1 case: objective per token = 1 (adv=1); KL term = 0 when policy == ref
    new_logp = torch.tensor([[0.0, 0.0]])
    old_logp = torch.tensor([[0.0, 0.0]])
    ref_logp = torch.tensor([[0.0, 0.0]])
    seq_adv = torch.tensor([1.0])
    token_mask = torch.tensor([[1.0, 1.0]])

    loss_g, diag_g = grpo_policy_loss(new_logp, old_logp, seq_adv, token_mask, ref_logp, eps=0.2, beta=0.1, loss_type="grpo")
    # canonical: sum(objective)=2, denom=2 -> policy term -1; kl 0
    assert math.isclose(float(diag_g["policy_term"]), -1.0, abs_tol=1e-6), diag_g["policy_term"]

    loss_d, diag_d = grpo_policy_loss(
        new_logp, old_logp, seq_adv, token_mask, ref_logp, eps=0.2, beta=0.1, loss_type="dr_grpo", max_completion_length=4
    )
    # dr_grpo: sum(objective)=2, denom=4 -> policy term -0.5
    assert math.isclose(float(diag_d["policy_term"]), -0.5, abs_tol=1e-6), diag_d["policy_term"]

    # Clipped case: ratio = 3 (outside [0.8, 1.2]) with adv = +1 -> objective = 1.2 per token
    new_clip = torch.tensor([[math.log(3.0), math.log(3.0)]])
    _, diag_c = grpo_policy_loss(new_clip, old_logp, seq_adv, token_mask, ref_logp, eps=0.2, beta=0.0, loss_type="grpo")
    assert math.isclose(float(diag_c["policy_term"]), -1.2, abs_tol=1e-5), diag_c["policy_term"]
    assert math.isclose(float(diag_c["clip_fraction"]), 1.0, abs_tol=1e-6)
    print("PASS  test_grpo_loss_denominators_and_clipping")


def test_grpo_truncation_masking():
    token_mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
    masked = mask_truncated_sequences(token_mask, [True, False])
    assert torch.allclose(masked, torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])), masked
    print("PASS  test_grpo_truncation_masking")


if __name__ == "__main__":
    test_gae_hand_computed()
    test_shaped_rewards_hand_computed()
    test_value_loss_and_advantage_normalization()
    test_grpo_loss_denominators_and_clipping()
    test_grpo_truncation_masking()
    print("\nAll RL helper tests passed.")
