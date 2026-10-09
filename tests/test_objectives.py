"""Objective-implementation validation tests for ATML PA2.

Each test encodes the equation from the assignment manual and checks the fixed
implementation against a hand-computed reference, plus checks that the original
(starter) defect would give a different answer.

Run standalone (no pytest needed):
    D:\\MyTools\\Anaconda\\envs\\atml-pa0\\python.exe tests/test_objectives.py
"""

from __future__ import annotations

import math
import pathlib
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from task1_dpo.dpo import dpo_loss  # noqa: E402
from task2_ppo.ppo import ppo_policy_loss  # noqa: E402
from task3_grpo.grpo import group_relative_advantages  # noqa: E402


def test_dpo_matches_manual_equation():
    pc = torch.tensor([-1.0, -2.0])
    pr = torch.tensor([-3.0, -1.5])
    rc = torch.tensor([-1.2, -2.2])
    rr = torch.tensor([-2.5, -2.2])
    beta = 0.1

    loss, diag = dpo_loss(pc, pr, rc, rr, beta)

    # Manual: logits = beta * [(pc - pr) - (rc - rr)]; loss = -mean(logsigmoid(logits))
    margin = (pc - pr) - (rc - rr)
    expected = -F.logsigmoid(beta * margin).mean()
    assert torch.allclose(loss, expected, atol=1e-6), (loss, expected)

    # The starter defect (adding the ref margin) gives a different objective.
    buggy = -F.logsigmoid(beta * ((pc - pr) + (rc - rr))).mean()
    assert not torch.allclose(loss, buggy, atol=1e-6)

    # Registered preference accuracy must use the policy-minus-ref margin.
    assert math.isclose(
        float(diag["preference_accuracy"]), float((margin > 0).float().mean()), abs_tol=1e-6
    )
    print("PASS  test_dpo_matches_manual_equation")


def test_ppo_clipped_surrogate_is_min():
    eps = 0.2
    mask = torch.ones(1)

    # Positive advantage, ratio far above 1+eps: min picks the clipped branch.
    new = torch.log(torch.tensor([3.0]))
    old = torch.zeros(1)
    loss, ratio, clip_frac = ppo_policy_loss(new, old, torch.tensor([1.0]), mask, eps=eps)
    assert torch.allclose(ratio, torch.tensor([3.0]), atol=1e-5)
    # manual: surr1 = 3*1 = 3 ; surr2 = 1.2*1 = 1.2 ; objective = min = 1.2 -> loss = -1.2
    assert math.isclose(loss.item(), -1.2, abs_tol=1e-5), loss.item()
    assert math.isclose(float(clip_frac), 1.0, abs_tol=1e-6)

    # Negative advantage, ratio far above 1+eps: min picks the UNCLIPPED branch.
    loss2, _, _ = ppo_policy_loss(new, old, torch.tensor([-1.0]), mask, eps=eps)
    # manual: surr1 = -3 ; surr2 = -1.2 ; objective = min = -3 -> loss = 3
    assert math.isclose(loss2.item(), 3.0, abs_tol=1e-5), loss2.item()

    # The starter defect (maximum) would give -3.0 for the first case.
    loss_buggy = -max(3.0, 1.2)
    assert not math.isclose(loss.item(), loss_buggy, abs_tol=1e-6)
    print("PASS  test_ppo_clipped_surrogate_is_min")


def test_grpo_advantages_are_within_group():
    rewards = torch.tensor([10.0, 12.0, 5.0, 6.0])
    group_ids = torch.tensor([0, 0, 1, 1])

    adv = group_relative_advantages(rewards, group_ids)
    # Hand-computed: group 0 -> mean 11, std 1 -> [-1, +1]; group 1 -> mean 5.5, std 0.5 -> [-1, +1]
    assert torch.allclose(adv, torch.tensor([-1.0, 1.0, -1.0, 1.0]), atol=1e-5), adv

    # Global normalization (the starter defect) gives very different values.
    global_adv = (rewards - rewards.mean()) / rewards.std(unbiased=False).clamp_min(1e-6)
    assert not torch.allclose(adv, global_adv, atol=1e-6)

    # Zero-variance group -> zero advantages (no relative signal).
    adv_zero = group_relative_advantages(torch.tensor([1.0, 1.0]), torch.tensor([0, 0]))
    assert torch.allclose(adv_zero, torch.zeros(2), atol=1e-6)

    # Size-1 group -> zero advantage by convention.
    adv_one = group_relative_advantages(torch.tensor([3.0]), torch.tensor([7]))
    assert torch.allclose(adv_one, torch.zeros(1), atol=1e-6)
    print("PASS  test_grpo_advantages_are_within_group")


if __name__ == "__main__":
    test_dpo_matches_manual_equation()
    test_ppo_clipped_surrogate_is_min()
    test_grpo_advantages_are_within_group()
    print("\nAll objective validation tests passed.")
