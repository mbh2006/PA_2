from __future__ import annotations

import torch
import torch.nn.functional as F


def dpo_loss(
    policy_chosen_logp: torch.Tensor,
    policy_rejected_logp: torch.Tensor,
    ref_chosen_logp: torch.Tensor,
    ref_rejected_logp: torch.Tensor,
    beta: float,
):
    """Return scalar DPO loss plus lightweight diagnostics.

    Validate this implementation against the equation in the assignment manual
    before using it.
    """

    policy_margin = (
        policy_chosen_logp
        - policy_rejected_logp
    )

    ref_margin = (
        ref_chosen_logp
        - ref_rejected_logp
    )

    # FIXED by student (2026-10-09), validated against the manual's L_DPO equation:
    # DPO logit = beta * [ (log pi(y+|x) - log pi(y-|x))
    #                      - (log pi_ref(y+|x) - log pi_ref(y-|x)) ]
    # The starter used "+ ref_margin", which is NOT the DPO objective (the reference
    # term must be SUBTRACTED because it defines the relative likelihood change).
    logits = beta * (
        policy_margin - ref_margin
    )

    loss = -F.logsigmoid(
        logits
    ).mean()

    return loss, {
        "logit_mean":
            logits.detach().mean(),

        "policy_margin_mean":
            policy_margin.detach().mean(),

        "preference_accuracy": (
            (policy_margin - ref_margin) > 0
        ).float().mean().detach(),
    }
