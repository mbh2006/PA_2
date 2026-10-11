from __future__ import annotations

import re
import numpy as np
import torch


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(dtype=x.dtype)
    return (x * mask).sum() / mask.sum().clamp_min(1.0)


def sampled_kl(policy_logp: torch.Tensor, ref_logp: torch.Tensor, mask: torch.Tensor):
    return masked_mean(policy_logp - ref_logp, mask)


def sample_entropy(sampled_logp: torch.Tensor, mask: torch.Tensor):
    return -masked_mean(sampled_logp, mask)


def full_vocab_entropy_sums(logits: torch.Tensor, mask: torch.Tensor, chunk_size: int = 1):
    """Full token-level policy entropy over the ENTIRE vocabulary, as defined in the manual:

        H_t = -sum_v pi(v | s_t) * log pi(v | s_t), averaged over valid response tokens.

    `logits` are raw next-token logits [B, R, V] (float32) at the response positions;
    `mask` marks valid response tokens [B, R]. Computed chunk-wise over the batch
    dimension to bound peak memory. Returns (sum_over_tokens, count) so callers can
    aggregate token-weighted across batches. Stable form: H = logsumexp(z) - sum(p * z).
    """
    if logits.numel() == 0:
        return 0.0, 0.0
    total = 0.0
    count = 0.0
    for i in range(0, logits.shape[0], chunk_size):
        z = logits[i : i + chunk_size].float()
        m = mask[i : i + chunk_size].float()
        lse = torch.logsumexp(z, dim=-1)  # [c, R]
        p = torch.exp(z - lse.unsqueeze(-1))  # softmax over vocab
        h = lse - (p * z).sum(dim=-1)  # per-token entropy [c, R]
        total += float((h * m).sum())
        count += float(m.sum())
    return total, count


def mean_entropy_full_vocab(logits: torch.Tensor, mask: torch.Tensor):
    total, count = full_vocab_entropy_sums(logits, mask)
    return (total / count) if count else None


def mean_response_length(mask: torch.Tensor):
    return float(mask.sum(-1).float().mean().item())


def preference_accuracy(chosen_logp, rejected_logp):
    return float((chosen_logp > rejected_logp).float().mean().item())


def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def parse_word_limit(prompt: str):
    patterns = [
        r"(?:at most|no more than|under|within)\s+(\d+)\s+words?",
        r"(?:in|use)\s+(\d+)\s+words?\s+(?:or fewer|max(?:imum)?)",
        r"(?:maximum|max)\s+(?:of\s+)?(\d+)\s+words?",
    ]
    lower = str(prompt).lower()
    for pattern in patterns:
        m = re.search(pattern, lower)
        if m:
            return int(m.group(1))
    return None


def word_limit_compliance(prompt: str, response: str):
    limit = parse_word_limit(prompt)
    if limit is None:
        return None
    return float(word_count(response) <= limit)


def safe_corr(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])
