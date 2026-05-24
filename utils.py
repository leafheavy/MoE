"""
utils.py — Loss functions, metrics, and training utilities
==========================================================
Key components:
  • entropy_loss      — minimize routing entropy (encourage confident decisions)
  • load_balance_loss — Shazeer-style auxiliary loss for uniform expert load
  • compute_moe_loss  — combines both auxiliary losses for all router outputs
  • compute_expert_stats — per-expert utilization stats for logging
  • get_lr_scheduler  — linear warmup + cosine decay
"""

import math
from typing import Dict, List

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Entropy loss (per-token, averaged over batch)
# ---------------------------------------------------------------------------

def entropy_loss(probs: torch.Tensor) -> torch.Tensor:
    """
    Mean entropy of router probability distributions.
    Minimizing this encourages each token to strongly prefer one expert/pool.

    Args:
        probs: [..., n_choices] — softmax probabilities
    Returns:
        scalar — mean entropy (lower = more confident routing)
    """
    # Clamp for numerical safety
    safe_probs = probs.clamp(min=1e-9)
    ent = -(safe_probs * safe_probs.log()).sum(dim=-1)  # [...] token-level entropy
    return ent.mean()


# ---------------------------------------------------------------------------
# Load balance loss (Shazeer-style auxiliary loss)
# ---------------------------------------------------------------------------

def load_balance_loss(probs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Encourages uniform load across experts within a pool.

    Follows Switch Transformer: L_balance = n_experts * sum(f_i * p_i)
    where:
      f_i = fraction of tokens dispatched to expert i (based on argmax)
      p_i = mean router probability assigned to expert i

    Args:
        probs: [BT, n_experts] — router probabilities for this pool
        mask:  [BT] — boolean mask of tokens belonging to this pool
    Returns:
        scalar auxiliary loss
    """
    if not mask.any():
        return probs.new_tensor(0.0)

    pool_probs = probs[mask]             # [M, n_experts]
    M, n_experts = pool_probs.shape

    # f_i: fraction of tokens routed to expert i (hard dispatch count)
    top1 = pool_probs.argmax(dim=-1)                    # [M]
    counts = torch.bincount(top1, minlength=n_experts).float()  # [n_experts]
    f = counts / M                                       # [n_experts]

    # p_i: mean soft probability for expert i
    p = pool_probs.mean(dim=0)                          # [n_experts]

    return n_experts * (f * p).sum()


# ---------------------------------------------------------------------------
# Combined MoE auxiliary loss over all router outputs
# ---------------------------------------------------------------------------

def compute_moe_loss(
    router_outputs,           # List[RouterOutput]
    lambda_entropy: float,
    lambda_balance: float,
) -> Dict[str, torch.Tensor]:
    """
    Aggregate entropy + load-balance losses over all MoE layers.

    Returns a dict with individual loss components and the total aux loss.
    """
    total_entropy = torch.tensor(0.0)
    total_balance = torch.tensor(0.0)
    device = None

    for ro in router_outputs:
        device = ro.stage1_probs.device
        total_entropy = total_entropy.to(device)
        total_balance = total_balance.to(device)

        # ── Entropy losses ─────────────────────────────────────────────
        total_entropy += entropy_loss(ro.stage1_probs)
        total_entropy += entropy_loss(ro.stable_probs)
        total_entropy += entropy_loss(ro.transfer_probs)

        # ── Load balance losses ────────────────────────────────────────
        total_balance += load_balance_loss(ro.stable_probs, ro.stable_mask)
        total_balance += load_balance_loss(ro.transfer_probs, ro.transfer_mask)

    n = max(len(router_outputs), 1)
    total_entropy /= n
    total_balance /= n

    loss_aux = lambda_entropy * total_entropy + lambda_balance * total_balance

    return {
        "loss_entropy": total_entropy,
        "loss_balance": total_balance,
        "loss_aux": loss_aux,
    }


# ---------------------------------------------------------------------------
# Expert utilization statistics (for logging / debugging)
# ---------------------------------------------------------------------------

def compute_expert_stats(router_outputs) -> Dict[str, float]:
    """
    Compute per-layer statistics for monitoring routing behavior.

    Returns dict with:
      - stable_frac:     mean fraction of tokens sent to stable pool
      - transfer_frac:   mean fraction sent to transfer pool
      - stable_entropy:  mean routing entropy within stable pool
      - transfer_entropy: mean routing entropy within transfer pool
      - stable_load_std: std of expert load in stable pool
      - transfer_load_std: std of expert load in transfer pool
    """
    if not router_outputs:
        return {}

    stable_fracs, transfer_fracs = [], []
    stable_ents, transfer_ents = [], []
    stable_stds, transfer_stds = [], []

    for ro in router_outputs:
        BT = ro.stable_mask.shape[0]
        stable_fracs.append(ro.stable_mask.float().mean().item())
        transfer_fracs.append(ro.transfer_mask.float().mean().item())

        stable_ents.append(entropy_loss(ro.stable_probs).item())
        transfer_ents.append(entropy_loss(ro.transfer_probs).item())

        # Load std: how unevenly are tokens distributed across experts?
        stable_load = ro.stable_probs.mean(dim=0)    # [n_stable]
        transfer_load = ro.transfer_probs.mean(dim=0)  # [n_transfer]
        stable_stds.append(stable_load.std().item())
        transfer_stds.append(transfer_load.std().item())

    def avg(lst):
        return sum(lst) / len(lst) if lst else 0.0

    return {
        "stable_frac": avg(stable_fracs),
        "transfer_frac": avg(transfer_fracs),
        "stable_entropy": avg(stable_ents),
        "transfer_entropy": avg(transfer_ents),
        "stable_load_std": avg(stable_stds),
        "transfer_load_std": avg(transfer_stds),
    }


# ---------------------------------------------------------------------------
# Learning rate scheduler: linear warmup + cosine decay
# ---------------------------------------------------------------------------

def get_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.1,
):
    """
    Linear warmup from 0 → lr, then cosine decay → min_lr_ratio * lr.
    """
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr_ratio, cosine)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Perplexity
# ---------------------------------------------------------------------------

def perplexity(loss_lm: float) -> float:
    """Convert mean cross-entropy loss to perplexity."""
    return math.exp(min(loss_lm, 20.0))


# ---------------------------------------------------------------------------
# Set global random seeds for reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
