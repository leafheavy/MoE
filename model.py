"""
Heterogeneous Two-Stage Mixture-of-Experts Model
=================================================
Architecture:
  - Expert: FFN with configurable hidden dim (small=stable, large=transferable)
  - TwoStageRouter:
      Stage 1 — lightweight linear classifier assigns each token to a pool
                 (stable vs transferable) with a hard argmax (straight-through).
      Stage 2 — within each pool, a learned linear router selects top-k experts
                 via noisy top-k gating.
  - HeterogeneousMoELayer: wraps both pools + router into a drop-in FFN replacement.
  - TransformerBlock: multi-head attention + MoE layer (every block) or standard
                       FFN (alternating blocks).
  - HeterogeneousMoEModel: full autoregressive language model.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data class for router outputs (returned alongside forward pass for logging)
# ---------------------------------------------------------------------------
@dataclass
class RouterOutput:
    stage1_probs: torch.Tensor      # [BT, 2]  pool assignment probabilities
    stable_mask: torch.Tensor       # [BT]     bool mask — tokens in stable pool
    transfer_mask: torch.Tensor     # [BT]     bool mask — tokens in transfer pool
    stable_probs: torch.Tensor      # [BT, n_stable]
    transfer_probs: torch.Tensor    # [BT, n_transfer]
    stable_topk_idx: torch.Tensor   # [BT, top_k]
    stable_topk_w: torch.Tensor     # [BT, top_k]
    transfer_topk_idx: torch.Tensor # [BT, top_k]
    transfer_topk_w: torch.Tensor   # [BT, top_k]


# ---------------------------------------------------------------------------
# Expert: a single FFN block
# ---------------------------------------------------------------------------
class Expert(nn.Module):
    """Feed-forward expert with GELU activation."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(F.gelu(self.fc1(x))))


# ---------------------------------------------------------------------------
# TwoStageRouter
# ---------------------------------------------------------------------------
class TwoStageRouter(nn.Module):
    """
    Stage 1: assign each token to 'stable' (0) or 'transferable' (1) pool
             via a lightweight linear + hard argmax (gradients flow through
             the stage-2 soft weights only).
    Stage 2: within each pool, select top-k experts with noisy softmax gating.
    """

    def __init__(
        self,
        d_model: int,
        n_stable: int,
        n_transfer: int,
        top_k: int = 2,
        noise_std: float = 1e-2,
    ):
        super().__init__()
        self.n_stable = n_stable
        self.n_transfer = n_transfer
        self.top_k = top_k
        self.noise_std = noise_std

        # Stage 1: pool classifier (2 logits → stable or transferable)
        self.stage1 = nn.Linear(d_model, 2, bias=False)

        # Stage 2: within-pool routers
        self.router_stable = nn.Linear(d_model, n_stable, bias=False)
        self.router_transfer = nn.Linear(d_model, n_transfer, bias=False)

    def _top_k_route(
        self, logits: torch.Tensor, top_k: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply noisy top-k gating.
        Returns:
            probs     [BT, n_experts] — full softmax probabilities
            topk_idx  [BT, top_k]    — selected expert indices
            topk_w    [BT, top_k]    — renormalized weights for selected experts
        """
        if self.training and self.noise_std > 0:
            logits = logits + torch.randn_like(logits) * self.noise_std
        probs = F.softmax(logits, dim=-1)
        topk_vals, topk_idx = probs.topk(top_k, dim=-1)
        # Renormalize selected weights so they sum to 1
        topk_w = topk_vals / (topk_vals.sum(dim=-1, keepdim=True) + 1e-9)
        return probs, topk_idx, topk_w

    def forward(self, x: torch.Tensor) -> RouterOutput:
        """
        Args:
            x: [B, T, d_model]
        Returns:
            RouterOutput dataclass
        """
        B, T, D = x.shape
        flat = x.reshape(B * T, D)  # [BT, D]

        # ── Stage 1: pool assignment ──────────────────────────────────────
        s1_logits = self.stage1(flat)           # [BT, 2]
        stage1_probs = F.softmax(s1_logits, dim=-1)
        pool_ids = stage1_probs.argmax(dim=-1)  # hard assignment [BT]
        stable_mask = pool_ids == 0             # [BT]
        transfer_mask = pool_ids == 1

        # ── Stage 2: within-pool routing ─────────────────────────────────
        stable_probs, stable_topk_idx, stable_topk_w = self._top_k_route(
            self.router_stable(flat), min(self.top_k, self.n_stable)
        )
        transfer_probs, transfer_topk_idx, transfer_topk_w = self._top_k_route(
            self.router_transfer(flat), min(self.top_k, self.n_transfer)
        )

        return RouterOutput(
            stage1_probs=stage1_probs,
            stable_mask=stable_mask,
            transfer_mask=transfer_mask,
            stable_probs=stable_probs,
            transfer_probs=transfer_probs,
            stable_topk_idx=stable_topk_idx,
            stable_topk_w=stable_topk_w,
            transfer_topk_idx=transfer_topk_idx,
            transfer_topk_w=transfer_topk_w,
        )


# ---------------------------------------------------------------------------
# Expert dispatch helper
# ---------------------------------------------------------------------------
def dispatch_to_experts(
    flat: torch.Tensor,       # [BT, D]
    mask: torch.Tensor,       # [BT] bool — which tokens use this pool
    topk_idx: torch.Tensor,   # [BT, top_k]
    topk_w: torch.Tensor,     # [BT, top_k]
    experts: nn.ModuleList,
) -> torch.Tensor:
    """
    Route tokens selected by `mask` through `experts` according to top-k
    indices and weights, then scatter back into a full [BT, D] output tensor.
    """
    BT, D = flat.shape
    top_k = topk_idx.shape[1]
    output = torch.zeros_like(flat)

    # Only process tokens that belong to this pool
    if not mask.any():
        return output

    pool_tokens = flat[mask]         # [M, D]
    pool_idx = topk_idx[mask]        # [M, top_k]
    pool_w = topk_w[mask]            # [M, top_k]
    pool_out = torch.zeros_like(pool_tokens)  # [M, D]

    for k in range(top_k):
        expert_ids = pool_idx[:, k]  # [M]
        weights = pool_w[:, k]       # [M]
        for e_id, expert in enumerate(experts):
            e_sel = expert_ids == e_id
            if not e_sel.any():
                continue
            e_out = expert(pool_tokens[e_sel])          # [m, D]
            pool_out[e_sel] += weights[e_sel].unsqueeze(-1) * e_out

    output[mask] = pool_out
    return output


# ---------------------------------------------------------------------------
# HeterogeneousMoELayer
# ---------------------------------------------------------------------------
class HeterogeneousMoELayer(nn.Module):
    """
    Replaces a standard FFN in a Transformer block.

    Maintains two expert pools:
      • stable_experts   — small FFNs (d_ff_stable)  for simple/common tokens
      • transfer_experts — large FFNs (d_ff_transfer) for complex/rare tokens
    """

    def __init__(
        self,
        d_model: int,
        n_stable: int,
        n_transfer: int,
        d_ff_stable: int,
        d_ff_transfer: int,
        top_k: int = 2,
        dropout: float = 0.1,
        noise_std: float = 1e-2,
    ):
        super().__init__()
        self.router = TwoStageRouter(d_model, n_stable, n_transfer, top_k, noise_std)

        self.stable_experts = nn.ModuleList(
            [Expert(d_model, d_ff_stable, dropout) for _ in range(n_stable)]
        )
        self.transfer_experts = nn.ModuleList(
            [Expert(d_model, d_ff_transfer, dropout) for _ in range(n_transfer)]
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, RouterOutput]:
        """
        Args:
            x: [B, T, d_model]
        Returns:
            out:          [B, T, d_model]
            router_out:   RouterOutput (for computing auxiliary losses)
        """
        B, T, D = x.shape
        flat = x.reshape(B * T, D)

        router_out = self.router(x)

        # Dispatch to stable pool
        stable_out = dispatch_to_experts(
            flat,
            router_out.stable_mask,
            router_out.stable_topk_idx,
            router_out.stable_topk_w,
            self.stable_experts,
        )
        # Dispatch to transfer pool
        transfer_out = dispatch_to_experts(
            flat,
            router_out.transfer_mask,
            router_out.transfer_topk_idx,
            router_out.transfer_topk_w,
            self.transfer_experts,
        )

        # Combine: each token gets output from exactly one pool
        combined = stable_out + transfer_out  # non-assigned tokens contribute 0
        out = combined.reshape(B, T, D)
        return out, router_out


# ---------------------------------------------------------------------------
# Standard FFN (used in non-MoE layers)
# ---------------------------------------------------------------------------
class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# TransformerBlock (with optional MoE FFN)
# ---------------------------------------------------------------------------
class TransformerBlock(nn.Module):
    """Standard pre-norm Transformer block.  If `moe_layer` is provided it
    replaces the FFN; otherwise a standard dense FFN is used."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        moe_layer: Optional[HeterogeneousMoELayer] = None,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.moe_layer = moe_layer
        if moe_layer is None:
            self.ffn = FeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[RouterOutput]]:
        # ── Multi-head self-attention ────────────────────────────────────
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed, attn_mask=attn_mask)
        x = x + self.dropout(attn_out)

        # ── FFN / MoE ────────────────────────────────────────────────────
        router_out = None
        if self.moe_layer is not None:
            moe_out, router_out = self.moe_layer(self.norm2(x))
            x = x + self.dropout(moe_out)
        else:
            x = x + self.dropout(self.ffn(self.norm2(x)))

        return x, router_out


# ---------------------------------------------------------------------------
# Full Autoregressive Language Model
# ---------------------------------------------------------------------------
class HeterogeneousMoEModel(nn.Module):
    """
    GPT-style language model with heterogeneous two-stage MoE replacing FFN
    in every other Transformer block (blocks 1, 3, 5, …).
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        max_seq_len: int,
        n_stable: int,
        n_transfer: int,
        d_ff_stable: int,
        d_ff_transfer: int,
        top_k: int = 2,
        dropout: float = 0.1,
        noise_std: float = 1e-2,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)

        d_ff_dense = d_model * 4  # standard FFN size for non-MoE layers
        self.blocks = nn.ModuleList()
        for i in range(n_layers):
            use_moe = (i % 2 == 1)  # MoE on odd-indexed layers
            moe = (
                HeterogeneousMoELayer(
                    d_model, n_stable, n_transfer,
                    d_ff_stable, d_ff_transfer,
                    top_k, dropout, noise_std,
                )
                if use_moe else None
            )
            self.blocks.append(
                TransformerBlock(d_model, n_heads, d_ff_dense, dropout, moe)
            )

        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        # Tie embedding weights
        self.lm_head.weight = self.token_emb.weight

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def _causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        """Upper-triangular causal attention mask for nn.MultiheadAttention."""
        mask = torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()
        return mask  # True = masked out

    def forward(
        self,
        input_ids: torch.Tensor,          # [B, T]
        labels: Optional[torch.Tensor] = None,  # [B, T]
    ) -> Dict[str, torch.Tensor]:
        B, T = input_ids.shape
        assert T <= self.max_seq_len, "Sequence too long"

        positions = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.drop(self.token_emb(input_ids) + self.pos_emb(positions))

        causal_mask = self._causal_mask(T, input_ids.device)
        all_router_outs: List[RouterOutput] = []

        for block in self.blocks:
            x, router_out = block(x, attn_mask=causal_mask)
            if router_out is not None:
                all_router_outs.append(router_out)

        x = self.norm(x)
        logits = self.lm_head(x)  # [B, T, vocab_size]

        result = {"logits": logits, "router_outputs": all_router_outs}

        if labels is not None:
            # Shift for next-token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            result["loss_lm"] = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return result

    def count_parameters(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        active = sum(
            p.numel()
            for p in self.parameters()
            if p.requires_grad
        )
        return {"total": total, "trainable": active}
