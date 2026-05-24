"""
test_model.py — Unit and integration tests
==========================================
Run with:  python -m pytest tests/test_model.py -v
or:        python tests/test_model.py

Tests cover:
  1. Expert forward pass — shape and gradient flow
  2. TwoStageRouter — output shapes, mask coverage, top-k validity
  3. HeterogeneousMoELayer — shape, router outputs, loss computation
  4. Loss functions — entropy_loss, load_balance_loss correctness
  5. Full model forward pass — logits shape, LM loss presence
  6. Reproducibility — same seed → same router output
  7. Dataset — SyntheticTextDataset indexing and collation
"""

import sys
from pathlib import Path

import torch

# Allow imports from src/
sys.path.insert(0, str(Path(__file__).parent))

from model import (
    Expert,
    TwoStageRouter,
    HeterogeneousMoELayer,
    HeterogeneousMoEModel,
    dispatch_to_experts,
)
from utils import (
    entropy_loss,
    load_balance_loss,
    compute_moe_loss,
    compute_expert_stats,
    set_seed,
)
from dataset import SyntheticTextDataset, collate_fn


# ── Shared small config ─────────────────────────────────────────────────────
D_MODEL = 64
N_STABLE = 3
N_TRANSFER = 3
D_FF_STABLE = 128
D_FF_TRANSFER = 256
TOP_K = 2
BATCH = 2
SEQ = 8


def make_input():
    return torch.randn(BATCH, SEQ, D_MODEL)


# ===========================================================================
# 1. Expert
# ===========================================================================

def test_expert_forward():
    expert = Expert(D_MODEL, D_FF_STABLE, dropout=0.0)
    x = make_input()
    out = expert(x)
    assert out.shape == (BATCH, SEQ, D_MODEL), f"Unexpected shape: {out.shape}"
    # Gradient check
    out.sum().backward()
    assert expert.fc1.weight.grad is not None, "No gradient on fc1.weight"
    print("  ✓ test_expert_forward")


# ===========================================================================
# 2. TwoStageRouter
# ===========================================================================

def test_router_shapes():
    router = TwoStageRouter(D_MODEL, N_STABLE, N_TRANSFER, TOP_K)
    x = make_input()
    ro = router(x)

    BT = BATCH * SEQ
    assert ro.stage1_probs.shape == (BT, 2), f"stage1_probs: {ro.stage1_probs.shape}"
    assert ro.stable_mask.shape == (BT,)
    assert ro.transfer_mask.shape == (BT,)
    assert ro.stable_probs.shape == (BT, N_STABLE)
    assert ro.transfer_probs.shape == (BT, N_TRANSFER)
    assert ro.stable_topk_idx.shape == (BT, TOP_K)
    assert ro.stable_topk_w.shape == (BT, TOP_K)
    assert ro.transfer_topk_idx.shape == (BT, TOP_K)
    assert ro.transfer_topk_w.shape == (BT, TOP_K)
    print("  ✓ test_router_shapes")


def test_router_mask_coverage():
    """Every token must be assigned to exactly one pool."""
    router = TwoStageRouter(D_MODEL, N_STABLE, N_TRANSFER, TOP_K)
    x = make_input()
    ro = router(x)
    # Stable XOR transfer (hard argmax → mutually exclusive)
    overlap = (ro.stable_mask & ro.transfer_mask).sum().item()
    union = (ro.stable_mask | ro.transfer_mask).sum().item()
    assert overlap == 0, f"Masks overlap for {overlap} tokens"
    assert union == BATCH * SEQ, f"Not all tokens assigned: {union} vs {BATCH * SEQ}"
    print("  ✓ test_router_mask_coverage")


def test_router_topk_valid():
    """Top-k indices must be within valid range and weights must sum ≈ 1."""
    router = TwoStageRouter(D_MODEL, N_STABLE, N_TRANSFER, TOP_K)
    x = make_input()
    ro = router(x)

    assert ro.stable_topk_idx.max() < N_STABLE, "Stable index out of range"
    assert ro.transfer_topk_idx.max() < N_TRANSFER, "Transfer index out of range"

    # Weights sum to 1 per token (within pool)
    stable_sum = ro.stable_topk_w.sum(dim=-1)  # [BT]
    transfer_sum = ro.transfer_topk_w.sum(dim=-1)
    assert torch.allclose(stable_sum, torch.ones_like(stable_sum), atol=1e-5)
    assert torch.allclose(transfer_sum, torch.ones_like(transfer_sum), atol=1e-5)
    print("  ✓ test_router_topk_valid")


# ===========================================================================
# 3. HeterogeneousMoELayer
# ===========================================================================

def test_moe_layer_forward():
    moe = HeterogeneousMoELayer(
        D_MODEL, N_STABLE, N_TRANSFER, D_FF_STABLE, D_FF_TRANSFER, TOP_K, dropout=0.0
    )
    x = make_input()
    out, router_out = moe(x)

    assert out.shape == (BATCH, SEQ, D_MODEL), f"MoE output shape: {out.shape}"
    assert router_out is not None
    print("  ✓ test_moe_layer_forward")


def test_moe_layer_gradient():
    moe = HeterogeneousMoELayer(
        D_MODEL, N_STABLE, N_TRANSFER, D_FF_STABLE, D_FF_TRANSFER, TOP_K, dropout=0.0
    )
    x = make_input()
    out, _ = moe(x)
    loss = out.sum()
    loss.backward()
    # Check at least one expert got a gradient
    assert moe.stable_experts[0].fc1.weight.grad is not None
    assert moe.transfer_experts[0].fc1.weight.grad is not None
    print("  ✓ test_moe_layer_gradient")


# ===========================================================================
# 4. Loss functions
# ===========================================================================

def test_entropy_loss_uniform():
    """Uniform distribution should have maximum entropy."""
    n = 4
    probs_uniform = torch.full((8, n), 1.0 / n)
    probs_onehot = torch.zeros(8, n)
    probs_onehot[:, 0] = 1.0

    ent_uniform = entropy_loss(probs_uniform).item()
    ent_onehot = entropy_loss(probs_onehot).item()

    assert ent_uniform > ent_onehot, (
        f"Uniform entropy ({ent_uniform:.4f}) should be > one-hot ({ent_onehot:.4f})"
    )
    print(f"  ✓ test_entropy_loss_uniform  [uniform={ent_uniform:.4f}, onehot={ent_onehot:.4f}]")


def test_load_balance_loss_uniform():
    """Perfectly balanced routing should give loss ≈ 1 (n * 1/n * 1/n)."""
    n_experts = 4
    BT = 16
    # Each expert gets exactly BT/n_experts tokens (one-hot hard routing)
    probs = torch.zeros(BT, n_experts)
    for i in range(BT):
        probs[i, i % n_experts] = 1.0
    mask = torch.ones(BT, dtype=torch.bool)
    loss = load_balance_loss(probs, mask).item()
    # With perfect balance: f_i = 1/n, p_i = 1/n → n * sum(1/n^2) = n * n/n^2 = 1
    assert abs(loss - 1.0) < 0.1, f"Expected ≈1.0, got {loss:.4f}"
    print(f"  ✓ test_load_balance_loss_uniform  [loss={loss:.4f}]")


def test_compute_moe_loss():
    """compute_moe_loss should return non-negative scalar losses."""
    moe = HeterogeneousMoELayer(
        D_MODEL, N_STABLE, N_TRANSFER, D_FF_STABLE, D_FF_TRANSFER, TOP_K, dropout=0.0
    )
    x = make_input()
    _, ro = moe(x)
    losses = compute_moe_loss([ro], lambda_entropy=0.01, lambda_balance=0.01)

    assert "loss_entropy" in losses
    assert "loss_balance" in losses
    assert "loss_aux" in losses
    assert losses["loss_aux"].item() >= 0
    print("  ✓ test_compute_moe_loss")


# ===========================================================================
# 5. Full model
# ===========================================================================

def test_full_model_forward():
    model = HeterogeneousMoEModel(
        vocab_size=100,
        d_model=D_MODEL,
        n_heads=2,
        n_layers=4,
        max_seq_len=SEQ,
        n_stable=N_STABLE,
        n_transfer=N_TRANSFER,
        d_ff_stable=D_FF_STABLE,
        d_ff_transfer=D_FF_TRANSFER,
        top_k=TOP_K,
        dropout=0.0,
        noise_std=0.0,
    )
    input_ids = torch.randint(0, 100, (BATCH, SEQ))
    labels = input_ids.clone()
    out = model(input_ids, labels=labels)

    assert "logits" in out
    assert "loss_lm" in out
    assert out["logits"].shape == (BATCH, SEQ, 100)
    assert out["loss_lm"].item() > 0
    print("  ✓ test_full_model_forward")


def test_full_model_backward():
    model = HeterogeneousMoEModel(
        vocab_size=100,
        d_model=D_MODEL,
        n_heads=2,
        n_layers=4,
        max_seq_len=SEQ,
        n_stable=N_STABLE,
        n_transfer=N_TRANSFER,
        d_ff_stable=D_FF_STABLE,
        d_ff_transfer=D_FF_TRANSFER,
        top_k=TOP_K,
        dropout=0.0,
        noise_std=0.0,
    )
    input_ids = torch.randint(0, 100, (BATCH, SEQ))
    labels = input_ids.clone()
    out = model(input_ids, labels=labels)
    aux = compute_moe_loss(out["router_outputs"], 0.01, 0.01)
    total_loss = out["loss_lm"] + aux["loss_aux"]
    total_loss.backward()

    # Check that at least one MoE-related parameter got a gradient
    for block in model.blocks:
        if block.moe_layer is not None:
            g = block.moe_layer.router.stage1.weight.grad
            assert g is not None, "stage1 router has no gradient"
            break
    print("  ✓ test_full_model_backward")


# ===========================================================================
# 6. Reproducibility
# ===========================================================================

def test_reproducibility():
    """Same seed → identical router outputs."""

    def get_router_output(seed: int):
        set_seed(seed)
        router = TwoStageRouter(D_MODEL, N_STABLE, N_TRANSFER, TOP_K, noise_std=0.1)
        router.eval()
        x = torch.randn(BATCH, SEQ, D_MODEL)
        with torch.no_grad():
            return router(x).stage1_probs.clone()

    out1 = get_router_output(42)
    out2 = get_router_output(42)
    out3 = get_router_output(99)  # Different seed

    assert torch.allclose(out1, out2), "Same seed produced different outputs"
    assert not torch.allclose(out1, out3), "Different seeds produced same output"
    print("  ✓ test_reproducibility")


# ===========================================================================
# 7. Dataset
# ===========================================================================

def test_synthetic_dataset():
    ds = SyntheticTextDataset(size=20, seq_len=SEQ, vocab_size=100, seed=0)
    assert len(ds) == 20
    item = ds[0]
    assert "input_ids" in item and "labels" in item
    assert item["input_ids"].shape == (SEQ,)
    print("  ✓ test_synthetic_dataset")


def test_collate_fn():
    ds = SyntheticTextDataset(size=4, seq_len=SEQ, vocab_size=100)
    batch = [ds[i] for i in range(4)]
    collated = collate_fn(batch)
    assert collated["input_ids"].shape == (4, SEQ)
    assert collated["labels"].shape == (4, SEQ)
    print("  ✓ test_collate_fn")


# ===========================================================================
# 8. Expert statistics
# ===========================================================================

def test_expert_stats():
    moe = HeterogeneousMoELayer(
        D_MODEL, N_STABLE, N_TRANSFER, D_FF_STABLE, D_FF_TRANSFER, TOP_K, dropout=0.0
    )
    x = make_input()
    _, ro = moe(x)
    stats = compute_expert_stats([ro])
    expected_keys = [
        "stable_frac", "transfer_frac",
        "stable_entropy", "transfer_entropy",
        "stable_load_std", "transfer_load_std",
    ]
    for k in expected_keys:
        assert k in stats, f"Missing key: {k}"
        assert isinstance(stats[k], float)
    # Fractions should sum to 1
    assert abs(stats["stable_frac"] + stats["transfer_frac"] - 1.0) < 1e-5
    print("  ✓ test_expert_stats")


# ===========================================================================
# Run all tests
# ===========================================================================

ALL_TESTS = [
    test_expert_forward,
    test_router_shapes,
    test_router_mask_coverage,
    test_router_topk_valid,
    test_moe_layer_forward,
    test_moe_layer_gradient,
    test_entropy_loss_uniform,
    test_load_balance_loss_uniform,
    test_compute_moe_loss,
    test_full_model_forward,
    test_full_model_backward,
    test_reproducibility,
    test_synthetic_dataset,
    test_collate_fn,
    test_expert_stats,
]

if __name__ == "__main__":
    print(f"\nRunning {len(ALL_TESTS)} tests ...\n")
    passed, failed = 0, 0
    for test_fn in ALL_TESTS:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  ✗ {test_fn.__name__}: {e}")
            failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed")
    if failed > 0:
        sys.exit(1)
