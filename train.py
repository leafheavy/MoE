"""
train.py — Training and evaluation pipeline
============================================
Usage:
    python src/train.py --config config/config.yaml

Supports:
  • Mixed-precision training (AMP) when a GPU is available
  • Gradient clipping
  • Configurable λ for entropy and load-balance auxiliary losses
  • TensorBoard logging (optional W&B)
  • Checkpoint save / resume
  • Ablation mode: --no_moe, --single_stage, --no_entropy_loss
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

# Make src importable when running from project root
sys.path.insert(0, str(Path(__file__).parent))

from model import HeterogeneousMoEModel
from hf_model import HFCausalLMWrapper
from dataset import build_dataloader, load_dataset_by_name
from utils import (
    compute_expert_stats,
    compute_moe_loss,
    get_lr_scheduler,
    perplexity,
    set_seed,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config helper
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg


def flatten_cfg(cfg: dict) -> dict:
    """Flatten nested config dict for easy attribute access."""
    flat = {}
    for section, values in cfg.items():
        if isinstance(values, dict):
            flat.update(values)
        else:
            flat[section] = values
    return flat


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """
    Encapsulates the training loop, evaluation, logging, and checkpointing.
    """

    def __init__(self, cfg: dict, ablation: Optional[Dict] = None):
        self.cfg = cfg
        self.ablation = ablation or {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info(f"Using device: {self.device}")

        set_seed(cfg.get("seed", 42))

        # ── Build model ───────────────────────────────────────────────────
        llm_backend = cfg.get("llm_backend", "custom_moe")
        if llm_backend == "custom_moe":
            self.model = HeterogeneousMoEModel(
                vocab_size=cfg["vocab_size"],
                d_model=cfg["d_model"],
                n_heads=cfg["n_heads"],
                n_layers=cfg["n_layers"],
                max_seq_len=cfg["max_seq_len"],
                n_stable=cfg["n_stable"],
                n_transfer=cfg["n_transfer"],
                d_ff_stable=cfg["d_ff_stable"],
                d_ff_transfer=cfg["d_ff_transfer"],
                top_k=cfg["top_k"],
                dropout=cfg["dropout"],
                noise_std=cfg["noise_std"],
            ).to(self.device)
        elif llm_backend in {"qwen", "llama"}:
            model_name = cfg.get("pretrained_model_name")
            if not model_name:
                raise ValueError("Please set training.pretrained_model_name when llm_backend is qwen/llama")
            self.model = HFCausalLMWrapper(model_name).to(self.device)
            if cfg.get("tokenizer_name") in (None, "", "gpt2"):
                cfg["tokenizer_name"] = model_name
                log.info(f"Set tokenizer_name to pretrained_model_name: {model_name}")
        else:
            raise ValueError(f"Unknown llm_backend: {llm_backend}")

        param_counts = self.model.count_parameters()
        log.info(f"Model params — total: {param_counts['total']:,}  "
                 f"trainable: {param_counts['trainable']:,}")

        # ── Data ──────────────────────────────────────────────────────────
        log.info(f"Loading dataset: {cfg['dataset_name']}")
        tokenizer = None
        if cfg["dataset_name"].lower() in {"wikitext", "c4"}:
            tok_name = cfg.get("tokenizer_name")
            if not tok_name:
                raise ValueError(
                    "For dataset_name in {'wikitext','c4'}, please set data.tokenizer_name in config.yaml"
                )
            try:
                from transformers import AutoTokenizer
            except ImportError as e:
                raise ImportError("Please install transformers: pip install transformers") from e
            tokenizer = AutoTokenizer.from_pretrained(tok_name)
        train_ds, val_ds = load_dataset_by_name(
            name=cfg["dataset_name"],
            vocab_size=cfg["vocab_size"],
            seq_len=cfg["max_length"],
            train_size=cfg["train_size"],
            val_size=cfg["val_size"],
            seed=cfg.get("seed", 42),
            tokenizer=tokenizer,
        )
        self.train_loader = build_dataloader(
            train_ds, cfg["batch_size"], shuffle=True,
            num_workers=cfg.get("num_workers", 0),
            pin_memory=(self.device.type == "cuda"),
        )
        self.val_loader = build_dataloader(
            val_ds, cfg["batch_size"], shuffle=False,
            num_workers=cfg.get("num_workers", 0),
        )
        log.info(f"Train batches: {len(self.train_loader)} | "
                 f"Val batches: {len(self.val_loader)}")

        # ── Optimizer & scheduler ─────────────────────────────────────────
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg["lr"],
            weight_decay=cfg.get("weight_decay", 0.01),
        )
        self.scheduler = get_lr_scheduler(
            self.optimizer,
            warmup_steps=cfg.get("warmup_steps", 200),
            total_steps=cfg["max_steps"],
        )

        # ── Mixed precision ───────────────────────────────────────────────
        self.use_amp = cfg.get("use_amp", False) and (self.device.type == "cuda")
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        # ── Logging ───────────────────────────────────────────────────────
        log_dir = cfg.get("log_dir", "logs/")
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)

        # ── State ─────────────────────────────────────────────────────────
        self.global_step = 0
        self.best_val_ppl = float("inf")

    # ── Training step ────────────────────────────────────────────────────

    def train_step(self, batch: Dict) -> Dict[str, float]:
        self.model.train()
        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)

        with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
            out = self.model(input_ids, labels=labels)
            loss_lm = out["loss_lm"]

            if out["router_outputs"]:
                # Auxiliary losses (entropy + load balance)
                aux = compute_moe_loss(
                    out["router_outputs"],
                    lambda_entropy=self.cfg.get("lambda_entropy", 0.01),
                    lambda_balance=self.cfg.get("lambda_balance", 0.01),
                )
                # Ablation: skip entropy loss if requested
                if self.ablation.get("no_entropy_loss", False):
                    total_loss = loss_lm + aux["loss_balance"] * self.cfg.get("lambda_balance", 0.01)
                else:
                    total_loss = loss_lm + aux["loss_aux"]
            else:
                aux = {
                    "loss_entropy": torch.tensor(0.0, device=self.device),
                    "loss_balance": torch.tensor(0.0, device=self.device),
                    "loss_aux": torch.tensor(0.0, device=self.device),
                }
                total_loss = loss_lm

        self.optimizer.zero_grad()
        self.scaler.scale(total_loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.cfg.get("grad_clip", 1.0)
        )
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()

        return {
            "loss_lm": loss_lm.item(),
            "loss_entropy": aux["loss_entropy"].item(),
            "loss_balance": aux["loss_balance"].item(),
            "loss_total": total_loss.item(),
            "lr": self.scheduler.get_last_lr()[0],
        }

    # ── Evaluation ───────────────────────────────────────────────────────

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        all_router_outs = []
        n_batches = 0

        for batch in self.val_loader:
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)
            out = self.model(input_ids, labels=labels)
            total_loss += out["loss_lm"].item()
            all_router_outs.extend(out["router_outputs"])
            n_batches += 1

        mean_loss = total_loss / max(n_batches, 1)
        stats = compute_expert_stats(all_router_outs) if all_router_outs else {}
        return {
            "val_loss": mean_loss,
            "val_ppl": perplexity(mean_loss),
            **stats,
        }

    # ── Main training loop ────────────────────────────────────────────────

    def train(self):
        max_steps = self.cfg["max_steps"]
        eval_every = self.cfg.get("eval_every", 500)
        save_every = self.cfg.get("save_every", 1000)
        log_every = 50

        log.info(f"Starting training for {max_steps} steps ...")
        t0 = time.time()
        step_metrics: Dict[str, float] = {}

        train_iter = iter(self.train_loader)

        while self.global_step < max_steps:
            # Cycle through dataloader
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                batch = next(train_iter)

            step_metrics = self.train_step(batch)
            self.global_step += 1

            # ── Log to TensorBoard ───────────────────────────────────────
            if self.global_step % log_every == 0:
                elapsed = time.time() - t0
                tokens_per_sec = (
                    log_every
                    * self.cfg["batch_size"]
                    * self.cfg["max_length"]
                    / max(elapsed, 1e-6)
                )
                log.info(
                    f"step={self.global_step:5d} | "
                    f"loss={step_metrics['loss_total']:.4f} | "
                    f"lm={step_metrics['loss_lm']:.4f} | "
                    f"ent={step_metrics['loss_entropy']:.4f} | "
                    f"bal={step_metrics['loss_balance']:.4f} | "
                    f"lr={step_metrics['lr']:.2e} | "
                    f"tok/s={tokens_per_sec:.0f}"
                )
                for k, v in step_metrics.items():
                    self.writer.add_scalar(f"train/{k}", v, self.global_step)
                t0 = time.time()

            # ── Evaluate ─────────────────────────────────────────────────
            if self.global_step % eval_every == 0:
                val_metrics = self.evaluate()
                log.info(
                    f"[EVAL step={self.global_step}] "
                    + " | ".join(f"{k}={v:.4f}" for k, v in val_metrics.items())
                )
                for k, v in val_metrics.items():
                    self.writer.add_scalar(f"val/{k}", v, self.global_step)

                if val_metrics["val_ppl"] < self.best_val_ppl:
                    self.best_val_ppl = val_metrics["val_ppl"]
                    self.save_checkpoint("best_model.pt")
                    log.info(f"  ✓ New best PPL={self.best_val_ppl:.2f}")

            # ── Save checkpoint ──────────────────────────────────────────
            if self.global_step % save_every == 0:
                self.save_checkpoint(f"step_{self.global_step}.pt")

        log.info(f"Training complete. Best val PPL: {self.best_val_ppl:.2f}")
        self.writer.close()

    # ── Checkpoint I/O ───────────────────────────────────────────────────

    def save_checkpoint(self, filename: str):
        ckpt_dir = "checkpoints/"
        os.makedirs(ckpt_dir, exist_ok=True)
        path = os.path.join(ckpt_dir, filename)
        torch.save({
            "global_step": self.global_step,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "best_val_ppl": self.best_val_ppl,
            "cfg": self.cfg,
        }, path)
        log.info(f"  Checkpoint saved → {path}")

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.scheduler.load_state_dict(ckpt["scheduler_state"])
        self.global_step = ckpt["global_step"]
        self.best_val_ppl = ckpt.get("best_val_ppl", float("inf"))
        log.info(f"Resumed from {path} at step {self.global_step}")


# ---------------------------------------------------------------------------
# Ablation runner
# ---------------------------------------------------------------------------

def run_ablation(cfg: dict, ablation_name: str, ablation_flags: dict):
    """Run a single ablation experiment variant and return val metrics."""
    log.info(f"\n{'='*60}\nAblation: {ablation_name}\n{'='*60}")
    # Reduce steps for ablation sweeps
    ablation_cfg = {**cfg, "max_steps": min(cfg["max_steps"], 2000)}
    trainer = Trainer(ablation_cfg, ablation=ablation_flags)
    trainer.train()
    val_metrics = trainer.evaluate()
    log.info(f"Ablation [{ablation_name}] val_ppl={val_metrics['val_ppl']:.2f}")
    return val_metrics


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Heterogeneous Two-Stage MoE Training")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--run_ablations", action="store_true",
                        help="Run ablation experiments after main training")
    parser.add_argument("--no_entropy_loss", action="store_true",
                        help="Ablation: disable entropy minimization loss")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg_raw = load_config(args.config)
    cfg = flatten_cfg(cfg_raw)

    ablation_flags = {
        "no_entropy_loss": args.no_entropy_loss,
    }

    trainer = Trainer(cfg, ablation=ablation_flags)

    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.train()

    # ── Optional ablation sweep ────────────────────────────────────────────
    if args.run_ablations:
        results = {}

        # Ablation 1: No entropy loss
        results["no_entropy"] = run_ablation(
            cfg, "no_entropy_loss", {"no_entropy_loss": True}
        )

        # Ablation 2: Force all tokens into stable pool (override stage1)
        # This requires a flag in the router, so we log a note here
        log.info("Ablation [force_stable]: set n_transfer=0 in config and retrain.")

        # Ablation 3: Single-stage routing (standard TopK MoE)
        log.info("Ablation [single_stage]: use standard TopK MoE as baseline.")

        # Save ablation summary
        summary_path = "logs/ablation_summary.json"
        with open(summary_path, "w") as f:
            json.dump(results, f, indent=2)
        log.info(f"Ablation results saved → {summary_path}")


if __name__ == "__main__":
    main()
