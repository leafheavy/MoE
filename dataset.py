"""
dataset.py — Data loading and preprocessing
============================================
Provides:
  • SyntheticTextDataset  — random token sequences (for unit testing)
  • HFTextDataset         — HuggingFace text dataset wrapper (wikitext / c4)
  • load_dataset_by_name  — factory that returns (train_ds, val_ds)
  • collate_fn            — pads batch and creates labels for LM
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Synthetic dataset (no external dependencies — good for quick tests)
# ---------------------------------------------------------------------------

class SyntheticTextDataset(Dataset):
    """
    Generates random integer sequences mimicking token IDs.
    Useful for verifying model forward/backward passes without real data.
    """

    def __init__(self, size: int, seq_len: int, vocab_size: int, seed: int = 42):
        super().__init__()
        rng = torch.Generator()
        rng.manual_seed(seed)
        # Pre-generate all sequences once
        self.data = torch.randint(0, vocab_size, (size, seq_len), generator=rng)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        tokens = self.data[idx]
        return {
            "input_ids": tokens,
            "labels": tokens.clone(),  # next-token: shifted inside model
        }


# ---------------------------------------------------------------------------
# HuggingFace text dataset wrapper
# ---------------------------------------------------------------------------

class HFTextDataset(Dataset):
    """
    Wraps a HuggingFace text dataset for causal language modelling.

    Tokenizes and chunks text into fixed-length windows.

    Args:
        hf_dataset: a HuggingFace Dataset split (e.g. ds["train"])
        tokenizer:  a HuggingFace tokenizer
        max_length: chunk size in tokens
        text_col:   name of the text column (default "text")
    """

    def __init__(self, hf_dataset, tokenizer, max_length: int = 512, text_col: str = "text"):
        self.max_length = max_length
        self.tokenizer = tokenizer

        # Tokenize all text and concatenate into one long sequence
        all_ids: List[int] = []
        for sample in hf_dataset:
            text = sample.get(text_col, "")
            if not text.strip():
                continue
            ids = tokenizer.encode(text, add_special_tokens=True)
            all_ids.extend(ids)

        # Chunk into non-overlapping windows of length max_length + 1
        # (the +1 is so that labels can be shifted by 1 inside the model)
        chunk_size = max_length + 1
        n_chunks = len(all_ids) // chunk_size
        all_ids = all_ids[: n_chunks * chunk_size]
        self.chunks = torch.tensor(all_ids, dtype=torch.long).reshape(n_chunks, chunk_size)

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        chunk = self.chunks[idx]
        return {
            "input_ids": chunk[:-1],   # [max_length]
            "labels": chunk[1:],       # [max_length] — already shifted
        }


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------

def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Pads variable-length sequences in a batch to the same length.
    Labels use -100 for padding positions (ignored by cross-entropy).
    """
    input_ids = [item["input_ids"] for item in batch]
    labels = [item["labels"] for item in batch]

    # Pad sequences
    max_len = max(x.shape[0] for x in input_ids)
    pad_input, pad_labels = [], []
    for ids, lbl in zip(input_ids, labels):
        pad_len = max_len - ids.shape[0]
        pad_input.append(torch.cat([ids, torch.zeros(pad_len, dtype=torch.long)]))
        pad_labels.append(torch.cat([lbl, torch.full((pad_len,), -100, dtype=torch.long)]))

    return {
        "input_ids": torch.stack(pad_input),   # [B, T]
        "labels": torch.stack(pad_labels),     # [B, T]
    }


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def load_dataset_by_name(
    name: str,
    vocab_size: int = 32000,
    seq_len: int = 512,
    train_size: int = 10000,
    val_size: int = 1000,
    seed: int = 42,
    tokenizer=None,
) -> Tuple[Dataset, Dataset]:
    """
    Returns (train_dataset, val_dataset) for the given dataset name.

    Supported names:
      "synthetic"  — random token sequences; no extra dependencies
      "wikitext"   — WikiText-103 from HuggingFace
      "c4"         — Colossal Clean Crawled Corpus (HuggingFace)

    For HuggingFace datasets a `tokenizer` must be provided.
    """
    name = name.lower()

    if name == "synthetic":
        train_ds = SyntheticTextDataset(train_size, seq_len, vocab_size, seed)
        val_ds = SyntheticTextDataset(val_size, seq_len, vocab_size, seed + 1)
        return train_ds, val_ds

    # ── HuggingFace datasets ──────────────────────────────────────────────
    try:
        from datasets import load_dataset as hf_load
    except ImportError:
        raise ImportError(
            "Install `datasets` for non-synthetic data: pip install datasets"
        )

    assert tokenizer is not None, "A tokenizer is required for HuggingFace datasets"

    if name == "wikitext":
        raw = hf_load("wikitext", "wikitext-103-raw-v1", split="train")
        raw_val = hf_load("wikitext", "wikitext-103-raw-v1", split="validation")
    elif name == "c4":
        raw = hf_load("c4", "en", split="train", streaming=True)
        raw_val = hf_load("c4", "en", split="validation", streaming=True)
        # Take a finite slice for streaming datasets
        raw = list(raw.take(train_size * 10))   # rough upper bound
        raw_val = list(raw_val.take(val_size * 10))
    else:
        raise ValueError(f"Unknown dataset: {name}")

    train_ds = HFTextDataset(raw, tokenizer, max_length=seq_len)
    val_ds = HFTextDataset(raw_val, tokenizer, max_length=seq_len)
    return train_ds, val_ds


# ---------------------------------------------------------------------------
# DataLoader builder
# ---------------------------------------------------------------------------

def build_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
