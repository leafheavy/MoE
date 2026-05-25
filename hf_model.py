"""HuggingFace CausalLM wrapper used as selectable lightweight backbones."""

from typing import Dict, List, Optional

import torch
import torch.nn as nn


class HFCausalLMWrapper(nn.Module):
    """Wrap AutoModelForCausalLM with the same output contract used by Trainer."""

    def __init__(self, pretrained_name: str):
        super().__init__()
        try:
            from transformers import AutoModelForCausalLM
        except ImportError as e:
            raise ImportError("Please install transformers: pip install transformers") from e

        self.model = AutoModelForCausalLM.from_pretrained(pretrained_name)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        out = self.model(input_ids=input_ids, labels=labels)
        return {
            "logits": out.logits,
            "loss_lm": out.loss if out.loss is not None else None,
            "router_outputs": [],
        }

    def count_parameters(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}
