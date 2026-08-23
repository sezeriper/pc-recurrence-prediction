from __future__ import annotations

from typing import final, override

from torch import Tensor, nn


@final
class RecurrenceHead(nn.Module):
    """A single affine recurrence logit over pooled patient embeddings."""

    def __init__(self, input_dimension: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dimension, 1, bias=True)

    @override
    def forward(self, embeddings: Tensor) -> Tensor:
        logits: Tensor = self.linear(embeddings)
        return logits.squeeze(-1)
