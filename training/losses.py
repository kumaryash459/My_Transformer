import torch
import torch.nn as nn


class LabelSmoothedCrossEntropyLoss(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        padding_idx: int,
        label_smoothing: float = 0.1
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        self.label_smoothing = label_smoothing

        self.loss_fn = nn.CrossEntropyLoss(
            ignore_index=padding_idx,
            label_smoothing=label_smoothing
        )

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:

        batch_size, sequence_length, vocab_size = logits.shape

        if vocab_size != self.vocab_size:
            raise ValueError(
                f"Expected vocabulary size {self.vocab_size}, "
                f"but received {vocab_size}."
            )

        logits = logits.reshape(
            batch_size * sequence_length,
            vocab_size
        )

        targets = targets.reshape(
            batch_size * sequence_length
        )

        loss = self.loss_fn(logits, targets)

        return loss