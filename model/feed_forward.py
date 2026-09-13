import torch
import torch.nn as nn


class PositionWiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.

    FFN(x) = Linear -> ReLU -> Dropout -> Linear

    Input:
        [batch_size, sequence_length, d_model]

    Output:
        [batch_size, sequence_length, d_model]
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1
    ):
        super().__init__()

        self.linear1 = nn.Linear(
            d_model,
            d_ff
        )

        self.activation = nn.ReLU()

        self.dropout = nn.Dropout(dropout)

        self.linear2 = nn.Linear(
            d_ff,
            d_model
        )

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:

        x = self.linear1(x)

        x = self.activation(x)

        x = self.dropout(x)

        x = self.linear2(x)

        return x