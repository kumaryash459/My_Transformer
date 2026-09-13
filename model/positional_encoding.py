import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):

    def __init__(
        self,
        d_model: int,
        max_seq_length: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()

        self.max_seq_length = max_seq_length

        self.dropout = nn.Dropout(
            dropout
        )

        # ----------------------------------------------------
        # Create position indices
        #
        # [0, 1, 2, ..., max_seq_length - 1]
        # ----------------------------------------------------

        position = torch.arange(
            max_seq_length,
            dtype=torch.float32
        ).unsqueeze(1)

        # ----------------------------------------------------
        # Compute frequency terms
        # ----------------------------------------------------

        div_term = torch.exp(
            torch.arange(
                0,
                d_model,
                2,
                dtype=torch.float32
            )
            * (
                -math.log(10000.0)
                / d_model
            )
        )

        # ----------------------------------------------------
        # Positional encoding matrix
        #
        # Shape:
        # [max_seq_length, d_model]
        # ----------------------------------------------------

        pe = torch.zeros(
            max_seq_length,
            d_model,
            dtype=torch.float32
        )

        # Even dimensions -> sine
        pe[:, 0::2] = torch.sin(
            position * div_term
        )

        # Odd dimensions -> cosine
        pe[:, 1::2] = torch.cos(
            position * div_term
        )

        # Add batch dimension
        #
        # [1, max_seq_length, d_model]
        #

        pe = pe.unsqueeze(0)

        # register_buffer means:
        # - moves automatically to CUDA with model
        # - saved in state_dict
        # - not trainable
        self.register_buffer(
            "pe",
            pe
        )

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:

        sequence_length = x.size(1)

        # ----------------------------------------------------
        # Safety check
        # ----------------------------------------------------

        if sequence_length > self.max_seq_length:

            raise ValueError(
                f"Sequence length {sequence_length} exceeds "
                f"maximum supported length "
                f"{self.max_seq_length}."
            )

        # ----------------------------------------------------
        # Add positional information
        # ----------------------------------------------------

        x = (
            x
            + self.pe[
                :,
                :sequence_length,
                :
            ]
        )

        # ----------------------------------------------------
        # Dropout
        # ----------------------------------------------------

        x = self.dropout(
            x
        )

        return x