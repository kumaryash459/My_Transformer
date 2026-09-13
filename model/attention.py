import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention.

    Attention(Q, K, V) =
        softmax(QK^T / sqrt(d_k)) V

    Input:
        Q: [batch, heads, query_length, d_k]
        K: [batch, heads, key_length, d_k]
        V: [batch, heads, key_length, d_v]

    Mask:
        [batch, 1, query_length, key_length]
        or
        [batch, heads, query_length, key_length]

    Output:
        context: [batch, heads, query_length, d_v]
        attention_weights:
            [batch, heads, query_length, key_length]
    """

    def __init__(self, dropout: float = 0.1):
        super().__init__()

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        d_k = query.size(-1)

        # QK^T
        scores = torch.matmul(
            query,
            key.transpose(-2, -1)
        )

        # Scaling
        scores = scores / math.sqrt(d_k)

        # Masking
        if mask is not None:
            scores = scores.masked_fill(
                mask == 0,
                torch.finfo(scores.dtype).min
            )

        # Softmax over key dimension
        attention_weights = F.softmax(
            scores,
            dim=-1
        )

        attention_weights = self.dropout(
            attention_weights
        )

        # Weighted sum of values
        context = torch.matmul(
            attention_weights,
            value
        )

        return context, attention_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention.

    Input:
        x: [batch_size, sequence_length, d_model]

    Output:
        [batch_size, sequence_length, d_model]
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1
    ):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError(
                "d_model must be divisible by num_heads."
            )

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Q, K, V projections
        self.query_projection = nn.Linear(
            d_model,
            d_model
        )

        self.key_projection = nn.Linear(
            d_model,
            d_model
        )

        self.value_projection = nn.Linear(
            d_model,
            d_model
        )

        # Final projection
        self.output_projection = nn.Linear(
            d_model,
            d_model
        )

        self.attention = ScaledDotProductAttention(
            dropout=dropout
        )

    def _split_heads(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:

        batch_size, sequence_length, _ = x.shape

        x = x.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.d_k
        )

        # [B, S, H, D] -> [B, H, S, D]
        x = x.transpose(1, 2)

        return x

    def _combine_heads(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:

        batch_size = x.size(0)
        sequence_length = x.size(2)

        # [B, H, S, D] -> [B, S, H, D]
        x = x.transpose(1, 2)

        # [B, S, H, D] -> [B, S, d_model]
        x = x.contiguous().view(
            batch_size,
            sequence_length,
            self.d_model
        )

        return x

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        # Linear projections
        Q = self.query_projection(query)
        K = self.key_projection(key)
        V = self.value_projection(value)

        # Split into multiple heads
        Q = self._split_heads(Q)
        K = self._split_heads(K)
        V = self._split_heads(V)

        # Attention
        context, attention_weights = self.attention(
            Q,
            K,
            V,
            mask
        )

        # Combine heads
        context = self._combine_heads(context)

        # Final projection
        output = self.output_projection(context)

        return output, attention_weights