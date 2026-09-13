import torch
import torch.nn as nn

from model.attention import MultiHeadAttention
from model.feed_forward import PositionWiseFeedForward


class EncoderLayer(nn.Module):
    """
    Single Transformer Encoder Layer.

    Architecture:

        x
        │
        ├───────────────┐
        ▼               │
    Self-Attention      │
        │               │
        ▼               │
      Dropout           │
        │               │
        ▼               │
    Add + LayerNorm ◄───┘
        │
        ▼
       FFN
        │
        ▼
      Dropout
        │
        ▼
    Add + LayerNorm
        │
        ▼
      Output
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1
    ):
        super().__init__()

        self.self_attention = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout
        )

        self.feed_forward = PositionWiseFeedForward(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout
        )

        self.dropout_attention = nn.Dropout(dropout)
        self.dropout_ffn = nn.Dropout(dropout)

        self.norm_attention = nn.LayerNorm(d_model)
        self.norm_ffn = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        src_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:

        
        # 1. Multi-Head Self-Attention
        

        attention_output, attention_weights = self.self_attention(
            query=x,
            key=x,
            value=x,
            mask=src_mask
        )

       
        # 2. Residual + LayerNorm
      

        x = self.norm_attention(
            x + self.dropout_attention(attention_output)
        )


        # 3. Feed-Forward Network


        ffn_output = self.feed_forward(x)

        
        # 4. Residual + LayerNorm
        x = self.norm_ffn(
            x + self.dropout_ffn(ffn_output)
        )

        return x, attention_weights
       
class TransformerEncoder(nn.Module):
    """
    Stack of Transformer Encoder Layers.

    Input:
        x: [batch_size, source_length, d_model]

    Output:
        x: [batch_size, source_length, d_model]
    """

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1
    ):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                EncoderLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    dropout=dropout
                )
                for _ in range(num_layers)
            ]
        )

        self.final_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        src_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:

        attention_maps = []

        for layer in self.layers:

            x, attention_weights = layer(
                x,
                src_mask
            )

            attention_maps.append(attention_weights)

        x = self.final_norm(x)

        return x, attention_maps