import torch
import torch.nn as nn

from model.attention import MultiHeadAttention
from model.feed_forward import PositionWiseFeedForward


class DecoderLayer(nn.Module):
    """
    Single Transformer Decoder Layer.

    Contains:

    1. Masked self-attention
    2. Encoder-decoder cross-attention
    3. Position-wise feed-forward network

    Uses post-normalization, following the
    original Transformer architecture.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1
    ):
        super().__init__()

        
        # 1. Masked Self-Attention
        

        self.self_attention = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout
        )

        
        # 2. Encoder-Decoder Attention
        

        self.cross_attention = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout
        )

        
        # 3. Feed-Forward Network
        

        self.feed_forward = PositionWiseFeedForward(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout
        )

        
        # Dropouts
        

        self.dropout_self_attention = nn.Dropout(dropout)
        self.dropout_cross_attention = nn.Dropout(dropout)
        self.dropout_ffn = nn.Dropout(dropout)

        
        # Layer Normalization
        

        self.norm_self_attention = nn.LayerNorm(d_model)
        self.norm_cross_attention = nn.LayerNorm(d_model)
        self.norm_ffn = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        src_mask: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor
    ]:

        
        # 1. MASKED SELF-ATTENTION
        

        self_attention_output, self_attention_weights = (
            self.self_attention(
                query=x,
                key=x,
                value=x,
                mask=tgt_mask
            )
        )

        # Residual + LayerNorm

        x = self.norm_self_attention(
            x + self.dropout_self_attention(
                self_attention_output
            )
        )

        
        # 2. CROSS-ATTENTION
        

        cross_attention_output, cross_attention_weights = (
            self.cross_attention(
                query=x,
                key=encoder_output,
                value=encoder_output,
                mask=src_mask
            )
        )

        # Residual + LayerNorm

        x = self.norm_cross_attention(
            x + self.dropout_cross_attention(
                cross_attention_output
            )
        )

        
        # 3. FEED-FORWARD NETWORK
        

        ffn_output = self.feed_forward(x)

        # Residual + LayerNorm

        x = self.norm_ffn(
            x + self.dropout_ffn(ffn_output)
        )

        return (
            x,
            self_attention_weights,
            cross_attention_weights
        )
        
class TransformerDecoder(nn.Module):
    """
    Stack of Transformer Decoder Layers.

    Input:
        x:
            [batch_size, target_length, d_model]

        encoder_output:
            [batch_size, source_length, d_model]

    Output:
        x:
            [batch_size, target_length, d_model]
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
                DecoderLayer(
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
        encoder_output: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        src_mask: torch.Tensor | None = None
    ) -> tuple[
        torch.Tensor,
        list[torch.Tensor],
        list[torch.Tensor]
    ]:

        self_attention_maps = []
        cross_attention_maps = []

        for layer in self.layers:

            (
                x,
                self_attention,
                cross_attention
            ) = layer(
                x=x,
                encoder_output=encoder_output,
                tgt_mask=tgt_mask,
                src_mask=src_mask
            )

            self_attention_maps.append(
                self_attention
            )

            cross_attention_maps.append(
                cross_attention
            )

        x = self.final_norm(x)

        return (
            x,
            self_attention_maps,
            cross_attention_maps
        )