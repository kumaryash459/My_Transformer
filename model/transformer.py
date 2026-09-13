import torch
import torch.nn as nn

from configs.config import TransformerConfig
from model.embeddings import TokenEmbedding
from model.positional_encoding import PositionalEncoding
from model.encoder import TransformerEncoder
from model.decoder import TransformerDecoder


class Transformer(nn.Module):
    """
    Complete Encoder-Decoder Transformer.

    Based on the original Transformer architecture
    introduced in "Attention Is All You Need".

    Input:
        src_tokens:
            [batch_size, source_length]

        tgt_tokens:
            [batch_size, target_length]

    Output:
        logits:
            [batch_size, target_length, tgt_vocab_size]
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()

        self.config = config

        
        # SOURCE EMBEDDING
        

        self.src_embedding = TokenEmbedding(
            vocab_size=config.src_vocab_size,
            d_model=config.d_model
        )

        
        # TARGET EMBEDDING


        self.tgt_embedding = TokenEmbedding(
            vocab_size=config.tgt_vocab_size,
            d_model=config.d_model
        )

        
        # POSITIONAL ENCODING
        

        self.src_position = PositionalEncoding(
            d_model=config.d_model,
            max_seq_length=config.max_seq_length,
            dropout=config.dropout
        )

        self.tgt_position = PositionalEncoding(
            d_model=config.d_model,
            max_seq_length=config.max_seq_length,
            dropout=config.dropout
        )

        
        # ENCODER
        

        self.encoder = TransformerEncoder(
            num_layers=config.num_encoder_layers,
            d_model=config.d_model,
            num_heads=config.num_heads,
            d_ff=config.d_ff,
            dropout=config.dropout
        )

        
        # DECODER
        

        self.decoder = TransformerDecoder(
            num_layers=config.num_decoder_layers,
            d_model=config.d_model,
            num_heads=config.num_heads,
            d_ff=config.d_ff,
            dropout=config.dropout
        )

        
        # FINAL VOCABULARY PROJECTION
        

        self.output_projection = nn.Linear(
            config.d_model,
            config.tgt_vocab_size,
            bias=False
        )

        # Weight tying
        self.output_projection.weight = (
            self.tgt_embedding.embedding.weight
        )

    def forward(
        self,
        src_tokens: torch.Tensor,
        tgt_tokens: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        tgt_mask: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:

        
        # 1. SOURCE REPRESENTATION
        

        src = self.src_embedding(src_tokens)

        src = self.src_position(src)

        
        # 2. ENCODER
        

        encoder_output, encoder_attention = (
            self.encoder(
                src,
                src_mask
            )
        )

        
        # 3. TARGET REPRESENTATION
        

        tgt = self.tgt_embedding(tgt_tokens)

        tgt = self.tgt_position(tgt)

        
        # 4. DECODER
        

        (
            decoder_output,
            decoder_self_attention,
            decoder_cross_attention
        ) = self.decoder(
            x=tgt,
            encoder_output=encoder_output,
            tgt_mask=tgt_mask,
            src_mask=src_mask
        )

        
        # 5. VOCABULARY PROJECTION
        

        logits = self.output_projection(
            decoder_output
        )

        return {
            "logits": logits,
            "encoder_output": encoder_output,
            "encoder_attention": encoder_attention,
            "decoder_self_attention": decoder_self_attention,
            "decoder_cross_attention": decoder_cross_attention
        }