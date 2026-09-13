import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """
    Converts token IDs into dense vector representations.

    Input:
        token_ids: [batch_size, sequence_length]

    Output:
        embeddings: [batch_size, sequence_length, d_model]
    """

    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model
        )

        self.d_model = d_model

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(token_ids) * (self.d_model ** 0.5)