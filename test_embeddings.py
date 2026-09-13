import torch

from configs.config import TransformerConfig
from device import get_device
from model.embeddings import TokenEmbedding
from model.positional_encoding import PositionalEncoding


def main() -> None:
    config = TransformerConfig()
    device = get_device()

    batch_size = 4
    sequence_length = 20

    token_ids = torch.randint(
        low=0,
        high=config.src_vocab_size,
        size=(batch_size, sequence_length),
        device=device
    )

    embedding = TokenEmbedding(
        vocab_size=config.src_vocab_size,
        d_model=config.d_model
    ).to(device)

    positional_encoding = PositionalEncoding(
        d_model=config.d_model,
        max_seq_length=config.max_seq_length,
        dropout=config.dropout
    ).to(device)

    x = embedding(token_ids)

    print("===== TOKEN EMBEDDING =====")
    print(f"Input shape:  {token_ids.shape}")
    print(f"Output shape: {x.shape}")
    print(f"Device:       {x.device}")

    x = positional_encoding(x)

    print()
    print("===== POSITIONAL ENCODING =====")
    print(f"Output shape: {x.shape}")
    print(f"Device:       {x.device}")

    assert x.shape == (
        batch_size,
        sequence_length,
        config.d_model
    )

    print()
    print("Embedding + Positional Encoding test PASSED.")


if __name__ == "__main__":
    main()