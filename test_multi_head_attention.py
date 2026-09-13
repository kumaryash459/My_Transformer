import torch

from configs.config import TransformerConfig
from device import get_device
from model.attention import MultiHeadAttention


def main() -> None:

    config = TransformerConfig()
    device = get_device()

    batch_size = 2
    sequence_length = 16

    x = torch.randn(
        batch_size,
        sequence_length,
        config.d_model,
        device=device
    )

    mha = MultiHeadAttention(
        d_model=config.d_model,
        num_heads=config.num_heads,
        dropout=0.0
    ).to(device)

    output, weights = mha(
        query=x,
        key=x,
        value=x
    )

    print("===== MULTI-HEAD ATTENTION =====")

    print(f"Input shape:   {x.shape}")
    print(f"Output shape:  {output.shape}")
    print(f"Weights shape: {weights.shape}")
    print(f"Device:        {output.device}")

    assert output.shape == (
        batch_size,
        sequence_length,
        config.d_model
    )

    assert weights.shape == (
        batch_size,
        config.num_heads,
        sequence_length,
        sequence_length
    )

    print()
    print("Multi-Head Attention test PASSED.")


if __name__ == "__main__":
    main()