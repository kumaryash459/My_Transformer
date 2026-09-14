import torch

from configs.config import TransformerConfig
from device import get_device
from model.attention import ScaledDotProductAttention


def main() -> None:

    config = TransformerConfig()
    device = get_device()

    batch_size = 2
    num_heads = 8
    sequence_length = 16

    d_k = config.d_model // num_heads

    query = torch.randn(
        batch_size,
        num_heads,
        sequence_length,
        d_k,
        device=device
    )

    key = torch.randn(
        batch_size,
        num_heads,
        sequence_length,
        d_k,
        device=device
    )

    value = torch.randn(
        batch_size,
        num_heads,
        sequence_length,
        d_k,
        device=device
    )

    attention = ScaledDotProductAttention(
        dropout=0.0
    ).to(device)

    context, weights = attention(
        query,
        key,
        value
    )

    print("===== SCALED DOT-PRODUCT ATTENTION =====")
    print(f"Query shape:   {query.shape}")
    print(f"Key shape:     {key.shape}")
    print(f"Value shape:   {value.shape}")
    print(f"Context shape: {context.shape}")
    print(f"Weights shape: {weights.shape}")
    print(f"Device:        {context.device}")

    assert context.shape == (
        batch_size,
        num_heads,
        sequence_length,
        d_k
    )

    assert weights.shape == (
        batch_size,
        num_heads,
        sequence_length,
        sequence_length
    )

    # Every attention distribution should sum approximately to 1.
    row_sums = weights.sum(dim=-1)

    assert torch.allclose(
        row_sums,
        torch.ones_like(row_sums),
        atol=1e-5
    )

    print()
    print("Attention test PASSED.")


if __name__ == "__main__":
    main()