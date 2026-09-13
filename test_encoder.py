import torch

from configs.config import TransformerConfig
from device import get_device
from model.encoder import EncoderLayer


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

    encoder_layer = EncoderLayer(
        d_model=config.d_model,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
        dropout=0.0
    ).to(device)

    output, attention_weights = encoder_layer(x)

    print("===== ENCODER LAYER =====")

    print(f"Input shape:            {x.shape}")
    print(f"Output shape:           {output.shape}")
    print(f"Attention shape:        {attention_weights.shape}")
    print(f"Device:                 {output.device}")

    assert output.shape == (
        batch_size,
        sequence_length,
        config.d_model
    )

    assert attention_weights.shape == (
        batch_size,
        config.num_heads,
        sequence_length,
        sequence_length
    )

    assert output.device == x.device

    print()
    print("Encoder Layer test PASSED.")


if __name__ == "__main__":
    main()