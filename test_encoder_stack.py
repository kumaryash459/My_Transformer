import torch

from configs.config import TransformerConfig
from device import get_device
from model.encoder import TransformerEncoder


def main() -> None:

    config = TransformerConfig()
    device = get_device()

    batch_size = 2
    sequence_length = 20

    x = torch.randn(
        batch_size,
        sequence_length,
        config.d_model,
        device=device
    )

    encoder = TransformerEncoder(
        num_layers=config.num_encoder_layers,
        d_model=config.d_model,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
        dropout=0.0
    ).to(device)

    output, attention_maps = encoder(x)

    print("===== TRANSFORMER ENCODER =====")

    print(f"Input:          {x.shape}")
    print(f"Output:         {output.shape}")
    print(f"Layers:         {len(attention_maps)}")
    print(f"Attention map:  {attention_maps[0].shape}")
    print(f"Device:          {output.device}")

    assert output.shape == x.shape

    assert len(attention_maps) == config.num_encoder_layers

    assert output.device == x.device

    print()
    print("Encoder Stack test PASSED.")


if __name__ == "__main__":
    main()