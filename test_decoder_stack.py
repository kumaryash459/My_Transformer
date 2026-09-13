import torch

from configs.config import TransformerConfig
from device import get_device
from model.decoder import TransformerDecoder


def main() -> None:

    config = TransformerConfig()
    device = get_device()

    batch_size = 2

    source_length = 20
    target_length = 16

    decoder_input = torch.randn(
        batch_size,
        target_length,
        config.d_model,
        device=device
    )

    encoder_output = torch.randn(
        batch_size,
        source_length,
        config.d_model,
        device=device
    )

    decoder = TransformerDecoder(
        num_layers=config.num_decoder_layers,
        d_model=config.d_model,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
        dropout=0.0
    ).to(device)

    (
        output,
        self_attention_maps,
        cross_attention_maps
    ) = decoder(
        decoder_input,
        encoder_output
    )

    print("===== TRANSFORMER DECODER =====")

    print(f"Decoder input:      {decoder_input.shape}")
    print(f"Encoder output:     {encoder_output.shape}")
    print(f"Decoder output:     {output.shape}")

    print(
        f"Layers:             "
        f"{len(self_attention_maps)}"
    )

    print(
        f"Self-attention:     "
        f"{self_attention_maps[0].shape}"
    )

    print(
        f"Cross-attention:    "
        f"{cross_attention_maps[0].shape}"
    )

    print(f"Device:             {output.device}")

    assert output.shape == decoder_input.shape

    assert len(self_attention_maps) == (
        config.num_decoder_layers
    )

    assert len(cross_attention_maps) == (
        config.num_decoder_layers
    )

    assert output.device == decoder_input.device

    print()
    print("Decoder Stack test PASSED.")


if __name__ == "__main__":
    main()