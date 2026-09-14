import torch

from configs.config import TransformerConfig
from device import get_device
from model.decoder import DecoderLayer


def main() -> None:

    config = TransformerConfig()
    device = get_device()

    batch_size = 2

    source_length = 20
    target_length = 16

    # Decoder input
    x = torch.randn(
        batch_size,
        target_length,
        config.d_model,
        device=device
    )

    # Output produced by encoder
    encoder_output = torch.randn(
        batch_size,
        source_length,
        config.d_model,
        device=device
    )

    decoder_layer = DecoderLayer(
        d_model=config.d_model,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
        dropout=0.0
    ).to(device)

    (
        output,
        self_attention_weights,
        cross_attention_weights
    ) = decoder_layer(
        x=x,
        encoder_output=encoder_output
    )

    print("===== DECODER LAYER =====")

    print(f"Decoder input shape:       {x.shape}")
    print(f"Encoder output shape:      {encoder_output.shape}")
    print(f"Decoder output shape:      {output.shape}")

    print()
    print(
        f"Self-attention shape:      "
        f"{self_attention_weights.shape}"
    )

    print(
        f"Cross-attention shape:     "
        f"{cross_attention_weights.shape}"
    )

    print(f"Device:                     {output.device}")

    
    # Shape checks
    

    assert output.shape == (
        batch_size,
        target_length,
        config.d_model
    )

    assert self_attention_weights.shape == (
        batch_size,
        config.num_heads,
        target_length,
        target_length
    )

    assert cross_attention_weights.shape == (
        batch_size,
        config.num_heads,
        target_length,
        source_length
    )

    assert output.device == x.device

    print()
    print("Decoder Layer test PASSED.")


if __name__ == "__main__":
    main()