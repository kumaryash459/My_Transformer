import torch

from configs.config import TransformerConfig
from device import get_device, print_device_info


def main() -> None:
    config = TransformerConfig()

    print("===== TRANSFORMER CONFIGURATION =====")
    print(f"Model dimension: {config.d_model}")
    print(f"Attention heads: {config.num_heads}")
    print(f"Encoder layers: {config.num_encoder_layers}")
    print(f"Decoder layers: {config.num_decoder_layers}")
    print(f"Feed-forward dimension: {config.d_ff}")

    print()

    print("===== DEVICE =====")
    print_device_info()

    print()

    device = get_device()

    x = torch.randn(
        4,
        config.max_seq_length,
        config.d_model,
        device=device,
    )

    print("===== TENSOR TEST =====")
    print(f"Shape: {x.shape}")
    print(f"Device: {x.device}")


if __name__ == "__main__":
    main()