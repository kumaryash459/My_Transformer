import torch

from configs.config import TransformerConfig
from device import get_device
from model.feed_forward import PositionWiseFeedForward


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

    ffn = PositionWiseFeedForward(
        d_model=config.d_model,
        d_ff=config.d_ff,
        dropout=0.0
    ).to(device)

    output = ffn(x)

    print("===== FEED-FORWARD NETWORK =====")

    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Device:       {output.device}")

    assert output.shape == x.shape

    assert output.device == x.device

    print()
    print("Feed-Forward Network test PASSED.")


if __name__ == "__main__":
    main()