import torch

from configs.config import TransformerConfig
from device import get_device
from model.masks import (
    create_padding_mask,
    create_causal_mask,
    create_decoder_mask,
)


def main() -> None:

    config = TransformerConfig()
    device = get_device()

    token_ids = torch.tensor(
        [
            [5, 8, 12, 7, 0, 0],
            [4, 9, 3, 6, 2, 0],
        ],
        device=device
    )

    print("===== PADDING MASK =====")

    padding_mask = create_padding_mask(
        token_ids,
        config.pad_token_id
    )

    print(f"Token shape: {token_ids.shape}")
    print(f"Mask shape:  {padding_mask.shape}")
    print(padding_mask)

    assert padding_mask.shape == (2, 1, 1, 6)

    print()
    print("===== CAUSAL MASK =====")

    causal_mask = create_causal_mask(
        sequence_length=6,
        device=device
    )

    print(f"Mask shape: {causal_mask.shape}")
    print(causal_mask[0, 0].int())

    assert causal_mask.shape == (1, 1, 6, 6)

    print()
    print("===== DECODER MASK =====")

    decoder_mask = create_decoder_mask(
        token_ids,
        config.pad_token_id
    )

    print(f"Mask shape: {decoder_mask.shape}")
    print(decoder_mask[0, 0].int())

    assert decoder_mask.shape == (2, 1, 6, 6)

    print()
    print("All mask tests PASSED.")


if __name__ == "__main__":
    main()