import torch

from configs.config import TransformerConfig
from device import get_device
from model.transformer import Transformer
from model.masks import (
    create_padding_mask,
    create_decoder_mask
)


def main() -> None:

    config = TransformerConfig()
    device = get_device()

    batch_size = 2

    source_length = 20
    target_length = 16

    
    # Fake source and target tokens
    

    src_tokens = torch.randint(
        3,
        config.src_vocab_size,
        (
            batch_size,
            source_length
        ),
        device=device
    )

    tgt_tokens = torch.randint(
        3,
        config.tgt_vocab_size,
        (
            batch_size,
            target_length
        ),
        device=device
    )

    
    # Masks
    

    src_mask = create_padding_mask(
        src_tokens,
        config.pad_token_id
    )

    tgt_mask = create_decoder_mask(
        tgt_tokens,
        config.pad_token_id
    )

    
    # Model
    

    model = Transformer(
        config
    ).to(device)

    
    # Forward Pass
    

    output = model(
        src_tokens=src_tokens,
        tgt_tokens=tgt_tokens,
        src_mask=src_mask,
        tgt_mask=tgt_mask
    )

    logits = output["logits"]

    
    # Print information
    

    print("===== FULL TRANSFORMER =====")

    print(
        f"Source tokens:        "
        f"{src_tokens.shape}"
    )

    print(
        f"Target tokens:        "
        f"{tgt_tokens.shape}"
    )

    print(
        f"Encoder output:       "
        f"{output['encoder_output'].shape}"
    )

    print(
        f"Final logits:          "
        f"{logits.shape}"
    )

    print(
        f"Expected logits:      "
        f"({batch_size}, {target_length}, "
        f"{config.tgt_vocab_size})"
    )

    print(
        f"Device:               "
        f"{logits.device}"
    )


    # Assertions
    

    assert logits.shape == (
        batch_size,
        target_length,
        config.tgt_vocab_size
    )

    assert output["encoder_output"].shape == (
        batch_size,
        source_length,
        config.d_model
    )

    assert logits.device == src_tokens.device

    
    # Parameter count
    

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print()
    print(
        f"Total parameters:     "
        f"{total_parameters:,}"
    )

    print(
        f"Trainable parameters: "
        f"{trainable_parameters:,}"
    )

    print()
    print("Full Transformer test PASSED.")


if __name__ == "__main__":
    main()