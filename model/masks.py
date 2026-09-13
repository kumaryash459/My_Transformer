import torch


def create_padding_mask(
    token_ids: torch.Tensor,
    pad_token_id: int = 0
) -> torch.Tensor:
    """
    Creates a padding mask.

    Input:
        token_ids: [batch_size, sequence_length]

    Output:
        mask: [batch_size, 1, 1, sequence_length]

    1 = valid token
    0 = padding token
    """

    mask = (token_ids != pad_token_id)

    return mask.unsqueeze(1).unsqueeze(2)


def create_causal_mask(
    sequence_length: int,
    device: torch.device
) -> torch.Tensor:
    """
    Creates a lower-triangular causal mask.

    Output:
        [1, 1, sequence_length, sequence_length]

    Example for sequence length 4:

        1 0 0 0
        1 1 0 0
        1 1 1 0
        1 1 1 1
    """

    mask = torch.tril(
        torch.ones(
            sequence_length,
            sequence_length,
            device=device,
            dtype=torch.bool
        )
    )

    return mask.unsqueeze(0).unsqueeze(0)


def create_decoder_mask(
    token_ids: torch.Tensor,
    pad_token_id: int = 0
) -> torch.Tensor:
    """
    Combines padding mask and causal mask.

    Output:
        [batch_size, 1, sequence_length, sequence_length]
    """

    batch_size, sequence_length = token_ids.shape

    padding_mask = create_padding_mask(
        token_ids,
        pad_token_id
    )

    causal_mask = create_causal_mask(
        sequence_length,
        token_ids.device
    )

    return padding_mask & causal_mask