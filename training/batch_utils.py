import torch


def prepare_decoder_inputs(
    target_tokens: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:

    if target_tokens.ndim != 2:
        raise ValueError(
            "target_tokens must have shape [batch_size, sequence_length]."
        )

    if target_tokens.size(1) < 2:
        raise ValueError(
            "Target sequence must contain at least two tokens."
        )

    decoder_input = target_tokens[:, :-1]
    expected_output = target_tokens[:, 1:]

    return decoder_input, expected_output