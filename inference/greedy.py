import torch

from model.masks import (
    create_padding_mask,
    create_decoder_mask
)


@torch.no_grad()
def greedy_decode(
    model,
    src_tokens: torch.Tensor,
    bos_token_id: int,
    eos_token_id: int,
    max_length: int,
    src_pad_token_id: int = 0,
    tgt_pad_token_id: int = 0
) -> torch.Tensor:

    model.eval()

    # ========================================================
    # SOURCE PADDING MASK
    # ========================================================

    src_mask = create_padding_mask(
        src_tokens,
        pad_token_id=src_pad_token_id
    )

    # ========================================================
    # ENCODER
    # ========================================================

    src = model.src_embedding(
        src_tokens
    )

    src = model.src_position(
        src
    )

    encoder_output, _ = model.encoder(
        src,
        src_mask
    )

    # ========================================================
    # START DECODER WITH BOS
    # ========================================================

    batch_size = src_tokens.size(0)

    generated = torch.full(
        (batch_size, 1),
        bos_token_id,
        dtype=torch.long,
        device=src_tokens.device
    )

    # ========================================================
    # AUTOREGRESSIVE GENERATION
    # ========================================================

    for _ in range(max_length - 1):

        # ----------------------------------------------------
        # TARGET MASK
        # ----------------------------------------------------

        tgt_mask = create_decoder_mask(
            generated,
            pad_token_id=tgt_pad_token_id
        )

        # ----------------------------------------------------
        # TARGET EMBEDDING
        # ----------------------------------------------------

        tgt = model.tgt_embedding(
            generated
        )

        tgt = model.tgt_position(
            tgt
        )

        # ----------------------------------------------------
        # DECODER
        # ----------------------------------------------------

        decoder_output, _, _ = model.decoder(
            x=tgt,
            encoder_output=encoder_output,
            tgt_mask=tgt_mask,
            src_mask=src_mask
        )

        # ----------------------------------------------------
        # VOCABULARY LOGITS
        # ----------------------------------------------------

        logits = model.output_projection(
            decoder_output
        )

        # ----------------------------------------------------
        # LAST POSITION
        # ----------------------------------------------------

        next_token_logits = logits[:, -1, :]

        # ----------------------------------------------------
        # GREEDY SELECTION
        # ----------------------------------------------------

        next_token = torch.argmax(
            next_token_logits,
            dim=-1,
            keepdim=True
        )

        # ----------------------------------------------------
        # APPEND TOKEN
        # ----------------------------------------------------

        generated = torch.cat(
            [
                generated,
                next_token
            ],
            dim=1
        )

        # ----------------------------------------------------
        # STOP AT EOS
        # ----------------------------------------------------

        if torch.all(
            next_token.squeeze(1)
            == eos_token_id
        ):
            break

    return generated