import torch
from torch.nn.utils.rnn import pad_sequence


def create_collate_fn(
    pad_token_id: int = 0
):

    def collate_fn(batch):

        source_sequences = [
            item["src"]
            for item in batch
        ]

        target_sequences = [
            item["tgt"]
            for item in batch
        ]

        src_batch = pad_sequence(
            source_sequences,
            batch_first=True,
            padding_value=pad_token_id
        )

        tgt_batch = pad_sequence(
            target_sequences,
            batch_first=True,
            padding_value=pad_token_id
        )

        return {
            "src": src_batch,
            "tgt": tgt_batch
        }

    return collate_fn