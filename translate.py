import torch

from configs.config import TransformerConfig
from data.preprocess import load_translation_data
from tokenizer.vocab import Vocabulary
from model.transformer import Transformer
from inference.greedy import greedy_decode


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def main():

    # ============================================================
    # DEVICE
    # ============================================================

    device = get_device()

    print("=" * 60)
    print("DEVICE")
    print("=" * 60)

    print(f"Device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ============================================================
    # CONFIGURATION
    # ============================================================

    # IMPORTANT:
    # These values MUST match the settings used during training.

    MAX_SAMPLES = 10_000
    MAX_LENGTH = 64
    VALIDATION_RATIO = 0.1
    MIN_FREQUENCY = 1

    CHECKPOINT_PATH = "checkpoints/best.pt"

    # ============================================================
    # LOADING DATA
    # ============================================================

    print()
    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)

    data = load_translation_data(
        validation_ratio=VALIDATION_RATIO,
        max_samples=MAX_SAMPLES,
        max_length=MAX_LENGTH,
    )

    train_source = data["train"]["source"]
    train_target = data["train"]["target"]

    val_source = data["validation"]["source"]
    val_target = data["validation"]["target"]

    # ============================================================
    # BUILD VOCABULARIES
    # ============================================================

    print("=" * 60)
    print("BUILDING VOCABULARIES")
    print("=" * 60)

    source_vocab = Vocabulary(
        min_frequency=MIN_FREQUENCY
    )

    target_vocab = Vocabulary(
        min_frequency=MIN_FREQUENCY
    )

    source_vocab.build(train_source)
    target_vocab.build(train_target)

    print(f"Source vocabulary: {len(source_vocab)}")
    print(f"Target vocabulary: {len(target_vocab)}")

    # ============================================================
    # CREATE CONFIG
    # ============================================================

    config = TransformerConfig(
        src_vocab_size=len(source_vocab),
        tgt_vocab_size=len(target_vocab),

        max_seq_length=MAX_LENGTH,

        d_model=256,
        num_heads=8,

        num_encoder_layers=4,
        num_decoder_layers=4,

        d_ff=1024,

        dropout=0.1,

        pad_token_id=source_vocab.token_to_id[
            source_vocab.pad_token
        ],

        bos_token_id=target_vocab.token_to_id[
            target_vocab.bos_token
        ],

        eos_token_id=target_vocab.token_to_id[
            target_vocab.eos_token
        ],

        batch_size=16,

        learning_rate=1e-4,

        weight_decay=0.01,

        device=device.type,
    )

    # ============================================================
    # CREATE MODEL
    # ============================================================

    print()
    print("=" * 60)
    print("CREATING MODEL")
    print("=" * 60)

    model = Transformer(config)

    model = model.to(device)

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(f"Model parameters: {total_parameters:,}")

    # ============================================================
    # LOAD CHECKPOINT
    # ============================================================

    print()
    print("=" * 60)
    print("LOADING CHECKPOINT")
    print("=" * 60)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print(
        f"Checkpoint loaded: {CHECKPOINT_PATH}"
    )

    print(
        f"Epoch: {checkpoint['epoch']}"
    )

    print(
        f"Validation loss: "
        f"{checkpoint['val_loss']}"
    )

    # ============================================================
    # TRANSLATION TEST
    # ============================================================

    print()
    print("=" * 60)
    print("TRANSLATION TEST")
    print("=" * 60)

    model.eval()

    # Test sentences from validation data.
    test_indices = [
        0,
        1,
        2,
        3,
        4,
    ]

    for index in test_indices:

        english_sentence = val_source[index]
        german_sentence = val_target[index]

        source_ids = source_vocab.encode(
            english_sentence,
            add_bos=True,
            add_eos=True
        )

        source_tensor = torch.tensor(
            [source_ids],
            dtype=torch.long,
            device=device
        )

        generated_ids = greedy_decode(
            model=model,
            src_tokens=source_tensor,

            bos_token_id=config.bos_token_id,
            eos_token_id=config.eos_token_id,

            max_length=MAX_LENGTH,

            src_pad_token_id=config.pad_token_id,
            tgt_pad_token_id=target_vocab.token_to_id[
                target_vocab.pad_token
            ]
        )

        predicted_ids = generated_ids[0].tolist()

        predicted_sentence = target_vocab.decode(
            predicted_ids
        )

        print()
        print("-" * 60)
        print(f"Example {index}")
        print("-" * 60)

        print(f"English : {english_sentence}")
        print(f"Actual  : {german_sentence}")
        print(f"Predicted: {predicted_sentence}")


if __name__ == "__main__":
    main()