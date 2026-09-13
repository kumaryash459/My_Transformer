from datasets import load_dataset

from tokenizer.tokenizer import BasicTokenizer


def load_translation_data(
    validation_ratio: float = 0.1,
    max_samples: int | None = None,
    max_length: int = 128
):
    dataset = load_dataset(
        "Helsinki-NLP/opus_books",
        "de-en",
        split="train"
    )

    if max_samples is not None:
        dataset = dataset.select(
            range(min(max_samples, len(dataset)))
        )

    tokenizer = BasicTokenizer()

    source_sentences = []
    target_sentences = []

    # --------------------------------------------------------
    # Reserve two positions:
    #
    # <BOS> + actual tokens + <EOS>
    #
    # Therefore actual content can contain at most:
    #
    # max_length - 2
    # --------------------------------------------------------

    content_max_length = max_length - 2

    if content_max_length <= 0:
        raise ValueError(
            "max_length must be greater than 2."
        )

    for example in dataset:

        translation = example["translation"]

        german = translation["de"].strip()
        english = translation["en"].strip()

        if not german or not english:
            continue

        # ----------------------------------------------------
        # IMPORTANT:
        # Use the SAME tokenizer used by Vocabulary.
        # ----------------------------------------------------

        german_tokens = tokenizer.tokenize(
            german
        )

        english_tokens = tokenizer.tokenize(
            english
        )

        # ----------------------------------------------------
        # Check ACTUAL tokenized length.
        # ----------------------------------------------------

        if len(german_tokens) > content_max_length:
            continue

        if len(english_tokens) > content_max_length:
            continue

        source_sentences.append(
            english
        )

        target_sentences.append(
            german
        )

    # --------------------------------------------------------
    # Safety check
    # --------------------------------------------------------

    if len(source_sentences) != len(target_sentences):
        raise ValueError(
            "Source and target lengths do not match."
        )

    total_samples = len(source_sentences)

    if total_samples == 0:
        raise ValueError(
            "No valid samples found after preprocessing."
        )

    # --------------------------------------------------------
    # Train / Validation split
    # --------------------------------------------------------

    validation_size = int(
        total_samples * validation_ratio
    )

    train_size = total_samples - validation_size

    train_source = source_sentences[
        :train_size
    ]

    train_target = target_sentences[
        :train_size
    ]

    val_source = source_sentences[
        train_size:
    ]

    val_target = target_sentences[
        train_size:
    ]

    return {
        "train": {
            "source": train_source,
            "target": train_target
        },
        "validation": {
            "source": val_source,
            "target": val_target
        }
    }


if __name__ == "__main__":

    data = load_translation_data(
        validation_ratio=0.1,
        max_samples=10_000,
        max_length=128
    )

    print(
        "Training samples:",
        len(data["train"]["source"])
    )

    print(
        "Validation samples:",
        len(data["validation"]["source"])
    )

    print("\nExample:")

    print(
        "EN:",
        data["train"]["source"][0]
    )

    print(
        "DE:",
        data["train"]["target"][0]
    )