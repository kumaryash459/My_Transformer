from datasets import load_dataset
from tokenizer.tokenizer import BasicTokenizer


def is_metadata_pair(source: str, target: str) -> bool:
    """
    Detect obvious metadata / book-header pairs.
    """

    metadata_patterns = [
        "source:",
        "project gutenberg",
        "http://",
        "https://",
        "www.",
    ]

    source_lower = source.lower().strip()
    target_lower = target.lower().strip()

    for pattern in metadata_patterns:
        if pattern in source_lower or pattern in target_lower:
            return True

    return False


def is_valid_length_ratio(
    source_length: int,
    target_length: int,
    min_ratio: float = 0.25,
    max_ratio: float = 4.0,
) -> bool:
    """
    Remove pairs where source and target lengths are
    extremely different.

    This is only a heuristic to remove obvious noisy pairs.
    """

    if source_length == 0 or target_length == 0:
        return False

    ratio = target_length / source_length

    return min_ratio <= ratio <= max_ratio


def load_translation_data(
    validation_ratio: float = 0.1,
    max_samples: int | None = None,
    max_length: int = 64,
    min_length: int = 3,
):
    print("Loading OPUS Books English-German dataset...")

    dataset = load_dataset(
        "Helsinki-NLP/opus_books",
        "de-en",
        split="train",
    )

    print(f"Raw dataset size: {len(dataset)}")

    if max_samples is not None:
        dataset = dataset.select(
            range(min(max_samples, len(dataset)))
        )

    tokenizer = BasicTokenizer()

    source_sentences = []
    target_sentences = []

    stats = {
        "raw": len(dataset),
        "metadata_removed": 0,
        "empty_removed": 0,
        "length_removed": 0,
        "ratio_removed": 0,
        "duplicate_removed": 0,
        "kept": 0,
    }

    seen_pairs = set()

    content_max_length = max_length - 2

    if content_max_length <= 0:
        raise ValueError(
            "max_length must be greater than 2."
        )

    for example in dataset:

        translation = example["translation"]

        german = translation["de"].strip()
        english = translation["en"].strip()

        # --------------------------------------------------
        # 1. Empty samples
        # --------------------------------------------------

        if not english or not german:
            stats["empty_removed"] += 1
            continue

        # --------------------------------------------------
        # 2. Metadata / book headers
        # --------------------------------------------------

        if is_metadata_pair(english, german):
            stats["metadata_removed"] += 1
            continue

        # --------------------------------------------------
        # 3. Tokenize
        # --------------------------------------------------

        english_tokens = tokenizer.tokenize(english)
        german_tokens = tokenizer.tokenize(german)

        source_length = len(english_tokens)
        target_length = len(german_tokens)

        # --------------------------------------------------
        # 4. Minimum length
        # --------------------------------------------------

        if (
            source_length < min_length
            or target_length < min_length
        ):
            stats["length_removed"] += 1
            continue

        # --------------------------------------------------
        # 5. Maximum length
        # --------------------------------------------------

        if (
            source_length > content_max_length
            or target_length > content_max_length
        ):
            stats["length_removed"] += 1
            continue

        # --------------------------------------------------
        # 6. Source / target length ratio
        # --------------------------------------------------

        if not is_valid_length_ratio(
            source_length,
            target_length,
        ):
            stats["ratio_removed"] += 1
            continue

        # --------------------------------------------------
        # 7. Remove exact duplicate pairs
        # --------------------------------------------------

        pair = (
            english.lower(),
            german.lower(),
        )

        if pair in seen_pairs:
            stats["duplicate_removed"] += 1
            continue

        seen_pairs.add(pair)

        # --------------------------------------------------
        # 8. Keep pair
        # --------------------------------------------------

        source_sentences.append(english)
        target_sentences.append(german)

        stats["kept"] += 1

    if len(source_sentences) == 0:
        raise ValueError(
            "No valid translation pairs remain after preprocessing."
        )

    # ------------------------------------------------------
    # Deterministic train / validation split
    # ------------------------------------------------------

    total_samples = len(source_sentences)

    validation_size = int(
        total_samples * validation_ratio
    )

    train_size = total_samples - validation_size

    train_source = source_sentences[:train_size]
    train_target = target_sentences[:train_size]

    val_source = source_sentences[train_size:]
    val_target = target_sentences[train_size:]

    # ------------------------------------------------------
    # Statistics
    # ------------------------------------------------------

    print()
    print("=" * 70)
    print("DATASET CLEANING STATISTICS")
    print("=" * 70)

    print(f"Raw samples:              {stats['raw']}")
    print(f"Metadata removed:         {stats['metadata_removed']}")
    print(f"Empty samples removed:    {stats['empty_removed']}")
    print(f"Length filtered:          {stats['length_removed']}")
    print(f"Ratio filtered:           {stats['ratio_removed']}")
    print(f"Duplicates removed:       {stats['duplicate_removed']}")
    print(f"Final usable samples:     {stats['kept']}")

    print()
    print(f"Training samples:         {len(train_source)}")
    print(f"Validation samples:       {len(val_source)}")

    return {
        "train": {
            "source": train_source,
            "target": train_target,
        },
        "validation": {
            "source": val_source,
            "target": val_target,
        },
    }


if __name__ == "__main__":

    data = load_translation_data(
        validation_ratio=0.1,
        max_samples=10_000,
        max_length=64,
    )

    print()
    print("=" * 70)
    print("EXAMPLES")
    print("=" * 70)

    for i in range(min(10, len(data["train"]["source"]))):

        print()
        print(f"[{i}] EN:")
        print(data["train"]["source"][i])

        print("DE:")
        print(data["train"]["target"][i])