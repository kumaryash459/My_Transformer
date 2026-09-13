from tokenizer.vocab import Vocabulary

from data.preprocess import load_translation_data


def build_translation_vocab(
    max_samples: int | None = 10_000,
    min_frequency: int = 2
):
    data = load_translation_data(
        validation_ratio=0.1,
        max_samples=max_samples
    )

    train_source = data["train"]["source"]
    train_target = data["train"]["target"]

    source_vocab = Vocabulary(
        min_frequency=min_frequency
    )

    target_vocab = Vocabulary(
        min_frequency=min_frequency
    )

    source_vocab.build(train_source)
    target_vocab.build(train_target)

    return (
        data,
        source_vocab,
        target_vocab
    )


if __name__ == "__main__":

    data, source_vocab, target_vocab = (
        build_translation_vocab()
    )

    print(
        "Source vocabulary size:",
        len(source_vocab)
    )

    print(
        "Target vocabulary size:",
        len(target_vocab)
    )

    print("\nSpecial tokens:")

    print(
        source_vocab.token_to_id
    )

    print(
        target_vocab.token_to_id
    )