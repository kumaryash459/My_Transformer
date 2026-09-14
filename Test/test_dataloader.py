from torch.utils.data import DataLoader

from tokenizer.vocab import Vocabulary
from data.dataset import TranslationDataset
from data.collate import create_collate_fn


def main():

    source_sentences = [
        "I love machine learning",
        "I like artificial intelligence",
        "Transformers are powerful models",
        "Deep learning is interesting",
        "Python is useful"
    ]

    target_sentences = [
        "Ich liebe maschinelles Lernen",
        "Ich mag künstliche Intelligenz",
        "Transformer sind leistungsfähige Modelle",
        "Deep Learning ist interessant",
        "Python ist nützlich"
    ]

    # ========================================
    # Build vocabularies
    # ========================================

    source_vocab = Vocabulary(
        min_frequency=1
    )

    target_vocab = Vocabulary(
        min_frequency=1
    )

    source_vocab.build(
        source_sentences
    )

    target_vocab.build(
        target_sentences
    )

    # ========================================
    # Dataset
    # ========================================

    dataset = TranslationDataset(
        source_sentences=source_sentences,
        target_sentences=target_sentences,
        source_vocab=source_vocab,
        target_vocab=target_vocab
    )

    # ========================================
    # DataLoader
    # ========================================

    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=create_collate_fn(
            pad_token_id=source_vocab.token_to_id[
                source_vocab.pad_token
            ]
        )
    )

    # ========================================
    # Test batch
    # ========================================

    batch = next(iter(dataloader))

    print("===== DATALOADER =====")

    print(
        f"Source batch shape: {batch['src'].shape}"
    )

    print(
        f"Target batch shape: {batch['tgt'].shape}"
    )

    print()
    print("Source batch:")
    print(batch["src"])

    print()
    print("Target batch:")
    print(batch["tgt"])

    print()

    assert batch["src"].dim() == 2
    assert batch["tgt"].dim() == 2

    print("DataLoader test PASSED.")


if __name__ == "__main__":
    main()