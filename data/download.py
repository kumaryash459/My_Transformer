from pathlib import Path

from datasets import load_dataset


DATA_DIR = Path("data/raw")
DATA_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def download_opus_books(
    max_samples: int | None = None
):
    print("Loading OPUS Books English-German dataset...")

    dataset = load_dataset(
        "Helsinki-NLP/opus_books",
        "de-en",
        split="train"
    )

    print(f"Total samples available: {len(dataset)}")

    if max_samples is not None:
        dataset = dataset.select(
            range(min(max_samples, len(dataset)))
        )

    return dataset


if __name__ == "__main__":
    dataset = download_opus_books()

    print("\nDataset loaded successfully.")
    print("Number of samples:", len(dataset))

    print("\nFirst sample:")
    print(dataset[0])