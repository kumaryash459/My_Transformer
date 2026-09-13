from torch.utils.data import DataLoader

from data.preprocess import load_translation_data
from data.dataset import TranslationDataset
from data.collate import create_collate_fn
from tokenizer.vocab import Vocabulary


# -----------------------------------------
# Load data
# -----------------------------------------

data = load_translation_data(
    validation_ratio=0.1,
    max_samples=10_000
)


train_source = data["train"]["source"]
train_target = data["train"]["target"]


# -----------------------------------------
# Build vocabulary
# -----------------------------------------

source_vocab = Vocabulary(
    min_frequency=2
)

target_vocab = Vocabulary(
    min_frequency=2
)

source_vocab.build(train_source)
target_vocab.build(train_target)


print("Source vocabulary:", len(source_vocab))
print("Target vocabulary:", len(target_vocab))


# -----------------------------------------
# Dataset
# -----------------------------------------

train_dataset = TranslationDataset(
    source_sentences=train_source,
    target_sentences=train_target,
    source_vocab=source_vocab,
    target_vocab=target_vocab
)


# -----------------------------------------
# DataLoader
# -----------------------------------------

collate_fn = create_collate_fn(
    pad_token_id=source_vocab.token_to_id[
        source_vocab.pad_token
    ]
)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    collate_fn=collate_fn
)


# -----------------------------------------
# Test batch
# -----------------------------------------

batch = next(iter(train_loader))

print("\nBatch information:")

print(
    "Source shape:",
    batch["src"].shape
)

print(
    "Target shape:",
    batch["tgt"].shape
)

print(
    "Source dtype:",
    batch["src"].dtype
)

print(
    "Target dtype:",
    batch["tgt"].dtype
)


# -----------------------------------------
# Display example
# -----------------------------------------

src_ids = batch["src"][0].tolist()
tgt_ids = batch["tgt"][0].tolist()

print("\nExample token IDs:")

print("SRC:", src_ids)
print("TGT:", tgt_ids)

print("\nDecoded example:")

print(
    "SRC:",
    source_vocab.decode(src_ids)
)

print(
    "TGT:",
    target_vocab.decode(tgt_ids)
)


# -----------------------------------------
# Assertions
# -----------------------------------------

assert batch["src"].ndim == 2
assert batch["tgt"].ndim == 2

assert batch["src"].dtype.is_floating_point is False
assert batch["tgt"].dtype.is_floating_point is False

assert len(source_vocab) > 4
assert len(target_vocab) > 4

print("\nReal dataset pipeline test passed.")