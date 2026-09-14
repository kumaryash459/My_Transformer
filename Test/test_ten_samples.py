import torch
from torch.utils.data import DataLoader

from configs.config import TransformerConfig
from data.preprocess import load_translation_data
from data.dataset import TranslationDataset
from data.collate import create_collate_fn
from tokenizer.vocab import Vocabulary

from model.transformer import Transformer
from training.optimizer import create_optimizer
from training.trainer import Trainer
from training.losses import LabelSmoothedCrossEntropyLoss

from device import get_device


# ============================================================
# CONFIG
# ============================================================

MAX_SEQUENCE_LENGTH = 64
NUM_EPOCHS = 300
BATCH_SIZE = 2

LEARNING_RATE = 1e-3

device = get_device()

print("=" * 70)
print("10-SAMPLE TRANSFORMER OVERFIT TEST")
print("=" * 70)

print("Device:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

print()


# ============================================================
# LOAD DATA
# ============================================================

data = load_translation_data(
    validation_ratio=0.0,
    max_samples=100,
    max_length=MAX_SEQUENCE_LENGTH
)

all_source = data["train"]["source"]
all_target = data["train"]["target"]


# ============================================================
# SELECT SHORT REAL TRANSLATION PAIRS
# ============================================================

selected_source = []
selected_target = []

for source, target in zip(all_source, all_target):

    source_tokens = source.split()
    target_tokens = target.split()

    # Remove obvious metadata
    if source.startswith("Source:"):
        continue

    if target.startswith("Source:"):
        continue

    if source in {
        "Jane Eyre",
        "Charlotte Bronte",
        "CHAPTER I"
    }:
        continue

    # Keep relatively short examples
    if len(source_tokens) < 5:
        continue

    if len(source_tokens) > 20:
        continue

    if len(target_tokens) < 5:
        continue

    if len(target_tokens) > 30:
        continue

    selected_source.append(source)
    selected_target.append(target)

    if len(selected_source) == 10:
        break


# ============================================================
# VERIFY DATA
# ============================================================

if len(selected_source) < 10:
    raise ValueError(
        f"Only found {len(selected_source)} suitable samples."
    )

print("Selected samples:")
print()

for i, (source, target) in enumerate(
    zip(selected_source, selected_target)
):

    print(f"[{i}] EN: {source}")
    print(f"    DE: {target}")
    print()


# ============================================================
# VOCABULARY
# ============================================================

source_vocab = Vocabulary(min_frequency=1)
target_vocab = Vocabulary(min_frequency=1)

source_vocab.build(selected_source)
target_vocab.build(selected_target)

print("=" * 70)
print("VOCABULARY")
print("=" * 70)

print("Source vocabulary:", len(source_vocab))
print("Target vocabulary:", len(target_vocab))
print()


# ============================================================
# DATASET
# ============================================================

dataset = TranslationDataset(
    source_sentences=selected_source,
    target_sentences=selected_target,
    source_vocab=source_vocab,
    target_vocab=target_vocab
)

collate_fn = create_collate_fn(
    pad_token_id=source_vocab.token_to_id[
        source_vocab.pad_token
    ]
)

dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn
)

print("Batches per epoch:", len(dataloader))
print()


# ============================================================
# MODEL
# ============================================================

config = TransformerConfig(
    src_vocab_size=len(source_vocab),
    tgt_vocab_size=len(target_vocab),

    max_seq_length=MAX_SEQUENCE_LENGTH,

    d_model=128,
    num_heads=4,

    num_encoder_layers=2,
    num_decoder_layers=2,

    d_ff=512,
    dropout=0.0,

    pad_token_id=source_vocab.token_to_id[
        source_vocab.pad_token
    ],

    bos_token_id=target_vocab.token_to_id[
        target_vocab.bos_token
    ],

    eos_token_id=target_vocab.token_to_id[
        target_vocab.eos_token
    ],

    batch_size=BATCH_SIZE,

    learning_rate=LEARNING_RATE,
    weight_decay=0.0,

    device=str(device)
)

model = Transformer(config).to(device)

parameter_count = sum(
    parameter.numel()
    for parameter in model.parameters()
)

print("Model parameters:", f"{parameter_count:,}")
print()


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    betas=(0.9, 0.98),
    eps=1e-9
)


# ============================================================
# LOSS
# ============================================================

loss_fn = LabelSmoothedCrossEntropyLoss(
    vocab_size=len(target_vocab),
    padding_idx=target_vocab.token_to_id[
        target_vocab.pad_token
    ],
    label_smoothing=0.0
)


# ============================================================
# FIXED SCHEDULER
# ============================================================

class FixedScheduler:

    def __init__(self, optimizer, learning_rate):

        self.optimizer = optimizer
        self.learning_rate = learning_rate

        for group in optimizer.param_groups:
            group["lr"] = learning_rate

    def step(self):
        pass

    def get_last_lr(self):
        return [
            group["lr"]
            for group in self.optimizer.param_groups
        ]

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass


scheduler = FixedScheduler(
    optimizer,
    LEARNING_RATE
)


# ============================================================
# TRAINER
# ============================================================

trainer = Trainer(
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    loss_fn=loss_fn,
    device=device,
    grad_clip=1.0,
    gradient_accumulation_steps=1,
    use_amp=False
)


# ============================================================
# TRAIN
# ============================================================

print("=" * 70)
print("TRAINING")
print("=" * 70)

best_loss = float("inf")

for epoch in range(1, NUM_EPOCHS + 1):

    loss = trainer.train_epoch(
        dataloader
    )

    best_loss = min(
        best_loss,
        loss
    )

    if (
        epoch == 1
        or epoch % 10 == 0
    ):

        print(
            f"Epoch {epoch:03d} | "
            f"Loss: {loss:.6f}"
        )


# ============================================================
# RESULT
# ============================================================

print()
print("=" * 70)
print("FINAL RESULT")
print("=" * 70)

print(
    f"Best training loss: {best_loss:.6f}"
)

if best_loss < 1.0:

    print("RESULT: PASS")
    print("The Transformer can memorize 10 translation pairs.")

elif best_loss < 2.0:

    print("RESULT: PARTIAL")

else:

    print("RESULT: FAIL")