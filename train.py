
import torch
from torch.utils.data import DataLoader

from configs.config import TransformerConfig
from device import get_device

from data.preprocess import load_translation_data
from data.dataset import TranslationDataset
from data.collate import create_collate_fn

from tokenizer.vocab import Vocabulary

from model.transformer import Transformer

from training.losses import LabelSmoothedCrossEntropyLoss
from training.optimizer import create_optimizer
from training.scheduler import NoamScheduler
from training.trainer import Trainer
from training.checkpoint import save_checkpoint


# ============================================================
# 1. CONFIGURATION
# ============================================================

MAX_SAMPLES = 10_000

MAX_LENGTH = 64

BATCH_SIZE = 16

NUM_EPOCHS = 15

# Process the complete dataset.
MAX_TRAIN_BATCHES = None
MAX_VAL_BATCHES = None

# Noam warmup.
WARMUP_STEPS = 2000

GRADIENT_ACCUMULATION_STEPS = 1

GRAD_CLIP = 1.0

# AMP disabled because it previously produced NaNs.
USE_AMP = False

# Word-level vocabulary.
# Use every token appearing in the training data.
MIN_FREQUENCY = 1

CHECKPOINT_DIR = "checkpoints"

LATEST_CHECKPOINT = f"{CHECKPOINT_DIR}/latest.pt"
BEST_CHECKPOINT = f"{CHECKPOINT_DIR}/best.pt"


# ============================================================
# 2. DEVICE
# ============================================================

device = get_device()

print("=" * 60)
print("DEVICE INFORMATION")
print("=" * 60)

print("Device:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA:", torch.version.cuda)

print()


# ============================================================
# 3. LOAD REAL DATASET
# ============================================================

print("=" * 60)
print("LOADING DATASET")
print("=" * 60)

data = load_translation_data(
    validation_ratio=0.1,
    max_samples=MAX_SAMPLES,
    max_length=MAX_LENGTH
)

train_source = data["train"]["source"]
train_target = data["train"]["target"]

val_source = data["validation"]["source"]
val_target = data["validation"]["target"]

print("Training samples:", len(train_source))
print("Validation samples:", len(val_source))

print()


# ============================================================
# 4. BUILD VOCABULARIES
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

# Vocabulary is built ONLY from training data.
# This prevents validation-data leakage.

source_vocab.build(train_source)
target_vocab.build(train_target)

print("Source vocabulary size:", len(source_vocab))
print("Target vocabulary size:", len(target_vocab))

print()

print("Source special tokens:")
print({
    "PAD": source_vocab.token_to_id[
        source_vocab.pad_token
    ],
    "BOS": source_vocab.token_to_id[
        source_vocab.bos_token
    ],
    "EOS": source_vocab.token_to_id[
        source_vocab.eos_token
    ],
    "UNK": source_vocab.token_to_id[
        source_vocab.unk_token
    ]
})

print()

print("Target special tokens:")
print({
    "PAD": target_vocab.token_to_id[
        target_vocab.pad_token
    ],
    "BOS": target_vocab.token_to_id[
        target_vocab.bos_token
    ],
    "EOS": target_vocab.token_to_id[
        target_vocab.eos_token
    ],
    "UNK": target_vocab.token_to_id[
        target_vocab.unk_token
    ]
})

print()


# ============================================================
# 5. CREATE DATASETS
# ============================================================

train_dataset = TranslationDataset(
    source_sentences=train_source,
    target_sentences=train_target,
    source_vocab=source_vocab,
    target_vocab=target_vocab
)

val_dataset = TranslationDataset(
    source_sentences=val_source,
    target_sentences=val_target,
    source_vocab=source_vocab,
    target_vocab=target_vocab
)


# ============================================================
# 6. CREATE DATALOADERS
# ============================================================

collate_fn = create_collate_fn(
    pad_token_id=source_vocab.token_to_id[
        source_vocab.pad_token
    ]
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn,
    pin_memory=(device.type == "cuda")
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn,
    pin_memory=(device.type == "cuda")
)

print("=" * 60)
print("DATALOADER")
print("=" * 60)

print("Train batches:", len(train_loader))
print("Validation batches:", len(val_loader))
print("Batch size:", BATCH_SIZE)

print()


# ============================================================
# 7. TRANSFORMER CONFIGURATION
# ============================================================

config = TransformerConfig(

    # Actual vocabulary sizes
    src_vocab_size=len(source_vocab),
    tgt_vocab_size=len(target_vocab),

    # Sequence configuration
    max_seq_length=MAX_LENGTH,

    # Transformer architecture
    d_model=256,
    num_heads=8,

    num_encoder_layers=4,
    num_decoder_layers=4,

    d_ff=1024,

    dropout=0.1,

    # SOURCE PAD token
    pad_token_id=source_vocab.token_to_id[
        source_vocab.pad_token
    ],

    # TARGET BOS token
    bos_token_id=target_vocab.token_to_id[
        target_vocab.bos_token
    ],

    # TARGET EOS token
    eos_token_id=target_vocab.token_to_id[
        target_vocab.eos_token
    ],

    batch_size=BATCH_SIZE,

    learning_rate=2e-4,

    weight_decay=0.01,

    device=str(device)
)

print("=" * 60)
print("TRANSFORMER CONFIGURATION")
print("=" * 60)

print(config)

print()


# ============================================================
# 8. CREATE MODEL
# ============================================================

print("=" * 60)
print("CREATING MODEL")
print("=" * 60)

model = Transformer(
    config
).to(device)

print(
    "Model parameters:",
    f"{sum(p.numel() for p in model.parameters()):,}"
)

print()


# ============================================================
# 9. LOSS FUNCTION
# ============================================================

loss_fn = LabelSmoothedCrossEntropyLoss(
    vocab_size=len(target_vocab),

    padding_idx=target_vocab.token_to_id[
        target_vocab.pad_token
    ],

    label_smoothing=0.0
)


# ============================================================
# 10. OPTIMIZER
# ============================================================

optimizer = create_optimizer(
    model=model,

    learning_rate=config.learning_rate,

    weight_decay=config.weight_decay
)


# ============================================================
# 11. NOAM LEARNING-RATE SCHEDULER
# ============================================================

scheduler = NoamScheduler(
    optimizer=optimizer,

    d_model=config.d_model,

    warmup_steps=WARMUP_STEPS
)


# ============================================================
# 12. TRAINER
# ============================================================

trainer = Trainer(

    model=model,

    optimizer=optimizer,

    scheduler=scheduler,

    loss_fn=loss_fn,

    device=device,

    grad_clip=GRAD_CLIP,

    gradient_accumulation_steps=(
        GRADIENT_ACCUMULATION_STEPS
    ),

    use_amp=USE_AMP
)


# ============================================================
# 13. TRAINING LOOP
# ============================================================

print("=" * 60)
print("STARTING TRAINING")
print("=" * 60)

print()

best_val_loss = float("inf")


for epoch in range(1, NUM_EPOCHS + 1):

    print("-" * 60)
    print(
        f"Epoch {epoch}/{NUM_EPOCHS}"
    )
    print("-" * 60)

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    train_loss = trainer.train_epoch(
        train_loader,
        max_batches=MAX_TRAIN_BATCHES
    )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    val_loss = trainer.validate(
        val_loader,
        max_batches=MAX_VAL_BATCHES
    )

    # --------------------------------------------------------
    # CURRENT LEARNING RATE
    # --------------------------------------------------------

    current_lr = scheduler.get_last_lr()[0]

    print()
    print(f"Train Loss   : {train_loss:.4f}")
    print(f"Val Loss     : {val_loss:.4f}")
    print(f"Learning Rate: {current_lr:.8f}")

    # --------------------------------------------------------
    # GPU MEMORY
    # --------------------------------------------------------

    if device.type == "cuda":

        allocated = (
            torch.cuda.memory_allocated(device)
            / 1024**3
        )

        reserved = (
            torch.cuda.memory_reserved(device)
            / 1024**3
        )

        peak = (
            torch.cuda.max_memory_allocated(device)
            / 1024**3
        )

        print(
            f"GPU Memory Allocated: {allocated:.2f} GB"
        )

        print(
            f"GPU Memory Reserved : {reserved:.2f} GB"
        )

        print(
            f"GPU Peak Memory     : {peak:.2f} GB"
        )

        torch.cuda.reset_peak_memory_stats(device)

    print()

    # --------------------------------------------------------
    # CHECKPOINT: LATEST
    # --------------------------------------------------------

    save_checkpoint(
        path=LATEST_CHECKPOINT,

        model=model,

        optimizer=optimizer,

        scheduler=scheduler,

        epoch=epoch,

        train_loss=train_loss,

        val_loss=val_loss,

        best_val_loss=best_val_loss
    )

    # --------------------------------------------------------
    # CHECKPOINT: BEST MODEL
    # --------------------------------------------------------

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        save_checkpoint(
            path=BEST_CHECKPOINT,

            model=model,

            optimizer=optimizer,

            scheduler=scheduler,

            epoch=epoch,

            train_loss=train_loss,

            val_loss=val_loss,

            best_val_loss=best_val_loss
        )

        print(
            f"New best validation loss: "
            f"{best_val_loss:.4f}"
        )

    else:

        print(
            f"Best validation loss remains: "
            f"{best_val_loss:.4f}"
        )

    print()


# ============================================================
# 14. TRAINING COMPLETE
# ============================================================

print("=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

print(
    f"Best validation loss: {best_val_loss:.4f}"
)

print(
    f"Latest checkpoint: {LATEST_CHECKPOINT}"
)

print(
    f"Best checkpoint: {BEST_CHECKPOINT}"
)

