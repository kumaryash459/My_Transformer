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

from device import get_device
from training.losses import LabelSmoothedCrossEntropyLoss

# ============================================================
# CONFIGURATION
# ============================================================

NUM_SAMPLES = 100
NUM_EPOCHS = 50
BATCH_SIZE = 8
MAX_SEQUENCE_LENGTH = 64

D_MODEL = 128
NUM_HEADS = 4
NUM_ENCODER_LAYERS = 2
NUM_DECODER_LAYERS = 2
D_FF = 512
DROPOUT = 0.0

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.0

CHECKPOINT_PATH = "checkpoints/overfit.pt"


# ============================================================
# DEVICE
# ============================================================

device = get_device()

print("=" * 70)
print("TRANSFORMER OVERFITTING TEST - FIXED LEARNING RATE")
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
    max_samples=NUM_SAMPLES,
    max_length=MAX_SEQUENCE_LENGTH
)

train_source = data["train"]["source"]
train_target = data["train"]["target"]

print("Training samples:", len(train_source))
print()


# ============================================================
# VOCABULARIES
# ============================================================

source_vocab = Vocabulary(min_frequency=1)
target_vocab = Vocabulary(min_frequency=1)

source_vocab.build(train_source)
target_vocab.build(train_target)

print("Source vocabulary size:", len(source_vocab))
print("Target vocabulary size:", len(target_vocab))
print()


# ============================================================
# DATASET
# ============================================================

dataset = TranslationDataset(
    source_sentences=train_source,
    target_sentences=train_target,
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
# MODEL CONFIG
# ============================================================

config = TransformerConfig(
    src_vocab_size=len(source_vocab),
    tgt_vocab_size=len(target_vocab),

    max_seq_length=MAX_SEQUENCE_LENGTH,

    d_model=D_MODEL,
    num_heads=NUM_HEADS,

    num_encoder_layers=NUM_ENCODER_LAYERS,
    num_decoder_layers=NUM_DECODER_LAYERS,

    d_ff=D_FF,
    dropout=DROPOUT,

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
    weight_decay=WEIGHT_DECAY,

    device=str(device)
)


# ============================================================
# MODEL
# ============================================================

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

optimizer = create_optimizer(
    model=model,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# ============================================================
# NO SCHEDULER
# ============================================================

class FixedScheduler:

    def __init__(self, optimizer, learning_rate):
        self.optimizer = optimizer
        self.learning_rate = learning_rate

        for parameter_group in self.optimizer.param_groups:
            parameter_group["lr"] = learning_rate

    def step(self):
        pass

    def get_last_lr(self):
        return [
            parameter_group["lr"]
            for parameter_group in self.optimizer.param_groups
        ]

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass


scheduler = FixedScheduler(
    optimizer=optimizer,
    learning_rate=LEARNING_RATE
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
print("STARTING OVERFIT TEST")
print("=" * 70)
print()

best_loss = float("inf")

for epoch in range(1, NUM_EPOCHS + 1):

    train_loss = trainer.train_epoch(
        dataloader=dataloader
    )

    current_lr = scheduler.get_last_lr()[0]

    print(
        f"Epoch {epoch:02d} | "
        f"Loss: {train_loss:.4f} | "
        f"LR: {current_lr:.7f}"
    )

    if train_loss < best_loss:

        best_loss = train_loss

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "source_vocab": source_vocab.token_to_id,
                "target_vocab": target_vocab.token_to_id,
                "loss": train_loss
            },
            CHECKPOINT_PATH
        )


# ============================================================
# RESULT
# ============================================================

print()
print("=" * 70)
print("OVERFIT TEST RESULT")
print("=" * 70)

print(f"Best training loss: {best_loss:.4f}")
print(f"Checkpoint saved: {CHECKPOINT_PATH}")

if best_loss < 1.0:

    print()
    print("RESULT: PASS")
    print("The Transformer can memorize the tiny dataset.")

elif best_loss < 2.0:

    print()
    print("RESULT: PARTIAL")
    print(
        "The Transformer is learning strongly, "
        "but did not completely memorize the dataset."
    )

else:

    print()
    print("RESULT: FAIL")
    print(
        "The Transformer is still unable to "
        "overfit the tiny dataset."
    )