import torch
from torch.utils.data import DataLoader

from configs.config import TransformerConfig
from model.transformer import Transformer

from training.losses import (
    LabelSmoothedCrossEntropyLoss
)

from training.optimizer import (
    create_optimizer
)

from training.scheduler import (
    NoamScheduler
)

from training.trainer import Trainer


# -----------------------------------------
# Configuration
# -----------------------------------------

config = TransformerConfig(
    src_vocab_size=50,
    tgt_vocab_size=50,
    max_seq_length=16,
    d_model=64,
    num_heads=4,
    num_encoder_layers=2,
    num_decoder_layers=2,
    d_ff=256,
    dropout=0.1
)


device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


# -----------------------------------------
# Model
# -----------------------------------------

model = Transformer(config).to(device)


# -----------------------------------------
# Fake dataset
# -----------------------------------------

dataset = []

for _ in range(20):

    src = torch.randint(
        4,
        50,
        (8,)
    )

    tgt = torch.randint(
        4,
        50,
        (8,)
    )

    src[0] = config.bos_token_id
    src[-1] = config.eos_token_id

    tgt[0] = config.bos_token_id
    tgt[-1] = config.eos_token_id

    dataset.append({
        "src": src,
        "tgt": tgt
    })


dataloader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True
)


# -----------------------------------------
# Loss
# -----------------------------------------

loss_fn = LabelSmoothedCrossEntropyLoss(
    vocab_size=config.tgt_vocab_size,
    padding_idx=config.pad_token_id,
    label_smoothing=0.1
)


# -----------------------------------------
# Optimizer
# -----------------------------------------

optimizer = create_optimizer(
    model=model,
    learning_rate=config.learning_rate,
    weight_decay=config.weight_decay
)


# -----------------------------------------
# Scheduler
# -----------------------------------------

scheduler = NoamScheduler(
    optimizer=optimizer,
    d_model=config.d_model,
    warmup_steps=100
)


# -----------------------------------------
# Trainer
# -----------------------------------------

trainer = Trainer(
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    loss_fn=loss_fn,
    device=device,
    grad_clip=1.0,
    gradient_accumulation_steps=1
)


# -----------------------------------------
# Training
# -----------------------------------------

loss = trainer.train_epoch(
    dataloader
)

print("Device:", device)
print("Training loss:", loss)
print("Current LR:", scheduler.get_last_lr()[0])

print("\nTrainer test passed.")