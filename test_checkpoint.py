import torch
import torch.nn as nn

from training.optimizer import create_optimizer
from training.scheduler import NoamScheduler

from training.checkpoint import (
    save_checkpoint,
    load_checkpoint
)


device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


# -----------------------------------------
# Create model
# -----------------------------------------

model = nn.Linear(10, 5).to(device)

optimizer = create_optimizer(
    model=model,
    learning_rate=3e-4,
    weight_decay=0.01
)

scheduler = NoamScheduler(
    optimizer=optimizer,
    d_model=64,
    warmup_steps=100
)


# -----------------------------------------
# Simulate training
# -----------------------------------------

for _ in range(5):

    x = torch.randn(
        4,
        10,
        device=device
    )

    output = model(x)

    loss = output.mean()

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    scheduler.step()


# -----------------------------------------
# Save
# -----------------------------------------

save_checkpoint(
    path="checkpoints/test.pt",
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    epoch=4,
    train_loss=2.5,
    val_loss=2.7,
    best_val_loss=2.7
)


# -----------------------------------------
# Create fresh model
# -----------------------------------------

new_model = nn.Linear(
    10,
    5
).to(device)

new_optimizer = create_optimizer(
    model=new_model,
    learning_rate=3e-4,
    weight_decay=0.01
)

new_scheduler = NoamScheduler(
    optimizer=new_optimizer,
    d_model=64,
    warmup_steps=100
)


# -----------------------------------------
# Load
# -----------------------------------------

state = load_checkpoint(
    path="checkpoints/test.pt",
    model=new_model,
    optimizer=new_optimizer,
    scheduler=new_scheduler,
    device=device
)


print("\nLoaded state:")
print(state)

assert state["start_epoch"] == 5

print("\nCheckpoint test passed.")