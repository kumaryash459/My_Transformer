import torch
import torch.nn as nn

from training.optimizer import create_optimizer
from training.scheduler import NoamScheduler


model = nn.Linear(10, 10)

optimizer = create_optimizer(
    model,
    learning_rate=3e-4,
    weight_decay=0.01
)

scheduler = NoamScheduler(
    optimizer=optimizer,
    d_model=256,
    warmup_steps=400
)


print("Learning rates:")

for _ in range(10):
    scheduler.step()

    lr = scheduler.get_last_lr()[0]

    print(
        f"Step {scheduler.step_num:02d} | "
        f"LR = {lr:.8f}"
    )

assert scheduler.step_num == 10

assert scheduler.get_last_lr()[0] > 0

print("\nScheduler test passed.")