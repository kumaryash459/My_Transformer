import torch

from training.losses import LabelSmoothedCrossEntropyLoss


batch_size = 4
sequence_length = 10
vocab_size = 100

logits = torch.randn(
    batch_size,
    sequence_length,
    vocab_size
)

targets = torch.randint(
    0,
    vocab_size,
    (batch_size, sequence_length)
)

# Assume token 0 is <PAD>
targets[0, -2:] = 0

loss_fn = LabelSmoothedCrossEntropyLoss(
    vocab_size=vocab_size,
    padding_idx=0,
    label_smoothing=0.1
)

loss = loss_fn(logits, targets)

print("Logits shape:", logits.shape)
print("Targets shape:", targets.shape)
print("Loss:", loss.item())

assert loss.ndim == 0
assert torch.isfinite(loss)

print("Loss test passed.")