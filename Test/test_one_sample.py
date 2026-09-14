import torch

from configs.config import TransformerConfig
from data.preprocess import load_translation_data
from tokenizer.vocab import Vocabulary

from model.transformer import Transformer
from model.masks import (
    create_padding_mask,
    create_decoder_mask
)
from training.batch_utils import prepare_decoder_inputs

from device import get_device


# ============================================================
# CONFIG
# ============================================================

MAX_SEQUENCE_LENGTH = 64
NUM_EPOCHS = 300
LEARNING_RATE = 1e-3

device = get_device()

print("=" * 70)
print("ONE-SAMPLE TRANSFORMER MEMORIZATION TEST")
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
    max_samples=10,
    max_length=MAX_SEQUENCE_LENGTH
)

source_sentence = data["train"]["source"][0]
target_sentence = data["train"]["target"][0]

print("English:")
print(source_sentence)

print()
print("German:")
print(target_sentence)
print()


# ============================================================
# VOCAB
# ============================================================

source_vocab = Vocabulary(min_frequency=1)
target_vocab = Vocabulary(min_frequency=1)

source_vocab.build([source_sentence])
target_vocab.build([target_sentence])

print("Source vocab:", len(source_vocab))
print("Target vocab:", len(target_vocab))
print()


# ============================================================
# CONFIG
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

    batch_size=1,
    learning_rate=LEARNING_RATE,
    weight_decay=0.0,

    device=str(device)
)


# ============================================================
# MODEL
# ============================================================

model = Transformer(config).to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    betas=(0.9, 0.98),
    eps=1e-9
)

loss_fn = torch.nn.CrossEntropyLoss(
    ignore_index=target_vocab.token_to_id[
        target_vocab.pad_token
    ]
)


# ============================================================
# ENCODE
# ============================================================

source_ids = source_vocab.encode(
    source_sentence,
    add_bos=True,
    add_eos=True
)

target_ids = target_vocab.encode(
    target_sentence,
    add_bos=True,
    add_eos=True
)

src = torch.tensor(
    [source_ids],
    dtype=torch.long,
    device=device
)

tgt = torch.tensor(
    [target_ids],
    dtype=torch.long,
    device=device
)

decoder_input, expected_output = prepare_decoder_inputs(tgt)

src_mask = create_padding_mask(
    src,
    pad_token_id=source_vocab.token_to_id[
        source_vocab.pad_token
    ]
)

tgt_mask = create_decoder_mask(
    decoder_input,
    pad_token_id=target_vocab.token_to_id[
        target_vocab.pad_token
    ]
)


print("Source shape:", src.shape)
print("Target shape:", tgt.shape)
print("Decoder input:", decoder_input.shape)
print("Expected output:", expected_output.shape)
print()


# ============================================================
# TRAIN
# ============================================================

print("=" * 70)
print("TRAINING")
print("=" * 70)
print()

for epoch in range(1, NUM_EPOCHS + 1):

    model.train()

    optimizer.zero_grad()

    output = model(
        src_tokens=src,
        tgt_tokens=decoder_input,
        src_mask=src_mask,
        tgt_mask=tgt_mask
    )

    logits = output["logits"]

    batch_size, sequence_length, vocab_size = logits.shape

    loss = loss_fn(
        logits.reshape(
            batch_size * sequence_length,
            vocab_size
        ),
        expected_output.reshape(
            batch_size * sequence_length
        )
    )

    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=1.0
    )

    optimizer.step()

    if epoch == 1 or epoch % 10 == 0:

        with torch.no_grad():

            predictions = torch.argmax(
                logits,
                dim=-1
            )

            correct = (
                predictions == expected_output
            ).sum().item()

            total = expected_output.numel()

            accuracy = correct / total * 100

        print(
            f"Epoch {epoch:03d} | "
            f"Loss: {loss.item():.6f} | "
            f"Accuracy: {accuracy:.2f}%"
        )


# ============================================================
# FINAL TOKEN PREDICTIONS
# ============================================================

model.eval()

with torch.no_grad():

    output = model(
        src_tokens=src,
        tgt_tokens=decoder_input,
        src_mask=src_mask,
        tgt_mask=tgt_mask
    )

    logits = output["logits"]

    predictions = torch.argmax(
        logits,
        dim=-1
    )


print()
print("=" * 70)
print("FINAL TOKEN PREDICTIONS")
print("=" * 70)

correct = 0
total = 0

for position in range(expected_output.size(1)):

    actual_id = expected_output[0, position].item()
    predicted_id = predictions[0, position].item()

    actual_token = target_vocab.id_to_token[actual_id]
    predicted_token = target_vocab.id_to_token[predicted_id]

    if actual_id != target_vocab.token_to_id[
        target_vocab.pad_token
    ]:

        total += 1

        if actual_id == predicted_id:
            correct += 1

    print(
        f"{position:3d} | "
        f"Actual: {actual_token:<20} | "
        f"Predicted: {predicted_token}"
    )


accuracy = correct / total * 100

print()
print("=" * 70)
print("FINAL RESULT")
print("=" * 70)

print(f"Final loss: {loss.item():.6f}")
print(f"Token accuracy: {accuracy:.2f}%")