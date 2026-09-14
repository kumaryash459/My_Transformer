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
# CONFIGURATION
# ============================================================

CHECKPOINT_PATH = "checkpoints/best.pt"

MAX_SAMPLES = 10_000

MAX_SEQUENCE_LENGTH = 64


# ============================================================
# DEVICE
# ============================================================

device = get_device()

print("=" * 60)
print("TEACHER FORCING SANITY CHECK")
print("=" * 60)

print("Device:", device)

if device.type == "cuda":
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

print()


# ============================================================
# LOAD DATA
# ============================================================

data = load_translation_data(
    validation_ratio=0.1,
    max_samples=MAX_SAMPLES,
    max_length=MAX_SEQUENCE_LENGTH
)

train_source = data["train"]["source"]
train_target = data["train"]["target"]


# ============================================================
# BUILD VOCABULARIES
# ============================================================

source_vocab = Vocabulary(
    min_frequency=1
)

target_vocab = Vocabulary(
    min_frequency=1
)

source_vocab.build(
    train_source
)

target_vocab.build(
    train_target
)


# ============================================================
# MODEL CONFIG
# ============================================================

config = TransformerConfig(

    src_vocab_size=len(source_vocab),

    tgt_vocab_size=len(target_vocab),

    max_seq_length=MAX_SEQUENCE_LENGTH,

    d_model=256,

    num_heads=8,

    num_encoder_layers=4,

    num_decoder_layers=4,

    d_ff=1024,

    dropout=0.1,

    pad_token_id=source_vocab.token_to_id[
        source_vocab.pad_token
    ],

    bos_token_id=target_vocab.token_to_id[
        target_vocab.bos_token
    ],

    eos_token_id=target_vocab.token_to_id[
        target_vocab.eos_token
    ],

    batch_size=16,

    learning_rate=1e-4,

    weight_decay=0.01,

    device=str(device)
)


# ============================================================
# CREATE MODEL
# ============================================================

model = Transformer(
    config
).to(device)


# ============================================================
# LOAD CHECKPOINT
# ============================================================

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print(
    "Checkpoint epoch:",
    checkpoint["epoch"]
)

print(
    "Validation loss:",
    checkpoint["val_loss"]
)

print()


# ============================================================
# SELECT TRAINING EXAMPLE
# ============================================================

index = 9

source_sentence = train_source[index]
target_sentence = train_target[index]

print("=" * 60)
print("TRAINING EXAMPLE")
print("=" * 60)

print()
print("English:")
print(source_sentence)

print()
print("German:")
print(target_sentence)

print()


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


# ============================================================
# PREPARE DECODER INPUT / EXPECTED OUTPUT
# ============================================================

decoder_input, expected_output = (
    prepare_decoder_inputs(tgt)
)


# ============================================================
# MASKS
# ============================================================

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


# ============================================================
# FORWARD PASS
# ============================================================

with torch.no_grad():

    output = model(
        src_tokens=src,
        tgt_tokens=decoder_input,
        src_mask=src_mask,
        tgt_mask=tgt_mask
    )

    logits = output["logits"]


# ============================================================
# PREDICT NEXT TOKENS
# ============================================================

predicted_ids = torch.argmax(
    logits,
    dim=-1
)


# ============================================================
# DISPLAY TOKEN-LEVEL PREDICTIONS
# ============================================================

print("=" * 60)
print("TOKEN-LEVEL PREDICTIONS")
print("=" * 60)

correct = 0
total = 0

for position in range(
    expected_output.size(1)
):

    actual_id = expected_output[
        0,
        position
    ].item()

    predicted_id = predicted_ids[
        0,
        position
    ].item()

    actual_token = target_vocab.id_to_token[
        actual_id
    ]

    predicted_token = target_vocab.id_to_token[
        predicted_id
    ]

    if actual_id == predicted_id:
        correct += 1

    if actual_id != target_vocab.token_to_id[
        target_vocab.pad_token
    ]:
        total += 1

    print(
        f"{position:3d} | "
        f"Actual: {actual_token:<20} | "
        f"Predicted: {predicted_token}"
    )


# ============================================================
# TOKEN ACCURACY
# ============================================================

accuracy = (
    correct / total
    if total > 0
    else 0.0
)

print()
print("=" * 60)
print("RESULT")
print("=" * 60)

print(
    f"Token Accuracy: {accuracy * 100:.2f}%"
)

print(
    f"Correct Tokens: {correct}/{total}"
)