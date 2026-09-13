import torch
from torch.utils.data import DataLoader

from configs.config import TransformerConfig
from model.transformer import Transformer

from data.preprocess import load_translation_data
from data.dataset import TranslationDataset
from data.collate import create_collate_fn

from tokenizer.vocab import Vocabulary

from model.masks import (
    create_padding_mask,
    create_decoder_mask
)

from training.batch_utils import (
    prepare_decoder_inputs
)

from training.losses import (
    LabelSmoothedCrossEntropyLoss
)


# =========================================================
# DEVICE
# =========================================================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)

if device.type == "cuda":
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# =========================================================
# LOAD DATA
# =========================================================

data = load_translation_data(
    validation_ratio=0.1,
    max_samples=10_000
)

train_source = data["train"]["source"]
train_target = data["train"]["target"]

print(
    "Training samples:",
    len(train_source)
)


# =========================================================
# BUILD VOCABULARIES
# =========================================================

source_vocab = Vocabulary(
    min_frequency=2
)

target_vocab = Vocabulary(
    min_frequency=2
)

source_vocab.build(
    train_source
)

target_vocab.build(
    train_target
)

print(
    "Source vocabulary size:",
    len(source_vocab)
)

print(
    "Target vocabulary size:",
    len(target_vocab)
)


# =========================================================
# DATASET
# =========================================================

train_dataset = TranslationDataset(
    source_sentences=train_source,
    target_sentences=train_target,
    source_vocab=source_vocab,
    target_vocab=target_vocab
)


# =========================================================
# DATALOADER
# =========================================================

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


# =========================================================
# CONFIGURATION
# =========================================================

config = TransformerConfig(
    src_vocab_size=len(source_vocab),
    tgt_vocab_size=len(target_vocab),

    max_seq_length=128,

    d_model=256,
    num_heads=8,

    num_encoder_layers=4,
    num_decoder_layers=4,

    d_ff=1024,

    dropout=0.1,

    pad_token_id=source_vocab.token_to_id[
        source_vocab.pad_token
    ],

    bos_token_id=source_vocab.token_to_id[
        source_vocab.bos_token
    ],

    eos_token_id=source_vocab.token_to_id[
        source_vocab.eos_token
    ],

    batch_size=16,

    learning_rate=3e-4,
    weight_decay=0.01,

    device=str(device)
)


# =========================================================
# MODEL
# =========================================================

model = Transformer(
    config
).to(device)

print(
    "\nModel created successfully."
)

print(
    "Total parameters:",
    sum(
        parameter.numel()
        for parameter in model.parameters()
    )
)


# =========================================================
# GET ONE REAL BATCH
# =========================================================

batch = next(
    iter(train_loader)
)

src = batch["src"].to(device)
tgt = batch["tgt"].to(device)

print("\nBatch:")
print("SRC:", src.shape)
print("TGT:", tgt.shape)


# =========================================================
# TARGET SHIFTING
# =========================================================

decoder_input, expected_output = (
    prepare_decoder_inputs(tgt)
)

print(
    "\nDecoder input:",
    decoder_input.shape
)

print(
    "Expected output:",
    expected_output.shape
)


# =========================================================
# MASKS
# =========================================================

src_mask = create_padding_mask(
    src,
    pad_token_id=config.pad_token_id
)

tgt_mask = create_decoder_mask(
    decoder_input,
    pad_token_id=config.pad_token_id
)

print(
    "\nSource mask:",
    src_mask.shape
)

print(
    "Target mask:",
    tgt_mask.shape
)


# =========================================================
# FORWARD PASS
# =========================================================

model.train()

output = model(
    src_tokens=src,
    tgt_tokens=decoder_input,
    src_mask=src_mask,
    tgt_mask=tgt_mask
)

logits = output["logits"]

print(
    "\nLogits:",
    logits.shape
)


# =========================================================
# LOSS
# =========================================================

loss_fn = LabelSmoothedCrossEntropyLoss(
    vocab_size=config.tgt_vocab_size,
    padding_idx=config.pad_token_id,
    label_smoothing=0.1
)

loss = loss_fn(
    logits,
    expected_output
)

print(
    "Loss:",
    loss.item()
)


# =========================================================
# VALIDATION
# =========================================================

assert logits.shape[0] == src.shape[0]

assert logits.shape[1] == expected_output.shape[1]

assert logits.shape[2] == len(target_vocab)

assert torch.isfinite(loss)

print(
    "\nReal dataset + Transformer "
    "forward-pass test passed."
)