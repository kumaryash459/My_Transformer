from collections import Counter

from data.preprocess import load_translation_data
from tokenizer.tokenizer import BasicTokenizer
from tokenizer.vocab import Vocabulary


MAX_SAMPLES = 10_000
MAX_LENGTH = 64

tokenizer = BasicTokenizer()

data = load_translation_data(
    validation_ratio=0.1,
    max_samples=MAX_SAMPLES,
    max_length=MAX_LENGTH,
)

train_source = data["train"]["source"]
train_target = data["train"]["target"]

source_vocab = Vocabulary(min_frequency=1)
target_vocab = Vocabulary(min_frequency=1)

source_vocab.build(train_source)
target_vocab.build(train_target)


def analyze_unknown_tokens(
    sentences,
    vocab,
    tokenizer,
):
    total_tokens = 0
    unknown_tokens = 0

    unknown_counter = Counter()

    for sentence in sentences:

        tokens = tokenizer.tokenize(sentence)

        for token in tokens:

            total_tokens += 1

            if token not in vocab.token_to_id:

                unknown_tokens += 1
                unknown_counter[token] += 1

    unknown_percentage = (
        100.0 * unknown_tokens / total_tokens
        if total_tokens > 0
        else 0.0
    )

    return (
        total_tokens,
        unknown_tokens,
        unknown_percentage,
        unknown_counter,
    )


(
    source_total,
    source_unknown,
    source_unknown_percentage,
    source_unknown_counter,
) = analyze_unknown_tokens(
    train_source,
    source_vocab,
    tokenizer,
)

(
    target_total,
    target_unknown,
    target_unknown_percentage,
    target_unknown_counter,
) = analyze_unknown_tokens(
    train_target,
    target_vocab,
    tokenizer,
)


print()
print("=" * 70)
print("VOCABULARY ANALYSIS")
print("=" * 70)

print()
print("Training samples:")
print("Source:", len(train_source))
print("Target:", len(train_target))

print()
print("Vocabulary:")
print("Source vocabulary:", len(source_vocab))
print("Target vocabulary:", len(target_vocab))

print()
print("Source tokens:")
print("Total:", source_total)
print("UNK:", source_unknown)
print(
    f"UNK percentage: {source_unknown_percentage:.2f}%"
)

print()
print("Target tokens:")
print("Total:", target_total)
print("UNK:", target_unknown)
print(
    f"UNK percentage: {target_unknown_percentage:.2f}%"
)

print()
print("=" * 70)
print("MOST COMMON UNKNOWN SOURCE TOKENS")
print("=" * 70)

for token, count in source_unknown_counter.most_common(20):
    print(f"{token!r}: {count}")

print()
print("=" * 70)
print("MOST COMMON UNKNOWN TARGET TOKENS")
print("=" * 70)

for token, count in target_unknown_counter.most_common(20):
    print(f"{token!r}: {count}")