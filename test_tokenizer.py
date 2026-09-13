from tokenizer.tokenizer import BasicTokenizer
from tokenizer.vocab import Vocabulary


def main():

    sentences = [
        "I love machine learning.",
        "Machine learning is powerful.",
        "I love artificial intelligence."
    ]

    tokenizer = BasicTokenizer()

    tokenized_sentences = [
        tokenizer.tokenize(sentence)
        for sentence in sentences
    ]

    vocabulary = Vocabulary(
        min_frequency=1
    )

    vocabulary.build(sentences)

    print("===== TOKENIZER =====")

    for sentence, tokens in zip(
        sentences,
        tokenized_sentences
    ):
        print()
        print(f"Sentence: {sentence}")
        print(f"Tokens:   {tokens}")

    print()
    print("===== VOCABULARY =====")

    print(
        f"Vocabulary size: {len(vocabulary)}"
    )

    print(
        vocabulary.token_to_id
    )

    encoded = vocabulary.encode(
        "I love machine learning."
    )

    decoded = vocabulary.decode(encoded)

    print()
    print("===== ENCODE / DECODE =====")

    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")

    assert encoded[0] == vocabulary.token_to_id[
        vocabulary.bos_token
    ]

    assert encoded[-1] == vocabulary.token_to_id[
        vocabulary.eos_token
    ]

    print()
    print("Tokenizer test PASSED.")


if __name__ == "__main__":
    main()