from collections import Counter
from tokenizer.tokenizer import BasicTokenizer


class Vocabulary:
    def __init__(self, min_frequency: int = 2):
        self.min_frequency = min_frequency

        self.token_to_id = {}
        self.id_to_token = {}

        self.pad_token = "<PAD>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
        self.unk_token = "<UNK>"

        self.tokenizer = BasicTokenizer()

        self._initialize_special_tokens()

    def _initialize_special_tokens(self):
        special_tokens = [
            self.pad_token,
            self.bos_token,
            self.eos_token,
            self.unk_token
        ]

        for token in special_tokens:
            self._add_token(token)

    def _add_token(self, token: str):
        token_id = len(self.token_to_id)

        self.token_to_id[token] = token_id
        self.id_to_token[token_id] = token

    def build(self, sentences: list[str]):
        counter = Counter()

        for sentence in sentences:
            tokens = self.tokenizer.tokenize(sentence)
            counter.update(tokens)

        for token, frequency in counter.items():
            if (
                frequency >= self.min_frequency
                and token not in self.token_to_id
            ):
                self._add_token(token)

    def encode(
        self,
        sentence: str,
        add_bos: bool = True,
        add_eos: bool = True
    ) -> list[int]:

        tokens = self.tokenizer.tokenize(sentence)

        ids = []

        if add_bos:
            ids.append(self.token_to_id[self.bos_token])

        for token in tokens:
            token_id = self.token_to_id.get(
                token,
                self.token_to_id[self.unk_token]
            )

            ids.append(token_id)

        if add_eos:
            ids.append(self.token_to_id[self.eos_token])

        return ids

    def decode(self, ids: list[int]) -> str:
        tokens = []

        for token_id in ids:
            token = self.id_to_token[token_id]

            if token in {
                self.pad_token,
                self.bos_token
            }:
                continue

            if token == self.eos_token:
                break

            tokens.append(token)

        return " ".join(tokens)

    def __len__(self):
        return len(self.token_to_id)