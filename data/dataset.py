import torch
from torch.utils.data import Dataset

from tokenizer.vocab import Vocabulary


class TranslationDataset(Dataset):
    def __init__(
        self,
        source_sentences,
        target_sentences,
        source_vocab: Vocabulary,
        target_vocab: Vocabulary
    ):
        if len(source_sentences) != len(target_sentences):
            raise ValueError(
                "Source and target datasets must have "
                "the same number of samples."
            )

        self.source_sentences = source_sentences
        self.target_sentences = target_sentences

        self.source_vocab = source_vocab
        self.target_vocab = target_vocab

    def __len__(self):
        return len(self.source_sentences)

    def __getitem__(self, index):
        source_text = self.source_sentences[index]
        target_text = self.target_sentences[index]

        source_ids = self.source_vocab.encode(source_text)
        target_ids = self.target_vocab.encode(target_text)

        return {
            "src": torch.tensor(source_ids, dtype=torch.long),
            "tgt": torch.tensor(target_ids, dtype=torch.long)
        }