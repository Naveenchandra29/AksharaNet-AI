import os
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader


# CONFIGURATION

LANGUAGE = "hin"

DATASET_PATH = os.path.join(
    "..",
    "data",
    "aksharantar_sampled",
    LANGUAGE
)

TRAIN_FILE = os.path.join(DATASET_PATH, "hin_train.csv")
VALID_FILE = os.path.join(DATASET_PATH, "hin_valid.csv")
TEST_FILE = os.path.join(DATASET_PATH, "hin_test.csv")

# SPECIAL TOKENS

PAD_TOKEN = "<PAD>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"
UNK_TOKEN = "<UNK>"

SPECIAL_TOKENS = [
    PAD_TOKEN,
    SOS_TOKEN,
    EOS_TOKEN,
    UNK_TOKEN,
]


# VOCABULARY CLASS

class Vocabulary:

    def __init__(self):

        self.char2idx = {}
        self.idx2char = {}

    def build(self, words):

        characters = set()

        for word in words:

            for character in word:

                characters.add(character)

        vocabulary = SPECIAL_TOKENS + sorted(list(characters))

        self.char2idx = {
            char: index
            for index, char in enumerate(vocabulary)
        }

        self.idx2char = {
            index: char
            for char, index in self.char2idx.items()
        }

    def encode(self, word):

        encoded = [self.char2idx[SOS_TOKEN]]

        for character in word:

            encoded.append(
                self.char2idx.get(
                    character,
                    self.char2idx[UNK_TOKEN]
                )
            )

        encoded.append(
            self.char2idx[EOS_TOKEN]
        )

        return encoded

    def decode(self, indices):

        characters = []

        for index in indices:

            character = self.idx2char[index]

            if character in SPECIAL_TOKENS:
                continue

            characters.append(character)

        return "".join(characters)

    def __len__(self):

        return len(self.char2idx)