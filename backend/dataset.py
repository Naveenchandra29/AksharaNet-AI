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
    # ==========================================================
# DATASET LOADER
# ==========================================================

def load_data(file_path):
    """
    Load transliteration pairs from a CSV file.

    Returns:
        source_words : List[str]
        target_words : List[str]
    """

    dataframe = pd.read_csv(
        file_path,
        header=None,
        names=["source", "target"]
    )

    source_words = dataframe["source"].astype(str).tolist()
    target_words = dataframe["target"].astype(str).tolist()

    return source_words, target_words


def build_vocabularies():

    train_source, train_target = load_data(TRAIN_FILE)

    source_vocab = Vocabulary()
    target_vocab = Vocabulary()

    source_vocab.build(train_source)
    target_vocab.build(train_target)

    return (
        source_vocab,
        target_vocab,
        train_source,
        train_target,
    )
# ==========================================================
# TEST
# ==========================================================

if __name__ == "__main__":

    source_vocab, target_vocab, source, target = build_vocabularies()

    print("=" * 50)
    print("Dataset Loaded Successfully")
    print("=" * 50)

    print(f"Training Samples : {len(source)}")
    print(f"Source Vocabulary : {len(source_vocab)}")
    print(f"Target Vocabulary : {len(target_vocab)}")

    print()

    print("Example")

    print(source[0])

    print(target[0])

    print(source_vocab.encode(source[0]))

    print(target_vocab.encode(target[0]))