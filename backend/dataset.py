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
# PYTORCH DATASET
# ==========================================================

class TransliterationDataset(Dataset):

    def __init__(
        self,
        source_words,
        target_words,
        source_vocab,
        target_vocab
    ):

        self.source_words = source_words
        self.target_words = target_words

        self.source_vocab = source_vocab
        self.target_vocab = target_vocab

    def __len__(self):

        return len(self.source_words)

    def __getitem__(self, index):

        source = self.source_vocab.encode(
            self.source_words[index]
        )

        target = self.target_vocab.encode(
            self.target_words[index]
        )

        return (
            torch.tensor(source),
            torch.tensor(target)
        )
    # ==========================================================
# COLLATE FUNCTION
# ==========================================================

def collate_batch(batch):

    source_batch = []
    target_batch = []

    for source, target in batch:

        source_batch.append(source)
        target_batch.append(target)

    source_batch = torch.nn.utils.rnn.pad_sequence(
        source_batch,
        batch_first=True,
        padding_value=0
    )

    target_batch = torch.nn.utils.rnn.pad_sequence(
        target_batch,
        batch_first=True,
        padding_value=0
    )

    return source_batch, target_batch
# ==========================================================
# DATALOADER
# ==========================================================


# ==========================================================
# CREATE TRAIN / VALID / TEST DATALOADERS
# ==========================================================

def create_dataloaders(batch_size=32):

    # ---------- Training ----------

    train_source, train_target = load_data(TRAIN_FILE)

    source_vocab = Vocabulary()
    target_vocab = Vocabulary()

    source_vocab.build(train_source)
    target_vocab.build(train_target)

    train_dataset = TransliterationDataset(
        train_source,
        train_target,
        source_vocab,
        target_vocab
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_batch
    )

    # ---------- Validation ----------

    valid_source, valid_target = load_data(VALID_FILE)

    valid_dataset = TransliterationDataset(
        valid_source,
        valid_target,
        source_vocab,
        target_vocab
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_batch
    )

    # ---------- Test ----------

    test_source, test_target = load_data(TEST_FILE)

    test_dataset = TransliterationDataset(
        test_source,
        test_target,
        source_vocab,
        target_vocab
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_batch
    )

    return (
        train_loader,
        valid_loader,
        test_loader,
        source_vocab,
        target_vocab
    )
# ==========================================================
# TEST
# ==========================================================

if __name__ == "__main__":

    (
        train_loader,
        valid_loader,
        test_loader,
        source_vocab,
        target_vocab
    ) = create_dataloaders()

    print("=" * 50)
    print("Dataset Ready")
    print("=" * 50)

    print("Train Batches :", len(train_loader))
    print("Valid Batches :", len(valid_loader))
    print("Test Batches  :", len(test_loader))

    print()

    source, target = next(iter(train_loader))

    print("Source Shape :", source.shape)
    print("Target Shape :", target.shape)

    print()

    print(source[0])

    print(target[0])