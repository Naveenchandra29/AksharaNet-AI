import pickle

import torch

from dataset import Vocabulary
from model import Encoder
from model import Decoder
from model import Seq2Seq


# ==========================================================
# CONFIGURATION
# ==========================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

EMBEDDING_DIM = 128
HIDDEN_DIM = 256
NUM_LAYERS = 1

CELL_TYPE = "LSTM"

MODEL_PATH = "saved_models/best_model.pth"

SOURCE_VOCAB = "saved_models/source_vocab.pkl"
TARGET_VOCAB = "saved_models/target_vocab.pkl"

with open(SOURCE_VOCAB, "rb") as file:
    source_vocab = pickle.load(file)

with open(TARGET_VOCAB, "rb") as file:
    target_vocab = pickle.load(file)

encoder = Encoder(
    input_dim=len(source_vocab),
    embedding_dim=EMBEDDING_DIM,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS,
    cell_type=CELL_TYPE,
)

decoder = Decoder(
    output_dim=len(target_vocab),
    embedding_dim=EMBEDDING_DIM,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS,
    cell_type=CELL_TYPE,
)

model = Seq2Seq(
    encoder,
    decoder,
    DEVICE,
).to(DEVICE)


model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model.eval()

def transliterate(word):

    # Encode input word
    source = source_vocab.encode(word)

    source = torch.tensor(
        source,
        dtype=torch.long
    ).unsqueeze(0).to(DEVICE)

    # Predict token ids
    predicted_ids = model.predict(source)

    # Decode prediction
    prediction = target_vocab.decode(predicted_ids)

    return prediction
print("Model Loaded Successfully")

if __name__ == "__main__":

    print("=" * 50)
    print("AksharaNet-AI Transliteration")
    print("=" * 50)

    while True:

        word = input("\nEnter English word (or 'exit'): ")

        if word.lower() == "exit":
            break

        prediction = transliterate(word)

        print(f"\nPrediction : {prediction}")