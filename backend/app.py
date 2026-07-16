import pickle
import torch

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from model import Encoder, Decoder, Seq2Seq

# =====================================================
# CONFIGURATION
# =====================================================

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

# =====================================================
# LOAD VOCABULARIES
# =====================================================

with open(SOURCE_VOCAB, "rb") as f:
    source_vocab = pickle.load(f)

with open(TARGET_VOCAB, "rb") as f:
    target_vocab = pickle.load(f)

    # =====================================================
# BUILD MODEL
# =====================================================

encoder = Encoder(
    input_dim=len(source_vocab),
    embedding_dim=EMBEDDING_DIM,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS,
    cell_type=CELL_TYPE
)

decoder = Decoder(
    output_dim=len(target_vocab),
    embedding_dim=EMBEDDING_DIM,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS,
    cell_type=CELL_TYPE
)

model = Seq2Seq(
    encoder,
    decoder,
    DEVICE
).to(DEVICE)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model.eval()
# =====================================================
# FASTAPI
# =====================================================

app = FastAPI(
    title="AksharaNet-AI"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Request(BaseModel):
    text: str

    # =====================================================
# TRANSLITERATION FUNCTION
# =====================================================

def transliterate(word: str):

    source = source_vocab.encode(word)

    source = torch.tensor(
        source,
        dtype=torch.long
    ).unsqueeze(0).to(DEVICE)

    predicted_ids = model.predict(source)

    prediction = target_vocab.decode(predicted_ids)

    return prediction


# =====================================================
# ROUTES
# =====================================================

@app.get("/")
def home():

    return {
        "message": "Welcome to AksharaNet-AI API"
    }


@app.post("/predict")
def predict(request: Request):

    prediction = transliterate(request.text)

    return {
        "input": request.text,
        "prediction": prediction
    }