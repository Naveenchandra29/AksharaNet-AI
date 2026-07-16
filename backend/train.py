"""
train.py

Training pipeline for the Character-Level Seq2Seq
Transliteration Model.
"""

import os
import time

import torch
import torch.nn as nn
import torch.optim as optim

from dataset import create_dataloaders
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

BATCH_SIZE = 32

LEARNING_RATE = 0.001

EPOCHS = 20

TEACHER_FORCING = 0.5

MODEL_PATH = "saved_models/best_model.pth"
# ==========================================================
# LOAD DATA
# ==========================================================

(
    train_loader,
    valid_loader,
    test_loader,
    source_vocab,
    target_vocab,
) = create_dataloaders(
    batch_size=BATCH_SIZE
)

print("=" * 60)
print("Dataset Loaded Successfully")
print("=" * 60)

print("Training Batches :", len(train_loader))
print("Validation Batches :", len(valid_loader))
print("Test Batches :", len(test_loader))

print()

print("Source Vocabulary :", len(source_vocab))
print("Target Vocabulary :", len(target_vocab))

print()

print("Device :", DEVICE)

print("=" * 60)
# ==========================================================
# BUILD MODEL
# ==========================================================

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
# ==========================================================
# LOSS & OPTIMIZER
# ==========================================================

criterion = nn.CrossEntropyLoss(
    ignore_index=0
)

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)
# ==========================================================
# PARAMETER COUNT
# ==========================================================

total_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)

print()

print(f"Trainable Parameters : {total_parameters:,}")

print("=" * 60)