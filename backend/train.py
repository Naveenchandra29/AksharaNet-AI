import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import pickle
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
# ==========================================================
# TRAIN FUNCTION
# ==========================================================

def train_one_epoch(model, loader, optimizer, criterion):

    model.train()

    epoch_loss = 0

    for source, decoder_input, decoder_target in loader:

        source = source.to(DEVICE)
        decoder_input = decoder_input.to(DEVICE)
        decoder_target = decoder_target.to(DEVICE)

        optimizer.zero_grad()

        output = model(
            source,
            decoder_input,
            teacher_forcing_ratio=TEACHER_FORCING
        )

        output = output.reshape(
            -1,
            output.shape[-1]
        )

        decoder_target = decoder_target.reshape(-1)

        loss = criterion(
            output,
            decoder_target
        )

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(loader)
# ==========================================================
# VALIDATION FUNCTION
# ==========================================================

def validate(model, loader, criterion):

    model.eval()

    epoch_loss = 0

    with torch.no_grad():

        for source, decoder_input, decoder_target in loader:

            source = source.to(DEVICE)
            decoder_input = decoder_input.to(DEVICE)
            decoder_target = decoder_target.to(DEVICE)

            output = model(
                source,
                decoder_input,
                teacher_forcing_ratio=0
            )

            output = output.reshape(
                -1,
                output.shape[-1]
            )

            decoder_target = decoder_target.reshape(-1)

            loss = criterion(
                output,
                decoder_target
            )

            epoch_loss += loss.item()

    return epoch_loss / len(loader)
# ==========================================================
# TRAINING LOOP
# ==========================================================

best_valid_loss = float("inf")

print("\nStarting Training...\n")

start_time = time.time()

for epoch in range(EPOCHS):

    train_loss = train_one_epoch(
        model,
        train_loader,
        optimizer,
        criterion
    )

    valid_loss = validate(
        model,
        valid_loader,
        criterion
    )

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] | "
        f"Train Loss: {train_loss:.4f} | "
        f"Valid Loss: {valid_loss:.4f}"
    )

    if valid_loss < best_valid_loss:

        best_valid_loss = valid_loss

        os.makedirs("saved_models", exist_ok=True)

        torch.save(
            model.state_dict(),
            MODEL_PATH 
        )

        with open(
            "saved_models/source_vocab.pkl",
            "wb"
        ) as file:

            pickle.dump(source_vocab, file)

        with open(
            "saved_models/target_vocab.pkl",
            "wb"
        ) as file:

            pickle.dump(target_vocab, file)

        print("✔ Best model and vocabularies saved")

end_time = time.time()

print("\n" + "=" * 60)
print("Training Completed")
print("=" * 60)

print(f"Best Validation Loss : {best_valid_loss:.4f}")
print(f"Training Time : {(end_time-start_time):.2f} seconds")
print(f"Model Saved : {MODEL_PATH}")