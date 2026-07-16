"""
model.py

Character-level Seq2Seq Model
Supports:
- RNN
- GRU
- LSTM
"""

import random
import torch
import torch.nn as nn


# ==========================================================
# ENCODER
# ==========================================================

class Encoder(nn.Module):

    def __init__(
        self,
        input_dim,
        embedding_dim,
        hidden_dim,
        num_layers=1,
        cell_type="LSTM",
        dropout=0.2
    ):

        super().__init__()

        self.cell_type = cell_type.upper()

        self.embedding = nn.Embedding(
            input_dim,
            embedding_dim
        )

        if self.cell_type == "RNN":

            self.rnn = nn.RNN(
                embedding_dim,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0
            )

        elif self.cell_type == "GRU":

            self.rnn = nn.GRU(
                embedding_dim,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0
            )

        elif self.cell_type == "LSTM":

            self.rnn = nn.LSTM(
                embedding_dim,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0
            )

        else:

            raise ValueError(
                "cell_type must be RNN, GRU or LSTM"
            )

    def forward(self, source):

        embedded = self.embedding(source)

        outputs, hidden = self.rnn(embedded)

        return outputs, hidden
    # ==========================================================
# DECODER
# ==========================================================

class Decoder(nn.Module):

    def __init__(
        self,
        output_dim,
        embedding_dim,
        hidden_dim,
        num_layers=1,
        cell_type="LSTM",
        dropout=0.2
    ):

        super().__init__()

        self.cell_type = cell_type.upper()

        self.embedding = nn.Embedding(
            output_dim,
            embedding_dim
        )

        if self.cell_type == "RNN":

            self.rnn = nn.RNN(
                embedding_dim,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0
            )

        elif self.cell_type == "GRU":

            self.rnn = nn.GRU(
                embedding_dim,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0
            )

        elif self.cell_type == "LSTM":

            self.rnn = nn.LSTM(
                embedding_dim,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0
            )

        else:

            raise ValueError(
                "cell_type must be RNN, GRU or LSTM"
            )

        self.fc = nn.Linear(
            hidden_dim,
            output_dim
        )

    def forward(
        self,
        input_token,
        hidden
    ):

        input_token = input_token.unsqueeze(1)

        embedded = self.embedding(input_token)

        output, hidden = self.rnn(
            embedded,
            hidden
        )

        prediction = self.fc(
            output.squeeze(1)
        )

        return prediction, hidden
    # ==========================================================
# SEQ2SEQ
# ==========================================================

class Seq2Seq(nn.Module):

    def __init__(
        self,
        encoder,
        decoder,
        device
    ):

        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(
        self,
        source,
        target,
        teacher_forcing_ratio=0.5
    ):

        batch_size = source.size(0)

        target_length = target.size(1)

        target_vocab_size = self.decoder.fc.out_features

        outputs = torch.zeros(
            batch_size,
            target_length,
            target_vocab_size,
            device=self.device
        )
    def predict(
    self,
    source,
    max_length=30
    ):

        self.eval()

        predictions = []

        with torch.no_grad():

            _, hidden = self.encoder(source)

            input_token = torch.tensor(
                [1],  # <SOS>
                device=self.device
            )

            for _ in range(max_length):

                output, hidden = self.decoder(
                    input_token,
                    hidden
                )

                predicted_token = output.argmax(1)

                token = predicted_token.item()

                if token == 2:      # <EOS>
                    break

                predictions.append(token)

                input_token = predicted_token

        return predictions    

        # --------------------------
        # Encoder
        # --------------------------

        _, hidden = self.encoder(source)

        # First decoder input = <SOS>
        input_token = target[:, 0]

        # --------------------------
        # Decoder
        # --------------------------

        for t in range(1, target_length):

            prediction, hidden = self.decoder(
                input_token,
                hidden
            )

            outputs[:, t, :] = prediction

            teacher_force = random.random() < teacher_forcing_ratio

            top_prediction = prediction.argmax(1)

            input_token = (
                target[:, t]
                if teacher_force
                else top_prediction
            )

        return outputs
    # ==========================================================
# TEST
# ==========================================================

if __name__ == "__main__":

    device = torch.device("cpu")

    encoder = Encoder(
        input_dim=30,
        embedding_dim=64,
        hidden_dim=128,
        num_layers=1,
        cell_type="LSTM"
    )

    decoder = Decoder(
        output_dim=68,
        embedding_dim=64,
        hidden_dim=128,
        num_layers=1,
        cell_type="LSTM"
    )

    model = Seq2Seq(
        encoder,
        decoder,
        device
    )

    source = torch.randint(
        0,
        30,
        (4, 12)
    )

    target = torch.randint(
        0,
        68,
        (4, 15)
    )

    output = model(
        source,
        target
    )

    print("=" * 50)
    print("Seq2Seq Model Ready")
    print("=" * 50)

    print("Input Shape :", source.shape)
    print("Target Shape :", target.shape)
    print("Output Shape :", output.shape)