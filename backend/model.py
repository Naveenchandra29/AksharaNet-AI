"""
model.py

Character-level Seq2Seq model for transliteration.

Supports:
- RNN
- LSTM
- GRU
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

        else:

            self.rnn = nn.LSTM(
                embedding_dim,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0
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

        else:

            self.rnn = nn.LSTM(
                embedding_dim,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0
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
        """
        input_token:
            (batch_size)

        hidden:
            Hidden state from encoder or previous decoder step.
        """

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
# SEQ2SEQ MODEL
# ==========================================================

class Seq2Seq(nn.Module):

    def __init__(
        self,
        encoder,
        decoder,
        device,
    ):

        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(
        self,
        source,
        target,
        teacher_forcing_ratio=0.5,
    ):
        """
        source : (batch_size, source_length)
        target : (batch_size, target_length)

        Returns:
            outputs : (batch_size, target_length, target_vocab_size)
        """

        batch_size = source.shape[0]

        target_length = target.shape[1]

        target_vocab_size = self.decoder.fc.out_features

        outputs = torch.zeros(
            batch_size,
            target_length,
            target_vocab_size,
            device=self.device
        )

        # -----------------------------
        # Encoder
        # -----------------------------

        _, hidden = self.encoder(source)

        # First decoder input = <SOS>
        input_token = target[:, 0]

        # -----------------------------
        # Decoder
        # -----------------------------

        for timestep in range(1, target_length):

            prediction, hidden = self.decoder(
                input_token,
                hidden
            )

            outputs[:, timestep] = prediction

            teacher_force = random.random() < teacher_forcing_ratio

            predicted_token = prediction.argmax(1)

            input_token = (
                target[:, timestep]
                if teacher_force
                else predicted_token
            )

        return outputs