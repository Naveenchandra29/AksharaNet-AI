# Table of Contents

- Project Overview
- How It Works
- Features
- Tech Stack
- Installation & Setup
- Deployment

---

# Project Overview

**AksharaNet-AI** is a character-level neural transliteration system that converts Romanized Hindi words into their corresponding Devanagari script using a Sequence-to-Sequence (Seq2Seq) deep learning model.

The primary objective of this project is to implement a configurable encoder-decoder architecture capable of learning character-level mappings from the AI4Bharat Aksharantar dataset. Unlike word-level translation, this implementation processes one character at a time, making it suitable for transliteration tasks where the pronunciation remains the same but the writing system changes.

The implemented workflow begins by preprocessing the Aksharantar dataset, constructing source and target vocabularies, encoding character sequences, training a configurable Seq2Seq model, saving the trained model and vocabularies, performing inference through greedy decoding, and finally exposing the trained model through a FastAPI backend with a simple web interface.

The project is named **AksharaNet-AI** because *Akshara* refers to characters or letters in many Indian languages, while *Net* represents the neural network architecture used to learn character-level transliteration patterns.

---

# How It Works

| Step | Implementation |
|------|----------------|
| **1. Dataset Loading** | The Aksharantar Hindi dataset (`hin_train.csv`, `hin_valid.csv`, `hin_test.csv`) is loaded and split into training, validation, and testing sets. |
| **2. Vocabulary Generation** | Separate source (Latin) and target (Devanagari) character vocabularies are created, including special tokens for sequence processing. |
| **3. Sequence Encoding** | Input and target words are converted into integer sequences suitable for neural network training. |
| **4. Data Preparation** | PyTorch Dataset and DataLoader objects generate mini-batches with dynamic padding and teacher-forcing decoder inputs/targets. |
| **5. Seq2Seq Model Construction** | A configurable Encoder-Decoder architecture is built using PyTorch. The implementation supports RNN, LSTM, and GRU cells with configurable embedding size, hidden dimension, and number of layers. |
| **6. Model Training** | The network is trained using CrossEntropy Loss and the Adam optimizer while monitoring validation loss after each epoch. |
| **7. Model Saving** | The best-performing model along with the generated vocabularies is stored for inference. |
| **8. Prediction** | During inference, the trained model performs greedy decoding to generate Devanagari characters sequentially until the output sequence is completed. |
| **9. API Integration** | FastAPI loads the trained model and exposes a prediction endpoint for transliteration requests. |
| **10. Web Interface** | A lightweight HTML, CSS, and JavaScript frontend sends user input to the API and displays the predicted transliterated word. |

---

# Features

- Character-level neural transliteration.
- Configurable Seq2Seq architecture.
- Support for **RNN**, **LSTM**, and **GRU** cells.
- Configurable embedding dimension, hidden dimension, and number of recurrent layers.
- Automatic vocabulary generation from the training dataset.
- Teacher forcing during training.
- Dynamic sequence padding using PyTorch DataLoader.
- Model checkpoint saving based on validation performance.
- Saved vocabularies for consistent inference.
- Greedy decoding for sequence prediction.
- FastAPI REST API for inference.
- Interactive browser-based frontend for transliteration.

---

# Tech Stack

| Category | Technology |
|----------|------------|
| Programming Language | Python |
| Deep Learning | PyTorch |
| Data Processing | Pandas |
| Numerical Computing | NumPy |
| API Framework | FastAPI |
| ASGI Server | Uvicorn |
| Frontend | HTML, CSS, JavaScript |
| Development Environment | Visual Studio Code |
| Dataset | AI4Bharat Aksharantar |
| Version Control | Git & GitHub |

---

# Installation & Setup

### Clone the repository

```bash
git clone https://github.com/Naveenchandra29/AksharaNet-AI.git
cd AksharaNet-AI
```

### Install dependencies

```bash
cd backend

pip install -r requirements.txt
```

### Train the model

```bash
python train.py
```

### Run inference

```bash
python predict.py
```

### Start the FastAPI server

```bash
uvicorn app:app --reload
```

The API will be available at:

```
http://127.0.0.1:8000
```

Interactive API documentation can be accessed at:

```
http://127.0.0.1:8000/docs
```

To use the graphical interface, open `frontend/index.html` (or serve it with Live Server) while the FastAPI backend is running.

---

# Deployment

The implementation follows a complete inference pipeline beginning with data preprocessing and model training, followed by deployment through a local FastAPI service.

During training, the Seq2Seq network learns character-level mappings from Romanized Hindi to Devanagari using teacher forcing. The model achieving the best validation loss is automatically stored in the `backend/saved_models/` directory together with the generated source and target vocabularies.

Generated outputs include:

- `best_model.pth` – trained Seq2Seq model weights.
- `source_vocab.pkl` – serialized source vocabulary.
- `target_vocab.pkl` – serialized target vocabulary.

For inference, the FastAPI backend loads these saved artifacts and performs greedy decoding to generate transliterated words. The frontend communicates with the `/predict` endpoint by sending user-entered Romanized text and displaying the predicted Devanagari output returned by the API.

The project can therefore be executed locally as a complete end-to-end transliteration system consisting of dataset preprocessing, neural network training, model inference, REST API integration, and a browser-based user interface.
