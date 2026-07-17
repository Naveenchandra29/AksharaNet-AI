# AksharaNet-AI

Character-Level Neural Transliteration using a Configurable Seq2Seq Network

---

# Table of Contents

- [Project Overview](#project-overview)
- [How It Works](#how-it-works)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Installation & Setup](#installation--setup)
- [Deployment](#deployment)

---

# 🎯Project Overview

**AksharaNet-AI** is a character-level neural transliteration system that converts Romanized Hindi words into their corresponding Devanagari script using a configurable Sequence-to-Sequence (Seq2Seq) neural network.

The project is built on the AI4Bharat Aksharantar dataset, where each Romanized word is mapped to its equivalent Hindi word. The implementation includes dataset preprocessing, vocabulary generation, Seq2Seq model training, greedy decoding for inference, a FastAPI backend, and a browser-based frontend.

The name **AksharaNet-AI** combines *Akshara* (character/letter in Indian languages) with *Net*, representing the neural network used for character-level transliteration.


---

# ⚙️How It Works

| Step | Description |
|------|-------------|
| 1 | Load the AI4Bharat Aksharantar Hindi dataset. |
| 2 | Build source and target character vocabularies. |
| 3 | Convert words into indexed character sequences. |
| 4 | Create padded DataLoaders with teacher forcing inputs. |
| 5 | Train a configurable Seq2Seq (RNN/LSTM/GRU) model. |
| 6 | Save the best model and vocabularies. |
| 7 | Perform greedy decoding for prediction. |
| 8 | Serve predictions through FastAPI and display them using the web interface. |

---

# ✨Features

- Character-level Seq2Seq transliteration
- Configurable RNN, LSTM and GRU cells
- Configurable embedding size, hidden size and layers
- Automatic vocabulary generation
- Teacher forcing during training
- Greedy decoding for inference
- Best-model checkpoint saving
- FastAPI REST API
- Browser-based frontend

---

# 💻Tech Stack

| Category | Technology |
|----------|------------|
| Language | Python |
| Deep Learning | PyTorch |
| Data Processing | Pandas, NumPy |
| Backend | FastAPI, Uvicorn |
| Frontend | HTML, CSS, JavaScript |
| IDE | Visual Studio Code |
| Dataset | AI4Bharat Aksharantar |
| Version Control | Git & GitHub |

---

# 🚀Installation & Setup

Clone the repository

```bash
git clone https://github.com/Naveenchandra29/AksharaNet-AI.git
cd AksharaNet-AI
```

Install dependencies

```bash
cd backend
pip install -r requirements.txt
```

Train the model

```bash
python train.py
```

Run inference

```bash
python predict.py
```

Start the API

```bash
uvicorn app:app --reload
```

Open `frontend/index.html` (or run it using Live Server) while the FastAPI server is running.

---

# 🌐Deployment

The project executes as a complete local transliteration pipeline. The dataset is preprocessed, the Seq2Seq model is trained using teacher forcing, and the best-performing model is saved as `best_model.pth` along with the generated vocabularies. During inference, FastAPI loads the saved model and performs greedy decoding to generate the corresponding Devanagari text, while the frontend displays the prediction through a simple web interface.
