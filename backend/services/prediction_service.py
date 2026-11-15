import pickle
from pathlib import Path

import numpy as np
import nltk
from tensorflow.keras.models import load_model

from backend.utils.preprocessing import prepare_input

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

MODEL_PATH = MODEL_DIR / "lstm_model.keras"
WORD_TO_ID_PATH = MODEL_DIR / "word_to_id.pkl"
ID_TO_WORD_PATH = MODEL_DIR / "id_to_word.pkl"

CONTEXT_SIZE = 10
DEFAULT_TOP_K = 5
MAX_TOP_K = 10

class PredictionService:

    def __init__(self):
        self.model = None
        self.word_to_id = None
        self.id_to_word = None

    def load_resources(self):
        """
        Load model and vocabulary only once.
        """

        if self.model is not None:
            return

        required_files = [
            MODEL_PATH,
            WORD_TO_ID_PATH,
            ID_TO_WORD_PATH
        ]

        for file_path in required_files:
            if not file_path.exists():
                raise FileNotFoundError(
                    f"Required file not found: {file_path}"
                )

        # Ensure the NLTK tokenizer data is available.
        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            nltk.download("punkt_tab", quiet=True)

        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)

        print("Loading LSTM model...")

        self.model = load_model(
            str(MODEL_PATH),
            compile=False
        )

        print("Loading vocabulary...")

        with open(WORD_TO_ID_PATH, "rb") as file:
            self.word_to_id = pickle.load(file)

        with open(ID_TO_WORD_PATH, "rb") as file:
            self.id_to_word = pickle.load(file)

        print("LSTM model and vocabulary loaded successfully.")

    def predict(self, text, top_k=DEFAULT_TOP_K):

        self.load_resources()

        if not isinstance(text, str) or not text.strip():
            raise ValueError("Text cannot be empty.")

        if not isinstance(top_k, int) or not 1 <= top_k <= MAX_TOP_K:
            raise ValueError(
                f"top_k must be an integer between 1 and {MAX_TOP_K}."
            )

        # Prepare input.
        input_ids, tokens = prepare_input(
            text,
            self.word_to_id,
            CONTEXT_SIZE
        )

        X = np.array([input_ids], dtype=np.int32)

        # Run the LSTM.
        probabilities = self.model.predict(
            X,
            verbose=0
        )[0]

        # Find the highest-probability words.
        ranked_indices = np.argsort(probabilities)[::-1]

        predictions = []

        for index in ranked_indices:

            word = self.id_to_word.get(
                int(index),
                "<UNK>"
            )

            # Avoid suggesting padding and unknown tokens.
            if word in ("<PAD>", "<UNK>"):
                continue

            predictions.append({
                "word": word,
                "probability": float(probabilities[index]),
                "percentage": round(
                    float(probabilities[index]) * 100,
                    2
                )
            })

            if len(predictions) == top_k:
                break

        return {
            "input_text": text,
            "tokens": tokens,
            "context_tokens": [
                self.id_to_word.get(token_id, "<UNK>")
                for token_id in input_ids
            ],
            "predictions": predictions,
            "model": "LSTM V3",
            "context_size": CONTEXT_SIZE
        }

# Shared service instance.
prediction_service = PredictionService()