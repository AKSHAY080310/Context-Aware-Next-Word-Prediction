
"""Lazy-loading model registry for ContextAI."""

from __future__ import annotations

import pickle
from pathlib import Path
from threading import Lock

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "models"


@keras.utils.register_keras_serializable(package="ContextAI")
class TiedOutputLayer(keras.layers.Layer):

    def __init__(self, embedding_layer=None, vocab_size=25000, **kwargs):
        super().__init__(**kwargs)

        if isinstance(embedding_layer, dict):
            embedding_layer = keras.layers.deserialize(
                embedding_layer
            )

        self.embedding_layer = embedding_layer

        if embedding_layer is not None:
            vocab_size = embedding_layer.input_dim

        self.vocab_size = int(vocab_size)

    def build(self, input_shape):

        self.bias = self.add_weight(
            name="bias",
            shape=(self.vocab_size,),
            initializer="zeros",
            trainable=True
        )

        super().build(input_shape)

    def call(self, inputs):

        logits = tf.linalg.matmul(
            inputs,
            self.embedding_layer.embeddings,
            transpose_b=True
        )

        return logits + self.bias

    def compute_output_shape(self, input_shape):

        return (
            input_shape[0],
            self.vocab_size
        )

    def get_config(self):

        config = super().get_config()

        config.update({
            "embedding_layer": keras.layers.serialize(
                self.embedding_layer
            ),
            "vocab_size": self.vocab_size
        })

        return config

    @classmethod
    def from_config(cls, config):

        embedding_config = config.pop("embedding_layer")

        embedding_layer = keras.layers.deserialize(
            embedding_config
        )

        return cls(
            embedding_layer=embedding_layer,
            **config
        )




MODEL_CONFIG = {
    "lstm": {
        "label": "Custom LSTM",
        "model_file": "lstm_model.keras",
        "word_to_id": "lstm_word_to_id.pkl",
        "id_to_word": "lstm_id_to_word.pkl",
        "context_size": 10,
        "kind": "keras",
    },

    "transformer": {
        "label": "Custom Transformer V2",
        "model_file": "transformer_best1.keras",
        "word_to_id": "transformer_word_to_id.pkl",
        "id_to_word": "transformer_id_to_word.pkl",
        "context_size": 32,
        "kind": "keras",
    },

    "gpt2": {
        "label": "GPT-2",
        "hf_name": "gpt2",
        "kind": "huggingface",
    },

    "distilgpt2": {
        "label": "DistilGPT-2",
        "hf_name": "distilgpt2",
        "kind": "huggingface",
    },
}


class ModelManager:

    def __init__(self):
        self._loaded = {}

        self._locks = {
            key: Lock()
            for key in MODEL_CONFIG
        }

    def _load_custom(self, model_key):

        cfg = MODEL_CONFIG[model_key]

        model_path = MODEL_DIR / cfg["model_file"]
        w2i_path = MODEL_DIR / cfg["word_to_id"]
        i2w_path = MODEL_DIR / cfg["id_to_word"]

        for path in (model_path, w2i_path, i2w_path):

            if not path.is_file():
                raise FileNotFoundError(
                    f"Required {cfg['label']} file is missing: {path}"
                )

        model = load_model(
            str(model_path),
            compile=False,
            custom_objects={
                "TiedOutputLayer": TiedOutputLayer
            },
            safe_mode=False,
        )

        with w2i_path.open("rb") as f:
            word_to_id = pickle.load(f)

        with i2w_path.open("rb") as f:
            id_to_word = pickle.load(f)

        if len(word_to_id) != len(id_to_word):
            raise ValueError(
                f"{cfg['label']} vocabulary files have different sizes."
            )

        if model.output_shape[-1] != len(word_to_id):
            raise ValueError(
                f"{cfg['label']} output size "
                f"({model.output_shape[-1]}) does not match "
                f"vocabulary size ({len(word_to_id)})."
            )

        return {
            "model": model,
            "word_to_id": word_to_id,
            "id_to_word": id_to_word,
        }

    def _load_huggingface(self, model_key):

        try:
            import torch

            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer
            )

        except ImportError as exc:
            raise RuntimeError(
                "GPT-2 models require the packages in "
                "requirements-ai.txt. Install them and restart "
                "the backend."
            ) from exc

        name = MODEL_CONFIG[model_key]["hf_name"]

        tokenizer = AutoTokenizer.from_pretrained(name)

        model = AutoModelForCausalLM.from_pretrained(name)

        model.eval()

        device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        model.to(device)

        return {
            "model": model,
            "tokenizer": tokenizer,
            "torch": torch,
            "device": device,
        }

    def get(self, model_key):

        if model_key not in MODEL_CONFIG:
            raise ValueError(
                "Unknown model. Choose lstm, transformer, "
                "gpt2 or distilgpt2."
            )

        if model_key not in self._loaded:

            with self._locks[model_key]:

                if model_key not in self._loaded:

                    cfg = MODEL_CONFIG[model_key]

                    print(f"Loading {cfg['label']}...")

                    if cfg["kind"] == "keras":

                        self._loaded[model_key] = (
                            self._load_custom(model_key)
                        )

                    else:

                        self._loaded[model_key] = (
                            self._load_huggingface(model_key)
                        )

                    print(f"{cfg['label']} loaded.")

        return self._loaded[model_key]

    def status(self):

        result = {}

        for key, cfg in MODEL_CONFIG.items():

            if cfg["kind"] == "keras":

                required = [
                    MODEL_DIR / cfg[k]
                    for k in (
                        "model_file",
                        "word_to_id",
                        "id_to_word"
                    )
                ]

                available = all(
                    path.is_file()
                    for path in required
                )

            else:

                # Hugging Face weights download on first use.
                available = True

            result[key] = {
                "label": cfg["label"],
                "available": available,
                "loaded": key in self._loaded,
                "context_size": cfg.get("context_size"),
                "kind": cfg["kind"],
            }

        return result
    
# ============================================================
# HUGGING FACE MODEL CONFIGURATION
# ============================================================

HF_MODEL_CONFIG = {
    "gpt2": {
        "model_name": "openai-community/gpt2",
        "display_name": "GPT-2"
    },

    "distilgpt2": {
        "model_name": "distilbert/distilgpt2",
        "display_name": "DistilGPT-2"
    }
}


# Cache for loaded Hugging Face models
hf_models = {}


# ============================================================
# LOAD HUGGING FACE MODEL
# ============================================================

def load_hf_model(model_key):

    if model_key not in HF_MODEL_CONFIG:
        raise ValueError(
            f"Unknown Hugging Face model: {model_key}"
        )

    # Return cached model if already loaded
    if model_key in hf_models:
        return hf_models[model_key]

    model_name = HF_MODEL_CONFIG[model_key]["model_name"]

    print(f"\nLoading {model_name}...")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load pretrained causal language model
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Select device
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Move model to selected device
    model.to(device)

    # Set evaluation mode
    model.eval()

    # Store model and tokenizer
    hf_models[model_key] = {
        "tokenizer": tokenizer,
        "model": model,
        "device": device,
        "display_name": HF_MODEL_CONFIG[model_key]["display_name"]
    }

    print(f"{model_name} loaded successfully on {device}")

    return hf_models[model_key]


# ============================================================
# LOAD ALL HUGGING FACE MODELS
# ============================================================

def load_all_hf_models():

    loaded_models = {}

    for model_key in HF_MODEL_CONFIG:

        loaded_models[model_key] = load_hf_model(model_key)

    return loaded_models


# ============================================================
# GET AVAILABLE HUGGING FACE MODELS
# ============================================================

def get_hf_model_list():

    return [
        {
            "key": model_key,
            "name": config["display_name"]
        }
        for model_key, config in HF_MODEL_CONFIG.items()
    ]    


model_manager = ModelManager()
