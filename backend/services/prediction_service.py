
from __future__ import annotations

import numpy as np
from nltk.tokenize import word_tokenize

from backend.services.model_manager import (
    MODEL_CONFIG,
    model_manager
)

DEFAULT_TOP_K = 5
MAX_TOP_K = 10


def _custom_prediction(model_key: str, text: str, top_k: int):

    cfg = MODEL_CONFIG[model_key]

    resources = model_manager.get(model_key)

    tokens = word_tokenize(text)

    if not tokens:
        raise ValueError("Please enter some text.")

    word_to_id = resources["word_to_id"]
    id_to_word = resources["id_to_word"]

    unk_id = word_to_id.get("<UNK>", 1)
    pad_id = word_to_id.get("<PAD>", 0)

    token_ids = [
        word_to_id.get(token, unk_id)
        for token in tokens
    ]

    token_ids = token_ids[-cfg["context_size"]:]

    token_ids = (
        [pad_id] * (cfg["context_size"] - len(token_ids))
        + token_ids
    )

    probabilities = resources["model"].predict(
        np.asarray([token_ids], dtype=np.int32),
        verbose=0
    )[0]

    ranked = np.argsort(probabilities)[::-1]

    predictions = []

    for idx in ranked:

        word = id_to_word.get(int(idx), "<UNK>")

        if word in ("<PAD>", "<UNK>"):
            continue

        prob = float(probabilities[idx])

        predictions.append({
            "word": word,
            "probability": prob,
            "percentage": round(prob * 100, 2),
            "unit": "word",
        })

        if len(predictions) >= top_k:
            break

    return {
        "input_text": text,
        "tokens": tokens,
        "context_tokens": [
            id_to_word.get(int(i), "<UNK>")
            for i in token_ids
            if int(i) != pad_id
        ],
        "predictions": predictions,
        "model": cfg["label"],
        "context_size": cfg["context_size"],
        "prediction_unit": "word",
    }


def _huggingface_prediction(
    model_key: str,
    text: str,
    top_k: int
):

    resources = model_manager.get(model_key)

    tokenizer = resources["tokenizer"]
    torch = resources["torch"]
    device = resources["device"]

    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )

    encoded = {
        key: value.to(device)
        for key, value in encoded.items()
    }

    if encoded["input_ids"].shape[1] == 0:
        raise ValueError("Please enter some text.")

    with torch.no_grad():

        logits = resources["model"](
            **encoded
        ).logits[0, -1]

        probs = torch.softmax(logits, dim=-1)

        values, indices = torch.topk(
            probs,
            k=top_k
        )

    predictions = []

    for value, index in zip(
        values.tolist(),
        indices.tolist()
    ):

        token_text = tokenizer.decode(
            [index],
            clean_up_tokenization_spaces=False
        )

        display = token_text.replace("\n", "\\n")

        predictions.append({
            "word": display,
            "probability": float(value),
            "percentage": round(float(value) * 100, 2),
            "unit": "token",
        })

    input_tokens = tokenizer.convert_ids_to_tokens(
        encoded["input_ids"][0].tolist()
    )

    return {
        "input_text": text,
        "tokens": input_tokens,
        "context_tokens": input_tokens[-12:],
        "predictions": predictions,
        "model": MODEL_CONFIG[model_key]["label"],
        "context_size": len(input_tokens),
        "prediction_unit": "token",
        "note": (
            "GPT-family models predict subword tokens, "
            "not necessarily complete words."
        ),
    }


def predict(
    text: str,
    model_key: str = "lstm",
    top_k: int = DEFAULT_TOP_K
):

    if not isinstance(text, str) or not text.strip():
        raise ValueError("Please provide some text.")

    if (
        isinstance(top_k, bool)
        or not isinstance(top_k, int)
        or not 1 <= top_k <= MAX_TOP_K
    ):
        raise ValueError(
            f"top_k must be an integer between 1 and {MAX_TOP_K}."
        )

    if model_key not in MODEL_CONFIG:
        raise ValueError("Invalid model selection.")

    if MODEL_CONFIG[model_key]["kind"] == "keras":

        return _custom_prediction(
            model_key,
            text,
            top_k
        )

    return _huggingface_prediction(
        model_key,
        text,
        top_k
    )
