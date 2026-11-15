from flask import Blueprint, request, jsonify

from backend.services.prediction_service import (
    prediction_service,
    MODEL_PATH,
    WORD_TO_ID_PATH,
    ID_TO_WORD_PATH
)


prediction_bp = Blueprint(
    "prediction",
    __name__,
    url_prefix="/api"
)


@prediction_bp.route("/health", methods=["GET"])
def health():

    files_available = all([
        MODEL_PATH.exists(),
        WORD_TO_ID_PATH.exists(),
        ID_TO_WORD_PATH.exists()
    ])

    return jsonify({
        "status": "healthy" if files_available else "model_files_missing",
        "model": "LSTM V3",
        "model_files_available": files_available
    })


@prediction_bp.route("/predict", methods=["POST"])
def predict():

    try:

        data = request.get_json(silent=True) or {}

        text = data.get("text", "")
        top_k = data.get("top_k", 5)

        if not isinstance(text, str) or not text.strip():
            return jsonify({
                "error": "Please provide some text."
            }), 400

        if (
            isinstance(top_k, bool)
            or not isinstance(top_k, int)
            or not 1 <= top_k <= 10
        ):
            return jsonify({
                "error": "top_k must be an integer between 1 and 10."
            }), 400

        result = prediction_service.predict(
            text=text,
            top_k=top_k
        )

        return jsonify(result), 200

    except ValueError as error:

        return jsonify({
            "error": str(error)
        }), 400

    except FileNotFoundError as error:

        return jsonify({
            "error": str(error)
        }), 500

    except Exception as error:

        print("Prediction error:", error)

        return jsonify({
            "error": "An internal prediction error occurred."
        }), 500