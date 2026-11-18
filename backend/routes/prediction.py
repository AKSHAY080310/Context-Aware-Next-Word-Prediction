
from flask import Blueprint, jsonify, request

from backend.services.model_manager import model_manager
from backend.services.prediction_service import predict

prediction_bp = Blueprint("prediction", __name__, url_prefix="/api")


@prediction_bp.get("/health")
def health():
    models = model_manager.status()

    return jsonify({
        "status": "healthy",
        "models": models,
        "model_files_available": (
            models["lstm"]["available"]
            or models["transformer"]["available"]
        ),
    })


@prediction_bp.get("/models")
def models():
    return jsonify({
        "models": model_manager.status()
    })


@prediction_bp.post("/predict")
def predict_route():
    data = request.get_json(silent=True) or {}

    text = data.get("text", "")
    model_key = data.get("model", "lstm")
    top_k = data.get("top_k", 5)

    try:
        if not isinstance(model_key, str):
            raise ValueError("model must be a string.")

        result = predict(
            text=text,
            model_key=model_key,
            top_k=top_k
        )

        return jsonify(result), 200

    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 500

    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 503

    except Exception as exc:
        print("Prediction error:", repr(exc))

        return jsonify({
            "error": (
                "An internal prediction error occurred. "
                "Check the backend terminal for details."
            )
        }), 500
