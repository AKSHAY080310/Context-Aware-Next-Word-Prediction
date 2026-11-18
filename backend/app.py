
from pathlib import Path

from flask import Flask, jsonify, render_template, abort
from backend.routes.prediction import prediction_bp

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

app = Flask(
    __name__,
    static_folder=str(FRONTEND_DIR),
    static_url_path="",
    template_folder=str(FRONTEND_DIR / "pages"),
)

app.register_blueprint(prediction_bp)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/workspace")
def workspace():
    return render_template("workspace.html")


@app.route("/workspace/<model_key>")
def model_workspace(model_key):
    if model_key not in (
        "lstm",
        "transformer",
        "gpt2",
        "distilgpt2"
    ):
        abort(404)

    return render_template(
        "index.html",
        initial_model=model_key
    )


@app.route("/comparison")
def comparison():
    return render_template("model_comparison.html")


@app.route("/explanation")
def explanation():
    return render_template("model_explanation.html")


@app.errorhandler(404)
def page_not_found(error):
    return jsonify({"error": "Page not found."}), 404


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )
