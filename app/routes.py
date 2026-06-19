"""
Routes de l'application Flask.
On separe les routes du fichier principal pour garder le code propre.
"""

from flask import Blueprint, render_template, request, jsonify, current_app
from src.utils import predict_sentiment, check_models_ready, load_metrics_json

main_bp = Blueprint("main", __name__)


@main_bp.route("/")
def index():
    """Page d'accueil avec le formulaire d'analyse."""
    return render_template("index.html")


@main_bp.route("/about")
def about():
    """Page de presentation du projet et de l'equipe."""
    return render_template("about.html")


@main_bp.route("/predict", methods=["POST"])
def predict():
    """
    Route principale d'API : recoit un texte et retourne le sentiment predit.
    Accepte JSON ou form-data.
    """
    if request.is_json:
        data = request.get_json()
        text = data.get("text", "")
        model_type = data.get("model", "logistic")
    else:
        text = request.form.get("text", "")
        model_type = request.form.get("model", "logistic")

    if not text or len(text.strip()) < 5:
        return jsonify({
            "error": "Texte trop court. Entre au moins 5 caracteres."
        }), 400

    try:
        cache = current_app.config.get("MODELS_CACHE", {})
        result = predict_sentiment(text.strip(), model_type=model_type, models_cache=cache)
        return jsonify(result)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        return jsonify({"error": f"Erreur interne : {str(e)}"}), 500


@main_bp.route("/status")
def status():
    """Retourne l'etat des modeles disponibles sur le serveur."""
    models_status = check_models_ready()
    return jsonify(models_status)


@main_bp.route("/metrics")
def metrics():
    """Retourne les dernieres metriques d'evaluation en JSON."""
    data = load_metrics_json()
    return jsonify(data)
