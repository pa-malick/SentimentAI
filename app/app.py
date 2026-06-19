"""
Point d'entree Flask.
Les modeles sont precharges au demarrage pour eviter les timeouts a la premiere requete.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask
from app.routes import main_bp

_models_cache = {}


def preload_models():
    """Charge tous les modeles disponibles au demarrage du serveur."""
    import joblib
    from src.utils import check_models_ready

    status = check_models_ready()

    if status["vectorizer"]:
        from src.preprocessing import load_vectorizer
        _models_cache["vectorizer"] = load_vectorizer()
        print("  Vectoriseur TF-IDF charge.")

    if status["logistic"]:
        _models_cache["logistic"] = joblib.load("models/tfidf_logistic.pkl")
        print("  Logistic Regression charge.")

    if status["random_forest"]:
        _models_cache["random_forest"] = joblib.load("models/tfidf_random_forest.pkl")
        print("  Random Forest charge.")

    if status["distilbert"]:
        from src.models import DistilBERTClassifier
        _models_cache["distilbert"] = DistilBERTClassifier.from_saved()
        print("  DistilBERT charge.")


def create_app():
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config["SECRET_KEY"] = "sentiment2050sn"
    app.config["JSON_AS_ASCII"] = False
    app.config["MODELS_CACHE"] = _models_cache
    app.register_blueprint(main_bp)
    return app


if __name__ == "__main__":
    print("\nChargement des modeles en memoire...")
    preload_models()
    app = create_app()
    port = int(os.environ.get("PORT", 5000))
    print(f"\nServeur Flask demarre sur http://localhost:{port}\n")
    app.run(debug=False, host="0.0.0.0", port=port)
