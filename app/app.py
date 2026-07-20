"""
Point d'entree Flask.
Les modeles sont precharges au demarrage pour eviter les timeouts a la premiere requete.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask
from app.routes import main_bp
from src.config import SECRET_KEY, PORT, LOGISTIC_PATH, RANDOM_FOREST_PATH, get_logger

log = get_logger()
_models_cache = {}


def preload_models():
    """
    Charge tous les modeles disponibles au demarrage du serveur.
    Le cache est un dictionnaire global : si il est deja rempli, on ne recharge rien
    (utile quand create_app() est appele plusieurs fois, par exemple dans les tests).
    """
    import joblib
    from src.utils import check_models_ready

    if _models_cache:
        return _models_cache

    log.info("Chargement des modeles en memoire...")
    status = check_models_ready()

    if status["vectorizer"]:
        from src.preprocessing import load_vectorizer
        _models_cache["vectorizer"] = load_vectorizer()
        log.info("  Vectoriseur TF-IDF charge.")

    if status["logistic"]:
        _models_cache["logistic"] = joblib.load(LOGISTIC_PATH)
        log.info("  Logistic Regression charge.")

    if status["random_forest"]:
        _models_cache["random_forest"] = joblib.load(RANDOM_FOREST_PATH)
        log.info("  Random Forest charge.")

    if status["distilbert"]:
        # DistilBERT est le plus fragile a charger (poids volumineux, dependances
        # optionnelles). Si ca echoue, on continue avec les modeles classiques
        # plutot que d'empecher le serveur de demarrer.
        try:
            from src.models import DistilBERTClassifier
            _models_cache["distilbert"] = DistilBERTClassifier.from_saved()
            log.info("  DistilBERT charge.")
        except Exception as e:
            log.warning(f"  DistilBERT non chargeable ({type(e).__name__}), on continue sans.")

    return _models_cache


def create_app(preload=True):
    """
    Cree l'application Flask.

    preload=True par defaut : les modeles sont charges en memoire des la creation
    de l'app. C'est important en production, ou gunicorn appelle create_app()
    directement sans passer par le bloc __main__ : sans ca, chaque requete
    rechargeait les modeles depuis le disque.
    Les tests peuvent passer preload=False pour demarrer instantanement.
    """
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config["SECRET_KEY"] = SECRET_KEY
    app.config["JSON_AS_ASCII"] = False

    if preload:
        preload_models()

    app.config["MODELS_CACHE"] = _models_cache
    app.register_blueprint(main_bp)
    return app


if __name__ == "__main__":
    app = create_app()
    log.info(f"\nServeur Flask demarre sur http://localhost:{PORT}\n")
    app.run(debug=False, host="0.0.0.0", port=PORT)
