"""
Fonctions utilitaires partagees dans tout le projet.
"""

import os
import json
import time
import joblib
from datetime import datetime


def save_metrics_json(metrics_dict, filename="metrics_results.json"):
    """Sauvegarde les metriques dans un fichier JSON pour consultation ulterieure."""
    os.makedirs("metrics", exist_ok=True)
    metrics_dict["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    path = os.path.join("metrics", filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=2, ensure_ascii=False)
    print(f"Metriques sauvegardees dans {path}")
    return path


def load_metrics_json(filename="metrics_results.json"):
    """Charge les metriques depuis le fichier JSON."""
    path = os.path.join("metrics", filename)
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def timer(func):
    """Decorateur qui mesure le temps d'execution d'une fonction."""
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"[Timer] {func.__name__} : {elapsed:.2f}s")
        return result
    return wrapper


def predict_sentiment(text, model_type="logistic", models_cache=None):
    """
    Fonction principale de prediction utilisee par l'API Flask.
    Utilise le cache des modeles precharges si disponible, sinon charge depuis le disque.

    model_type : "logistic", "random_forest", ou "distilbert"
    """
    from src.preprocessing import clean_text

    label_map = {0: "Negatif", 1: "Positif"}
    cache = models_cache or {}

    if model_type == "distilbert":
        if "distilbert" in cache:
            bert = cache["distilbert"]
        else:
            from src.models import DistilBERTClassifier
            if not os.path.isdir("models/distilbert"):
                raise FileNotFoundError("DistilBERT non entraine. Lance : python main.py --bert")
            bert = DistilBERTClassifier.from_saved()
        pred, probs = bert.predict_text(text)
        confidence = round(max(probs) * 100, 2)
        return {
            "sentiment": label_map[pred],
            "confidence": confidence,
            "label": pred,
            "model": "DistilBERT"
        }

    if "vectorizer" in cache:
        vectorizer = cache["vectorizer"]
    else:
        from src.preprocessing import load_vectorizer
        vectorizer = load_vectorizer()

    cleaned = clean_text(text)
    X = vectorizer.transform([cleaned])

    key = "logistic" if model_type == "logistic" else "random_forest"
    if key in cache:
        model = cache[key]
    else:
        model_filename = "tfidf_logistic.pkl" if model_type == "logistic" else "tfidf_random_forest.pkl"
        model_path = os.path.join("models", model_filename)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modele introuvable : {model_path}. Lance l'entrainement d'abord.")
        model = joblib.load(model_path)

    pred = int(model.predict(X)[0])
    proba = model.predict_proba(X)[0]
    confidence = round(float(max(proba)) * 100, 2)
    model_display = "Logistic Regression" if model_type == "logistic" else "Random Forest"

    return {
        "sentiment": label_map[pred],
        "confidence": confidence,
        "label": pred,
        "model": model_display
    }


def list_available_models():
    """Retourne la liste des modeles disponibles dans le dossier models/."""
    if not os.path.exists("models"):
        return []
    return os.listdir("models")


def check_models_ready():
    """Verifie quels modeles sont deja entraines et disponibles."""
    status = {
        "logistic": os.path.exists("models/tfidf_logistic.pkl"),
        "random_forest": os.path.exists("models/tfidf_random_forest.pkl"),
        "distilbert": os.path.isdir("models/distilbert"),
        "vectorizer": os.path.exists("models/tfidf_vectorizer.pkl")
    }
    return status
