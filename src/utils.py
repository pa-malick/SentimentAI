"""
Fonctions utilitaires partagees dans tout le projet.
"""

import os
import json
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import joblib

from src.config import (
    METRICS_DIR, METRICS_JSON_PATH, MODELS_DIR,
    VECTORIZER_PATH, LOGISTIC_PATH, RANDOM_FOREST_PATH, DISTILBERT_DIR,
    RANDOM_SEED, get_logger,
)

log = get_logger()


def set_seed(seed: int = RANDOM_SEED) -> int:
    """
    Fixe la graine aleatoire partout pour que les resultats soient reproductibles.

    Sans ca, deux entrainements identiques peuvent donner des scores legerement
    differents (initialisation des poids, melange des batchs, echantillonnage).
    Pour un projet ou l'on compare des modeles, c'est genant : on ne sait plus
    si un ecart vient du modele ou du hasard.
    """
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)

    # torch seulement s'il est installe (extras BERT)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

    log.info(f"Graine aleatoire fixee a {seed}")
    return seed


def save_metrics_json(metrics_dict: Dict[str, Any], path: str = METRICS_JSON_PATH) -> str:
    """Sauvegarde les metriques dans un fichier JSON pour consultation ulterieure."""
    os.makedirs(METRICS_DIR, exist_ok=True)
    metrics_dict["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=2, ensure_ascii=False)
    log.info(f"Metriques sauvegardees dans {path}")
    return path


def load_metrics_json(path: str = METRICS_JSON_PATH) -> Dict[str, Any]:
    """Charge les metriques depuis le fichier JSON."""
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


def predict_sentiment(
    text: str,
    model_type: str = "logistic",
    models_cache: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Fonction principale de prediction utilisee par l'API Flask.
    Utilise le cache des modeles precharges si disponible, sinon charge depuis le disque.

    model_type : "logistic", "random_forest", ou "distilbert"
    """
    from src.preprocessing import clean_text

    label_map = {0: "Négatif", 1: "Positif"}
    cache = models_cache or {}

    if model_type == "distilbert":
        if "distilbert" in cache:
            bert = cache["distilbert"]
        else:
            from src.models import DistilBERTClassifier
            if not os.path.isdir(DISTILBERT_DIR):
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
        model_path = LOGISTIC_PATH if model_type == "logistic" else RANDOM_FOREST_PATH
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


def list_available_models() -> List[str]:
    """Retourne la liste des modeles disponibles dans le dossier models/."""
    if not os.path.exists(MODELS_DIR):
        return []
    return os.listdir(MODELS_DIR)


def distilbert_disponible() -> bool:
    """
    Verifie que DistilBERT est reellement utilisable.

    Il ne suffit pas que le dossier existe : les poids (model.safetensors ou
    pytorch_model.bin) sont volumineux donc gitignores, alors que les fichiers
    de config le sont parfois. Sur un clone frais, on peut donc avoir un dossier
    distilbert/ present mais inutilisable. Se fier a isdir() ferait planter le
    prechargement au demarrage du serveur.
    """
    if not os.path.isdir(DISTILBERT_DIR):
        return False

    poids = ["model.safetensors", "pytorch_model.bin"]
    return any(os.path.exists(os.path.join(DISTILBERT_DIR, f)) for f in poids)


def check_models_ready() -> Dict[str, bool]:
    """Verifie quels modeles sont deja entraines et disponibles."""
    status = {
        "logistic": os.path.exists(LOGISTIC_PATH),
        "random_forest": os.path.exists(RANDOM_FOREST_PATH),
        "distilbert": distilbert_disponible(),
        "vectorizer": os.path.exists(VECTORIZER_PATH)
    }
    return status
