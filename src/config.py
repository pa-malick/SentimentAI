"""
Configuration centrale du projet.

Toute la config qui etait avant eparpillee en dur dans les differents fichiers
(chemins des modeles, du cache, des metriques) est regroupee ici.
Comme ca, si on change l'organisation des dossiers, on ne modifie qu'un seul endroit.

On y met aussi un petit helper de logging pour avoir des messages propres
et coherents dans tout le projet, au lieu de print() un peu partout.
"""

import os
import logging

# Ce projet est 100% PyTorch. Si TensorFlow / Flax traine dans l'environnement,
# transformers essaie quand meme de les importer, ce qui peut casser (conflits
# protobuf par exemple). On lui dit explicitement de ne charger que PyTorch.
# setdefault : on ne force pas si l'utilisateur a deja mis une valeur.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

# Racine du projet (le dossier qui contient src/, app/, etc.)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Dossiers principaux
MODELS_DIR = os.path.join(BASE_DIR, "models")
METRICS_DIR = os.path.join(BASE_DIR, "metrics")

# Fichiers modeles et cache
DATA_CACHE_PATH = os.path.join(MODELS_DIR, "imdb_data.pkl")
VECTORIZER_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
LOGISTIC_PATH = os.path.join(MODELS_DIR, "tfidf_logistic.pkl")
RANDOM_FOREST_PATH = os.path.join(MODELS_DIR, "tfidf_random_forest.pkl")
DISTILBERT_DIR = os.path.join(MODELS_DIR, "distilbert")

# Fichier de metriques
METRICS_JSON_PATH = os.path.join(METRICS_DIR, "metrics_results.json")

# Parametres applicatifs (surchargables via variables d'environnement)
SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-change-me")
PORT = int(os.environ.get("PORT", 5000))

# Contraintes de validation sur le texte recu par l'API.
# Le plafond evite qu'on envoie un texte enorme qui ferait ramer le serveur.
MIN_TEXT_LENGTH = 5
MAX_TEXT_LENGTH = int(os.environ.get("MAX_TEXT_LENGTH", 5000))

# Nombre max de textes acceptes en une seule requete sur /predict/batch
MAX_BATCH_SIZE = int(os.environ.get("MAX_BATCH_SIZE", 100))

# Les seuls modeles que l'API accepte. Sert a rejeter proprement une valeur
# inconnue au lieu de retomber silencieusement sur la Random Forest.
VALID_MODELS = ("logistic", "random_forest", "distilbert")

# Graine aleatoire globale pour la reproductibilite
RANDOM_SEED = 42


def get_logger(name="sentiment"):
    """
    Retourne un logger configure simplement.
    On evite d'ajouter plusieurs handlers si la fonction est appelee plusieurs fois.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger
