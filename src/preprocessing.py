"""
Preprocessing du texte avant de passer aux modeles.
On nettoie, on tokenise, et on vectorise selon le modele cible.
"""

import re
import os
from typing import List, Tuple

import nltk
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import VECTORIZER_PATH, MODELS_DIR, get_logger

log = get_logger()

# On a seulement besoin de la liste des stopwords anglais.
# (Pas besoin de "punkt" : on tokenise avec un simple .split(), pas avec nltk.)
nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords

STOP_WORDS = set(stopwords.words("english"))


def clean_text(text: str) -> str:
    """
    Nettoie un texte brut :
    - mise en minuscule
    - suppression des balises HTML (les critiques IMDB en ont parfois)
    - suppression de la ponctuation et des chiffres
    - suppression des stopwords
    """
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]
    return " ".join(tokens)


def preprocess_batch(texts: List[str], verbose: bool = True) -> List[str]:
    """Applique clean_text sur une liste de textes."""
    if verbose:
        log.info(f"Nettoyage de {len(texts)} textes...")
    cleaned = [clean_text(t) for t in texts]
    return cleaned


def build_tfidf_vectorizer(
    train_texts: List[str],
    max_features: int = 30000,
    ngram_range: Tuple[int, int] = (1, 2),
) -> TfidfVectorizer:
    """
    Entraine un vectoriseur TF-IDF sur les textes d'entrainement.
    On utilise des bigrammes (1, 2) pour capturer des expressions comme "not good".
    """
    log.info("Construction du vectoriseur TF-IDF...")
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=True
    )
    vectorizer.fit(train_texts)
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    log.info(f"Vectoriseur sauvegarde dans {VECTORIZER_PATH}")
    return vectorizer


def transform_texts(vectorizer: TfidfVectorizer, texts: List[str]):
    """Transforme une liste de textes en matrice TF-IDF."""
    return vectorizer.transform(texts)


def load_vectorizer() -> TfidfVectorizer:
    """Charge le vectoriseur deja entraine depuis le disque."""
    if not os.path.exists(VECTORIZER_PATH):
        raise FileNotFoundError("Vectoriseur introuvable. Lance d'abord l'entrainement.")
    return joblib.load(VECTORIZER_PATH)
