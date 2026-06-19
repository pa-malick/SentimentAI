"""
Preprocessing du texte avant de passer aux modeles.
On nettoie, on tokenise, et on vectorise selon le modele cible.
"""

import re
import string
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os

nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
from nltk.corpus import stopwords

STOP_WORDS = set(stopwords.words("english"))
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"


def clean_text(text):
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


def preprocess_batch(texts, verbose=True):
    """Applique clean_text sur une liste de textes."""
    if verbose:
        print(f"Nettoyage de {len(texts)} textes...")
    cleaned = [clean_text(t) for t in texts]
    return cleaned


def build_tfidf_vectorizer(train_texts, max_features=30000, ngram_range=(1, 2)):
    """
    Entraine un vectoriseur TF-IDF sur les textes d'entrainement.
    On utilise des bigrammes (1, 2) pour capturer des expressions comme "not good".
    """
    print("Construction du vectoriseur TF-IDF...")
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=True
    )
    vectorizer.fit(train_texts)
    os.makedirs("models", exist_ok=True)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f"Vectoriseur sauvegarde dans {VECTORIZER_PATH}")
    return vectorizer


def transform_texts(vectorizer, texts):
    """Transforme une liste de textes en matrice TF-IDF."""
    return vectorizer.transform(texts)


def load_vectorizer():
    """Charge le vectoriseur deja entraine depuis le disque."""
    if not os.path.exists(VECTORIZER_PATH):
        raise FileNotFoundError("Vectoriseur introuvable. Lance d'abord l'entrainement.")
    return joblib.load(VECTORIZER_PATH)


def preprocess_for_bert(text, tokenizer, max_length=256):
    """
    Tokenisation specifique a DistilBERT.
    On tronque a max_length pour eviter des sequences trop longues.
    On ne supprime pas les stopwords ici car BERT les comprend en contexte.
    """
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    encoding = tokenizer(
        text,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    return encoding
