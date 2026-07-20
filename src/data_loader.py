"""
Chargement et preparation du dataset IMDB.
On utilise HuggingFace datasets pour avoir un acces simple et rapide aux donnees.
"""

import os
import pickle
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

from src.config import DATA_CACHE_PATH, MODELS_DIR, RANDOM_SEED, get_logger

log = get_logger()


def load_imdb_dataset(cache=True):
    """
    Charge le dataset IMDB depuis HuggingFace.
    Si les donnees sont deja en cache local, on les recharge depuis le fichier.
    Retourne quatre listes : textes train, labels train, textes test, labels test.
    """
    if cache and os.path.exists(DATA_CACHE_PATH):
        log.info("Chargement des donnees depuis le cache local...")
        with open(DATA_CACHE_PATH, "rb") as f:
            data = pickle.load(f)
        return data["train_texts"], data["train_labels"], data["test_texts"], data["test_labels"]

    log.info("Telechargement du dataset IMDB depuis HuggingFace...")
    dataset = load_dataset("stanfordnlp/imdb")

    train_texts = dataset["train"]["text"]
    train_labels = dataset["train"]["label"]
    test_texts = dataset["test"]["text"]
    test_labels = dataset["test"]["label"]

    if cache:
        os.makedirs(MODELS_DIR, exist_ok=True)
        with open(DATA_CACHE_PATH, "wb") as f:
            pickle.dump({
                "train_texts": train_texts,
                "train_labels": train_labels,
                "test_texts": test_texts,
                "test_labels": test_labels
            }, f)
        log.info(f"Donnees mises en cache dans {DATA_CACHE_PATH}")

    return train_texts, train_labels, test_texts, test_labels


def stratified_sample(texts, labels, n, seed=RANDOM_SEED):
    """
    Tire n exemples au hasard en gardant la proportion de chaque classe.

    On passe par train_test_split de sklearn plutot que de decouper a la main :
    l'ancienne version prenait les n/2 PREMIERS de chaque classe, donc toujours
    exactement les memes critiques (celles du debut du fichier). Un tirage
    aleatoire est bien plus representatif du dataset, et random_state le rend
    quand meme reproductible d'un run a l'autre.
    """
    texts, labels = list(texts), list(labels)

    if n >= len(texts):
        return texts, labels

    sous_textes, _, sous_labels, _ = train_test_split(
        texts,
        labels,
        train_size=n,
        stratify=labels,
        random_state=seed,
        shuffle=True,
    )
    return sous_textes, sous_labels


def get_sample(train_texts, train_labels, test_texts, test_labels,
               n_train=5000, n_test=1000, seed=RANDOM_SEED):
    """
    Retourne un sous-ensemble equilibre (stratifie) des donnees pour aller vite.
    Utilise en mode --sample.
    """
    t_texts, t_labels = stratified_sample(train_texts, train_labels, n_train, seed)
    te_texts, te_labels = stratified_sample(test_texts, test_labels, n_test, seed)
    return t_texts, t_labels, te_texts, te_labels


def to_dataframe(texts, labels):
    """Convertit les listes en DataFrame pandas pour une manipulation plus facile."""
    label_map = {0: "negative", 1: "positive"}
    df = pd.DataFrame({
        "text": texts,
        "label": labels,
        "sentiment": [label_map[l] for l in labels]
    })
    return df


def get_dataset_info(train_texts, train_labels, test_texts, test_labels):
    """Affiche quelques stats de base sur le dataset."""
    log.info(f"Taille train : {len(train_texts)} exemples")
    log.info(f"Taille test  : {len(test_texts)} exemples")
    positifs_train = sum(train_labels)
    log.info(f"Distribution train - Positif: {positifs_train} | Negatif: {len(train_labels) - positifs_train}")
    positifs_test = sum(test_labels)
    log.info(f"Distribution test  - Positif: {positifs_test} | Negatif: {len(test_labels) - positifs_test}")
