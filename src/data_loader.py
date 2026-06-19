"""
Chargement et preparation du dataset IMDB.
On utilise HuggingFace datasets pour avoir un acces simple et rapide aux donnees.
"""

import os
import pickle
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm


DATA_CACHE_PATH = "models/imdb_data.pkl"


def load_imdb_dataset(cache=True):
    """
    Charge le dataset IMDB depuis HuggingFace.
    Si les donnees sont deja en cache local, on les recharge depuis le fichier.
    Retourne quatre listes : textes train, labels train, textes test, labels test.
    """
    if cache and os.path.exists(DATA_CACHE_PATH):
        print("Chargement des donnees depuis le cache local...")
        with open(DATA_CACHE_PATH, "rb") as f:
            data = pickle.load(f)
        return data["train_texts"], data["train_labels"], data["test_texts"], data["test_labels"]

    print("Telechargement du dataset IMDB depuis HuggingFace...")
    dataset = load_dataset("stanfordnlp/imdb")

    train_texts = dataset["train"]["text"]
    train_labels = dataset["train"]["label"]
    test_texts = dataset["test"]["text"]
    test_labels = dataset["test"]["label"]

    if cache:
        os.makedirs("models", exist_ok=True)
        with open(DATA_CACHE_PATH, "wb") as f:
            pickle.dump({
                "train_texts": train_texts,
                "train_labels": train_labels,
                "test_texts": test_texts,
                "test_labels": test_labels
            }, f)
        print(f"Donnees mises en cache dans {DATA_CACHE_PATH}")

    return train_texts, train_labels, test_texts, test_labels


def get_sample(train_texts, train_labels, test_texts, test_labels, n_train=5000, n_test=1000):
    """
    Retourne un sous-ensemble equilibre (stratifie) des donnees pour des tests rapides.
    Le dataset stanfordnlp/imdb est trie par label, donc on prend la moitie de chaque classe.
    """
    def stratified_split(texts, labels, n):
        pos_idx = [i for i, l in enumerate(labels) if l == 1][:n // 2]
        neg_idx = [i for i, l in enumerate(labels) if l == 0][:n // 2]
        indices = neg_idx + pos_idx
        return [texts[i] for i in indices], [labels[i] for i in indices]

    t_texts, t_labels = stratified_split(list(train_texts), list(train_labels), n_train)
    te_texts, te_labels = stratified_split(list(test_texts), list(test_labels), n_test)
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
    print(f"Taille train : {len(train_texts)} exemples")
    print(f"Taille test  : {len(test_texts)} exemples")
    positifs_train = sum(train_labels)
    print(f"Distribution train - Positif: {positifs_train} | Negatif: {len(train_labels) - positifs_train}")
    positifs_test = sum(test_labels)
    print(f"Distribution test  - Positif: {positifs_test} | Negatif: {len(test_labels) - positifs_test}")
