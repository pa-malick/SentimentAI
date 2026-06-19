"""
Definition des trois modeles utilises dans ce projet :
  1. TF-IDF + Logistic Regression (modele classique rapide)
  2. TF-IDF + Random Forest (ensemble learning)
  3. DistilBERT fine-tune (modele deep learning)
"""

import os
import joblib
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

MODELS_DIR = "models"


class TFIDFLogisticModel:
    """
    Modele classique : TF-IDF + Regression Logistique.
    Simple, rapide et souvent tres competitif en NLP binaire.
    """

    def __init__(self, C=1.0, max_iter=1000):
        self.model = LogisticRegression(C=C, max_iter=max_iter, solver="lbfgs", n_jobs=-1)
        self.name = "tfidf_logistic"

    def fit(self, X_train, y_train):
        print("Entrainement Logistic Regression...")
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def save(self):
        path = os.path.join(MODELS_DIR, f"{self.name}.pkl")
        os.makedirs(MODELS_DIR, exist_ok=True)
        joblib.dump(self.model, path)
        print(f"Modele sauvegarde : {path}")

    def load(self):
        path = os.path.join(MODELS_DIR, f"{self.name}.pkl")
        self.model = joblib.load(path)
        return self


class TFIDFRandomForestModel:
    """
    Modele classique : TF-IDF + Random Forest.
    Plus lent que la regression logistique mais capture mieux les non-linearites.
    """

    def __init__(self, n_estimators=200, n_jobs=-1):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            n_jobs=n_jobs,
            random_state=42
        )
        self.name = "tfidf_random_forest"

    def fit(self, X_train, y_train):
        print("Entrainement Random Forest (ca peut prendre quelques minutes)...")
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def save(self):
        path = os.path.join(MODELS_DIR, f"{self.name}.pkl")
        os.makedirs(MODELS_DIR, exist_ok=True)
        joblib.dump(self.model, path)
        print(f"Modele sauvegarde : {path}")

    def load(self):
        path = os.path.join(MODELS_DIR, f"{self.name}.pkl")
        self.model = joblib.load(path)
        return self


class DistilBERTClassifier:
    """
    Modele deep learning base sur DistilBERT pre-entraine.
    DistilBERT est une version allegee de BERT (60% de la taille, 97% des performances).
    On fine-tune la couche de classification sur IMDB.
    """

    MODEL_NAME = "distilbert-base-uncased"

    def __init__(self, num_labels=2, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"DistilBERT utilise le device : {self.device}")
        self.tokenizer = DistilBertTokenizer.from_pretrained(self.MODEL_NAME)
        self.model = DistilBertForSequenceClassification.from_pretrained(
            self.MODEL_NAME,
            num_labels=num_labels
        )
        self.model.to(self.device)
        self.name = "distilbert"

    def tokenize(self, texts, max_length=256):
        """Tokenise une liste de textes pour DistilBERT."""
        return self.tokenizer(
            texts,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

    def predict_text(self, text):
        """
        Predit le sentiment d'un texte individuel.
        Retourne le label predit et les probabilites.
        """
        import re
        text = re.sub(r"<[^>]+>", " ", text).strip()
        self.model.eval()
        with torch.no_grad():
            inputs = self.tokenize([text])
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
            pred = int(probs.argmax())
        return pred, probs.tolist()

    def save(self):
        path = os.path.join(MODELS_DIR, self.name)
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"DistilBERT sauvegarde dans : {path}")

    def load(self):
        path = os.path.join(MODELS_DIR, self.name)
        self.model = DistilBertForSequenceClassification.from_pretrained(path)
        self.tokenizer = DistilBertTokenizer.from_pretrained(path)
        self.model.to(self.device)
        return self

    @classmethod
    def from_saved(cls):
        """Charge un modele DistilBERT deja fine-tune depuis le disque."""
        instance = cls.__new__(cls)
        instance.device = "cuda" if torch.cuda.is_available() else "cpu"
        instance.name = "distilbert"
        path = os.path.join(MODELS_DIR, instance.name)
        instance.model = DistilBertForSequenceClassification.from_pretrained(path)
        instance.tokenizer = DistilBertTokenizer.from_pretrained(path)
        instance.model.to(instance.device)
        return instance
