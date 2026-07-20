"""
Definition des trois modeles utilises dans ce projet :
  1. TF-IDF + Logistic Regression (modele classique rapide)
  2. TF-IDF + Random Forest (ensemble learning)
  3. DistilBERT fine-tune (modele deep learning)

Note sur les imports : torch et transformers ne sont PAS importes en haut du
fichier. Ils ne sont charges qu'au moment ou on utilise reellement DistilBERT.
Comme ca, les modeles classiques fonctionnent meme sans avoir installe torch
(qui pese plusieurs centaines de Mo). Voir requirements-bert.txt.
"""

import os
from typing import Any, List, Tuple

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.config import MODELS_DIR, RANDOM_SEED, get_logger

log = get_logger()


class SklearnTextModel:
    """
    Classe de base commune aux modeles sklearn du projet.

    Les deux modeles classiques (LR et Random Forest) partagent exactement la
    meme mecanique : entrainer, predire, sauvegarder, recharger. On la met donc
    ici une seule fois, et chaque modele concret n'a plus qu'a preciser
    son nom de fichier et l'estimateur sklearn qu'il utilise.
    """

    name: str = "base_model"
    fit_message: str = "Entrainement du modele..."

    def __init__(self, model: Any):
        self.model = model

    @property
    def path(self) -> str:
        """Chemin du fichier .pkl associe a ce modele."""
        return os.path.join(MODELS_DIR, f"{self.name}.pkl")

    def fit(self, X_train: Any, y_train: Any) -> "SklearnTextModel":
        log.info(self.fit_message)
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X: Any) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: Any) -> np.ndarray:
        return self.model.predict_proba(X)

    def save(self) -> str:
        os.makedirs(MODELS_DIR, exist_ok=True)
        joblib.dump(self.model, self.path)
        log.info(f"Modele sauvegarde : {self.path}")
        return self.path

    def load(self) -> "SklearnTextModel":
        if not os.path.exists(self.path):
            raise FileNotFoundError(
                f"Modele introuvable : {self.path}. Lance d'abord l'entrainement."
            )
        self.model = joblib.load(self.path)
        return self


class TFIDFLogisticModel(SklearnTextModel):
    """
    Modele classique : TF-IDF + Regression Logistique.
    Simple, rapide et souvent tres competitif en NLP binaire.
    """

    name = "tfidf_logistic"
    fit_message = "Entrainement Logistic Regression..."

    def __init__(self, C: float = 1.0, max_iter: int = 1000):
        super().__init__(
            LogisticRegression(C=C, max_iter=max_iter, solver="lbfgs", n_jobs=-1)
        )


class TFIDFRandomForestModel(SklearnTextModel):
    """
    Modele classique : TF-IDF + Random Forest.
    Plus lent que la regression logistique mais capture mieux les non-linearites.
    """

    name = "tfidf_random_forest"
    fit_message = "Entrainement Random Forest (ca peut prendre quelques minutes)..."

    def __init__(self, n_estimators: int = 200, n_jobs: int = -1):
        super().__init__(
            RandomForestClassifier(
                n_estimators=n_estimators,
                n_jobs=n_jobs,
                random_state=RANDOM_SEED,
            )
        )


class DistilBERTClassifier:
    """
    Modele deep learning base sur DistilBERT pre-entraine.
    DistilBERT est une version allegee de BERT (60% de la taille, 97% des performances).
    On fine-tune la couche de classification sur IMDB.
    """

    MODEL_NAME = "distilbert-base-uncased"
    name = "distilbert"

    def __init__(self, num_labels: int = 2, device: str = None):
        torch, AutoTokenizer, DistilBertForSequenceClassification = _load_bert_deps()

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        log.info(f"DistilBERT utilise le device : {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.model = DistilBertForSequenceClassification.from_pretrained(
            self.MODEL_NAME,
            num_labels=num_labels,
        )
        self.model.to(self.device)

    @property
    def path(self) -> str:
        """Dossier ou DistilBERT est sauvegarde (format HuggingFace)."""
        return os.path.join(MODELS_DIR, self.name)

    def tokenize(self, texts: List[str], max_length: int = 256):
        """Tokenise une liste de textes pour DistilBERT."""
        return self.tokenizer(
            texts,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

    def predict_text(self, text: str) -> Tuple[int, List[float]]:
        """
        Predit le sentiment d'un texte individuel.
        Retourne le label predit et les probabilites.
        """
        import re
        import torch

        text = re.sub(r"<[^>]+>", " ", text).strip()
        self.model.eval()
        with torch.no_grad():
            inputs = self.tokenize([text])
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
            pred = int(probs.argmax())
        return pred, probs.tolist()

    def save(self) -> str:
        os.makedirs(self.path, exist_ok=True)
        self.model.save_pretrained(self.path)
        self.tokenizer.save_pretrained(self.path)
        log.info(f"DistilBERT sauvegarde dans : {self.path}")
        return self.path

    def load(self) -> "DistilBERTClassifier":
        _, AutoTokenizer, DistilBertForSequenceClassification = _load_bert_deps()
        self.model = DistilBertForSequenceClassification.from_pretrained(self.path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.path)
        self.model.to(self.device)
        return self

    @classmethod
    def from_saved(cls) -> "DistilBERTClassifier":
        """
        Charge un modele DistilBERT deja fine-tune depuis le disque.
        On passe par __new__ pour eviter de retelecharger le modele pre-entraine
        depuis HuggingFace, ce que ferait __init__.
        """
        torch, _, _ = _load_bert_deps()

        instance = cls.__new__(cls)
        instance.device = "cuda" if torch.cuda.is_available() else "cpu"
        if not os.path.isdir(instance.path):
            raise FileNotFoundError(
                f"DistilBERT introuvable dans {instance.path}. "
                "Lance : python main.py --bert"
            )
        return instance.load()


def _load_bert_deps():
    """
    Importe torch et transformers a la demande.

    Le message d'erreur est explicite si les dependances manquent : c'est le cas
    quand on a installe seulement requirements.txt sans les extras BERT.
    """
    try:
        # src.config est deja importe plus haut, donc USE_TF=0 est bien pose
        # avant que transformers ne soit charge.
        import torch
        from transformers import AutoTokenizer, DistilBertForSequenceClassification
    except ImportError as e:
        raise ImportError(
            "DistilBERT necessite torch et transformers. "
            "Installe-les avec : pip install -r requirements-bert.txt"
        ) from e

    return torch, AutoTokenizer, DistilBertForSequenceClassification
