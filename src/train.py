"""
Entrainement des modeles classiques (TF-IDF + sklearn).
Le pipeline est : chargement donnees -> preprocessing -> entrainement -> sauvegarde.

Le fine-tuning de DistilBERT vit dans src/train_bert.py, car il demande torch
et transformers alors que ce module-ci n'a besoin que de scikit-learn.
"""

from typing import Any, List, Tuple

from src.data_loader import load_imdb_dataset, get_sample
from src.preprocessing import preprocess_batch, build_tfidf_vectorizer, transform_texts
from src.models import TFIDFLogisticModel, TFIDFRandomForestModel
from src.config import get_logger

log = get_logger()


def train_classical_models(sample_mode: bool = False) -> Tuple:
    """
    Entraine les deux modeles classiques : LR et Random Forest.
    Avec sample_mode=True on utilise un sous-ensemble pour aller vite.

    Retourne (modele_lr, modele_rf, vectoriseur, X_test, labels_test).
    """
    train_texts, train_labels, test_texts, test_labels = load_imdb_dataset()

    if sample_mode:
        log.info("Mode sample active : utilisation d'un sous-ensemble...")
        train_texts, train_labels, test_texts, test_labels = get_sample(
            train_texts, train_labels, test_texts, test_labels
        )

    log.info("\n[1/3] Nettoyage des textes...")
    clean_train = preprocess_batch(train_texts)
    clean_test = preprocess_batch(test_texts)

    log.info("\n[2/3] Vectorisation TF-IDF...")
    vectorizer = build_tfidf_vectorizer(clean_train)
    X_train = transform_texts(vectorizer, clean_train)
    X_test = transform_texts(vectorizer, clean_test)

    log.info("\n[3/3] Entrainement des modeles classiques...")

    lr_model = TFIDFLogisticModel()
    lr_model.fit(X_train, train_labels)
    lr_model.save()

    rf_model = TFIDFRandomForestModel()
    rf_model.fit(X_train, train_labels)
    rf_model.save()

    return lr_model, rf_model, vectorizer, X_test, test_labels
