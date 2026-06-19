"""
Fonctions d'entrainement pour chaque type de modele.
Le pipeline est : chargement donnees -> preprocessing -> entrainement -> sauvegarde.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

from src.data_loader import load_imdb_dataset, get_sample
from src.preprocessing import preprocess_batch, build_tfidf_vectorizer, transform_texts
from src.models import TFIDFLogisticModel, TFIDFRandomForestModel, DistilBERTClassifier


class IMDBDataset(Dataset):
    """Dataset PyTorch pour DistilBERT."""

    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }


def train_classical_models(sample_mode=False):
    """
    Entraine les deux modeles classiques : LR et Random Forest.
    Avec sample_mode=True on utilise un sous-ensemble pour aller vite.
    """
    train_texts, train_labels, test_texts, test_labels = load_imdb_dataset()

    if sample_mode:
        print("Mode sample active : utilisation d'un sous-ensemble...")
        from src.data_loader import get_sample
        train_texts, train_labels, test_texts, test_labels = get_sample(
            train_texts, train_labels, test_texts, test_labels
        )

    print("\n[1/3] Nettoyage des textes...")
    clean_train = preprocess_batch(train_texts)
    clean_test = preprocess_batch(test_texts)

    print("\n[2/3] Vectorisation TF-IDF...")
    vectorizer = build_tfidf_vectorizer(clean_train)
    X_train = transform_texts(vectorizer, clean_train)
    X_test = transform_texts(vectorizer, clean_test)

    print("\n[3/3] Entrainement des modeles classiques...")

    lr_model = TFIDFLogisticModel()
    lr_model.fit(X_train, train_labels)
    lr_model.save()

    rf_model = TFIDFRandomForestModel()
    rf_model.fit(X_train, train_labels)
    rf_model.save()

    return lr_model, rf_model, vectorizer, X_test, test_labels


def train_distilbert(epochs=3, batch_size=16, lr=2e-5, sample_mode=False):
    """
    Fine-tuning de DistilBERT sur le dataset IMDB.
    On utilise un scheduler avec warmup pour stabiliser l'entrainement.
    """
    train_texts, train_labels, test_texts, test_labels = load_imdb_dataset()

    if sample_mode:
        from src.data_loader import get_sample
        train_texts, train_labels, _, _ = get_sample(
            train_texts, train_labels, test_texts, test_labels, n_train=2000, n_test=500
        )
        print(f"Mode sample : {len(train_texts)} exemples d'entrainement")

    bert_model = DistilBERTClassifier()

    dataset = IMDBDataset(train_texts, train_labels, bert_model.tokenizer)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    optimizer = AdamW(bert_model.model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps
    )

    bert_model.model.train()
    print(f"\nFine-tuning DistilBERT sur {epochs} epochs...")

    for epoch in range(epochs):
        total_loss = 0
        progress = tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for batch in progress:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(bert_model.device)
            attention_mask = batch["attention_mask"].to(bert_model.device)
            labels = batch["label"].to(bert_model.device)

            outputs = bert_model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(bert_model.model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            progress.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch + 1} terminee. Loss moyenne : {avg_loss:.4f}")

    bert_model.save()
    return bert_model
