"""
Fine-tuning de DistilBERT.

Ce module est separe de src/train.py exprès : il a besoin de torch et de
transformers, alors que l'entrainement des modeles classiques n'en a pas besoin.
On peut donc utiliser tout le reste du projet sans installer les grosses
dependances deep learning.
"""

from typing import List

from src.data_loader import load_imdb_dataset, get_sample
from src.models import DistilBERTClassifier
from src.utils import set_seed
from src.config import get_logger

log = get_logger()


def _torch_imports():
    """Importe les briques PyTorch necessaires, avec un message clair si absentes."""
    try:
        import torch
        from torch.utils.data import Dataset, DataLoader
        from torch.optim import AdamW
        from transformers import get_linear_schedule_with_warmup
        from tqdm import tqdm
    except ImportError as e:
        raise ImportError(
            "L'entrainement de DistilBERT necessite torch et transformers. "
            "Installe-les avec : pip install -r requirements-bert.txt"
        ) from e
    return torch, Dataset, DataLoader, AdamW, get_linear_schedule_with_warmup, tqdm


def build_dataset_class():
    """
    Construit la classe Dataset PyTorch.

    On la fabrique dans une fonction car elle herite de torch.utils.data.Dataset,
    donc on ne peut pas la definir tant que torch n'est pas importe.
    """
    torch, Dataset, _, _, _, _ = _torch_imports()

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
                return_tensors="pt",
            )
            return {
                "input_ids": encoding["input_ids"].squeeze(),
                "attention_mask": encoding["attention_mask"].squeeze(),
                "label": torch.tensor(self.labels[idx], dtype=torch.long),
            }

    return IMDBDataset


def train_distilbert(
    epochs: int = 3,
    batch_size: int = 16,
    lr: float = 2e-5,
    sample_mode: bool = False,
) -> DistilBERTClassifier:
    """
    Fine-tuning de DistilBERT sur le dataset IMDB.
    On utilise un scheduler avec warmup pour stabiliser l'entrainement.
    """
    torch, _, DataLoader, AdamW, get_linear_schedule_with_warmup, tqdm = _torch_imports()
    IMDBDataset = build_dataset_class()

    # Fixe les graines torch aussi : sans ca, le melange des batchs et
    # l'initialisation de la couche de classification varient a chaque run.
    set_seed()

    train_texts, train_labels, test_texts, test_labels = load_imdb_dataset()

    if sample_mode:
        train_texts, train_labels, _, _ = get_sample(
            train_texts, train_labels, test_texts, test_labels,
            n_train=2000, n_test=500,
        )
        log.info(f"Mode sample : {len(train_texts)} exemples d'entrainement")

    bert_model = DistilBERTClassifier()

    dataset = IMDBDataset(train_texts, train_labels, bert_model.tokenizer)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    optimizer = AdamW(bert_model.model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps,
    )

    bert_model.model.train()
    log.info(f"\nFine-tuning DistilBERT sur {epochs} epochs...")

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
                labels=labels,
            )
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(bert_model.model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            progress.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(loader)
        log.info(f"Epoch {epoch + 1} terminee. Loss moyenne : {avg_loss:.4f}")

    bert_model.save()
    return bert_model
