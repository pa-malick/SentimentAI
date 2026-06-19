"""
Evaluation des modeles avec les metriques classiques en NLP :
accuracy, precision, recall, F1-score, et la matrice de confusion.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)

METRICS_DIR = "metrics"
os.makedirs(METRICS_DIR, exist_ok=True)


def compute_metrics(y_true, y_pred, model_name="model"):
    """
    Calcule et affiche toutes les metriques importantes.
    Retourne un dictionnaire avec les valeurs.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted")
    rec = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")

    print(f"\n{'='*50}")
    print(f"Resultats pour : {model_name}")
    print(f"{'='*50}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"\nRapport complet :")
    print(classification_report(y_true, y_pred, target_names=["Negative", "Positive"]))

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def plot_confusion_matrix(y_true, y_pred, model_name="model"):
    """Genere et sauvegarde la matrice de confusion."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"]
    )
    plt.title(f"Matrice de Confusion - {model_name}", fontsize=14, pad=12)
    plt.xlabel("Prediction")
    plt.ylabel("Vraie valeur")
    plt.tight_layout()
    path = os.path.join(METRICS_DIR, f"confusion_matrix_{model_name}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Matrice de confusion sauvegardee : {path}")
    return path


def compare_models(results_dict):
    """
    Genere un graphique comparatif de toutes les metriques entre les modeles.
    results_dict : {"nom_modele": {"accuracy": x, "f1": x, ...}}
    """
    models = list(results_dict.keys())
    metrics = ["accuracy", "precision", "recall", "f1"]
    x = np.arange(len(metrics))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    for i, model in enumerate(models):
        vals = [results_dict[model].get(m, 0) for m in metrics]
        bars = ax.bar(x + i * width, vals, width, label=model, color=colors[i % len(colors)], alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=8
            )

    ax.set_xlabel("Metriques", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Comparaison des modeles", fontsize=14, pad=12)
    ax.set_xticks(x + width)
    ax.set_xticklabels([m.capitalize() for m in metrics])
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    path = os.path.join(METRICS_DIR, "comparaison_modeles.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Graphique comparatif sauvegarde : {path}")
    return path


def evaluate_classical_model(model, X_test, y_test, model_name):
    """Evalue un modele classique sklearn sur les donnees de test."""
    print(f"\nEvaluation de {model_name}...")
    y_pred = model.predict(X_test)
    metrics = compute_metrics(y_test, y_pred, model_name)
    plot_confusion_matrix(y_test, y_pred, model_name)
    return metrics


def evaluate_bert_model(bert_model, test_texts, test_labels, batch_size=32):
    """
    Evalue DistilBERT sur le dataset de test.
    On fait l'inference par batch pour ne pas saturer la memoire.
    """
    import torch
    from torch.utils.data import DataLoader
    from src.train import IMDBDataset

    print("\nEvaluation de DistilBERT...")
    dataset = IMDBDataset(test_texts, test_labels, bert_model.tokenizer)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_preds = []
    bert_model.model.eval()
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(bert_model.device)
            attention_mask = batch["attention_mask"].to(bert_model.device)
            outputs = bert_model.model(input_ids=input_ids, attention_mask=attention_mask)
            preds = outputs.logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)

    metrics = compute_metrics(test_labels[:len(all_preds)], all_preds, "DistilBERT")
    plot_confusion_matrix(test_labels[:len(all_preds)], all_preds, "DistilBERT")
    return metrics
