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
    f1_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve
)

from src.config import METRICS_DIR, get_logger

log = get_logger()
os.makedirs(METRICS_DIR, exist_ok=True)


def compute_metrics(y_true, y_pred, model_name="model", y_proba=None):
    """
    Calcule et affiche toutes les metriques importantes.

    y_proba (optionnel) : probabilites de la classe positive. Si on les fournit,
    on calcule en plus le ROC AUC, qui mesure la qualite du classement des
    predictions independamment du seuil de decision (0.5 par defaut).

    Retourne un dictionnaire avec les valeurs.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted")
    rec = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")

    resultats = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

    log.info(f"\n{'='*50}")
    log.info(f"Resultats pour : {model_name}")
    log.info(f"{'='*50}")
    log.info(f"  Accuracy  : {acc:.4f}")
    log.info(f"  Precision : {prec:.4f}")
    log.info(f"  Recall    : {rec:.4f}")
    log.info(f"  F1-Score  : {f1:.4f}")

    if y_proba is not None:
        auc = roc_auc_score(y_true, y_proba)
        resultats["roc_auc"] = auc
        log.info(f"  ROC AUC   : {auc:.4f}")

    # Detail par classe : utile pour reperer si le modele se trompe surtout
    # sur les critiques negatives ou surtout sur les positives.
    prec_classe = precision_score(y_true, y_pred, average=None)
    rec_classe = recall_score(y_true, y_pred, average=None)
    f1_classe = f1_score(y_true, y_pred, average=None)

    resultats["par_classe"] = {
        "Negative": {
            "precision": float(prec_classe[0]),
            "recall": float(rec_classe[0]),
            "f1": float(f1_classe[0]),
        },
        "Positive": {
            "precision": float(prec_classe[1]),
            "recall": float(rec_classe[1]),
            "f1": float(f1_classe[1]),
        },
    }

    log.info("\nRapport complet :")
    log.info(classification_report(y_true, y_pred, target_names=["Negative", "Positive"]))

    return resultats


def plot_roc_curve(y_true, y_proba, model_name="model"):
    """
    Trace la courbe ROC : taux de vrais positifs en fonction des faux positifs.
    Plus la courbe monte vite vers le coin haut-gauche, meilleur est le modele.
    La diagonale represente un modele qui repondrait au hasard.
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f"{model_name} (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.6, label="Hasard (AUC = 0.5)")
    plt.xlabel("Taux de faux positifs")
    plt.ylabel("Taux de vrais positifs")
    plt.title(f"Courbe ROC - {model_name}", fontsize=14, pad=12)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    path = os.path.join(METRICS_DIR, f"roc_curve_{model_name}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    log.info(f"Courbe ROC sauvegardee : {path}")
    return path


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
    log.info(f"Matrice de confusion sauvegardee : {path}")
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
    log.info(f"Graphique comparatif sauvegarde : {path}")
    return path


def evaluate_classical_model(model, X_test, y_test, model_name):
    """Evalue un modele classique sklearn sur les donnees de test."""
    log.info(f"\nEvaluation de {model_name}...")
    y_pred = model.predict(X_test)

    # Probabilites de la classe positive, pour le ROC AUC et la courbe ROC
    y_proba = None
    try:
        y_proba = model.predict_proba(X_test)[:, 1]
    except (AttributeError, NotImplementedError):
        log.info("  (pas de predict_proba : ROC AUC non calcule)")

    metrics = compute_metrics(y_test, y_pred, model_name, y_proba=y_proba)
    plot_confusion_matrix(y_test, y_pred, model_name)
    if y_proba is not None:
        plot_roc_curve(y_test, y_proba, model_name)
    return metrics


def evaluate_bert_model(bert_model, test_texts, test_labels, batch_size=32):
    """
    Evalue DistilBERT sur le dataset de test.
    On fait l'inference par batch pour ne pas saturer la memoire.
    """
    import torch
    from torch.utils.data import DataLoader
    from src.train_bert import build_dataset_class

    IMDBDataset = build_dataset_class()

    log.info("\nEvaluation de DistilBERT...")
    dataset = IMDBDataset(test_texts, test_labels, bert_model.tokenizer)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_preds = []
    all_probas = []
    bert_model.model.eval()
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(bert_model.device)
            attention_mask = batch["attention_mask"].to(bert_model.device)
            outputs = bert_model.model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1)
            all_preds.extend(probs.argmax(dim=1).cpu().numpy())
            # Probabilite de la classe positive, pour le ROC AUC
            all_probas.extend(probs[:, 1].cpu().numpy())

    y_true = test_labels[:len(all_preds)]
    metrics = compute_metrics(y_true, all_preds, "DistilBERT", y_proba=all_probas)
    plot_confusion_matrix(y_true, all_preds, "DistilBERT")
    plot_roc_curve(y_true, all_probas, "DistilBERT")
    return metrics
