"""
Point d'entree principal du projet.
Lance l'entrainement des modeles et l'evaluation complete.

Usage :
  python main.py                  -> entrainement complet
  python main.py --sample         -> mode rapide avec sous-ensemble
  python main.py --eval-only      -> evaluation sans reentrainement
  python main.py --bert           -> inclut DistilBERT (long)
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_imdb_dataset, get_sample, get_dataset_info
from src.preprocessing import preprocess_batch, build_tfidf_vectorizer, transform_texts, load_vectorizer
from src.models import TFIDFLogisticModel, TFIDFRandomForestModel
from src.train import train_classical_models, train_distilbert
from src.evaluation import evaluate_classical_model, evaluate_bert_model, compare_models
from src.utils import save_metrics_json, check_models_ready


def parse_args():
    parser = argparse.ArgumentParser(description="Analyse de Sentiment IMDB")
    parser.add_argument("--sample", action="store_true", help="Utilise un sous-ensemble des donnees")
    parser.add_argument("--eval-only", action="store_true", help="Evaluation sans reentrainement")
    parser.add_argument("--bert", action="store_true", help="Inclut l'entrainement DistilBERT")
    return parser.parse_args()


def main():
    args = parse_args()

    print("\n" + "="*60)
    print("  SENTIMENT ANALYSIS - NLP Project")
    print("  Dataset : IMDB Movie Reviews")
    print("="*60 + "\n")

    status = check_models_ready()

    if not args.eval_only:
        print("[ETAPE 1] Entrainement des modeles classiques\n")
        lr_model, rf_model, vectorizer, X_test, y_test = train_classical_models(
            sample_mode=args.sample
        )

        if args.bert:
            print("\n[ETAPE 2] Fine-tuning DistilBERT\n")
            bert_model = train_distilbert(epochs=3, sample_mode=args.sample)
    else:
        print("Mode eval-only : chargement des modeles existants...")
        if not (status["logistic"] and status["vectorizer"]):
            print("Erreur : modeles introuvables. Lance d'abord l'entrainement.")
            sys.exit(1)

        train_texts, train_labels, test_texts, test_labels = load_imdb_dataset()
        if args.sample:
            _, _, test_texts, test_labels = get_sample(
                train_texts, train_labels, test_texts, test_labels
            )

        clean_test = preprocess_batch(test_texts, verbose=False)
        vectorizer = load_vectorizer()
        X_test = transform_texts(vectorizer, clean_test)
        y_test = test_labels

        lr_model = TFIDFLogisticModel().load()
        rf_model = TFIDFRandomForestModel().load()

    print("\n[ETAPE FINALE] Evaluation et comparaison des modeles\n")

    all_results = {}

    lr_metrics = evaluate_classical_model(lr_model, X_test, y_test, "Logistic_Regression")
    all_results["Logistic Regression"] = lr_metrics

    rf_metrics = evaluate_classical_model(rf_model, X_test, y_test, "Random_Forest")
    all_results["Random Forest"] = rf_metrics

    if args.bert and not args.eval_only:
        _, _, raw_test_texts, raw_test_labels = load_imdb_dataset()
        pos = [(t, l) for t, l in zip(raw_test_texts, raw_test_labels) if l == 1][:500]
        neg = [(t, l) for t, l in zip(raw_test_texts, raw_test_labels) if l == 0][:500]
        bert_test_texts = [x[0] for x in neg + pos]
        bert_test_labels = [x[1] for x in neg + pos]
        bert_metrics = evaluate_bert_model(bert_model, bert_test_texts, bert_test_labels)
        all_results["DistilBERT"] = bert_metrics

    compare_models(all_results)
    save_metrics_json(all_results)

    print("\n" + "="*60)
    print("  Entrainement et evaluation termines !")
    print("  Les graphiques sont dans le dossier metrics/")
    print("  Pour lancer l'interface web : python -m app.app")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
