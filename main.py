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

from src.data_loader import load_imdb_dataset, get_sample, get_dataset_info, stratified_sample
from src.preprocessing import preprocess_batch, build_tfidf_vectorizer, transform_texts, load_vectorizer
from src.models import TFIDFLogisticModel, TFIDFRandomForestModel
from src.train import train_classical_models
from src.evaluation import evaluate_classical_model, evaluate_bert_model, compare_models
from src.utils import save_metrics_json, check_models_ready, set_seed
from src.config import get_logger

log = get_logger()


# Nombre de critiques utilisees pour evaluer DistilBERT.
# Sur CPU, l'inference sur les 25 000 du test prendrait plusieurs heures.
BERT_EVAL_SIZE = 2000


def parse_args():
    parser = argparse.ArgumentParser(description="Analyse de Sentiment IMDB")
    parser.add_argument("--sample", action="store_true", help="Utilise un sous-ensemble des donnees")
    parser.add_argument("--eval-only", action="store_true", help="Evaluation sans reentrainement")
    parser.add_argument("--bert", action="store_true", help="Inclut l'entrainement DistilBERT")
    return parser.parse_args()


def main():
    args = parse_args()

    log.info("\n" + "="*60)
    log.info("  SENTIMENT ANALYSIS - NLP Project")
    log.info("  Dataset : IMDB Movie Reviews")
    log.info("="*60 + "\n")

    # Reproductibilite : deux runs identiques doivent donner les memes scores
    set_seed()

    status = check_models_ready()

    if not args.eval_only:
        log.info("[ETAPE 1] Entrainement des modeles classiques\n")
        lr_model, rf_model, vectorizer, X_test, y_test = train_classical_models(
            sample_mode=args.sample
        )

        if args.bert:
            log.info("\n[ETAPE 2] Fine-tuning DistilBERT\n")
            # Import ici et pas en haut du fichier : sans --bert, on n'a pas
            # besoin de torch ni de transformers.
            from src.train_bert import train_distilbert
            bert_model = train_distilbert(epochs=3, sample_mode=args.sample)
    else:
        log.info("Mode eval-only : chargement des modeles existants...")
        if not (status["logistic"] and status["vectorizer"]):
            log.error("Erreur : modeles introuvables. Lance d'abord l'entrainement.")
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

    log.info("\n[ETAPE FINALE] Evaluation et comparaison des modeles\n")

    all_results = {}

    lr_metrics = evaluate_classical_model(lr_model, X_test, y_test, "Logistic_Regression")
    all_results["Logistic Regression"] = lr_metrics

    rf_metrics = evaluate_classical_model(rf_model, X_test, y_test, "Random_Forest")
    all_results["Random Forest"] = rf_metrics

    if args.bert:
        # En mode eval-only, on recharge le DistilBERT deja fine-tune au lieu
        # de le reentrainer : c'est le cas d'usage "je veux juste ses metriques".
        if args.eval_only:
            if not status["distilbert"]:
                log.error("DistilBERT introuvable. Lance d'abord : python main.py --bert")
                sys.exit(1)
            log.info("\nChargement du DistilBERT deja entraine...")
            from src.models import DistilBERTClassifier
            bert_model = DistilBERTClassifier.from_saved()

        # On evalue BERT sur un sous-ensemble stratifie du test : l'inference
        # sur les 25 000 critiques prendrait des heures sur CPU.
        _, _, raw_test_texts, raw_test_labels = load_imdb_dataset()
        bert_test_texts, bert_test_labels = stratified_sample(
            raw_test_texts, raw_test_labels, BERT_EVAL_SIZE
        )
        log.info(f"Evaluation DistilBERT sur {len(bert_test_texts)} critiques de test")
        bert_metrics = evaluate_bert_model(bert_model, bert_test_texts, bert_test_labels)
        all_results["DistilBERT"] = bert_metrics

    compare_models(all_results)
    save_metrics_json(all_results)

    log.info("\n" + "="*60)
    log.info("  Entrainement et evaluation termines !")
    log.info("  Les graphiques sont dans le dossier metrics/")
    log.info("  Pour lancer l'interface web : python -m app.app")
    log.info("="*60 + "\n")


if __name__ == "__main__":
    main()
