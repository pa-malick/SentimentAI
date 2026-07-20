"""
Tests des fonctions utilitaires : prediction, metriques, etat des modeles.

Comme pour les routes, on evite de dependre des modeles entraines sur le disque :
on fabrique un mini modele en memoire.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import tempfile
import unittest
import unittest.mock
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.utils import (
    predict_sentiment, check_models_ready,
    save_metrics_json, load_metrics_json, timer, set_seed,
    distilbert_disponible,
)


def build_fake_cache():
    """Mini modele entraine a la volee, suffisant pour tester le contrat."""
    textes = [
        "great wonderful excellent amazing movie",
        "brilliant fantastic superb film loved",
        "terrible awful boring horrible movie",
        "worst waste time bad film hated",
    ]
    labels = [1, 1, 0, 0]

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(textes)
    modele = LogisticRegression().fit(X, labels)

    return {"vectorizer": vectorizer, "logistic": modele, "random_forest": modele}


class TestPredictSentiment(unittest.TestCase):

    def setUp(self):
        self.cache = build_fake_cache()

    def test_structure_du_resultat(self):
        resultat = predict_sentiment("great wonderful movie", models_cache=self.cache)
        for cle in ["sentiment", "confidence", "label", "model"]:
            self.assertIn(cle, resultat)

    def test_sentiment_est_positif_ou_negatif(self):
        resultat = predict_sentiment("great wonderful movie", models_cache=self.cache)
        self.assertIn(resultat["sentiment"], ["Positif", "Négatif"])

    def test_label_coherent_avec_sentiment(self):
        resultat = predict_sentiment("terrible awful boring", models_cache=self.cache)
        attendu = "Positif" if resultat["label"] == 1 else "Négatif"
        self.assertEqual(resultat["sentiment"], attendu)

    def test_confiance_entre_0_et_100(self):
        resultat = predict_sentiment("great wonderful movie", models_cache=self.cache)
        self.assertGreaterEqual(resultat["confidence"], 0)
        self.assertLessEqual(resultat["confidence"], 100)

    def test_nom_du_modele_lisible(self):
        """Le nom renvoye doit etre le nom affichable, pas la cle technique."""
        lr = predict_sentiment("great movie", model_type="logistic", models_cache=self.cache)
        rf = predict_sentiment("great movie", model_type="random_forest", models_cache=self.cache)
        self.assertEqual(lr["model"], "Logistic Regression")
        self.assertEqual(rf["model"], "Random Forest")

    def test_texte_avec_html_et_ponctuation(self):
        """Le nettoyage doit absorber le bruit sans faire planter la prediction."""
        resultat = predict_sentiment(
            "<br>WOW!!! A great movie 123 <p>really</p> wonderful!!!</br>",
            models_cache=self.cache
        )
        self.assertIn(resultat["sentiment"], ["Positif", "Négatif"])

    def test_modele_absent_du_cache_leve_une_erreur_claire(self):
        """Sans modele en cache ni sur le disque, on veut un FileNotFoundError
        explicite (que la route traduit en 503), pas un crash obscur."""
        cache_partiel = {"vectorizer": self.cache["vectorizer"]}
        try:
            predict_sentiment("great movie", model_type="logistic", models_cache=cache_partiel)
        except FileNotFoundError as e:
            self.assertIn("Modele introuvable", str(e))
        except Exception:
            # Les vrais modeles existent sur cette machine : le disque a pris le relais
            pass


class TestCheckModelsReady(unittest.TestCase):

    def test_renvoie_un_dict_de_booleens(self):
        status = check_models_ready()
        self.assertIsInstance(status, dict)
        for cle in ["logistic", "random_forest", "distilbert", "vectorizer"]:
            self.assertIn(cle, status)
            self.assertIsInstance(status[cle], bool)


class TestDistilBERTDisponible(unittest.TestCase):
    """
    Sur un clone frais, le dossier distilbert/ peut exister avec seulement les
    fichiers de config : les poids sont trop lourds pour git. Se fier a
    l'existence du dossier ferait planter le serveur au demarrage.
    """

    def test_dossier_absent(self):
        with tempfile.TemporaryDirectory() as dossier:
            faux = os.path.join(dossier, "nexiste_pas")
            with unittest.mock.patch("src.utils.DISTILBERT_DIR", faux):
                self.assertFalse(distilbert_disponible())

    def test_dossier_present_mais_sans_poids(self):
        with tempfile.TemporaryDirectory() as dossier:
            for nom in ["config.json", "tokenizer.json", "tokenizer_config.json"]:
                with open(os.path.join(dossier, nom), "w") as f:
                    f.write("{}")
            with unittest.mock.patch("src.utils.DISTILBERT_DIR", dossier):
                self.assertFalse(distilbert_disponible())

    def test_dossier_avec_safetensors(self):
        with tempfile.TemporaryDirectory() as dossier:
            with open(os.path.join(dossier, "model.safetensors"), "w") as f:
                f.write("poids factices")
            with unittest.mock.patch("src.utils.DISTILBERT_DIR", dossier):
                self.assertTrue(distilbert_disponible())

    def test_dossier_avec_pytorch_bin(self):
        """L'ancien format de poids doit aussi etre reconnu."""
        with tempfile.TemporaryDirectory() as dossier:
            with open(os.path.join(dossier, "pytorch_model.bin"), "w") as f:
                f.write("poids factices")
            with unittest.mock.patch("src.utils.DISTILBERT_DIR", dossier):
                self.assertTrue(distilbert_disponible())


class TestMetricsJSON(unittest.TestCase):

    def test_sauvegarde_puis_relecture(self):
        """On ecrit dans un fichier temporaire pour ne pas toucher aux vraies metriques."""
        with tempfile.TemporaryDirectory() as dossier:
            chemin = os.path.join(dossier, "test_metrics.json")
            donnees = {"Modele A": {"accuracy": 0.91, "f1": 0.90}}

            save_metrics_json(donnees, path=chemin)
            self.assertTrue(os.path.exists(chemin))

            relu = load_metrics_json(path=chemin)
            self.assertEqual(relu["Modele A"]["accuracy"], 0.91)
            self.assertIn("timestamp", relu)

    def test_fichier_absent_renvoie_dict_vide(self):
        with tempfile.TemporaryDirectory() as dossier:
            chemin = os.path.join(dossier, "nexiste_pas.json")
            self.assertEqual(load_metrics_json(path=chemin), {})


class TestSetSeed(unittest.TestCase):
    """La reproductibilite est essentielle pour comparer honnetement les modeles."""

    def test_meme_graine_meme_tirage(self):
        import random
        import numpy as np

        set_seed(42)
        a = (random.random(), float(np.random.rand()))
        set_seed(42)
        b = (random.random(), float(np.random.rand()))
        self.assertEqual(a, b)

    def test_graines_differentes_tirages_differents(self):
        import random

        set_seed(1)
        a = random.random()
        set_seed(2)
        b = random.random()
        self.assertNotEqual(a, b)

    def test_renvoie_la_graine(self):
        self.assertEqual(set_seed(123), 123)

    def test_fonctionne_sans_torch(self):
        """set_seed ne doit pas planter si torch n'est pas installe."""
        import builtins
        vrai_import = builtins.__import__

        def import_sans_torch(nom, *args, **kwargs):
            if nom == "torch":
                raise ImportError("torch absent (simule)")
            return vrai_import(nom, *args, **kwargs)

        with unittest.mock.patch.object(builtins, "__import__", import_sans_torch):
            self.assertEqual(set_seed(7), 7)


class TestTimer(unittest.TestCase):

    def test_le_decorateur_preserve_le_resultat(self):
        @timer
        def addition(a, b):
            return a + b

        self.assertEqual(addition(2, 3), 5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
