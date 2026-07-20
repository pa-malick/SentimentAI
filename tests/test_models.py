"""
Tests pour verifier que les modeles s'initialisent et predisent correctement.
On ne fait pas d'entrainement ici : trop long pour des tests unitaires.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Importer la config en premier : elle pose USE_TF=0, ce qui evite que
# transformers essaie de charger TensorFlow si un jour il est importe.
import src.config  # noqa: F401

# src/models.py n'importe plus torch ni transformers au niveau du module :
# ces dependances ne sont chargees que si on utilise vraiment DistilBERT.
# Ces tests tournent donc meme sans les extras BERT installes.
from src.models import TFIDFLogisticModel, TFIDFRandomForestModel, SklearnTextModel


class TestTFIDFLogisticModel(unittest.TestCase):

    def setUp(self):
        self.model = TFIDFLogisticModel()
        # Donnees factices en format sparse-compatible
        from scipy.sparse import csr_matrix
        self.X = csr_matrix(np.random.rand(100, 50))
        self.y = [0, 1] * 50

    def test_init(self):
        self.assertIsInstance(self.model.model, LogisticRegression)
        self.assertEqual(self.model.name, "tfidf_logistic")

    def test_fit_predict(self):
        self.model.fit(self.X, self.y)
        preds = self.model.predict(self.X)
        self.assertEqual(len(preds), 100)
        self.assertTrue(all(p in [0, 1] for p in preds))

    def test_predict_proba(self):
        self.model.fit(self.X, self.y)
        probs = self.model.predict_proba(self.X)
        self.assertEqual(probs.shape, (100, 2))
        for row in probs:
            self.assertAlmostEqual(sum(row), 1.0, places=5)


class TestTFIDFRandomForestModel(unittest.TestCase):

    def setUp(self):
        self.model = TFIDFRandomForestModel(n_estimators=10)
        from scipy.sparse import csr_matrix
        self.X = csr_matrix(np.random.rand(60, 30))
        self.y = [0, 1] * 30

    def test_init(self):
        self.assertIsInstance(self.model.model, RandomForestClassifier)
        self.assertEqual(self.model.name, "tfidf_random_forest")

    def test_fit_predict(self):
        self.model.fit(self.X, self.y)
        preds = self.model.predict(self.X)
        self.assertEqual(len(preds), 60)


class TestClasseDeBase(unittest.TestCase):
    """La mecanique commune (chemin, save, load) vit dans SklearnTextModel."""

    def test_les_deux_modeles_heritent_de_la_base(self):
        self.assertTrue(issubclass(TFIDFLogisticModel, SklearnTextModel))
        self.assertTrue(issubclass(TFIDFRandomForestModel, SklearnTextModel))

    def test_chemin_derive_du_nom(self):
        self.assertTrue(TFIDFLogisticModel().path.endswith("tfidf_logistic.pkl"))
        self.assertTrue(TFIDFRandomForestModel().path.endswith("tfidf_random_forest.pkl"))

    def test_noms_distincts(self):
        """Deux modeles ne doivent pas ecraser le fichier l'un de l'autre."""
        self.assertNotEqual(TFIDFLogisticModel().path, TFIDFRandomForestModel().path)

    def test_save_puis_load(self):
        """Sauvegarde puis rechargement doivent redonner les memes predictions."""
        import tempfile
        from scipy.sparse import csr_matrix
        from unittest.mock import patch

        X = csr_matrix(np.random.rand(40, 20))
        y = [0, 1] * 20

        modele = TFIDFLogisticModel()
        modele.fit(X, y)
        avant = modele.predict(X)

        # On redirige MODELS_DIR vers un dossier temporaire pour ne pas
        # ecraser les vrais modeles du projet pendant les tests.
        with tempfile.TemporaryDirectory() as dossier:
            with patch("src.models.MODELS_DIR", dossier):
                chemin = modele.save()
                self.assertTrue(os.path.exists(chemin))

                recharge = TFIDFLogisticModel().load()
                apres = recharge.predict(X)

        np.testing.assert_array_equal(avant, apres)

    def test_load_modele_absent_leve_une_erreur_claire(self):
        import tempfile
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as dossier:
            with patch("src.models.MODELS_DIR", dossier):
                with self.assertRaises(FileNotFoundError):
                    TFIDFLogisticModel().load()


class TestImportsParesseux(unittest.TestCase):
    """
    Point important de l'architecture : utiliser les modeles classiques ne doit
    jamais charger torch ni transformers (plusieurs centaines de Mo).
    """

    def test_modeles_classiques_sans_torch(self):
        import subprocess
        code = (
            "import sys; "
            "import src.models, src.train, src.utils; "
            "from app.app import create_app; "
            "create_app(preload=False); "
            "print('torch' in sys.modules, 'transformers' in sys.modules)"
        )
        racine = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        resultat = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, cwd=racine,
        )
        self.assertEqual(resultat.stdout.strip(), "False False", resultat.stderr[-500:])


class TestUtilsPredict(unittest.TestCase):

    def test_check_models_ready_returns_dict(self):
        from src.utils import check_models_ready
        status = check_models_ready()
        self.assertIsInstance(status, dict)
        self.assertIn("logistic", status)
        self.assertIn("distilbert", status)
        self.assertIn("vectorizer", status)


if __name__ == "__main__":
    unittest.main(verbosity=2)
