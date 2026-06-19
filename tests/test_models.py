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
from src.models import TFIDFLogisticModel, TFIDFRandomForestModel


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
