"""
Tests des metriques d'evaluation.

On travaille sur de petits vecteurs de predictions calcules a la main, pour
pouvoir verifier les valeurs attendues sans lancer le moindre entrainement.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tempfile
import unittest
from unittest.mock import patch

from src.evaluation import compute_metrics, plot_confusion_matrix, plot_roc_curve


class TestComputeMetrics(unittest.TestCase):

    def setUp(self):
        self.y_true = [0, 0, 0, 0, 1, 1, 1, 1]
        self.y_pred = [0, 0, 0, 1, 1, 1, 1, 0]
        self.y_proba = [0.1, 0.2, 0.3, 0.6, 0.9, 0.8, 0.7, 0.4]

    def test_metriques_de_base_presentes(self):
        m = compute_metrics(self.y_true, self.y_pred, "Test")
        for cle in ["accuracy", "precision", "recall", "f1"]:
            self.assertIn(cle, m)

    def test_accuracy_correcte(self):
        """6 bonnes reponses sur 8 = 0.75."""
        m = compute_metrics(self.y_true, self.y_pred, "Test")
        self.assertAlmostEqual(m["accuracy"], 0.75, places=4)

    def test_prediction_parfaite(self):
        m = compute_metrics(self.y_true, self.y_true, "Parfait")
        self.assertAlmostEqual(m["accuracy"], 1.0)
        self.assertAlmostEqual(m["f1"], 1.0)

    def test_roc_auc_ajoute_si_probas_fournies(self):
        m = compute_metrics(self.y_true, self.y_pred, "Test", y_proba=self.y_proba)
        self.assertIn("roc_auc", m)
        self.assertGreaterEqual(m["roc_auc"], 0.0)
        self.assertLessEqual(m["roc_auc"], 1.0)

    def test_roc_auc_absent_sans_probas(self):
        """Sans probabilites, on ne doit pas inventer de ROC AUC."""
        m = compute_metrics(self.y_true, self.y_pred, "Test")
        self.assertNotIn("roc_auc", m)

    def test_detail_par_classe(self):
        m = compute_metrics(self.y_true, self.y_pred, "Test")
        self.assertIn("par_classe", m)
        for classe in ["Negative", "Positive"]:
            self.assertIn(classe, m["par_classe"])
            for cle in ["precision", "recall", "f1"]:
                self.assertIn(cle, m["par_classe"][classe])

    def test_metriques_entre_0_et_1(self):
        m = compute_metrics(self.y_true, self.y_pred, "Test", y_proba=self.y_proba)
        for cle in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            self.assertGreaterEqual(m[cle], 0.0)
            self.assertLessEqual(m[cle], 1.0)


class TestGraphiques(unittest.TestCase):
    """Les graphiques sont ecrits dans un dossier temporaire pendant les tests."""

    def setUp(self):
        self.y_true = [0, 0, 1, 1]
        self.y_pred = [0, 1, 1, 1]
        self.y_proba = [0.2, 0.6, 0.8, 0.9]

    def test_matrice_de_confusion_creee(self):
        with tempfile.TemporaryDirectory() as dossier:
            with patch("src.evaluation.METRICS_DIR", dossier):
                chemin = plot_confusion_matrix(self.y_true, self.y_pred, "Test")
                self.assertTrue(os.path.exists(chemin))

    def test_courbe_roc_creee(self):
        with tempfile.TemporaryDirectory() as dossier:
            with patch("src.evaluation.METRICS_DIR", dossier):
                chemin = plot_roc_curve(self.y_true, self.y_proba, "Test")
                self.assertTrue(os.path.exists(chemin))
                self.assertTrue(chemin.endswith(".png"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
