"""
Tests de l'echantillonnage des donnees.

On ne teste pas le telechargement du dataset IMDB (trop lourd et dependant du
reseau) : on se concentre sur la logique de sous-echantillonnage, qui est la
partie ou l'on peut vraiment se tromper.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from collections import Counter

from src.data_loader import stratified_sample, get_sample, to_dataframe


def jeu_de_donnees(n=1000):
    """Dataset factice trie par label, comme l'est le vrai IMDB."""
    textes = [f"critique numero {i}" for i in range(n)]
    labels = [0] * (n // 2) + [1] * (n // 2)
    return textes, labels


class TestStratifiedSample(unittest.TestCase):

    def setUp(self):
        self.textes, self.labels = jeu_de_donnees()

    def test_taille_demandee(self):
        textes, labels = stratified_sample(self.textes, self.labels, 100)
        self.assertEqual(len(textes), 100)
        self.assertEqual(len(labels), 100)

    def test_classes_equilibrees(self):
        """Le tirage doit conserver la proportion de chaque classe."""
        _, labels = stratified_sample(self.textes, self.labels, 100)
        repartition = Counter(labels)
        self.assertEqual(repartition[0], 50)
        self.assertEqual(repartition[1], 50)

    def test_reproductible(self):
        """Meme graine, meme echantillon : indispensable pour comparer des runs."""
        a, _ = stratified_sample(self.textes, self.labels, 50)
        b, _ = stratified_sample(self.textes, self.labels, 50)
        self.assertEqual(a, b)

    def test_graines_differentes_donnent_echantillons_differents(self):
        a, _ = stratified_sample(self.textes, self.labels, 50, seed=1)
        b, _ = stratified_sample(self.textes, self.labels, 50, seed=2)
        self.assertNotEqual(a, b)

    def test_ne_prend_pas_que_les_premiers(self):
        """
        L'ancienne version prenait les n/2 premiers de chaque classe, donc
        toujours les memes critiques. On verifie que ce n'est plus le cas.
        """
        textes, _ = stratified_sample(self.textes, self.labels, 100)
        premiers = [f"critique numero {i}" for i in range(50)]
        self.assertNotEqual(textes[:50], premiers)

    def test_n_superieur_au_dataset_renvoie_tout(self):
        textes, labels = stratified_sample(self.textes, self.labels, 99999)
        self.assertEqual(len(textes), len(self.textes))

    def test_textes_et_labels_restent_alignes(self):
        """
        Piege classique : melanger textes et labels separement casserait
        la correspondance. Ici le label doit rester celui du bon texte.
        """
        textes = [f"texte_{i}" for i in range(100)]
        labels = [i % 2 for i in range(100)]
        correspondance = dict(zip(textes, labels))

        sous_textes, sous_labels = stratified_sample(textes, labels, 40)
        for texte, label in zip(sous_textes, sous_labels):
            self.assertEqual(correspondance[texte], label)


class TestGetSample(unittest.TestCase):

    def test_retourne_quatre_listes(self):
        textes, labels = jeu_de_donnees()
        resultat = get_sample(textes, labels, textes, labels, n_train=100, n_test=40)
        self.assertEqual(len(resultat), 4)
        tr_t, tr_l, te_t, te_l = resultat
        self.assertEqual(len(tr_t), 100)
        self.assertEqual(len(te_t), 40)


class TestToDataFrame(unittest.TestCase):

    def test_colonnes_et_mapping(self):
        df = to_dataframe(["bon film", "mauvais film"], [1, 0])
        for colonne in ["text", "label", "sentiment"]:
            self.assertIn(colonne, df.columns)
        self.assertEqual(df.loc[0, "sentiment"], "positive")
        self.assertEqual(df.loc[1, "sentiment"], "negative")


if __name__ == "__main__":
    unittest.main(verbosity=2)
