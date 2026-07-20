"""
Tests de l'API Flask.

Le but est de verifier le comportement des routes SANS dependre des modeles
entraines sur le disque (sinon les tests ne tourneraient pas en CI).
Pour les cas nominaux, on injecte un petit modele factice entraine a la volee.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from app.app import create_app
from src.config import MAX_TEXT_LENGTH, MIN_TEXT_LENGTH, MAX_BATCH_SIZE


def build_fake_cache():
    """
    Entraine un mini modele en memoire (quelques phrases) pour simuler
    un modele reel. Ca suffit pour tester le contrat de l'API.
    """
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


class BaseRouteTest(unittest.TestCase):
    """Classe de base : monte une app de test avec les modeles factices."""

    def setUp(self):
        # preload=False : on ne charge pas les vrais modeles du disque
        self.app = create_app(preload=False)
        self.app.config["TESTING"] = True
        self.app.config["MODELS_CACHE"] = build_fake_cache()
        self.client = self.app.test_client()


class TestPagesHTML(BaseRouteTest):

    def test_accueil(self):
        reponse = self.client.get("/")
        self.assertEqual(reponse.status_code, 200)

    def test_about(self):
        reponse = self.client.get("/about")
        self.assertEqual(reponse.status_code, 200)

    def test_accueil_affiche_la_limite(self):
        """La limite serveur doit etre injectee dans le HTML (maxlength)."""
        html = self.client.get("/").get_data(as_text=True)
        self.assertIn(f'maxlength="{MAX_TEXT_LENGTH}"', html)


class TestRoutesInfo(BaseRouteTest):

    def test_status_renvoie_les_cles_attendues(self):
        data = self.client.get("/status").get_json()
        for cle in ["logistic", "random_forest", "distilbert", "vectorizer"]:
            self.assertIn(cle, data)

    def test_metrics_renvoie_du_json(self):
        reponse = self.client.get("/metrics")
        self.assertEqual(reponse.status_code, 200)
        self.assertIsInstance(reponse.get_json(), dict)

    def test_health_structure(self):
        """/health doit toujours repondre avec status, models_ready et models."""
        reponse = self.client.get("/health")
        data = reponse.get_json()
        self.assertIn(reponse.status_code, [200, 503])
        self.assertIn(data["status"], ["ok", "degraded"])
        self.assertIn("models_ready", data)
        self.assertIn("models", data)

    def test_health_coherent_avec_models_ready(self):
        """Si models_ready est faux, le code HTTP doit etre 503 (et non 200)."""
        reponse = self.client.get("/health")
        data = reponse.get_json()
        attendu = 200 if data["models_ready"] else 503
        self.assertEqual(reponse.status_code, attendu)


class TestPredictValidation(BaseRouteTest):
    """Tous ces cas doivent etre rejetes AVANT d'atteindre un modele."""

    def test_texte_trop_court(self):
        reponse = self.client.post("/predict", json={"text": "ab", "model": "logistic"})
        self.assertEqual(reponse.status_code, 400)
        self.assertIn("error", reponse.get_json())

    def test_texte_vide(self):
        reponse = self.client.post("/predict", json={"text": "", "model": "logistic"})
        self.assertEqual(reponse.status_code, 400)

    def test_texte_que_des_espaces(self):
        """Un texte d'espaces doit etre traite comme vide grace au strip()."""
        reponse = self.client.post("/predict", json={"text": "        ", "model": "logistic"})
        self.assertEqual(reponse.status_code, 400)

    def test_texte_trop_long(self):
        trop_long = "a" * (MAX_TEXT_LENGTH + 1)
        reponse = self.client.post("/predict", json={"text": trop_long, "model": "logistic"})
        self.assertEqual(reponse.status_code, 413)

    def test_texte_a_la_limite_exacte_est_accepte(self):
        """Pile MAX_TEXT_LENGTH caracteres doit passer (limite inclusive)."""
        pile = "great movie " * 10
        pile = pile + "a" * (MAX_TEXT_LENGTH - len(pile))
        self.assertEqual(len(pile), MAX_TEXT_LENGTH)
        reponse = self.client.post("/predict", json={"text": pile, "model": "logistic"})
        self.assertEqual(reponse.status_code, 200)

    def test_modele_inconnu_rejete(self):
        """Point important : avant, un modele inconnu retombait silencieusement
        sur la Random Forest au lieu de signaler l'erreur."""
        reponse = self.client.post("/predict", json={"text": "This film was great", "model": "gpt5"})
        self.assertEqual(reponse.status_code, 400)
        self.assertIn("gpt5", reponse.get_json()["error"])

    def test_texte_non_string(self):
        reponse = self.client.post("/predict", json={"text": 12345, "model": "logistic"})
        self.assertEqual(reponse.status_code, 400)

    def test_champ_text_absent(self):
        reponse = self.client.post("/predict", json={"model": "logistic"})
        self.assertEqual(reponse.status_code, 400)

    def test_erreurs_ont_toujours_la_cle_error(self):
        """Le front lit data.error : le format doit etre uniforme partout."""
        cas = [
            {"text": "ab", "model": "logistic"},
            {"text": "a" * (MAX_TEXT_LENGTH + 1), "model": "logistic"},
            {"text": "This film was great", "model": "inconnu"},
            {"text": 42, "model": "logistic"},
        ]
        for payload in cas:
            data = self.client.post("/predict", json=payload).get_json()
            self.assertIn("error", data, f"Pas de cle 'error' pour {payload}")
            self.assertIsInstance(data["error"], str)


class TestPredictNominal(BaseRouteTest):

    def test_prediction_json(self):
        reponse = self.client.post(
            "/predict",
            json={"text": "great wonderful excellent movie", "model": "logistic"}
        )
        self.assertEqual(reponse.status_code, 200)
        data = reponse.get_json()
        for cle in ["sentiment", "confidence", "label", "model"]:
            self.assertIn(cle, data)

    def test_prediction_form_data(self):
        """L'API doit continuer d'accepter le form-data, pas seulement le JSON."""
        reponse = self.client.post(
            "/predict",
            data={"text": "great wonderful excellent movie", "model": "logistic"}
        )
        self.assertEqual(reponse.status_code, 200)

    def test_modele_par_defaut_est_logistic(self):
        """Sans champ 'model', on doit retomber sur logistic."""
        reponse = self.client.post("/predict", json={"text": "great wonderful movie"})
        self.assertEqual(reponse.status_code, 200)
        self.assertEqual(reponse.get_json()["model"], "Logistic Regression")

    def test_confiance_est_un_pourcentage(self):
        data = self.client.post(
            "/predict",
            json={"text": "terrible awful boring movie", "model": "logistic"}
        ).get_json()
        self.assertGreaterEqual(data["confidence"], 0)
        self.assertLessEqual(data["confidence"], 100)

    def test_label_et_sentiment_coherents(self):
        data = self.client.post(
            "/predict",
            json={"text": "great wonderful excellent movie", "model": "logistic"}
        ).get_json()
        self.assertIn(data["label"], [0, 1])
        attendu = "Positif" if data["label"] == 1 else "Négatif"
        self.assertEqual(data["sentiment"], attendu)


class TestPredictBatch(BaseRouteTest):
    """Analyse de plusieurs textes en une requete."""

    def test_lot_nominal(self):
        reponse = self.client.post("/predict/batch", json={
            "texts": [
                "great wonderful excellent movie",
                "terrible awful boring film",
                "brilliant fantastic superb",
            ],
            "model": "logistic",
        })
        self.assertEqual(reponse.status_code, 200)
        data = reponse.get_json()
        self.assertEqual(data["count"], 3)
        self.assertEqual(data["success_count"], 3)
        self.assertEqual(data["error_count"], 0)

    def test_index_conserve_l_ordre(self):
        """Chaque resultat doit pouvoir etre remis en face de son texte d'origine."""
        reponse = self.client.post("/predict/batch", json={
            "texts": ["great movie here", "terrible awful film", "wonderful superb"],
            "model": "logistic",
        })
        indices = [r["index"] for r in reponse.get_json()["results"]]
        self.assertEqual(indices, [0, 1, 2])

    def test_lot_partiellement_invalide(self):
        """Un texte invalide ne doit pas faire echouer tout le lot."""
        reponse = self.client.post("/predict/batch", json={
            "texts": ["great wonderful movie", "ab", "terrible awful film", 123],
            "model": "logistic",
        })
        self.assertEqual(reponse.status_code, 200)
        data = reponse.get_json()
        self.assertEqual(data["success_count"], 2)
        self.assertEqual(data["error_count"], 2)

        resultats = {r["index"]: r for r in data["results"]}
        self.assertIn("error", resultats[1])   # texte trop court
        self.assertIn("error", resultats[3])   # pas une chaine
        self.assertIn("sentiment", resultats[0])
        self.assertIn("sentiment", resultats[2])

    def test_texts_pas_une_liste(self):
        reponse = self.client.post("/predict/batch", json={"texts": "coucou"})
        self.assertEqual(reponse.status_code, 400)

    def test_liste_vide(self):
        reponse = self.client.post("/predict/batch", json={"texts": []})
        self.assertEqual(reponse.status_code, 400)

    def test_champ_texts_absent(self):
        reponse = self.client.post("/predict/batch", json={"model": "logistic"})
        self.assertEqual(reponse.status_code, 400)

    def test_lot_trop_gros(self):
        reponse = self.client.post("/predict/batch", json={
            "texts": ["great movie"] * (MAX_BATCH_SIZE + 1),
            "model": "logistic",
        })
        self.assertEqual(reponse.status_code, 413)

    def test_lot_a_la_taille_max_accepte(self):
        reponse = self.client.post("/predict/batch", json={
            "texts": ["great wonderful movie"] * MAX_BATCH_SIZE,
            "model": "logistic",
        })
        self.assertEqual(reponse.status_code, 200)
        self.assertEqual(reponse.get_json()["count"], MAX_BATCH_SIZE)

    def test_modele_inconnu(self):
        reponse = self.client.post("/predict/batch", json={
            "texts": ["great movie"], "model": "gpt5",
        })
        self.assertEqual(reponse.status_code, 400)

    def test_meme_regles_que_predict_simple(self):
        """
        Les deux routes partagent valider_texte, donc un texte refuse par
        /predict doit aussi etre refuse dans un lot.
        """
        texte_court = "ab"
        simple = self.client.post("/predict", json={"text": texte_court, "model": "logistic"})
        lot = self.client.post("/predict/batch", json={"texts": [texte_court], "model": "logistic"})

        self.assertEqual(simple.status_code, 400)
        self.assertIn("error", lot.get_json()["results"][0])


class TestModelesNonEntraines(unittest.TestCase):
    """Si aucun modele n'est disponible, l'API doit repondre 503, pas planter."""

    def setUp(self):
        self.app = create_app(preload=False)
        self.app.config["TESTING"] = True
        self.app.config["MODELS_CACHE"] = {}   # cache vide
        self.client = self.app.test_client()

    def test_predict_sans_modele_renvoie_503_ou_200(self):
        """503 si rien sur le disque, 200 si les vrais modeles sont la."""
        reponse = self.client.post(
            "/predict",
            json={"text": "This movie was great", "model": "logistic"}
        )
        self.assertIn(reponse.status_code, [200, 503])
        if reponse.status_code == 503:
            self.assertIn("error", reponse.get_json())


if __name__ == "__main__":
    unittest.main(verbosity=2)
