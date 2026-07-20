<p align="center">
  <img src="app/static/img/logo.svg" alt="SentimentAI" width="90" />
</p>

<h1 align="center">SentimentAI</h1>

<p align="center">
  Analyse de sentiment sur des critiques de films.<br/>
  Comparaison de trois approches d'apprentissage automatique.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/Flask-3.0-000000?logo=flask&logoColor=white" alt="Flask" />
  <img src="https://img.shields.io/badge/scikit--learn-1.3-F7931E?logo=scikitlearn&logoColor=white" alt="scikit-learn" />
  <img src="https://img.shields.io/badge/PyTorch-2.0-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white" alt="Docker" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/tests-91%20passed-ccff00" alt="Tests" />
  <img src="https://img.shields.io/badge/meilleur%20score-89.2%25-ccff00" alt="Accuracy" />
  <img src="https://img.shields.io/badge/dataset-IMDB%2050K-6b6b73" alt="Dataset" />
</p>

---

Projet realise dans le cadre du Master 2 Intelligence Artificielle et Data Science.

## Modeles implementes

Les trois modeles sont entraines sur IMDB puis evalues sur des critiques
jamais vues pendant l'entrainement.

| Modele | Type | Score | Remarque |
|---|---|---|---|
| TF-IDF + Logistic Regression | Classique | **89,2 %** | Rapide, interpretable, le meilleur ici |
| TF-IDF + Random Forest | Ensemble | 85,9 % | Robuste, plus lourd |
| DistilBERT fine-tune | Deep Learning | 87,5 % | Comprend le contexte, sous-entraine |

> **Un resultat contre-intuitif.** La regression logistique devance DistilBERT.
> La comparaison n'est pas equitable : la premiere a ete entrainee sur les
> 25 000 critiques disponibles, le second sur 2 000 seulement, faute de GPU.
> A moyens limites, une methode classique bien reglee reste tres competitive.

## Installation

Les dependances sont separees pour ne pas imposer torch (plusieurs centaines
de Mo) a ceux qui n'utilisent que les modeles classiques.

```bash
# Modeles classiques + interface web (leger)
pip install -r requirements.txt

# Ajouter DistilBERT (torch + transformers)
pip install -r requirements-bert.txt

# Pour developper et lancer les tests
pip install -r requirements-dev.txt
```

## Configuration

Copier `.env.example` en `.env` et ajuster si besoin.

| Variable | Defaut | Role |
|---|---|---|
| `SECRET_KEY` | `dev-secret-change-me` | Cle secrete Flask (a changer en production) |
| `PORT` | `5000` | Port du serveur web |
| `MAX_TEXT_LENGTH` | `5000` | Longueur max d'un texte analyse |
| `MAX_BATCH_SIZE` | `100` | Nombre max de textes par requete batch |

## Entrainement

```bash
# Entrainement complet (modeles classiques)
python main.py

# Mode rapide avec sous-ensemble
python main.py --sample

# Inclure DistilBERT (long, GPU recommande)
python main.py --bert
```

## Interface Web

```bash
python -m app.app
# Ouvrir http://localhost:5000
```

### Routes de l'API

| Route | Methode | Description |
|---|---|---|
| `/` | GET | Page d'analyse |
| `/about` | GET | Presentation du projet |
| `/predict` | POST | Analyse un texte (JSON ou form-data) |
| `/predict/batch` | POST | Analyse plusieurs textes en une requete |
| `/health` | GET | Healthcheck : 200 si operationnel, 503 sinon |
| `/status` | GET | Detail des modeles disponibles |
| `/metrics` | GET | Dernieres metriques d'evaluation |

Exemple d'appel :

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This movie was absolutely brilliant", "model": "distilbert"}'
```

Analyse par lot (jusqu'a 100 textes) :

```bash
curl -X POST http://localhost:5000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Great movie", "Boring and slow"], "model": "logistic"}'
```

Un texte invalide dans un lot n'annule pas les autres : il est renvoye avec sa
propre erreur et son index.

Modeles acceptes : `logistic`, `random_forest`, `distilbert`.

## Docker

```bash
# Lancer l'interface web
docker-compose up

# Lancer l'entrainement en container
docker-compose --profile train up sentiment-train
```

Deux images sont disponibles :

| Fichier | Contenu | Usage |
|---|---|---|
| `Dockerfile` | Sans torch, ~200 Mo | Deploiement en ligne |
| `Dockerfile.bert` | Avec DistilBERT, ~2 Go | Local, machine avec assez de RAM |

## Deploiement

Le site est deploye sur Render a partir de `render.yaml`. Chaque push sur
`main` declenche automatiquement un nouveau deploiement.

Seuls le vectoriseur TF-IDF et la regression logistique sont versionnes
(1.4 Mo au total) : c'est ce qui permet au site de fonctionner des le premier
deploiement. La Random Forest (31 Mo) et DistilBERT (268 Mo) restent en local,
car ils ne tiennent pas dans une offre gratuite.

L'interface s'adapte automatiquement : les modeles absents du serveur sont
grises au lieu de renvoyer une erreur.

## Tests

```bash
python -m pytest tests/ -v

# Avec le taux de couverture
python -m pytest tests/ --cov=src --cov=app
```

Les tests tournent sans avoir besoin des modeles entraines : les cas nominaux
utilisent un mini modele construit en memoire. Ils sont aussi lances
automatiquement par GitHub Actions a chaque push (voir `.github/workflows/ci.yml`).

## Structure

```
src/
  config.py       Configuration centrale (chemins, logging, limites)
  data_loader.py  Chargement du dataset IMDB
  preprocessing.py Nettoyage du texte et vectorisation TF-IDF
  models.py       Les trois modeles (torch importe seulement si besoin)
  train.py        Entrainement des modeles classiques
  train_bert.py   Fine-tuning DistilBERT (necessite les extras BERT)
  evaluation.py   Metriques et graphiques
  utils.py        Prediction et fonctions partagees
app/              Interface web Flask
tests/            Tests unitaires
models/           Modeles sauvegardes
metrics/          Graphiques et rapports
```

## Dataset

IMDB Movie Reviews - 50 000 critiques etiquetees (positif/negatif).
Source : HuggingFace Datasets (`load_dataset("imdb")`)

## Resultats mesures

Evaluation sur les 25 000 critiques de test pour les modeles classiques,
et sur 2 000 critiques tirees aleatoirement pour DistilBERT.

| Modele | Accuracy | F1-score | AUC |
|---|---|---|---|
| Logistic Regression | 89,22 % | 89,22 % | 0,959 |
| DistilBERT | 87,50 % | 87,49 % | 0,948 |
| Random Forest | 85,89 % | 85,89 % | 0,937 |

Chaque evaluation produit dans `metrics/` : matrice de confusion, courbe ROC
avec AUC, graphique comparatif, et un JSON contenant les metriques globales
ainsi que le detail par classe.

Les graines aleatoires sont fixees (`RANDOM_SEED = 42`), donc deux entrainements
identiques donnent les memes scores.

## Rapport

Un rapport de synthese au format Word est genere a partir des metriques
courantes :

```bash
python generate_rapport.py
```
