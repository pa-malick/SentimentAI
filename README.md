
# SentimentAI 2050 - Analyse de Sentiment NLP

Projet complet d'analyse de sentiment sur le dataset IMDB.
Realise dans le cadre du Master 2 Intelligence Artificielle et Data Science.

## Modeles implementes

| Modele | Type | Avantage |
|---|---|---|
| TF-IDF + Logistic Regression | Classique | Rapide, interpretable |
| TF-IDF + Random Forest | Ensemble | Robuste, stable |
| DistilBERT fine-tune | Deep Learning | Precis, etat de l'art |

## Installation

```bash
pip install -r requirements.txt
```

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

## Docker

```bash
# Lancer l'interface web
docker-compose up

# Lancer l'entrainement en container
docker-compose --profile train up sentiment-train
```

## Tests

```bash
python -m pytest tests/ -v
```

## Structure

```
src/            Code metier (chargement, preprocessing, modeles, evaluation)
app/            Interface web Flask
tests/          Tests unitaires
models/         Modeles sauvegardes
metrics/        Graphiques et rapports
```

## Dataset

IMDB Movie Reviews - 50 000 critiques etiquetees (positif/negatif).
Source : HuggingFace Datasets (`load_dataset("imdb")`)

## Resultats typiques

- Logistic Regression : ~89-91% accuracy
- Random Forest : ~84-87% accuracy
- DistilBERT fine-tune : ~92-94% accuracy

## Vision Senegal 2050

Ce projet s'inscrit dans la dynamique de formation en IA au Senegal,
contribuant a l'objectif de souverainete numerique et technologique du continent.
