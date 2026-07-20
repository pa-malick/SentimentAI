FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Version legere : requirements.txt ne contient ni torch ni transformers.
# L'app tourne avec le vectoriseur TF-IDF et la regression logistique, qui sont
# versionnes dans git. Pour une image avec DistilBERT, voir Dockerfile.bert
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Les stopwords NLTK sont telecharges au build, pas au demarrage :
# sinon le premier lancement echoue si le conteneur n'a pas de reseau.
RUN python -c "import nltk; nltk.download('stopwords')"

COPY . .

RUN mkdir -p models metrics

ENV PYTHONUNBUFFERED=1
ENV FLASK_ENV=production

EXPOSE 5000

# 2 workers suffisent : sans DistilBERT, l'empreinte memoire est faible.
CMD gunicorn --bind 0.0.0.0:${PORT:-5000} --workers 2 --timeout 120 "app.app:create_app()"
