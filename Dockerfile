FROM python:3.10-slim

WORKDIR /app

# Pas de gcc/g++ : scikit-learn, numpy et scipy fournissent des paquets
# precompiles pour Linux. Installer un compilateur ajouterait ~300 Mo pour rien.

# Image de service : requirements-web.txt ne contient que le strict necessaire
# pour repondre aux requetes (ni pandas, ni matplotlib, ni datasets).
# L'app tourne avec le vectoriseur TF-IDF et la regression logistique, tous deux
# versionnes dans git. Pour une image avec DistilBERT, voir Dockerfile.bert
COPY requirements-web.txt .
RUN pip install --no-cache-dir -r requirements-web.txt

# Les mots vides NLTK sont telecharges au build, pas au demarrage :
# sinon le premier lancement echoue si le conteneur n'a pas de reseau.
RUN python -c "import nltk; nltk.download('stopwords')"

COPY . .

RUN mkdir -p models metrics

ENV PYTHONUNBUFFERED=1
ENV FLASK_ENV=production

EXPOSE 5000

# Forme exec avec sh -c : permet d'utiliser $PORT (impose par Render)
# tout en gerant correctement les signaux d'arret.
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-5000} --workers 2 --timeout 120 'app.app:create_app()'"]
