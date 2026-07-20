"""
Routes de l'application Flask.
On separe les routes du fichier principal pour garder le code propre.
"""

from flask import Blueprint, render_template, request, jsonify, current_app

from src.utils import predict_sentiment, check_models_ready, load_metrics_json
from src.config import (
    MIN_TEXT_LENGTH, MAX_TEXT_LENGTH, MAX_BATCH_SIZE, VALID_MODELS, get_logger,
)

log = get_logger()
main_bp = Blueprint("main", __name__)


def error(message, status):
    """
    Petit helper pour que TOUTES les erreurs de l'API aient la meme forme.
    Le front lit simplement data.error, donc on garde cette cle partout.
    """
    return jsonify({"error": message}), status


def valider_texte(texte):
    """
    Verifie qu'un texte est analysable.
    Retourne (texte_nettoye, message_erreur). Un des deux vaut None.

    Cette fonction est partagee par /predict et /predict/batch pour que les
    deux routes appliquent exactement les memes regles.
    """
    if not isinstance(texte, str):
        return None, "Le champ 'text' doit etre une chaine de caracteres."

    texte = texte.strip()

    if len(texte) < MIN_TEXT_LENGTH:
        return None, f"Texte trop court. Entre au moins {MIN_TEXT_LENGTH} caracteres."

    if len(texte) > MAX_TEXT_LENGTH:
        return None, (
            f"Texte trop long ({len(texte)} caracteres). "
            f"Maximum autorise : {MAX_TEXT_LENGTH}."
        )

    return texte, None


def extract_payload():
    """
    Recupere le texte et le modele demandes, que la requete soit en JSON
    ou en form-data (le formulaire HTML classique).
    """
    if request.is_json:
        data = request.get_json(silent=True) or {}
        return data.get("text", ""), data.get("model", "logistic")
    return request.form.get("text", ""), request.form.get("model", "logistic")


@main_bp.route("/")
def index():
    """Page d'accueil avec le formulaire d'analyse."""
    # On passe la limite au template pour que le compteur et le maxlength
    # du textarea restent alignes avec la validation cote serveur.
    return render_template("index.html", max_length=MAX_TEXT_LENGTH)


@main_bp.route("/about")
def about():
    """Page de presentation du projet et de l'equipe."""
    return render_template("about.html")


@main_bp.route("/predict", methods=["POST"])
def predict():
    """
    Route principale d'API : recoit un texte et retourne le sentiment predit.
    Accepte JSON ou form-data.
    """
    text, model_type = extract_payload()

    text, souci = valider_texte(text)
    if souci:
        # 413 pour un texte trop long, 400 pour le reste
        return error(souci, 413 if "trop long" in souci else 400)

    if model_type not in VALID_MODELS:
        return error(
            f"Modele inconnu : '{model_type}'. Choix possibles : {', '.join(VALID_MODELS)}.",
            400
        )

    try:
        cache = current_app.config.get("MODELS_CACHE", {})
        result = predict_sentiment(text, model_type=model_type, models_cache=cache)
        return jsonify(result)
    except FileNotFoundError as e:
        # Le modele demande n'a pas encore ete entraine
        return error(str(e), 503)
    except Exception as e:
        # On loggue la vraie erreur cote serveur mais on reste sobre cote client
        log.exception("Erreur pendant la prediction")
        return error(f"Erreur interne : {e}", 500)


@main_bp.route("/predict/batch", methods=["POST"])
def predict_batch():
    """
    Analyse plusieurs textes en une seule requete.

    Corps attendu : {"texts": ["...", "..."], "model": "logistic"}

    Un texte invalide n'annule pas tout le lot : il est renvoye avec sa propre
    erreur et son index, et les autres sont analyses normalement. C'est plus
    pratique quand on traite un fichier de critiques ou une ou deux lignes
    sont vides.
    """
    data = request.get_json(silent=True) or {}
    textes = data.get("texts")
    model_type = data.get("model", "logistic")

    if not isinstance(textes, list):
        return error("Le champ 'texts' doit etre une liste de textes.", 400)

    if len(textes) == 0:
        return error("La liste 'texts' est vide.", 400)

    if len(textes) > MAX_BATCH_SIZE:
        return error(
            f"Trop de textes ({len(textes)}). Maximum par requete : {MAX_BATCH_SIZE}.",
            413
        )

    if model_type not in VALID_MODELS:
        return error(
            f"Modele inconnu : '{model_type}'. Choix possibles : {', '.join(VALID_MODELS)}.",
            400
        )

    cache = current_app.config.get("MODELS_CACHE", {})
    resultats = []

    for i, texte in enumerate(textes):
        texte_propre, souci = valider_texte(texte)
        if souci:
            resultats.append({"index": i, "error": souci})
            continue

        try:
            resultat = predict_sentiment(texte_propre, model_type=model_type, models_cache=cache)
            resultat["index"] = i
            resultats.append(resultat)
        except FileNotFoundError as e:
            # Le modele n'est pas entraine : inutile de continuer le lot
            return error(str(e), 503)
        except Exception as e:
            log.exception(f"Erreur sur le texte {i} du lot")
            resultats.append({"index": i, "error": f"Erreur interne : {e}"})

    reussites = sum(1 for r in resultats if "error" not in r)

    return jsonify({
        "results": resultats,
        "count": len(resultats),
        "success_count": reussites,
        "error_count": len(resultats) - reussites,
    })


@main_bp.route("/health")
def health():
    """
    Healthcheck simple pour les plateformes de deploiement (Render, Railway...).
    Repond 200 si le serveur tourne et qu'au moins un modele est utilisable.
    """
    status = check_models_ready()
    ready = status["vectorizer"] and (status["logistic"] or status["random_forest"])
    ready = ready or status["distilbert"]

    payload = {
        "status": "ok" if ready else "degraded",
        "models_ready": ready,
        "models": status,
    }
    return jsonify(payload), 200 if ready else 503


@main_bp.route("/status")
def status():
    """Retourne l'etat des modeles disponibles sur le serveur."""
    return jsonify(check_models_ready())


@main_bp.route("/metrics")
def metrics():
    """Retourne les dernieres metriques d'evaluation en JSON."""
    return jsonify(load_metrics_json())
