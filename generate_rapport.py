"""
Genere le rapport du projet au format Word (rapport_projet.docx).

Le rapport se veut simple et pedagogique : les grandes lignes du projet,
sans entrer dans le detail du code. Les chiffres sont lus directement depuis
metrics/metrics_results.json, donc le rapport reste toujours a jour.

Usage : python generate_rapport.py
"""

import os
import json

from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

from src.config import METRICS_DIR, METRICS_JSON_PATH

SORTIE = "rapport_projet.docx"

# Bleu sobre pour les titres
BLEU = RGBColor(0x1F, 0x4E, 0x79)


def charger_metriques():
    """Lit les metriques produites par la derniere evaluation."""
    if not os.path.exists(METRICS_JSON_PATH):
        raise FileNotFoundError(
            "metrics_results.json introuvable. Lance d'abord : python main.py"
        )
    with open(METRICS_JSON_PATH, encoding="utf-8") as f:
        return json.load(f)


def titre(doc, texte, niveau=1):
    """Ajoute un titre colore."""
    h = doc.add_heading(texte, level=niveau)
    for run in h.runs:
        run.font.color.rgb = BLEU
    return h


def para(doc, texte, gras=False, taille=11):
    """Ajoute un paragraphe justifie."""
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    run = p.add_run(texte)
    run.bold = gras
    run.font.size = Pt(taille)
    return p


def image(doc, nom_fichier, largeur=5.5):
    """Insere une figure du dossier metrics/ si elle existe."""
    chemin = os.path.join(METRICS_DIR, nom_fichier)
    if os.path.exists(chemin):
        doc.add_picture(chemin, width=Inches(largeur))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        return True
    return False


def page_de_garde(doc):
    for _ in range(6):
        doc.add_paragraph()

    t = doc.add_heading("SentimentAI", level=0)
    t.alignment = WD_ALIGN_PARAGRAPH.CENTER

    for texte, taille, gras in [
        ("Analyse de sentiment par apprentissage automatique", 16, False),
        ("Comparaison de trois approches sur le dataset IMDB", 13, False),
        ("", 11, False),
        ("Master Data Science et Génie Logiciel", 12, True),
    ]:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(texte)
        run.font.size = Pt(taille)
        run.bold = gras

    doc.add_page_break()


def introduction(doc):
    titre(doc, "1. Introduction")
    para(doc,
         "L'analyse de sentiment consiste a determiner automatiquement si un texte "
         "exprime une opinion positive ou negative. C'est une tache classique du "
         "traitement automatique du langage, utilisee par exemple pour suivre l'avis "
         "des clients sur un produit ou l'accueil reserve a un film.")
    para(doc,
         "Ce projet compare trois facons de resoudre ce probleme, de la plus simple "
         "a la plus moderne, sur un meme jeu de donnees. L'objectif n'est pas seulement "
         "d'obtenir le meilleur score, mais de comprendre ce que chaque approche apporte "
         "et ce qu'elle coute.")

    titre(doc, "Le jeu de donnees", 2)
    para(doc,
         "Nous utilisons IMDB, un ensemble de 50 000 critiques de films en anglais, "
         "reparties en deux moities egales : 25 000 pour l'entrainement et 25 000 pour "
         "le test. Chaque critique est etiquetee positive ou negative, et les deux classes "
         "sont parfaitement equilibrees. Cet equilibre est important : il evite qu'un modele "
         "obtienne un bon score en repondant toujours la meme chose.")


def approches(doc):
    titre(doc, "2. Les trois approches comparees")

    titre(doc, "Preparer le texte : TF-IDF", 2)
    para(doc,
         "Un ordinateur ne comprend pas les mots, il faut donc les transformer en nombres. "
         "La methode TF-IDF donne a chaque mot un poids : eleve s'il est frequent dans une "
         "critique donnee mais rare dans l'ensemble des critiques. Autrement dit, elle met "
         "en valeur les mots qui distinguent un texte des autres. Les mots tres courants "
         "comme the ou and sont retires au prealable, car ils n'apportent aucune information "
         "sur le sentiment.")
    para(doc,
         "Nous conservons aussi les paires de mots consecutifs. Cela permet de capturer des "
         "expressions comme not good, dont le sens se perd si l'on regarde les mots isolement.")

    titre(doc, "Approche 1 : regression logistique", 2)
    para(doc,
         "Le modele le plus simple des trois. A partir des poids TF-IDF, il apprend quels "
         "mots penchent vers le positif et lesquels vers le negatif, puis combine ces indices "
         "pour trancher. Son grand avantage est la transparence : on peut lire directement "
         "quels mots ont pese dans la decision. Il s'entraine en quelques minutes.")

    titre(doc, "Approche 2 : forets aleatoires", 2)
    para(doc,
         "Cette methode construit deux cents arbres de decision, chacun apprenant sur une "
         "partie differente des donnees, puis fait voter l'ensemble. L'idee est que les "
         "erreurs individuelles se compensent. Elle capture des combinaisons de mots plus "
         "complexes, mais reste plus lourde et moins lisible.")

    titre(doc, "Approche 3 : DistilBERT", 2)
    para(doc,
         "DistilBERT est un reseau de neurones deja pre-entraine sur d'enormes quantites de "
         "texte anglais. Il connait donc la langue avant meme de voir nos critiques. "
         "Contrairement aux deux methodes precedentes, il ne traite pas les mots isolement "
         "mais tient compte de leur contexte : il peut comprendre que dans je ne pensais pas "
         "aimer ce film, mais, la suite renverse le sens.")
    para(doc,
         "Nous l'avons specialise sur nos donnees par une phase de fine-tuning, c'est-a-dire "
         "un reentrainement leger a partir de ses connaissances existantes.")


def resultats(doc, metriques):
    titre(doc, "3. Resultats")

    para(doc,
         "Chaque modele est juge sur quatre indicateurs. La precision globale indique la "
         "proportion de bonnes reponses. Le F1-score equilibre deux risques : passer a cote "
         "de critiques positives, ou en declarer a tort. L'AUC mesure la capacite du modele "
         "a bien classer les critiques par ordre de confiance, independamment du seuil choisi. "
         "Plus ces valeurs sont proches de 1, meilleur est le modele.")

    tableau = doc.add_table(rows=1, cols=4)
    tableau.style = "Light Grid Accent 1"
    entetes = tableau.rows[0].cells
    for i, nom in enumerate(["Modele", "Precision globale", "F1-score", "AUC"]):
        entetes[i].text = nom
        for p in entetes[i].paragraphs:
            for run in p.runs:
                run.bold = True

    ordre = ["Logistic Regression", "Random Forest", "DistilBERT"]
    noms_lisibles = {
        "Logistic Regression": "Regression logistique",
        "Random Forest": "Forets aleatoires",
        "DistilBERT": "DistilBERT",
    }

    for cle in ordre:
        if cle not in metriques:
            continue
        m = metriques[cle]
        ligne = tableau.add_row().cells
        ligne[0].text = noms_lisibles[cle]
        ligne[1].text = f"{m['accuracy']*100:.2f} %"
        ligne[2].text = f"{m['f1']*100:.2f} %"
        ligne[3].text = f"{m.get('roc_auc', 0):.4f}" if m.get("roc_auc") else "-"

    doc.add_paragraph()
    image(doc, "comparaison_modeles.png")
    para(doc, "Figure 1 : comparaison des trois modeles.", taille=9)

    doc.add_page_break()
    titre(doc, "Lecture des courbes ROC", 2)
    para(doc,
         "La courbe ROC montre le compromis entre bien reconnaitre les critiques positives "
         "et eviter les fausses alertes. Plus la courbe monte vite vers le coin superieur "
         "gauche, meilleur est le modele. La diagonale represente un modele qui repondrait "
         "au hasard.")
    image(doc, "roc_curve_Logistic_Regression.png", largeur=4.8)
    para(doc, "Figure 2 : courbe ROC de la regression logistique.", taille=9)


def analyse(doc, metriques):
    doc.add_page_break()
    titre(doc, "4. Analyse : un resultat contre-intuitif")

    lr = metriques.get("Logistic Regression", {})
    db = metriques.get("DistilBERT", {})

    para(doc,
         f"Le resultat le plus interessant de ce projet est inattendu : la regression "
         f"logistique ({lr.get('accuracy', 0)*100:.1f} %) devance DistilBERT "
         f"({db.get('accuracy', 0)*100:.1f} %), alors que ce dernier est de loin le modele "
         f"le plus sophistique. Ce constat merite d'etre explique plutot que masque.",
         gras=True)

    titre(doc, "L'explication", 2)
    para(doc,
         "La comparaison n'est pas equitable en termes de moyens. La regression logistique "
         "a ete entrainee sur la totalite des 25 000 critiques disponibles. DistilBERT, lui, "
         "n'a vu que 2 000 critiques, car son entrainement demande une puissance de calcul "
         "dont nous ne disposions pas : sans carte graphique, un entrainement complet aurait "
         "demande plusieurs heures.")
    para(doc,
         "Nous comparons donc un modele simple bien nourri a un modele puissant sous-alimente. "
         "Un reseau de neurones a besoin de beaucoup d'exemples pour exprimer son potentiel ; "
         "prive de donnees, il n'exploite qu'une fraction de ses capacites.")

    titre(doc, "Ce que cela nous apprend", 2)
    para(doc,
         "Cette observation constitue un enseignement utile plutot qu'un echec. Elle rappelle "
         "qu'un modele plus complexe n'est pas automatiquement meilleur : sa superiorite depend "
         "des moyens qu'on peut lui consacrer. A budget limite, en donnees comme en calcul, une "
         "methode classique bien reglee reste tres competitive.")
    para(doc,
         "C'est un critere concret de choix technique. La regression logistique s'entraine en "
         "quelques minutes, occupe 240 kilo-octets et repond instantanement. DistilBERT pese "
         "268 megaoctets et exige une infrastructure bien plus consequente. Pour un gain de "
         "performance qui, ici, n'est pas au rendez-vous, le surcout n'est pas justifie.")
    para(doc,
         "Signalons toutefois la limite de cette conclusion : elle vaut pour nos conditions "
         "d'experimentation. Avec un entrainement sur l'integralite des donnees, la litterature "
         "situe DistilBERT autour de 92 a 93 % sur IMDB, soit nettement devant nos modeles "
         "classiques. Ce prolongement constitue la suite naturelle du projet.")


def application(doc):
    doc.add_page_break()
    titre(doc, "5. De l'experimentation a l'application")

    para(doc,
         "Un modele n'a d'interet que s'il est utilisable. Nous avons donc developpe une "
         "application web ou l'on saisit un texte et obtient immediatement le sentiment "
         "predit, accompagne d'un indice de confiance. Le modele employe est selectionnable, "
         "ce qui permet de comparer les approches en direct.")

    titre(doc, "Choix techniques", 2)
    para(doc,
         "L'application est ecrite en Python avec le framework Flask. Elle expose egalement "
         "une interface de programmation permettant d'analyser un texte isole ou une serie de "
         "textes en une seule requete, ce qui est utile pour traiter un fichier d'avis.")
    para(doc,
         "Le service est mis en ligne automatiquement : chaque modification deposee sur le "
         "depot declenche un nouveau deploiement. Seule la regression logistique y est "
         "embarquee, car sa taille reduite lui permet de fonctionner sur un hebergement "
         "gratuit. C'est une consequence directe de l'analyse precedente : le modele le plus "
         "leger se revele aussi le plus performant et le seul deployable.")

    titre(doc, "Qualite du code", 2)
    para(doc,
         "Le projet comporte 91 tests automatiques verifiant le nettoyage du texte, le "
         "comportement des modeles et les reponses de l'interface web. Ils sont executes "
         "automatiquement a chaque modification. Les tirages aleatoires sont fixes, si bien "
         "que deux entrainements identiques produisent exactement les memes resultats, "
         "condition indispensable a une comparaison honnete.")


def conclusion(doc, metriques):
    titre(doc, "6. Conclusion")

    lr = metriques.get("Logistic Regression", {})
    para(doc,
         f"Ce projet a permis de construire une chaine complete, du jeu de donnees brut "
         f"jusqu'a une application en ligne. Le meilleur modele, la regression logistique, "
         f"atteint {lr.get('accuracy', 0)*100:.1f} % de bonnes reponses sur 25 000 critiques "
         f"jamais vues pendant l'entrainement.")
    para(doc,
         "Son principal enseignement n'est pas un score, mais une nuance : la sophistication "
         "d'un modele ne garantit pas sa superiorite. Les moyens disponibles, en donnees comme "
         "en calcul, pesent autant que le choix de l'algorithme.")

    titre(doc, "Perspectives", 2)
    for texte in [
        "Entrainer DistilBERT sur l'integralite des donnees avec une carte graphique, "
        "pour verifier l'ecart attendu.",
        "Etendre l'analyse a d'autres langues, notamment le francais.",
        "Depasser la distinction positif/negatif en detectant des nuances comme l'ironie.",
    ]:
        p = doc.add_paragraph(texte, style="List Bullet")
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY


def main():
    metriques = charger_metriques()

    doc = Document()
    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(11)

    page_de_garde(doc)
    introduction(doc)
    approches(doc)
    resultats(doc, metriques)
    analyse(doc, metriques)
    application(doc)
    conclusion(doc, metriques)

    doc.save(SORTIE)
    print(f"Rapport genere : {SORTIE}")
    print(f"Metriques utilisees (evaluation du {metriques.get('timestamp', 'inconnue')})")


if __name__ == "__main__":
    main()
