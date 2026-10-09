"""
Site web du TP N°1 : le travail s'effectue dans le navigateur.

Lancer :  python app.py   puis ouvrir http://127.0.0.1:5000
"""
import base64
import io
import os

import matplotlib
matplotlib.use("Agg")               # pas de fenêtre : les graphiques sont envoyés au navigateur
import matplotlib.pyplot as plt
from flask import Flask, render_template, request

from analyse import executer_pipeline

DOSSIER = os.path.dirname(os.path.abspath(__file__))
DOSSIER_DONNEES = os.path.join(DOSSIER, "data")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024   # fichiers de 5 Mo maximum


def figure_en_base64(fig):
    """Transforme une figure matplotlib en image intégrable dans la page."""
    tampon = io.BytesIO()
    fig.savefig(tampon, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(tampon.getvalue()).decode("ascii")


def fichiers_disponibles():
    return sorted(f for f in os.listdir(DOSSIER_DONNEES) if f.lower().endswith((".csv", ".txt")))


@app.route("/", methods=["GET", "POST"])
def accueil():
    contexte = {"fichiers": fichiers_disponibles(), "erreur": None, "resultats": None,
                "choix": {"fichier": "consommation_batiment.txt", "methode": "voisins", "date_coupure": "2024-01-20"}}

    if request.method == "POST":
        choix = {
            "fichier": request.form.get("fichier", ""),
            "methode": request.form.get("methode", "voisins"),
            "date_coupure": request.form.get("date_coupure", "2024-01-20"),
        }
        contexte["choix"] = choix
        try:
            televerse = request.files.get("televerse")
            if televerse and televerse.filename:
                source = io.StringIO(televerse.read().decode("utf-8", errors="replace"))
            elif choix["fichier"] in contexte["fichiers"]:
                source = os.path.join(DOSSIER_DONNEES, choix["fichier"])
            else:
                raise ValueError("Choisissez un fichier de données.")
            if choix["methode"] not in ("voisins", "moyenne"):
                raise ValueError("Méthode de correction inconnue.")

            res = executer_pipeline(source, choix["methode"], choix["date_coupure"])
            res["images"] = {nom: figure_en_base64(fig) for nom, fig in res["figures"].items()}
            res["apercu"] = res["brut"].head(8).to_html(classes="tableau", index=False, border=0)
            res["rapport_html"] = res["rapport"].to_html(classes="tableau", index=False, border=0, na_rep="")
            res["coef_html"] = res["coefficients"].round(3).to_html(classes="tableau", index=False, border=0)
            contexte["resultats"] = res
        except Exception as erreur:        # message lisible plutôt qu'une page d'erreur
            contexte["erreur"] = f"{type(erreur).__name__} : {erreur}"

    return render_template("index.html", **contexte)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
