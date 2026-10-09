"""
Exécute tout le TP d'un seul coup et prépare les réponses aux questions
à partir des résultats réels (utilisé par le site web app.py).
"""
import numpy as np
import pandas as pd

import prevision as P

JOURS = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]


def _reponses(df, cible, train, test, tab_coef, metriques, residus):
    """Construit les réponses rédigées à partir des valeurs calculées."""
    rep = {}

    # --- Étape 1 : cycles journalier et hebdomadaire
    par_heure = df.groupby("heure")[cible].mean()
    par_jour = df.groupby("jour_semaine")[cible].mean()
    semaine = df[df["est_weekend"] == 0][cible].mean()
    weekend = df[df["est_weekend"] == 1][cible].mean()
    rep["cycles"] = (
        f"Cycle journalier : la consommation moyenne est minimale à {par_heure.idxmin()} h "
        f"({par_heure.min():.0f} kW) et maximale à {par_heure.idxmax()} h ({par_heure.max():.0f} kW). "
        f"Elle monte fortement le matin à l'arrivée des occupants puis redescend en soirée. "
        f"Cycle hebdomadaire : la consommation moyenne est de {semaine:.0f} kW en semaine contre "
        f"{weekend:.0f} kW le week-end ; le jour le plus consommateur est le "
        f"{JOURS[int(par_jour.idxmax())]} ({par_jour.max():.0f} kW) et le moins consommateur le "
        f"{JOURS[int(par_jour.idxmin())]} ({par_jour.min():.0f} kW)."
    )

    # --- Étape 4 : variable la plus influente
    premiere = tab_coef.iloc[0]
    deuxieme = tab_coef.iloc[1]
    rep["influence"] = (
        f"La variable la plus influente est {premiere['variable']} "
        f"(influence {premiere['influence']:+.1f}), suivie de {deuxieme['variable']} "
        f"({deuxieme['influence']:+.1f}). L'influence est le coefficient multiplié par l'écart-type de la variable, "
        f"ce qui permet de comparer des variables d'échelles différentes."
    )
    if train["est_weekend"].nunique() == 1:
        rep["influence"] += (
            " Remarque : la période d'entraînement ne contient aucun week-end, donc le coefficient "
            "de est_weekend est nul : le modèle n'a jamais vu de week-end."
        )

    # --- Étape 5 : métriques
    mae, rmse, mape = metriques["MAE"], metriques["RMSE"], metriques["MAPE"]
    rep["metriques"] = (
        f"MAE = {mae:.2f} kW : l'erreur absolue moyenne. RMSE = {rmse:.2f} kW : plus sensible aux grosses erreurs. "
        f"MAPE = {mape:.2f} % : l'erreur moyenne représente {mape:.1f} % de la consommation réelle."
    )
    rep["mape18"] = (
        "Un MAPE de 18 % signifie qu'en moyenne la prédiction s'écarte de 18 % de la valeur réelle : "
        "pour 100 kW réels, l'erreur moyenne est d'environ 18 kW."
    )
    rep["rmse_mae"] = (
        f"La RMSE élève les erreurs au carré avant de moyenner, donc les grosses erreurs pèsent plus que dans la MAE "
        f"(RMSE ≥ MAE toujours). Ici RMSE / MAE = {rmse / mae:.2f} : il y a des erreurs ponctuelles importantes "
        f"(maximum {np.abs(residus).max():.0f} kW)."
    )

    # --- Étape 6 : résidus
    t = test.assign(residu=residus)
    par_h = t.groupby("heure")["residu"].mean()
    sous = par_h.nlargest(3)
    sur = par_h.nsmallest(3)
    res_we = t[t["est_weekend"] == 1]["residu"].mean()
    rep["residus"] = (
        "Sous-estimation systématique (résidu positif) vers "
        + ", ".join(f"{h} h ({v:+.0f} kW)" for h, v in sous.items())
        + " ; surestimation systématique (résidu négatif) vers "
        + ", ".join(f"{h} h ({v:+.0f} kW)" for h, v in sur.items())
        + f". Pour les week-ends du test, le résidu moyen est de {res_we:+.0f} kW."
    )
    return rep


def executer_pipeline(source, methode="voisins", date_coupure="2024-01-20"):
    """Exécute les 6 étapes du TP et retourne tous les résultats dans un dictionnaire."""
    brut, cible = P.charger_donnees(source)
    dates = brut["date_heure"].dropna()
    audit = P.auditer_donnees(brut, cible)
    propre, rapport = P.nettoyer_donnees(brut, cible, methode)

    df = P.ajouter_features(propre)
    train, test, X_train, X_test, y_train, y_test = P.separer_train_test(df, cible, date_coupure)
    if len(train) == 0 or len(test) == 0:
        debut = dates.min().strftime("%Y-%m-%d")
        fin = dates.max().strftime("%Y-%m-%d")
        raise ValueError(
            f"La date de coupure {date_coupure} ne sépare pas les données. "
            f"Votre fichier contient des données du {debut} au {fin}. "
            "Choisissez une date située entre ces deux dates."
        )

    modele = P.entrainer_modele(X_train, y_train)
    tab_coef = P.tableau_coefficients(modele, X_train)
    y_pred = modele.predict(X_test)
    metriques = P.calculer_metriques(y_test, y_pred)
    residus = y_test.values - y_pred

    figures = {
        "serie": P.graphique_serie(df, cible),
        "profils": P.graphique_profils(df, cible),
        "h48": P.graphique_48h(test, y_test, y_pred),
        "residus": P.graphique_residus(test, residus),
    }
    if len(rapport) > 0:
        figures["corrections"] = P.graphique_corrections(brut, propre, cible)

    return {
        "cible": cible, "audit": audit, "rapport": rapport, "brut": brut, "df": df,
        "train": train, "test": test, "coefficients": tab_coef, "intercept": float(modele.intercept_),
        "metriques": metriques, "figures": figures,
        "reponses": _reponses(df, cible, train, test, tab_coef, metriques, residus),
    }
