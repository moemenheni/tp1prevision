"""
TP N°1 - Prévision de la consommation électrique d'un bâtiment.

Ce module regroupe toutes les étapes du TP sous forme de fonctions.
Il est utilisé à la fois par le script tp_prevision.py (VS Code) et par le site web app.py.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Variables explicatives utilisées par le modèle
FEATURES = ["heure", "jour_semaine", "est_weekend", "temperature_c", "temperature2"]

# Règles de contrôle qualité (modifiables)
TEMP_MIN, TEMP_MAX = -20.0, 45.0   # plage plausible de température extérieure (°C)
SAUT_TEMP_MAX = 5.0                # écart maximal toléré entre une température et la médiane de ses voisines (°C)
SAUT_ZERO_MAX = 2.0                # un 0 °C exact est suspect s'il s'écarte de plus de 2 °C de ses voisines
SEUIL_Z = 5.0                      # seuil de détection d'une consommation aberrante (z-score robuste)


# --------------------------------------------------------------------------
# Étape 1 : chargement, contrôle qualité et correction des données
# --------------------------------------------------------------------------
def charger_donnees(source):
    """Charge le fichier txt ou csv (séparateur détecté automatiquement)."""
    df = pd.read_csv(source, sep=None, engine="python")
    df.columns = df.columns.str.strip()
    df["date_heure"] = pd.to_datetime(df["date_heure"], errors="coerce")
    cible = [c for c in df.columns if "conso" in c.lower()][0]
    return df, cible


def auditer_donnees(df, cible):
    """Compte les anomalies présentes dans les données brutes."""
    d = df.copy()
    conso = pd.to_numeric(d[cible], errors="coerce")
    temp = pd.to_numeric(d["temperature_c"], errors="coerce")
    dates = d["date_heure"].dropna()
    grille = pd.date_range(dates.min(), dates.max(), freq="h")
    return {
        "Nombre de lignes": len(d),
        "Période": f"{dates.min()} -> {dates.max()}",
        "Dates invalides": int(d["date_heure"].isna().sum()),
        "Dates en double": int(dates.duplicated().sum()),
        "Heures manquantes": len(grille.difference(dates)),
        "Consommation manquante": int(conso.isna().sum()),
        "Consommation nulle ou négative": int((conso <= 0).sum()),
        "Température manquante": int(temp.isna().sum()),
        f"Température hors de [{TEMP_MIN:.0f}, {TEMP_MAX:.0f}] °C": int(((temp < TEMP_MIN) | (temp > TEMP_MAX)).sum()),
        "Température exactement égale à 0 °C": int((temp == 0).sum()),
    }


def nettoyer_donnees(df, cible, methode="voisins"):
    """
    Détecte et corrige les valeurs fausses.

    Valeurs détectées : manquantes, consommation <= 0, température hors plage,
    température isolée très différente de ses voisines, 0 °C suspect, consommation aberrante
    par rapport au profil habituel de la même heure (jour ouvré / week-end).

    methode = "voisins" : interpolation linéaire entre la valeur précédente et la suivante.
    methode = "moyenne" : moyenne des valeurs de la même heure (même type de jour).

    Retourne le DataFrame corrigé et le tableau des corrections effectuées.
    """
    colonnes = [cible, "temperature_c"]
    d = df.copy()
    rapport = []

    # 1) dates invalides et doublons
    for _ in d.loc[d["date_heure"].isna()].index:
        rapport.append({"date_heure": pd.NaT, "colonne": "ligne", "valeur_avant": np.nan,
                        "valeur_corrigee": np.nan, "raison": "date invalide : ligne supprimée"})
    d = d.dropna(subset=["date_heure"]).sort_values("date_heure")
    doublons = d["date_heure"].duplicated()
    for date in d.loc[doublons, "date_heure"]:
        rapport.append({"date_heure": date, "colonne": "ligne", "valeur_avant": np.nan,
                        "valeur_corrigee": np.nan, "raison": "date en double : ligne supprimée"})
    d = d[~doublons]

    # 2) grille horaire complète : les heures absentes deviennent des valeurs manquantes
    grille = pd.date_range(d["date_heure"].min(), d["date_heure"].max(), freq="h")
    d = d.set_index("date_heure").reindex(grille).rename_axis("date_heure")
    for col in colonnes:
        d[col] = pd.to_numeric(d[col], errors="coerce")

    avant = d[colonnes].copy()
    raisons = {col: pd.Series("", index=d.index) for col in colonnes}
    valeurs = d[colonnes].copy()

    def marquer(col, masque, raison):
        masque = masque.fillna(False) & (raisons[col] == "")
        raisons[col][masque] = raison
        valeurs.loc[raisons[col] != "", col] = np.nan

    # 3) règles de détection
    marquer(cible, valeurs[cible].isna(), "valeur manquante")
    marquer(cible, valeurs[cible] <= 0, "consommation nulle ou négative")
    marquer("temperature_c", valeurs["temperature_c"].isna(), "valeur manquante")
    marquer("temperature_c", (valeurs["temperature_c"] < TEMP_MIN) | (valeurs["temperature_c"] > TEMP_MAX),
            "température hors plage plausible")

    # référence = médiane glissante sur 5 heures (robuste : un pic isolé ne la déforme pas)
    t = valeurs["temperature_c"]
    mediane_voisines = t.rolling(5, center=True, min_periods=3).median()
    ecart = (t - mediane_voisines).abs()
    marquer("temperature_c", ecart > SAUT_TEMP_MAX,
            "température isolée très différente de ses voisines")
    marquer("temperature_c", (t == 0) & (ecart > SAUT_ZERO_MAX),
            "0 °C suspect (valeur par défaut d'un capteur ?)")

    type_jour = (d.index.dayofweek >= 5).astype(int)
    heure = d.index.hour
    c = valeurs[cible]
    profil = c.groupby([type_jour, heure]).transform("median")
    residu = c - profil
    mad = (residu - residu.median()).abs().median() * 1.4826
    if mad > 0:
        z = (residu - residu.median()).abs() / mad
        marquer(cible, z > SEUIL_Z, "consommation aberrante par rapport au profil de cette heure")

    # 4) correction
    for col in colonnes:
        s = valeurs[col]
        if methode == "moyenne":
            moyenne_type = s.groupby([type_jour, heure]).transform("mean")
            moyenne_heure = s.groupby(heure).transform("mean")
            rempli = s.fillna(moyenne_type).fillna(moyenne_heure).fillna(s.mean())
        else:
            rempli = s.interpolate(method="linear", limit_direction="both")
        for date in d.index[raisons[col] != ""]:
            rapport.append({"date_heure": date, "colonne": col, "valeur_avant": avant.loc[date, col],
                            "valeur_corrigee": round(float(rempli.loc[date]), 2), "raison": raisons[col][date]})
        d[col] = rempli

    colonnes_rapport = ["date_heure", "colonne", "valeur_avant", "valeur_corrigee", "raison"]
    return d.reset_index(), pd.DataFrame(rapport, columns=colonnes_rapport)


# --------------------------------------------------------------------------
# Étape 2 : feature engineering
# --------------------------------------------------------------------------
def ajouter_features(df):
    """Ajoute heure, jour_semaine, est_weekend et temperature2."""
    df = df.copy()
    df["heure"] = df["date_heure"].dt.hour
    df["jour_semaine"] = df["date_heure"].dt.dayofweek
    df["est_weekend"] = (df["jour_semaine"] >= 5).astype(int)
    df["temperature2"] = df["temperature_c"] ** 2
    return df


# --------------------------------------------------------------------------
# Étape 3 : séparation temporelle
# --------------------------------------------------------------------------
def separer_train_test(df, cible, date_coupure="2024-01-20"):
    """Sépare chronologiquement : avant la date de coupure = train, à partir de la date = test."""
    date_coupure = pd.Timestamp(date_coupure)
    train = df[df["date_heure"] < date_coupure].copy()
    test = df[df["date_heure"] >= date_coupure].copy()
    return train, test, train[FEATURES], test[FEATURES], train[cible], test[cible]


# --------------------------------------------------------------------------
# Étapes 4 et 5 : modèle et évaluation
# --------------------------------------------------------------------------
def entrainer_modele(X_train, y_train):
    modele = LinearRegression()
    modele.fit(X_train, y_train)
    return modele


def tableau_coefficients(modele, X_train):
    """Coefficients bruts et influence = coefficient x écart-type de la variable."""
    tab = pd.DataFrame({
        "variable": FEATURES,
        "coefficient": modele.coef_,
        "influence": modele.coef_ * X_train.std().values,
    })
    tab["influence_absolue"] = tab["influence"].abs()
    return tab.sort_values("influence_absolue", ascending=False).reset_index(drop=True)


def calculer_metriques(y_test, y_pred):
    """MAE, RMSE et MAPE (en %)."""
    y_test = np.asarray(y_test)
    return {
        "MAE": float(mean_absolute_error(y_test, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "MAPE": float(np.mean(np.abs((y_test - y_pred) / y_test)) * 100),
    }


# --------------------------------------------------------------------------
# Graphiques
# --------------------------------------------------------------------------
def graphique_serie(df, cible):
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(df["date_heure"], df[cible], linewidth=1, color="tab:blue")
    for jour in pd.date_range(df["date_heure"].min().normalize(), df["date_heure"].max().normalize()):
        if jour.dayofweek >= 5:
            ax.axvspan(jour, jour + pd.Timedelta(days=1), color="grey", alpha=0.2)
    ax.set_xlabel("Date")
    ax.set_ylabel("Consommation (kW)")
    ax.set_title("Consommation électrique horaire (week-ends en gris)")
    fig.tight_layout()
    return fig


def graphique_profils(df, cible):
    profil_heure = df.groupby(df["date_heure"].dt.hour)[cible].mean()
    profil_jour = df.groupby(df["date_heure"].dt.dayofweek)[cible].mean()
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    profil_heure.plot(ax=axes[0], marker="o")
    axes[0].set_title("Cycle journalier (moyenne par heure)")
    axes[0].set_xlabel("Heure")
    axes[0].set_ylabel("Consommation moyenne (kW)")
    profil_jour.plot(kind="bar", ax=axes[1], color="tab:orange")
    axes[1].set_title("Cycle hebdomadaire (0 = lundi, 6 = dimanche)")
    axes[1].set_xlabel("Jour de la semaine")
    axes[1].set_ylabel("Consommation moyenne (kW)")
    fig.tight_layout()
    return fig


def graphique_corrections(brut, propre, cible):
    """Compare les données brutes et corrigées (utile quand des valeurs ont été corrigées)."""
    fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
    axes[0].plot(brut["date_heure"], pd.to_numeric(brut[cible], errors="coerce"), "o-", markersize=2, label="brut")
    axes[0].plot(propre["date_heure"], propre[cible], "-", linewidth=1, label="corrigé")
    axes[0].set_ylabel("Consommation (kW)")
    axes[0].legend()
    axes[1].plot(brut["date_heure"], pd.to_numeric(brut["temperature_c"], errors="coerce"), "o-", markersize=2, label="brut")
    axes[1].plot(propre["date_heure"], propre["temperature_c"], "-", linewidth=1, label="corrigé")
    axes[1].set_ylabel("Température (°C)")
    axes[1].legend()
    fig.suptitle("Données brutes et données corrigées")
    fig.tight_layout()
    return fig


def graphique_48h(test, y_test, y_pred, n=48):
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(test["date_heure"].values[:n], np.asarray(y_test)[:n], label="Valeurs réelles", marker="o", markersize=3)
    ax.plot(test["date_heure"].values[:n], np.asarray(y_pred)[:n], label="Prédictions", marker="s", markersize=3, linestyle="--")
    ax.set_xlabel("Date")
    ax.set_ylabel("Consommation (kW)")
    ax.set_title(f"Réel vs prédit sur les {n} premières heures du test")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def graphique_residus(test, residus):
    fig, axes = plt.subplots(2, 1, figsize=(13, 7))
    axes[0].plot(test["date_heure"].values, residus, color="tab:red", linewidth=1)
    axes[0].axhline(0, color="black", linestyle="--")
    axes[0].set_ylabel("Résidu (kW)")
    axes[0].set_title("Résidus du modèle linéaire sur le jeu de test")
    par_heure = pd.Series(residus, index=test["heure"].values).groupby(level=0).mean()
    axes[1].bar(par_heure.index, par_heure.values, color="tab:purple")
    axes[1].axhline(0, color="black")
    axes[1].set_xlabel("Heure de la journée")
    axes[1].set_ylabel("Résidu moyen (kW)")
    axes[1].set_title("Résidu moyen par heure")
    fig.tight_layout()
    return fig
