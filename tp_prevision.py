# %% [markdown]
# # TP N°1 : Mise en œuvre d'un premier modèle de prévision
# Script pour VS Code : chaque cellule `# %%` s'exécute avec le bouton « Run Cell » (ou Maj + Entrée).
# Il fonctionne aussi dans un terminal : `python tp_prevision.py`.

# %% Étape 1.1 : bibliothèques
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import prevision as P

sns.set_theme(style="whitegrid")
FICHIER = "data/consommation_batiment.txt"     # ou data/consommation_batiment.csv
DATE_COUPURE = "2024-01-20"
METHODE_CORRECTION = "voisins"                 # "voisins" (interpolation) ou "moyenne" (moyenne de la même heure)

# %% Étape 1.2 et 1.3 : chargement, premières lignes, informations, valeurs manquantes
brut, CIBLE = P.charger_donnees(FICHIER)
print(brut.head())
print()
brut.info()
print()
print("Valeurs manquantes :")
print(brut.isnull().sum())

# %% Étape 1.4 : contrôle qualité et correction des valeurs fausses
audit = P.auditer_donnees(brut, CIBLE)
for nom, valeur in audit.items():
    print(f"{nom:45s} : {valeur}")

df_propre, rapport = P.nettoyer_donnees(brut, CIBLE, METHODE_CORRECTION)

if len(rapport) == 0:
    print("\nAucune valeur fausse détectée : aucune correction nécessaire.")
else:
    print(f"\n{len(rapport)} correction(s) effectuée(s) :")
    print(rapport.to_string(index=False))
    P.graphique_corrections(brut, df_propre, CIBLE)
    plt.show()

# %% Étape 1.5 : courbe de consommation, cycles journalier et hebdomadaire
P.graphique_serie(df_propre, CIBLE)
plt.show()
df = P.ajouter_features(df_propre)       # (les variables sont créées ici pour tracer les profils)
P.graphique_profils(df, CIBLE)
plt.show()

# %% Étape 2 : feature engineering (heure, jour_semaine, est_weekend, temperature2)
print(df[["date_heure", "heure", "jour_semaine", "est_weekend", "temperature_c", "temperature2"]].head(10))

# %% Étape 3 : séparation temporelle
train, test, X_train, X_test, y_train, y_test = P.separer_train_test(df, CIBLE, DATE_COUPURE)
print(f"Entraînement : {len(train)} lignes ({train['date_heure'].min()} -> {train['date_heure'].max()})")
print(f"Test         : {len(test)} lignes ({test['date_heure'].min()} -> {test['date_heure'].max()})")

# %% Étape 4 : régression linéaire et coefficients
modele = P.entrainer_modele(X_train, y_train)
print("Ordonnée à l'origine :", round(modele.intercept_, 3))
print(P.tableau_coefficients(modele, X_train).round(3).to_string(index=False))

# %% Étape 5 : prédictions, métriques, graphique sur 48 h
y_pred = modele.predict(X_test)
metriques = P.calculer_metriques(y_test, y_pred)
print(f"MAE  : {metriques['MAE']:.2f} kW")
print(f"RMSE : {metriques['RMSE']:.2f} kW")
print(f"MAPE : {metriques['MAPE']:.2f} %")

P.graphique_48h(test, y_test, y_pred)
plt.show()

# %% Étape 6 : résidus
residus = y_test.values - y_pred
P.graphique_residus(test, residus)
plt.show()
