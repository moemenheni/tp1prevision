# TP N°1 : Prévision de la consommation électrique

Le TP contient un notebook, un script et un **site web** (Flask) qui exécute les 6 étapes dans le navigateur.

```
tp_prevision/
├── data/                       consommation_batiment.txt et .csv
├── templates/index.html        page du site
├── static/style.css            style du site
├── prevision.py                fonctions du TP (chargement, nettoyage, modèle, graphiques)
├── analyse.py                  exécute tout le TP et rédige les réponses
├── app.py                      site web
├── tp_prevision.py             script à cellules pour VS Code
├── TP_Prevision_Heni_Moemen.ipynb   notebook (VS Code ou Colab)
└── requirements.txt
```

## 1. Installer (une seule fois)

1. Installer **Python 3.10 ou plus** et **VS Code**.
2. Dans VS Code, installer les extensions **Python** et **Jupyter** (Microsoft).
3. Ouvrir le dossier : **Fichier > Ouvrir un dossier…** puis choisir `tp_prevision`.
4. Ouvrir le terminal (**Terminal > Nouveau terminal**) et taper :

```bash
python -m venv .venv
# Windows :
.venv\Scripts\activate
# macOS / Linux :
source .venv/bin/activate

pip install -r requirements.txt
```

5. Choisir l'interpréteur : `Ctrl + Maj + P` > **Python: Select Interpreter** > celui de `.venv`.

## 1 bis. Télécharger le TP depuis GitHub

Pour récupérer le projet sur un autre ordinateur :

```bash
git clone https://github.com/moemenheni/tp1prevision.git
cd tp1prevision
python -m venv .venv
```

Puis suivre les étapes d'installation ci-dessus et lancer l'application avec
`python app.py`.

## 2. Travailler dans VS Code

- **Notebook :** ouvrir `TP_Prevision_Heni_Moemen.ipynb`, choisir le noyau `.venv` en haut à droite, puis **Run All**.
- **Script :** ouvrir `tp_prevision.py`. Chaque cellule `# %%` a un bouton **Run Cell**. Les graphiques s'ouvrent dans la fenêtre interactive.

## 3. Lancer le site web

```bash
python app.py
```

Ouvrir **http://127.0.0.1:5000** dans le navigateur, choisir le fichier, la méthode de correction et la date de coupure, puis cliquer sur **Lancer le TP**. Arrêter le site avec `Ctrl + C`.

## 4. Contrôle qualité des données

Les règles se trouvent en haut de `prevision.py` (`TEMP_MIN`, `TEMP_MAX`, `SAUT_TEMP_MAX`, `SAUT_ZERO_MAX`, `SEUIL_Z`). Sont détectées puis corrigées :

| Problème | Règle |
|---|---|
| Date invalide ou en double | ligne supprimée |
| Heure absente | ajoutée puis corrigée |
| Valeur manquante | corrigée |
| Consommation ≤ 0 | corrigée |
| Température hors de −20 à 45 °C | corrigée |
| Température très différente de ses voisines (> 5 °C de la médiane sur 5 h) | corrigée |
| 0 °C exact qui s'écarte de plus de 2 °C de ses voisines | corrigée |
| Consommation très éloignée du profil de cette heure (z-score robuste > 5) | corrigée |

Deux méthodes de correction sont proposées :
- **voisins** : interpolation linéaire entre la valeur précédente et la suivante ;
- **moyenne** : moyenne des valeurs de la même heure (même type de jour : semaine ou week-end).

Les corrections sont listées dans un tableau (valeur avant, valeur après, raison).
