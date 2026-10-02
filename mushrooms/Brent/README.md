# Mushroom classification

Machine-learningmodellen voor de classificatie van paddenstoelen als eetbaar of giftig/onbekend, in het kader van de CloudAI Challenge.

## Bestanden

- `01_mushroom_random_forest.ipynb` — Random Forest
- `02_mushroom_logistic_regression.ipynb` — Logistic Regression
- `03_mushroom_gradient_boosting.ipynb` — Gradient Boosting
- `04_mushroom_catboost.ipynb` — CatBoost
- `05_mushroom_stacking_ensemble.ipynb` — stacking ensemble
- `06_mushroom_model_comparison.ipynb` — vergelijking van de modellen
- `project assignment.md` — opdrachtbeschrijving

## Dataset

De notebooks verwachten `mushroom_project_dataset.csv` in deze map. Dit bestand wordt niet automatisch door Git toegevoegd omdat CSV-bestanden in de hoofdrepo worden genegeerd. Plaats het datasetbestand daarom lokaal in deze map, of voeg het expliciet toe met:

```powershell
git add -f mushroom_project_dataset.csv
```

## Installatie

Maak vanuit de hoofdrepo een virtuele omgeving en installeer de dependencies:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install catboost
```

Open daarna de notebooks in Jupyter of VS Code. Voer eerst de modelnotebooks uit en daarna `06_mushroom_model_comparison.ipynb` om de resultaten naast elkaar te bekijken. De notebooks laden het datasetbestand automatisch vanuit de huidige map.

## Reproduceerbaarheid

Gebruik dezelfde Python-omgeving voor alle notebooks. De resultaten kunnen licht variëren wanneer random seeds, packageversies of de uitvoeringsvolgorde worden gewijzigd.
