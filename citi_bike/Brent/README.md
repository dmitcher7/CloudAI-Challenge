# NYC Citi Bike

**Team:** Git Push & pray

Dit project voorspelt vóór vertrek de duur van een Citi Bike-rit. De volledige workflow loopt van datacontrole en cleaning tot modeltraining, evaluatie en een herbruikbaar modelartifact.

## Notebooks

> **De twee notebooks in `notebooks/gecombineerd/` bevatten samen alles uit de oorspronkelijke losse notebooks. Voor de volledige uitleg en uitvoering hoef je dus alleen deze twee bestanden te gebruiken.**

| Volgorde | Bestand | Inhoud |
|---|---|---|
| 1 | [`01_data_cleaning_en_eda.ipynb`](notebooks/gecombineerd/01_data_cleaning_en_eda.ipynb) | Downloadconfiguratie, data-audit, voorbereiding, cleaning, aggregaties, grafieken en EDA. |
| 2 | [`02_modellen_en_evaluatie.ipynb`](notebooks/gecombineerd/02_modellen_en_evaluatie.ipynb) | Hypothesetoets, modellen, AutoML-resultaten, modelselectie, finale evaluatie, foutanalyse en dashboard. |

De notebooks `01_...` tot en met `07_...` in de hoofdmap `notebooks/` bevatten dezelfde inhoud, maar opgesplitst in kleinere stappen.

## Uitvoeren

Gebruik Python 3.11 of 3.12 en open de map `Brent` als workspace.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Selecteer daarna `.venv\Scripts\python.exe` als notebookkernel en voer de twee gecombineerde notebooks in bovenstaande volgorde uit.

- `RUN_DOWNLOAD=False` laat de bestaande dataset staan en voorkomt een nieuwe download.
- `RUN_AUTOML=False` gebruikt de opgeslagen AutoML-resultaten en vermijdt een nieuwe lange AutoML-run.

## Resultaten

De actuele run gebruikt de volledige lokale historie van juni 2013 tot en met september 2026. Voor de modelvergelijking worden één miljoen ritten deterministisch over deze tijdlijn bemonsterd.

- Geselecteerd model: `hist_gradient_boosting_absolute`
- Validatie-MAE: 3,62 minuten
- Finale hold-out-MAE: 3,51 minuten
- Finale hold-out-R²: 0,454
- Mediane ritduur casual riders: 6,31 minuten langer dan members

De exacte resultaten staan in `reports/`; het getrainde model staat in `models/duration_model.pkl`.

## Projectstructuur

```text
notebooks/gecombineerd/  volledige workflow in twee notebooks
notebooks/               oorspronkelijke opgesplitste notebooks
data/                    raw, weer- en processed data
src/                     cleaning, features, analyse, training en voorspelling
models/                  getraind model en metadata
reports/                 metrics en analyseresultaten
tests/                   automatische controles
```

## Methodiek en beperkingen

De modellen gebruiken alleen informatie die bij vertrek bekend is. De data worden chronologisch gesplitst, zodat toekomstige ritten niet naar de training lekken. MAE in minuten is de primaire metric; RMSE, MedAE en R² worden aanvullend gerapporteerd.

De gegevens komen van [Citi Bike System Data](https://citibikenyc.com/system-data) en het weer van [Open-Meteo](https://open-meteo.com/). De resultaten tonen associaties, geen causaliteit. Verkeer, evenementen, fiets- en stationsbeschikbaarheid en lokale straatomstandigheden ontbreken.

## GenAI-disclosure

Generatieve AI is gebruikt voor projectstructuur, documentatie en codevoorstellen. Het team blijft verantwoordelijk voor het uitvoeren, controleren, interpreteren en uitleggen van alle stappen.
