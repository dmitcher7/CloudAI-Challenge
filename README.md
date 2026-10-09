# CloudAI Challenge

## Team & Auteurs
* **Teamnaam:** Git Push & Pray
* **Leden:**
  * Dmitrij Tchervonisjenko
  * Kobe Vandenberk
  * Cisse Vandeweyer
  * Brent Van Gelder

---

## Projectoverzicht

1. **Secondary Mushroom Dataset:**
   * Binaire classificatie (eetbaar vs. giftig/onbekend).
   * Locatie: map `mushrooms/`
2. **NYC Citi Bike System Data:**
   * Exploratory Data Analysis, hypothesevorming en voorspellende modellen.
   * Locatie: map `citi_bike/`
## Mappenstructuur & Werkwijze
Binnen beide dataset-mappen hebben we een strikte structuur gehanteerd om efficiënt samen te werken en versieconflicten te voorkomen:
* **Individuele mappen (`Brent/`, `Cisse/`, `Dimi/`, `Kobe/`):** 
  Elk teamlid heeft in zijn eigen map onafhankelijk de data verkend (EDA), opgeschoond en geëxperimenteerd met verschillende Machine Learning modellen en AutoML (PyCaret). Hierdoor konden we out-of-the-box ideeën testen en meerdere modellen onafhankelijk van elkaar trainen.
  
* **De `FINAL/` map:** 
  Zodra de individuele experimenten klaar waren, hebben we de metrics (zoals accuracy en de foutenanalyse) naast elkaar gelegd. De code met de beste Exploratory Data Analysis, de slimste aanpak voor data cleaning (zoals het oplossen van bewuste multicollineariteit/valstrikken) en het best presterende model is vervolgens geselecteerd en samengevoegd in de `FINAL/` map. Deze map bevat onze definitieve pipeline die gekoppeld is aan de web-deployment en API.
## Deployment (Oracle VM)

```
deploy/                      ← alles wat naar de VM gaat
├── app.py                   ← start de API en koppelt alle modellen
├── requirements.txt         ← pakketten voor de VM
├── check_model.py           ← test API + modellen vóór elke deploy
├── web/                     ← index.html (keuzepagina), mushrooms.html, citi_bike_models.html (keuze Citi Bike), citi_bike.html, citi_bike_demand.html
├── mushrooms/               ← api.py + stacking.pkl
├── citi_bike/               ← api.py + model.pkl (ritduur)
└── citi_bike_demand/        ← api.py + model.pkl (vraag per zone per uur, zie citi_bike/Kobe/)
```

| Adres | Wat |
|---|---|
| `/` | keuzepagina: mushrooms of Citi Bike |
| `/citi_bike_models.html` | keuzepagina tussen de twee Citi Bike-modellen |
| `/mushrooms.html`, `/citi_bike.html`, `/citi_bike_demand.html` | webpagina per model |
| `GET /mushrooms/`, `POST /mushrooms/predict` | mushroom-model (eetbaar 0 / giftig 1) |
| `GET /citi_bike/`, `POST /citi_bike/predict` | Citi Bike-model (ritduur in minuten) |
| `GET /citi_bike_demand/`, `POST /citi_bike_demand/predict`, `POST /citi_bike_demand/predict_day` | Citi Bike-vraagmodel (vertrekken per zone per uur), zie `citi_bike/Kobe/` |
| `/health`, `/docs` | status en Swagger |

Lokaal starten: `cd deploy && pip install -r requirements.txt && uvicorn app:app --port 8000`.
Testen: `python deploy/check_model.py`.

Alle modellen draaien in hetzelfde proces en moeten dus met **dezelfde scikit-learn-versie** opgeslagen zijn
als in `deploy/requirements.txt` (nu 1.4.2). Een merge naar `main` die iets in `deploy/` wijzigt, deployt
automatisch via `.github/workflows/deploy-api.yml` (met rollback als de API niet start).
