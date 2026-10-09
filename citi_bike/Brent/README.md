# Green Wheels — NYC Citi Bike

> **Groep:** Green Wheels  
> **Teamleden:** _vul hier de drie namen in_

Een reproduceerbaar machine-learningproject dat vóór vertrek de verwachte duur van een Citi Bike-rit voorspelt. De repository bevat de volledige keten: downloaden → valideren → opschonen → EDA en hypothesetoetsing → modelvergelijking → API → webfrontend → CI-pipeline.

## Onderzoeksvraag

**Kunnen we de ritduur voorspellen met uitsluitend informatie die bij de start bekend is?**

Dat voorkomt target leakage. `ended_at`, eindstation en de gerealiseerde afstand worden nooit als features gebruikt. De primaire metric is MAE in minuten, omdat die makkelijk uit te leggen en minder gevoelig voor extreem lange ritten is dan RMSE. We rapporteren daarnaast RMSE, MedAE en R².

De vooraf geformuleerde hypothese is:

- H₀: de verdelingen van ritduur voor members en casual riders zijn gelijk.
- H₁: casual riders maken typisch langere ritten dan members.

Naast een Mann–Whitney-U-toets rapporteren we Cliff's delta en een bootstrap-betrouwbaarheidsinterval voor het verschil in medianen. Zo onderscheiden we statistische van praktische relevantie.

## Snel starten

Gebruik bij voorkeur Python 3.11 of 3.12 (een deel van de AutoML-stack ondersteunt nieuwere Python-versies mogelijk nog niet).

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
# alleen voor notebook 07:
# python -m pip install -r requirements-aws.txt
python -m src.data.download
python -m src.data.weather
python -m src.data.prepare
python -m src.models.train --data data/processed/trips.parquet
uvicorn app.main:app --reload
```

De ritdownload gebruikt standaard het volledige kalenderjaar 2024 (`2024-01` t/m `2024-12`) en kan daardoor meerdere GB groot zijn. `src.data.weather` downloadt voor dezelfde periode een compacte uur-weertabel voor New York. De training neemt standaard maximaal 1.000.000 deterministisch en gelijkmatig over het hele jaar verdeelde ritten, zodat alle seizoenen vertegenwoordigd blijven zonder onnodig geheugengebruik. Gebruik `--max-rows 0` alleen wanneer de machine alle rijen aankan. Open na training <http://127.0.0.1:8000>. Vermeld de gebruikte maanden en steekproefgrootte altijd in de presentatie.

## Notebookvolgorde

Open in VS Code de map `Brent` als workspace en selecteer rechtsboven in een notebook de kernel
`venv\Scripts\python.exe` (`Python (venv)`). Iedere notebook begint met een bootstrapcel die de
projectroot automatisch aan het Python-pad toevoegt. Daardoor werken imports uit `src` ook wanneer
VS Code de notebookkernel vanuit `notebooks/` start. Notebook 01 laat `RUN_DOWNLOAD=False` staan als
de jaarbestanden al aanwezig zijn; notebook 07 vereist daarnaast een geconfigureerd AWS-account.

1. `notebooks/01_download_and_audit.ipynb` — geautomatiseerde download en schema-audit.
2. `notebooks/02_eda_and_cleaning.ipynb` — cleaning, grafieken en inhoudelijke EDA.
3. `notebooks/03_prepare_data.ipynb` — compacte, grafiekloze voorbereiding van raw naar prepared.
4. `notebooks/04_hypothesis.ipynb` — vooraf vastgelegde hypothese, toets en effectgroottes.
5. `notebooks/05_automl_and_models.ipynb` — baseline, AutoML en handmatig gekozen modellen.
6. `notebooks/06_model_comparison.ipynb` — vergelijking, foutanalyse en definitieve keuze.
7. `notebooks/07_aws_sagemaker.ipynb` — reproduceerbare SageMaker-training (vereist eigen AWS-account; installeer daarvoor ook `requirements-aws.txt`).
8. `notebooks/08_model_dashboard.ipynb` — grafisch eindrapport met alle features, technieken, resultaten, importance en foutanalyse.

De notebooks zijn dunne, uitlegbare onderzoekslagen boven herbruikbare code in `src/`. Genereer ze opnieuw met `python scripts/build_notebooks.py`.

## Projectstructuur

```text
app/                 FastAPI-backend en statische frontend
data/                raw/interim/processed (niet in Git)
notebooks/           genummerde onderzoeksnotebooks
src/data/            downloader, validatie en cleaning
src/features/        lekvrije feature engineering
src/models/          training, evaluatie en predictie
tests/               unit- en integratietests
.github/workflows/   CI: test, train smoke-model, artifact
```

## Reproduceerbaarheid en data

De data worden nooit handmatig toegevoegd. `src.data.download` bouwt de officiële maand-URL's op, controleert ZIP-bestanden en schrijft een manifest met URL, tijdstip, bestandsgrootte en SHA-256. `src.data.weather` haalt historische uurwaarden op voor temperatuur, luchtvochtigheid, neerslag en wind. `src.data.prepare` harmoniseert historische kolomnamen, verwijdert technisch ongeldige records, begrenst alleen onrealistische ritduur en bewaart de opgeschoonde data als Parquet.

Bron: [Citi Bike System Data](https://citibikenyc.com/system-data). De link in de opdracht bevat per ongeluk tweemaal de URL; dit is de correcte officiële pagina. Gebruik van de trip histories valt onder de [Citi Bike Data Use Policy](https://citibikenyc.com/data-sharing-policy).

Historisch en verwacht weer komt van [Open-Meteo](https://open-meteo.com/). De training koppelt weer per lokaal vertrekuur; de API probeert voor nieuwe ritten een forecast op te halen en valt bij onbeschikbaarheid terug op de trainingsmediaan.

## Modellen en eerlijke evaluatie

- naïeve mediaanbaseline;
- Ridge op one-hot/cyclische features;
- histogram gradient boosting met log-, raw- en absolute-error-targetvarianten;
- Extra Trees;
- CatBoost met native categorische features;
- FLAML als reproduceerbare AutoML-vergelijking (meegeleverd in `requirements.txt`);
- SageMaker-training met dezelfde train/testgrens.

De steekproef wordt chronologisch gesplitst: de eerste 80% is ontwikkeldata en de laatste 20% is de finale hold-out. Binnen de ontwikkeldata vormt opnieuw de laatste 20% de selectievalidatie. Kandidaten worden dus nooit op de finale hold-out gekozen. Om zowel nauwkeurigheid als verklaarde variantie te verbeteren, wint binnen 0,10 minuut van de beste validatie-MAE het model met de hoogste validatie-R². Route- en stationshistoriek gebruikt uitsluitend eerder waargenomen ritten. Selectiemetrics staan in `reports/metrics.csv`, de eenmalige eindscore in `reports/final_holdout_metrics.csv` en de stabiliteit per maand in `reports/metrics_by_month.csv`. De uiteindelijke pipeline staat in `models/duration_model.joblib`; zonder dit artifact weigert de API te starten.

### Uitgevoerde volledige-jaar-run

De lokale reproduceerbare run op januari–december 2024 leverde 44.199.909 geldige ritten op. Voor
modelvergelijking werden 1.000.000 ritten gelijkmatig over het jaar geselecteerd: 640.000 voor
selectietraining, 160.000 voor selectievalidatie en de laatste 200.000 als aparte finale tijdshold-out.
Met geplande bestemming, geometrie, uurweer en lekvrije routehistoriek behaalde histogram gradient
boosting op `log1p`-ritduur daar een MAE van 3,12 minuten, RMSE van 7,36 minuten en R² van 0,518.
De maandelijkse hold-out-R² loopt van 0,488 in oktober tot 0,554 in december. De exacte waarden staan
in `reports/`; notebooks 04, 06 en 08 berekenen aanvullend hypothese-, subgroep- en dashboardresultaten.

## Deployment

```powershell
docker build -t citibike-duration .
docker run --rm -p 8000:8000 citibike-duration
```

`GET /health` geeft modelstatus; `POST /predict` geeft een puntschatting en duidelijke beperkingen. De GitHub Actions-workflow valideert iedere push en traint een smoke-model op deterministische testdata. Een handmatige workflow-run kan officiële maanden downloaden en een nieuw modelartifact produceren. Voor publieke hosting kan dezelfde container naar AWS App Runner, Render, Railway of een VM worden gebracht.

## Belangrijkste beperkingen

- Trip histories bevatten ritten, geen niet-gerealiseerde vraag: causaliteit of volledige vraagvoorspelling is niet mogelijk.
- Evenementen, verkeer, beschikbaarheid en fietspadcondities ontbreken.
- Weer is een uurwaarde voor de omgeving New York en vangt geen lokale straatverschillen.
- De historische eindlocatie fungeert als proxy voor een vooraf geplande bestemming; afwijkingen van de werkelijk bedoelde route blijven onbekend.
- Stations en gebruikersgedrag veranderen in de tijd; driftmonitoring en periodiek hertrainen zijn nodig.
- Afgekapt ongeldige/extreme ritten worden gedocumenteerd en niet stilzwijgend als normale ritten behandeld.
- De AWS-notebook is uitvoerbaar, maar cloudtraining en hosting vereisen eigen credentials, budget en expliciete uitvoering.

## GenAI-disclosure

Generatieve AI is gebruikt om de eerste projectstructuur, documentatie en codevoorstellen op te stellen. Het team blijft verantwoordelijk voor het uitvoeren, controleren, interpreteren en mondeling verdedigen van iedere stap. Voeg vóór indiening concrete menselijke wijzigingen, uitgevoerde experimenten en teamnamen toe.
