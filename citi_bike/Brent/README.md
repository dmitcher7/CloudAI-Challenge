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

De ritdownload gebruikt standaard de volledige officiële historie vanaf juni 2013 tot de nieuwste gepubliceerde maand. De downloader leest de officiële S3-inventaris: jaarbundels voor 2013–2023 en maandarchieven vanaf 2024 worden automatisch herkend; Jersey City-bestanden worden uitgesloten. `src.data.weather` haalt dezelfde periode op. De training leest standaard maximaal 1.000.000 deterministisch over de volledige tijdlijn verdeelde ritten rechtstreeks uit Parquet, zodat ieder jaar vertegenwoordigd blijft zonder eerst de volledige dataset in RAM te laden. Gebruik `--max-rows 0` alleen wanneer de machine alle rijen aankan. Open na training <http://127.0.0.1:8000>.

## Notebookvolgorde

Open in VS Code de map `Brent` als workspace en selecteer rechtsboven in een notebook de kernel
`.venv\Scripts\python.exe` (`Python (.venv)`). Iedere notebook begint met een bootstrapcel die de
projectroot automatisch aan het Python-pad toevoegt. Daardoor werken imports uit `src` ook wanneer
VS Code de notebookkernel vanuit `notebooks/` start. Notebook 01 laat `RUN_DOWNLOAD=False` staan als
de jaarbestanden al aanwezig zijn.

1. `notebooks/01_download_and_audit.ipynb` — geautomatiseerde download en schema-audit.
2. `notebooks/02_eda_and_cleaning.ipynb` — cleaning, grafieken en inhoudelijke EDA.
3. `notebooks/03_prepare_data.ipynb` — compacte, grafiekloze voorbereiding van raw naar prepared.
4. `notebooks/04_hypothesis.ipynb` — vooraf vastgelegde hypothese, toets en effectgroottes.
5. `notebooks/05_automl_and_models.ipynb` — baseline, AutoML en handmatig gekozen modellen.
6. `notebooks/06_model_comparison.ipynb` — vergelijking, foutanalyse en definitieve keuze.
7. `notebooks/08_model_dashboard.ipynb` — grafisch eindrapport met alle features, technieken, resultaten, importance en foutanalyse.

De notebooks zijn dunne, uitlegbare onderzoekslagen boven herbruikbare code in `src/`.

## Projectstructuur

```text
notebooks/           genummerde analyse-, training- en presentatienotebooks
data/                lokale raw-, weer- en processed data (niet in Git)
models/              getrainde pipeline (.pkl) en metadata
reports/             reproduceerbare metrics en hypotheseresultaten
src/                 download, cleaning, features, training en predictie
app/                 minimale FastAPI-backend en webinterface
aws/                 SageMaker-entrypoint voor de verplichte cloudtraining
.github/workflows/   automatische tests, training en modelartifact
tests/               compacte controles op de volledige pipeline
scripts/             deterministische kleine dataset voor CI-controles
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
- FLAML als reproduceerbare AutoML-vergelijking van HistGradientBoosting, Extra Trees,
  Random Forest, CatBoost en XGBoost (meegeleverd in `requirements.txt`); LightGBM is in de
  gebruikte Windows/Python 3.13-runtime uitgesloten wegens een reproduceerbare native crash;
- SageMaker-training met dezelfde train/testgrens.

De steekproef wordt chronologisch gesplitst: de eerste 80% is ontwikkeldata en de laatste 20% is de finale hold-out. Binnen de ontwikkeldata vormt opnieuw de laatste 20% de selectievalidatie. Kandidaten worden dus nooit op de finale hold-out gekozen. Om zowel nauwkeurigheid als verklaarde variantie te verbeteren, wint binnen 0,10 minuut van de beste validatie-MAE het model met de hoogste validatie-R². Route- en stationshistoriek gebruikt uitsluitend eerder waargenomen ritten. Selectiemetrics staan in `reports/metrics.csv`, de eenmalige eindscore in `reports/final_holdout_metrics.csv` en de stabiliteit per maand in `reports/metrics_by_month.csv`. De uiteindelijke pipeline staat in `models/duration_model.pkl`; zonder dit artifact weigert de API te starten.

### Actuele all-years-run

De hertraining van 12 oktober 2026 gebruikt 327.931.287 geldige ritten uit 44
officiële archieven, van juni 2013 tot en met september 2026. Voor de
modelvergelijking zijn 1.000.000 ritten deterministisch over die volledige
tijdlijn verdeeld. `hist_gradient_boosting_absolute` is geselecteerd en behaalt
op de aparte finale tijdshold-out (30 mei 2025–30 september 2026) een MAE van
3,51 minuten, RMSE van 8,75 minuten en R² van 0,454. De exacte actuele waarden
staan in `reports/` en `models/duration_model.metadata.json`.

De eerlijke FLAML-run gebruikt dezelfde 639.865 selectie-trainingsritten en
159.967 chronologische validatieritten. Alle vijf stabiele learners zijn geprobeerd;
FLAML koos XGBoost met een validatie-MAE van 4,48 minuten. Dat is slechter dan de
3,62 minuten van `hist_gradient_boosting_absolute`, dat daarom het eindmodel blijft.
De AutoML-resultaten staan in `reports/automl_metrics.csv` en
`reports/automl_result.json`.

### Eerdere referentierun (alleen 2024)

Deze cijfers zijn uitsluitend de eerdere referentie vóór de all-years-hertraining. De lokale run op januari–december 2024 leverde 44.199.909 geldige ritten op. Voor
modelvergelijking werden 1.000.000 ritten gelijkmatig over het jaar geselecteerd: 640.000 voor
selectietraining, 160.000 voor selectievalidatie en de laatste 200.000 als aparte finale tijdshold-out.
Met geplande bestemming, geometrie, uurweer en lekvrije routehistoriek behaalde histogram gradient
boosting op `log1p`-ritduur daar een MAE van 3,12 minuten, RMSE van 7,36 minuten en R² van 0,518.
De maandelijkse hold-out-R² loopt van 0,488 in oktober tot 0,554 in december. De exacte waarden staan
in `reports/`; notebooks 04, 06 en 08 berekenen aanvullend hypothese-, subgroep- en dashboardresultaten.

## Deployment

```powershell
uvicorn app.main:app --reload
```

Open daarna <http://127.0.0.1:8000>. `GET /health` geeft de modelstatus en `POST /predict` geeft een puntschatting met duidelijke beperkingen. De GitHub Actions-workflow valideert iedere push en traint een smoke-model op deterministische testdata. Een handmatige workflow-run kan officiële maanden downloaden en een nieuw modelartifact produceren.

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
