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
python -m src.data.download --start 2024-01 --end 2024-03
python -m src.data.prepare
python -m src.models.train --data data/processed/trips.parquet
uvicorn app.main:app --reload
```

Open daarna <http://127.0.0.1:8000>. De gekozen standaardperiode is klein genoeg voor een laptop; voor seizoenseffecten hoort de eindanalyse minimaal twaalf maanden te gebruiken, bijvoorbeeld `--start 2024-01 --end 2024-12`. Alle maanden en steekproefgroottes moeten in de presentatie worden genoemd.

## Notebookvolgorde

1. `notebooks/01_download_and_audit.ipynb` — geautomatiseerde download en schema-audit.
2. `notebooks/02_eda_and_cleaning.ipynb` — cleaning, grafieken en inhoudelijke EDA.
3. `notebooks/03_prepare_data.ipynb` — compacte, grafiekloze voorbereiding van raw naar prepared.
4. `notebooks/04_hypothesis.ipynb` — vooraf vastgelegde hypothese, toets en effectgroottes.
5. `notebooks/05_automl_and_models.ipynb` — baseline, AutoML en handmatig gekozen modellen.
6. `notebooks/06_model_comparison.ipynb` — vergelijking, foutanalyse en definitieve keuze.
7. `notebooks/07_aws_sagemaker.ipynb` — reproduceerbare SageMaker-training (vereist eigen AWS-account; installeer daarvoor ook `requirements-aws.txt`).

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

De data worden nooit handmatig toegevoegd. `src.data.download` bouwt de officiële maand-URL's op, controleert ZIP-bestanden en schrijft een manifest met URL, tijdstip, bestandsgrootte en SHA-256. `src.data.prepare` harmoniseert historische kolomnamen, verwijdert technisch ongeldige records, begrenst alleen onrealistische ritduur en bewaart de opgeschoonde data als Parquet.

Bron: [Citi Bike System Data](https://citibikenyc.com/system-data). De link in de opdracht bevat per ongeluk tweemaal de URL; dit is de correcte officiële pagina. Gebruik van de trip histories valt onder de [Citi Bike Data Use Policy](https://citibikenyc.com/data-sharing-policy).

## Modellen en eerlijke evaluatie

- naïeve mediaanbaseline;
- Ridge op one-hot/cyclische features;
- histogram gradient boosting;
- Extra Trees;
- FLAML als reproduceerbare AutoML-vergelijking (meegeleverd in `requirements.txt`);
- SageMaker-training met dezelfde train/testgrens.

De laatste 20% in de tijd is de hold-outset. Hyperparameters worden alleen op het eerdere train-gedeelte gekozen. Metrics staan na uitvoering in `reports/metrics.csv`; de uiteindelijke pipeline in `models/duration_model.joblib`. De API weigert te starten zonder getraind artifact, behalve met `ALLOW_FALLBACK_MODEL=true` voor een expliciete demo.

### Uitgevoerde Q1-2024-run

De lokale reproduceerbare run op januari–maart 2024 leverde 6.658.731 geldige ritten op. Casual
ritten hadden een mediaan van 11,54 minuten tegenover 7,70 voor members: verschil 3,85 minuten,
95%-bootstrap-BI [3,57; 4,12], Cliff’s δ = 0,297. Op een tijdshold-out (100.000 ritten) haalde
histogram gradient boosting MAE 6,69 minuten, tegenover 7,04 voor de mediaanbaseline; RMSE was
11,72 minuten en R² 0,049. Dit is een eerste Q1-run, geen claim dat het model alle seizoenen
generaliseert. De exacte waarden staan in `reports/` en worden door de notebooks opnieuw berekend.

## Deployment

```powershell
docker build -t citibike-duration .
docker run --rm -p 8000:8000 citibike-duration
```

`GET /health` geeft modelstatus; `POST /predict` geeft een puntschatting en duidelijke beperkingen. De GitHub Actions-workflow valideert iedere push en traint een smoke-model op deterministische testdata. Een handmatige workflow-run kan officiële maanden downloaden en een nieuw modelartifact produceren. Voor publieke hosting kan dezelfde container naar AWS App Runner, Render, Railway of een VM worden gebracht.

## Belangrijkste beperkingen

- Trip histories bevatten ritten, geen niet-gerealiseerde vraag: causaliteit of volledige vraagvoorspelling is niet mogelijk.
- Weer, evenementen, beschikbaarheid en fietspadcondities ontbreken.
- Stations en gebruikersgedrag veranderen in de tijd; driftmonitoring en periodiek hertrainen zijn nodig.
- Afgekapt ongeldige/extreme ritten worden gedocumenteerd en niet stilzwijgend als normale ritten behandeld.
- De AWS-notebook is uitvoerbaar, maar cloudtraining en hosting vereisen eigen credentials, budget en expliciete uitvoering.

## GenAI-disclosure

Generatieve AI is gebruikt om de eerste projectstructuur, documentatie en codevoorstellen op te stellen. Het team blijft verantwoordelijk voor het uitvoeren, controleren, interpreteren en mondeling verdedigen van iedere stap. Voeg vóór indiening concrete menselijke wijzigingen, uitgevoerde experimenten en teamnamen toe.
