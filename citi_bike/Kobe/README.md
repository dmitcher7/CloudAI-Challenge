# Citi Bike – vraag per zone per uur (Kobe)

**Vraag:** hoeveel Citi Bikes vertrekken er in een bepaalde zone van New York in een bepaald uur?
Dat is wat Citi Bike moet weten om fietsen op tijd te herverdelen. Een leeg station in de spits is een
gemiste groene rit (thema *Going green*).

- **Data:** alle **328,8 miljoen ritten** sinds de start van Citi Bike (juni 2013 – september 2026), rechtstreeks uit de officiële S3-bucket, plus het uurlijkse weer (Open-Meteo).
- **Target:** vertrekken per zone per uur. De stations zijn met KMeans gegroepeerd tot 30 zones (op het netwerk van de laatste 12 maanden).
- **Kenmerken** (allemaal gekend vóór het uur begint): zone, uur, weekdag, maand, weekend, feestdag, temperatuur, neerslag, wind, jaar, aantal actieve stations in de zone, sneeuwhoogte.
- **Gedeployd model:** HistGradientBoosting met Poisson-verlies, via `/citi_bike_demand` op de bestaande API en de webpagina `citi_bike_demand.html` (bereikbaar via de Citi Bike-keuzepagina).

## Aanpak in twee stappen

1. **Modelselectie op 2024.** Op één jaar (44 miljoen ritten) vergeleken we grondig zes modellen. Gradient boosting met Poisson-verlies won, in elke fold.
2. **Het definitieve model op alle data.** De gekozen techniek, getraind op 13 jaar. Meer jaren betekent meer voorbeelden van zeldzame situaties (feestdagen, stormen, sneeuw) maar ook een netwerk dat sterk veranderde. Daarom komen er drie kenmerken bij en wordt er tijdsgebonden gevalideerd.

## Notebooks (in volgorde uitvoeren)

| Notebook | Data | Wat |
|---|---|---|
| [01_download.ipynb](01_download.ipynb) | alle | alle NYC-bestanden downloaden (manifest + SHA-256), dubbele versies weglaten, in stukken samenvatten tot vertrekken per station per uur, weer ophalen |
| [02_eda_cleaning.ipynb](02_eda_cleaning.ipynb) | 2024 | EDA + cleaning in detail: ontbrekende waarden, station-id's, uitschieters, niet-publieke stations, dagritme, weer, **zones (KMeans)** en **stationsprofielen (KMeans op gedrag)** |
| [02b_eda_all_years.ipynb](02b_eda_all_years.ipynb) | alle | wat er bijkomt met 13 jaar: drie generaties kolomnamen en datumnotaties, dubbele bestanden, twee stelsels van station-id's, groei van het netwerk, corona, gesloten dagen door sneeuwstormen |
| [03_hypotheses.ipynb](03_hypotheses.ipynb) | 2024 | H1 (regen verlaagt de vraag) en H2 (ander dagritme in het weekend): toetsen + effectgroottes |
| [04_prepare_data.ipynb](04_prepare_data.ipynb) | alle | alle cleaning zonder grafieken → `data/model_table.parquet` |
| [05a_baseline_2024.ipynb](05a_baseline_2024.ipynb) | 2024 | baselines + snel eerste model |
| [05b_pycaret_2024.ipynb](05b_pycaret_2024.ipynb) | 2024 | AutoML met PyCaret (zelfde split en folds) |
| [05c_poisson_glm_2024.ipynb](05c_poisson_glm_2024.ipynb) | 2024 | Poisson-GLM: het uitlegbare model (effecten als factoren) |
| [05d_random_forest_2024.ipynb](05d_random_forest_2024.ipynb) | 2024 | random forest, twee zoekrondes |
| [05e_hist_gradient_boosting_2024.ipynb](05e_hist_gradient_boosting_2024.ipynb) | 2024 | gradient boosting (Poisson), randomized search + experiment met lag-kenmerken |
| [06_model_selectie_2024.ipynb](06_model_selectie_2024.ipynb) | 2024 | vergelijking, bootstrap per dag, foutenanalyse → keuze van de techniek |
| [07_baseline_all_data.ipynb](07_baseline_all_data.ipynb) | alle | baselines en eerste model op alle data, eerste versie van de deployment |
| [08_hgb_all_data.ipynb](08_hgb_all_data.ipynb) | alle | opnieuw tunen (backtesting), **helpt meer data?**, **helpen de nieuwe kenmerken?**, model voor de API |
| [09_model_comparison.ipynb](09_model_comparison.ipynb) | alle | vergelijking, foutenanalyse (feestdagen, sneeuw, regen, zones), permutation importance, conclusie |

Gedeelde code staat in [citibike.py](citibike.py): de notebooks en `train.py` verwerken de data zo op exact dezelfde manier.
De 2024-studie gebruikt `data/2024/` en `models/2024/`; de studie op alle data `data/` en `models/`.

## Validatie

| | Modelselectie (2024) | Definitief model (alle data) |
|---|---|---|
| Testset | laatste 7 dagen van elke maand | de laatste 12 maanden (oktober 2025 – september 2026) |
| Validatie bij tunen | `GroupKFold(5)` op kalenderweek | **backtesting**: 3 folds, telkens trainen op alle jaren vóór een validatiejaar |
| Waarom | elk seizoen in de test, geen lek via naburige uren | met 13 jaar is er een sterke trend; een model moet altijd vooruit voorspellen |

Metrics: MAE (primair, "x vertrekken per uur ernaast"), RMSE, R², Poisson-deviance.

## Resultaten

**Hypotheses (03, 2024):** regen kost −27% vertrekken per nat uur (95%-BI −32% tot −21%), met een dosis-effect tot 5 mm/u. Het dagritme verschilt tussen weekdag en weekend (Cramér's V = 0,20; ochtendspits 17,9% vs. 8,3% van de vertrekken).

**Modelselectie op 2024 (06):**

| Model | CV-MAE | Test-MAE | Test-R² |
|---|---|---|---|
| Baseline: gemiddelde per zone × uur × weekend | 59,2 | 59,3 | 0,72 |
| Poisson-GLM (zone × uur × weekend) | 30,5 | 33,2 | 0,89 |
| Random forest (getuned) | 27,4 | 30,3 | 0,90 |
| PyCaret: Extra Trees (getuned) | 26,6 | 30,6 | 0,89 |
| HistGradientBoosting, standaard | 27,6 | 30,9 | 0,90 |
| **HistGradientBoosting, getuned** | **24,9** | **29,0** | **0,91** |

**Definitief model op alle data (07–09):**

Testset: oktober 2025 – september 2026 (262.080 zone-uren, gemiddeld 173 vertrekken per zone-uur).

| Model | Backtest-MAE | Test-MAE | Test-R² |
|---|---|---|---|
| Baseline: zone × uur × weekend, vorig jaar | 57,6 | 71,0 | 0,68 |
| HistGradientBoosting, standaard | 36,4 | 36,1 | 0,90 |
| HistGradientBoosting, getuned, alleen het laatste jaar | | 30,3 | 0,925 |
| **HistGradientBoosting, getuned, alle data (gedeployd)** | **31,8** | **28,9** | **0,931** |

- **Meer data helpt**, met afnemende winst: 1 jaar 30,3 → 2 jaar 29,8 → 3 jaar 29,3 → 5 jaar 29,1 → alles 28,9 (bootstrap-winst tegenover één jaar: 1,4, 95%-BI 0,7 tot 2,1).
- **Nieuwe kenmerken:** zonder `year` stijgt de fout naar 35,1; zonder `year`, `active_stations` en `snow_depth_cm` samen naar 79 (het model middelt dan 13 jaar groei uit).
- **Feestdagen** gaan nu goed (Thanksgiving +5%, Kerstmis +26%; in de 2024-studie +165% en +213%). De grootste fouten zitten nu in de kerstvakantie en vlak na sneeuwstormen.
- **Iteratie na de foutenanalyse:** de sneeuwhoogte kwam erbij omdat de grootste fouten op droge dagen na een sneeuwstorm vielen (test-MAE 30,1 → 28,9; in de backtest neutraal, want die jaren hadden weinig sneeuw).

## Van notebook naar productie

Alle modellen worden lokaal getraind.

```
08 (tuning) -> models/best_params.json
  -> train.py: hertrainen op alle uren + kwaliteitscontrole (MAE niet > 2% slechter dan het model dat er staat)
  -> deploy/citi_bike_demand/model.pkl
  -> pull request naar main -> merge -> deploy-api.yml (Oracle VM, met rollback) -> api_smoke_test.yml
```

- `data/model_table.parquet` (28 MB, alle jaren), `data/2024/model_table.parquet` (2,3 MB) en `data/zones.json` staan in Git (uitzondering in `.gitignore`), zodat iedereen de modellen kan hertrainen zonder 31 GB te downloaden. De ruwe zip's en tussenbestanden staan niet in Git; `01_download` maakt ze opnieuw (± 1 uur download + ± 1 uur verwerking).
- De API gebruikt voor nieuwe voorspellingen het **huidige** aantal actieve stations per zone (uit `zones.json`), en de webpagina haalt temperatuur, neerslag, wind en sneeuwhoogte op bij Open-Meteo.
- `make_features()` staat in `citibike.py` en (gekopieerd) in `deploy/citi_bike_demand/api.py`, omdat alleen `deploy/` naar de VM gaat. Draai `python -m pytest test_features.py` na een wijziging: de test faalt zodra de twee uit elkaar lopen.

## Lokaal uitvoeren

```powershell
py -3.11 -m venv venv
venv\Scripts\python -m pip install -r requirements.txt
cd citi_bike\Kobe
..\..\venv\Scripts\jupyter nbconvert --to notebook --execute --inplace 01_download.ipynb   # download + verwerking
# ... daarna 02 t/m 09 op dezelfde manier (08 duurt ± 2 uur)
python train.py                       # model.pkl voor de API
python -m pytest test_features.py
```

API lokaal: `cd deploy && uvicorn app:app --port 8000`, open dan <http://localhost:8000/>.

## Op AWS SageMaker draaien

De modelnotebooks hebben alleen `citibike.py`, `data/model_table.parquet` (of `data/2024/model_table.parquet`) en `data/zones.json` nodig, niet de 31 GB ruwe data. Upload die bestanden (in dezelfde mappenstructuur) samen met het notebook naar SageMaker Studio en installeer `requirements.txt`. Alle paden zijn relatief tot de map van `citibike.py`.
Om een AWS-model mee te nemen in `09_model_comparison`, bewaar je de resultaten met `cb.save_results("aws_xgboost", metrics, predictions)`: `models/metrics/aws_xgboost.json` (minstens `model`, `test_mae`, `test_rmse`, `test_r2`, `test_poisson_deviance`, en `cv_mae` als lijst van de 3 backtest-folds) en `models/predictions/aws_xgboost.parquet` (kolommen `zone`, `time`, `trips`, `pred` voor de testset). 09 pikt elk extra bestand automatisch op.

## Beperkingen

- De data bevat **gerealiseerde** ritten, geen echte vraag: een leeg station telt als 0 vertrekken, ook als er mensen wachtten. Het model leert de vraag zoals het systeem die kon bedienen.
- Het weer komt van één punt in NYC uit een weermodel (Open-Meteo); lokale buien kunnen afwijken.
- Evenementen, wegenwerken, stationsstoringen en vakantieperiodes (zoals de week tussen Kerstmis en Nieuwjaar) zitten niet in het model.
- Voor de toekomst neemt het model aan dat het netwerk blijft zoals het nu is (huidig aantal actieve stations per zone).

## GenAI

De code en teksten zijn opgesteld met hulp van Claude (Anthropic). Alle keuzes, resultaten en interpretaties zijn nagekeken door Kobe en moeten mondeling verdedigd kunnen worden.
