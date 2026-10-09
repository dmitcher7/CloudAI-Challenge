# Citi Bike – vraag per zone per uur (Kobe)

**Vraag:** hoeveel Citi Bikes vertrekken er in een bepaalde zone van New York in een bepaald uur?
Dat is wat Citi Bike moet weten om fietsen op tijd te herverdelen. Een leeg station in de spits is een
gemiste groene rit (thema *Going green*).

- **Data:** alle 44,3 miljoen ritten van 2024 (officiële S3-bucket), plus het uurlijkse weer (Open-Meteo).
- **Target:** vertrekken per zone per uur. De 2.211 publieke stations zijn met KMeans gegroepeerd tot 30 zones.
- **Kenmerken** (allemaal gekend vóór het uur begint): zone, uur, weekdag, maand, weekend, feestdag, temperatuur, neerslag, wind.
- **Gedeployd model:** HistGradientBoosting met Poisson-verlies, via `/citi_bike_demand` op de bestaande API en webpagina `citi_bike_demand.html`.

## Notebooks (in volgorde uitvoeren)

| Notebook | Wat |
|---|---|
| [01_download.ipynb](01_download.ipynb) | 12 maanden automatisch downloaden (manifest + SHA-256), in stukken samenvatten tot vertrekken per station per uur, weer ophalen |
| [02_eda_cleaning.ipynb](02_eda_cleaning.ipynb) | EDA + cleaning: ontbrekende waarden, station-id's, uitschieters, niet-publieke stations, dagritme, weer, **zones (KMeans)** en **stationsprofielen (KMeans op gedrag)** |
| [03_hypotheses.ipynb](03_hypotheses.ipynb) | H1 (regen verlaagt de vraag) en H2 (ander dagritme in het weekend): toetsen + effectgroottes |
| [04_prepare_data.ipynb](04_prepare_data.ipynb) | Alle cleaning zonder grafieken: ruwe aggregaten → `data/model_table.parquet` |
| [05a_baseline.ipynb](05a_baseline.ipynb) | Baselines (gemiddelden) + snel eerste model voor de deployment |
| [05b_pycaret.ipynb](05b_pycaret.ipynb) | AutoML-vergelijking met PyCaret (zelfde split en folds) |
| [05c_poisson_glm.ipynb](05c_poisson_glm.ipynb) | Poisson-GLM: het uitlegbare model (effecten als factoren) |
| [05d_random_forest.ipynb](05d_random_forest.ipynb) | Random forest, randomized search |
| [05e_hist_gradient_boosting.ipynb](05e_hist_gradient_boosting.ipynb) | Gradient boosting (Poisson), randomized search + experiment met lag-kenmerken |
| [06_model_comparison.ipynb](06_model_comparison.ipynb) | Vergelijking, bootstrap per dag, foutenanalyse, permutation importance, conclusie |

Gedeelde code staat in [citibike.py](citibike.py): de notebooks en de pipeline verwerken de data zo op exact dezelfde manier.

## Validatie

- **Testset:** de laatste 7 dagen van elke maand (23% van de uren). Zo zit elk seizoen in de test. Een split "laatste twee maanden" zou alleen winter testen.
- **Cross-validatie:** `GroupKFold(5)` op kalenderweek binnen de trainset. Uren van dezelfde week komen nooit tegelijk in train en validatie.
- **Metrics:** MAE (primair, "x vertrekken per uur ernaast"), RMSE, R², Poisson-deviance.

## Resultaten

**Hypotheses (03):** regen kost −27% vertrekken per nat uur (95%-BI −32% tot −21%), met een dosis-effect tot 5 mm/u. Het dagritme verschilt tussen weekdag en weekend (Cramér's V = 0,20; ochtendspits 17,9% vs. 8,3% van de vertrekken).

| Model | CV-MAE | Test-MAE | Test-R² |
|---|---|---|---|
| Baseline: gemiddelde per zone × uur × weekend | 59,2 | 59,3 | 0,72 |
| Poisson-GLM (zone × uur × weekend) | 30,5 | 33,2 | 0,89 |
| Random forest (getuned) | 27,4 | 30,3 | 0,90 |
| PyCaret: Extra Trees (getuned) | 26,6 | 30,6 | 0,89 |
| HistGradientBoosting, standaard | 27,6 | 30,9 | 0,90 |
| **HistGradientBoosting, getuned (gedeployd)** | **24,9** | **29,0** | **0,91** |
| HistGB + lags 24u/168u (niet deploybaar) | 27,5 | 26,9 | 0,92 |

De gemiddelde vraag is 168 vertrekken per zone per uur, dus het gedeployde model zit gemiddeld ± 17% naast. Het wint in alle 5 folds van elk ander model. De grootste fouten zitten op Kerstmis, Thanksgiving en de dagen errond, en bij zware regen (zie `06`).

## Pipeline: van code naar productie

```
push naar citi_bike/Kobe/ (data, features, best_params.json, train.py)
  └─ .github/workflows/citi_bike_demand_retrain.yml
       ├─ test_features.py      API en training berekenen dezelfde kenmerken?
       ├─ train.py              hertrainen + kwaliteitspoort (MAE niet > 2% slechter dan productie)
       ├─ deploy/check_model.py API werkt met het nieuwe model?
       └─ pull request "Nieuw Citi Bike-vraagmodel" met deploy/citi_bike_demand/model.pkl
            └─ merge → deploy-api.yml (Oracle VM, met rollback) → api_smoke_test.yml
```

- `data/model_table.parquet` (2,3 MB) en `data/zones.json` staan in Git (uitzondering in `.gitignore`), zodat de pipeline kan trainen zonder 8,7 GB te downloaden. De ruwe zip's en tussenbestanden staan niet in Git; `01_download` maakt ze opnieuw.
- **Vereiste repo-instelling:** *Settings → Actions → General → "Allow GitHub Actions to create and approve pull requests"*.
- `make_features()` staat in `citibike.py` en (gekopieerd) in `deploy/citi_bike_demand/api.py`, omdat alleen `deploy/` naar de VM gaat. `test_features.py` faalt zodra de twee uit elkaar lopen.

## Lokaal uitvoeren

```powershell
py -3.11 -m venv venv
venv\Scripts\python -m pip install -r requirements.txt
cd citi_bike\Kobe
..\..\venv\Scripts\jupyter nbconvert --to notebook --execute --inplace 01_download.ipynb   # ± 10 min + download
# ... 02 t/m 06 op dezelfde manier
python train.py                       # model.pkl voor de API
python -m pytest test_features.py
```

API lokaal: `cd deploy && uvicorn app:app --port 8000`, open dan <http://localhost:8000/citi_bike_demand.html>.

## Op AWS SageMaker draaien

De modelnotebooks (05a–05e, 06) hebben alleen `citibike.py`, `data/model_table.parquet` en `data/zones.json` nodig, niet de 8,7 GB ruwe data. Upload die drie bestanden (in dezelfde mappenstructuur) samen met het notebook naar SageMaker Studio en installeer `requirements.txt`. Alle paden zijn relatief tot de map van `citibike.py`.
Om het AWS-model mee te nemen in `06_model_comparison`, bewaar je de resultaten in hetzelfde formaat met `cb.save_results("aws_xgboost", metrics, predictions)`. Dat schrijft `models/metrics/aws_xgboost.json` (met minstens `model`, `test_mae`, `test_rmse`, `test_r2`, `test_poisson_deviance`, en `cv_mae` als lijst van 5 folds) en `models/predictions/aws_xgboost.parquet` (kolommen `zone`, `time`, `trips`, `pred` voor de testset). 06 pikt elk extra bestand automatisch op.

## Beperkingen

- De data bevat **gerealiseerde** ritten, geen echte vraag: een leeg station telt als 0 vertrekken, ook als er mensen wachtten. Het model leert dus de vraag zoals het huidige systeem die kon bedienen.
- Eén jaar data (2024): het model kent geen groei of trends tussen jaren.
- Het weer komt van één punt in NYC uit een weermodel (Open-Meteo); lokale buien kunnen afwijken.
- Evenementen, wegenwerken en stationsstoringen zitten niet in het model.

## GenAI

De code en teksten zijn opgesteld met hulp van Claude (Anthropic). Alle keuzes, resultaten en interpretaties zijn nagekeken door Kobe en moeten mondeling verdedigd kunnen worden.
