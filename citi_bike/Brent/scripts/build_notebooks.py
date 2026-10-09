"""Generate the ordered notebooks from reviewable source cells.

This keeps notebook JSON noise out of hand-written code while still committing runnable
`.ipynb` files required by the assignment.
"""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": dedent(text).strip().splitlines(True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(text).strip().splitlines(True),
    }


def notebook(title: str, purpose: str, cells: list[dict]) -> dict:
    intro = md(
        f"""
        # {title}

        **Auteurs:** _vul namen en concrete bijdragen in vóór indiening_  
        **Doel:** {purpose}

        Elke codecel wordt voorafgegaan door uitleg. Voer de notebooks in nummervolgorde uit
        vanaf de repository-root. Resultaten moeten door het team zelf worden gecontroleerd en
        geïnterpreteerd; synthetische CI-data gelden nooit als onderzoeksbewijs.
        """
    )
    bootstrap = code(
        """
        # Maak lokale projectimports betrouwbaar in VS Code, JupyterLab en nbconvert.
        from pathlib import Path
        import sys

        PROJECT_ROOT = next(
            (
                path
                for path in [Path.cwd(), *Path.cwd().parents]
                if (path / "pyproject.toml").exists() and (path / "src").is_dir()
            ),
            None,
        )
        if PROJECT_ROOT is None:
            raise RuntimeError("Projectroot niet gevonden; open de map Brent als VS Code-workspace.")
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))
        print(f"Projectroot: {PROJECT_ROOT}")
        """
    )
    all_cells = [intro, bootstrap, *cells]
    for index, cell in enumerate(all_cells):
        cell["id"] = f"cell-{index:03d}"
    return {
        "cells": all_cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python (venv)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


NOTEBOOKS = {
    "01_download_and_audit.ipynb": notebook(
        "01 · Download en data-audit",
        "Officiële maandarchieven met code downloaden, valideren en de herkomst vastleggen.",
        [
            md(
                """
                ## Keuze van periode

                De standaardrun gebruikt het volledige kalenderjaar 2024, zodat alle seizoenen in de
                analyse voorkomen. De periode is centraal vastgelegd, zodat scripts en notebooks
                dezelfde grenzen gebruiken en cherry-picking achteraf wordt voorkomen.
                """
            ),
            code(
                """
                from pathlib import Path
                import json
                import pandas as pd
                from src.config import DEFAULT_END_MONTH, DEFAULT_START_MONTH, RAW_DIR, WEATHER_PATH
                from src.data.download import download_range
                from src.data.weather import download_hourly_weather

                START_MONTH = DEFAULT_START_MONTH
                END_MONTH = DEFAULT_END_MONTH
                RUN_DOWNLOAD = False    # zet bewust op True; download kan meerdere GB zijn
                if RUN_DOWNLOAD:
                    download_range(START_MONTH, END_MONTH)
                    download_hourly_weather(START_MONTH, END_MONTH, WEATHER_PATH)
                """
            ),
            md("## Controle van provenance\n\nHet manifest maakt elke bronfile controleerbaar met URL, grootte en SHA-256."),
            code(
                """
                manifest_path = RAW_DIR / "manifest.json"
                if not manifest_path.exists():
                    print("Nog geen manifest. Zet RUN_DOWNLOAD=True en voer de vorige cel uit.")
                else:
                    manifest = pd.DataFrame(json.loads(manifest_path.read_text()))
                    display(manifest)
                    print(f"Totaal: {manifest['bytes'].sum() / 1e9:.2f} GB")
                    assert manifest["sha256"].str.len().eq(64).all()
                """
            ),
            md("## Schema-audit zonder alles in geheugen te laden\n\nWe lezen alleen de headers en vergelijken kolommen per CSV-lid."),
            code(
                """
                from zipfile import ZipFile

                schemas = []
                for path in sorted(RAW_DIR.glob("*-citibike-tripdata.zip")):
                    with ZipFile(path) as archive:
                        for member in archive.namelist():
                            if member.lower().endswith(".csv"):
                                with archive.open(member) as handle:
                                    columns = pd.read_csv(handle, nrows=0).columns.tolist()
                                schemas.append({"archive": path.name, "member": member, "columns": columns})
                display(pd.DataFrame(schemas))
                """
            ),
            md(
                """
                ## Besluit

                Het downloadscript faalt expliciet bij corrupte ZIP's en schrijft pas na succesvolle
                validatie. Schema-afwijkingen worden hier zichtbaar voordat cleaning start. Noteer in
                de definitieve versie de gedownloade periode, omvang en eventuele afwijkingen.
                """
            ),
        ],
    ),
    "02_eda_and_cleaning.ipynb": notebook(
        "02 · EDA en cleaning",
        "Datakwaliteit onderzoeken en een inhoudelijk verhaal met aggregaties en grafieken bouwen.",
        [
            md("## Laden\n\nNotebook 03 maakt het Parquet-bestand. Voor de EDA nemen we maximaal één miljoen gelijkmatig verdeelde rijen om plots responsief te houden."),
            code(
                """
                import json
                import numpy as np
                import pandas as pd
                import matplotlib.pyplot as plt
                import seaborn as sns
                from src.config import PROCESSED_DIR

                sns.set_theme(style="whitegrid")
                path = PROCESSED_DIR / "trips.parquet"
                trips = pd.read_parquet(path).sort_values("started_at")
                if len(trips) > 1_000_000:
                    trips = trips.iloc[np.linspace(0, len(trips)-1, 1_000_000, dtype=int)].copy()
                print(f"EDA-steekproef: {len(trips):,} ritten, {trips.started_at.min()} t/m {trips.started_at.max()}")
                display(trips.head())
                """
            ),
            md("## Datakwaliteit\n\nWe kwantificeren ontbrekende waarden en duplicaten. Missende eindstations mogen voor ons voorspeldoel blijven bestaan; een ontbrekend startstation niet."),
            code(
                """
                quality = pd.DataFrame({
                    "dtype": trips.dtypes.astype(str),
                    "missing_n": trips.isna().sum(),
                    "missing_pct": trips.isna().mean().mul(100).round(2),
                    "unique": trips.nunique(dropna=False),
                }).sort_values("missing_pct", ascending=False)
                display(quality)
                print("Dubbele ride_id's:", trips.ride_id.duplicated().sum())
                display(trips.duration_minutes.describe(percentiles=[.5, .9, .95, .99]).to_frame())
                """
            ),
            md("## Cleaning-impact\n\nDe audit telt redenen niet exclusief: één rij kan tegelijk een ongeldige duur én locatie hebben. Daarom tellen redenkolommen niet op tot het totaal verwijderd."),
            code(
                """
                audit_path = PROCESSED_DIR / "cleaning_audit.json"
                if audit_path.exists():
                    audit = json.loads(audit_path.read_text())
                    summary = {k: v for k, v in audit.items() if k != "chunks"}
                    display(pd.Series(summary, name="waarde"))
                    print(f"Behoud: {audit['output_rows'] / audit['input_rows']:.1%}")
                """
            ),
            md("## Aggregatie 1 — wanneer wordt gereden?\n\nAantal ritten en mediane duur beantwoorden verschillende vragen: gebruiksvolume versus typisch gedrag."),
            code(
                """
                trips["hour"] = trips.started_at.dt.hour
                hourly = trips.groupby(["hour", "member_casual"], observed=True).agg(
                    rides=("ride_id", "size"), median_minutes=("duration_minutes", "median")
                ).reset_index()
                fig, axes = plt.subplots(1, 2, figsize=(14, 4))
                sns.lineplot(data=hourly, x="hour", y="rides", hue="member_casual", ax=axes[0])
                sns.lineplot(data=hourly, x="hour", y="median_minutes", hue="member_casual", ax=axes[1])
                axes[0].set(title="Ritten per startuur", ylabel="ritten in steekproef")
                axes[1].set(title="Mediane ritduur per startuur", ylabel="minuten")
                plt.tight_layout()
                display(hourly.head())
                """
            ),
            md("## Aggregatie 2 — weekritme\n\nWeekdag wordt geordend, zodat alfabetische sortering geen misleidend patroon maakt."),
            code(
                """
                order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
                trips["weekday"] = pd.Categorical(trips.started_at.dt.day_name().str[:3], order, ordered=True)
                daily = trips.groupby(["weekday", "member_casual"], observed=True).agg(
                    rides=("ride_id", "size"), median_minutes=("duration_minutes", "median")
                ).reset_index()
                fig, axes = plt.subplots(1, 2, figsize=(14, 4))
                sns.barplot(data=daily, x="weekday", y="rides", hue="member_casual", ax=axes[0])
                sns.pointplot(data=daily, x="weekday", y="median_minutes", hue="member_casual", ax=axes[1])
                axes[0].set(title="Ritvolume per weekdag", ylabel="ritten")
                axes[1].set(title="Typische ritduur per weekdag", ylabel="mediane minuten")
                plt.tight_layout()
                """
            ),
            md("## Aggregatie 3 — stations\n\nAlleen stations met minstens 500 ritten worden voor mediane duur gerangschikt; kleine groepen leveren instabiele extremen."),
            code(
                """
                stations = trips.groupby(["start_station_id", "start_station_name"], observed=True).agg(
                    rides=("ride_id", "size"), median_minutes=("duration_minutes", "median"),
                    casual_share=("member_casual", lambda s: (s == "casual").mean()),
                ).reset_index()
                stable = stations.query("rides >= 500").nlargest(15, "rides")
                plt.figure(figsize=(9, 6))
                sns.barplot(data=stable, y="start_station_name", x="rides", color="#4f8f67")
                plt.title("Top-15 startstations in de EDA-steekproef")
                plt.tight_layout()
                display(stations.sort_values("rides", ascending=False).head(20))
                """
            ),
            md("## Verdeling en outliers\n\nDe rechterstaart motiveert MAE en een log-getransformeerde target bij modelleren. De grens van 180 minuten is een vooraf gedocumenteerde plausibiliteitsregel, geen op de testscore afgestemde keuze."),
            code(
                """
                fig, axes = plt.subplots(1, 2, figsize=(14, 4))
                sns.histplot(trips.duration_minutes, bins=80, ax=axes[0])
                sns.boxplot(data=trips, x="member_casual", y="duration_minutes", showfliers=False, ax=axes[1])
                axes[0].set(xlabel="duur (minuten)", title="Ritduur na plausibiliteitsfilter")
                axes[1].set(xlabel="type rijder", ylabel="duur (minuten)", title="Verdeling zonder getekende uitschieters")
                plt.tight_layout()
                """
            ),
            md("## Interpretatiechecklist\n\nVul na uitvoering concrete getallen in: periode en N, grootste piekuren, member/casual-verschil, missings, cleaning-impact en beperkingen. Formuleer associaties; deze observationele data ondersteunen geen causale claims."),
        ],
    ),
    "03_prepare_data.ipynb": notebook(
        "03 · Definitieve datavoorbereiding",
        "Zonder grafieken de volledige raw→prepared-transformatie reproduceerbaar uitvoeren.",
        [
            md("## Eén bron van waarheid\n\nAlle cleaningregels leven in `src/data/prepare.py`; dit notebook roept die code aan en voert acceptatiechecks uit."),
            code(
                """
                from pathlib import Path
                import json
                import pandas as pd
                from src.config import RAW_DIR, PROCESSED_DIR
                from src.data.prepare import prepare_archives

                OUTPUT = PROCESSED_DIR / "trips.parquet"
                summary = prepare_archives(RAW_DIR, OUTPUT)
                display(pd.Series({k: v for k, v in summary.items() if k != "chunks"}))
                """
            ),
            md("## Acceptatiechecks\n\nDeze invarianten voorkomen dat schemawijzigingen stilletjes verkeerde trainingsdata opleveren."),
            code(
                """
                data = pd.read_parquet(OUTPUT)
                assert len(data) == summary["output_rows"]
                assert data.ride_id.is_unique
                assert data.duration_minutes.between(1, 180).all()
                assert data.member_casual.isin(["member", "casual"]).all()
                assert data.start_lat.between(40.4, 41.1).all()
                assert data.start_lng.between(-74.4, -73.5).all()
                assert data.started_at.notna().all()
                assert data.start_station_id.notna().all()
                print(f"PASS — {len(data):,} rijen, {data.started_at.min()} t/m {data.started_at.max()}")
                """
            ),
            md("## Outputcontract\n\nDe Parquet-data en audit worden niet gecommit vanwege omvang. Het downloadmanifest, codeversie en requirements maken reconstructie mogelijk."),
        ],
    ),
    "04_hypothesis.ipynb": notebook(
        "04 · Toetsbare hypothese",
        "Vóór modelleren aantonen of een praktisch betekenisvol voorspelpatroon bestaat.",
        [
            md(
                """
                ## Vooraf vastgelegd

                - **H₀:** ritduur heeft dezelfde verdeling voor casual riders en members.
                - **H₁ (eenzijdig):** casual riders maken typisch langere ritten.
                - **α = 0,05.**
                - Primair praktisch effect: verschil in mediaan met 95%-bootstrap-BI.
                - Secundair: Cliff's delta; de Mann–Whitney-toets levert statistische evidentie.

                Dit is associationeel, niet causaal: rijderstype kan samenhangen met route, dag en seizoen.
                """
            ),
            code(
                """
                import json
                import pandas as pd
                import matplotlib.pyplot as plt
                import seaborn as sns
                from src.analysis.hypothesis import test_member_difference
                from src.config import PROCESSED_DIR, REPORT_DIR

                trips = pd.read_parquet(PROCESSED_DIR / "trips.parquet", columns=["member_casual", "duration_minutes"])
                result = test_member_difference(trips)
                print(json.dumps(result, indent=2))
                REPORT_DIR.mkdir(exist_ok=True)
                (REPORT_DIR / "hypothesis_result.json").write_text(json.dumps(result, indent=2))
                """
            ),
            md("## Visualisatie van effect, niet alleen p-waarde\n\nWe tonen medianen en het bootstrapinterval van hun verschil."),
            code(
                """
                medians = trips.groupby("member_casual").duration_minutes.median().sort_index()
                ax = medians.plot.bar(color=["#d6ff70", "#283d32"], figsize=(7, 4))
                ax.set(title="Mediane ritduur per type rijder", xlabel="", ylabel="minuten")
                plt.xticks(rotation=0)
                plt.show()
                print("95%-BI mediaanverschil casual − member:", result["median_difference_ci95"])
                """
            ),
            md(
                """
                ## Beslisregel en conclusie

                Verwerp H₀ alleen wanneer p < 0,05 én bespreek of effectgrootte/BI praktisch relevant
                zijn. Een klein effect met een minuscule p-waarde is bij miljoenen ritten geen sterke
                productreden. Noteer hier na uitvoering de concrete conclusie en hoe dit de feature
                `member_casual` motiveert.
                """
            ),
        ],
    ),
    "05_automl_and_models.ipynb": notebook(
        "05 · AutoML en handmatige modellen",
        "Een baseline, AutoML en meerdere uitlegbare kandidaten eerlijk vergelijken.",
        [
            md(
                """
                ## Evaluatieontwerp

                De laatste 20% van ritten vormt de onaangeraakte hold-outset. Een random split zou
                toekomstige stations- en gedragspatronen naar training lekken. MAE is primair; RMSE,
                MedAE en R² maken verschillende foutaspecten zichtbaar. Features bevatten uitsluitend
                informatie die bij vertrek bekend is, inclusief de vooraf gekozen bestemming en
                het weer op het geplande vertrekuur.
                """
            ),
            code(
                """
                from src.config import PROCESSED_DIR, MODEL_DIR, REPORT_DIR
                from src.models.train import train_and_compare

                bundle, metrics = train_and_compare(
                    PROCESSED_DIR / "trips.parquet",
                    MODEL_DIR / "duration_model.joblib",
                    REPORT_DIR / "metrics.csv",
                    max_rows=500_000,
                    quick=False,
                )
                display(metrics.style.format({"mae_minutes":"{:.2f}", "rmse_minutes":"{:.2f}", "r2":"{:.3f}"}))
                """
            ),
            md(
                """
                ## Waarom deze modellen?

                De mediaan is de eerlijke naïeve referentie. Ridge test een eenvoudige lineaire relatie.
                Histogram Gradient Boosting vangt niet-lineaire interacties efficiënt. Extra Trees kan
                complexere stations- en tijdspatronen leren, maar overfit makkelijker. De target krijgt
                een log1p-transformatie wegens de lange rechterstaart. De hemelsbrede afstand is een
                lekvrije ondergrens voor de routeafstand, omdat de bestemming vooraf wordt ingevoerd.
                """
            ),
            md("## AutoML (FLAML)\n\nFLAML vervangt hier PyCaret als toegestane AutoML-vergelijking. Het tijdsbudget maakt de run reproduceerbaar qua kosten; de uitkomst kan per hardware iets verschillen."),
            code(
                """
                RUN_AUTOML = False
                if RUN_AUTOML:
                    from src.models.automl import run_automl
                    automl_result = run_automl(PROCESSED_DIR / "trips.parquet", seconds=600)
                    display(automl_result)
                else:
                    print("Zet RUN_AUTOML=True na installatie van de optionele FLAML dependency.")
                """
            ),
            md("## Selectie\n\nDe pipeline kiest de laagste hold-out-MAE, niet de trainingsscore. Controleer in notebook 06 of winst breed gedragen wordt en niet uit één subgroup komt."),
        ],
    ),
    "06_model_comparison.ipynb": notebook(
        "06 · Modelvergelijking en foutanalyse",
        "De winnaar toetsen op tijd, subgroepen en concrete grote fouten voordat deployment volgt.",
        [
            md("## Scorekaart\n\nEen model moet de mediaanbaseline merkbaar verslaan en acceptabele staartfouten hebben."),
            code(
                """
                import joblib
                import numpy as np
                import pandas as pd
                import matplotlib.pyplot as plt
                import seaborn as sns
                from src.config import MODEL_DIR, PROCESSED_DIR, REPORT_DIR, TARGET, WEATHER_PATH
                from src.data.weather import attach_hourly_weather
                from src.features.build import make_features
                from src.models.train import temporal_split

                metrics = pd.read_csv(REPORT_DIR / "metrics.csv").sort_values("mae_minutes")
                display(metrics)
                ax = metrics.plot.barh(x="model", y="mae_minutes", legend=False, color="#4f8f67")
                ax.set(xlabel="hold-out MAE (minuten)", ylabel="", title="Lagere fout is beter")
                plt.show()
                """
            ),
            md("## Residuen van het geselecteerde model\n\nWe reconstrueren exact dezelfde chronologische hold-outset en bewaren de grootste fouten voor bespreking."),
            code(
                """
                bundle = joblib.load(MODEL_DIR / "duration_model.joblib")
                columns = [
                    "started_at", "rideable_type", "member_casual", "start_station_id",
                    "end_station_id", "start_lat", "start_lng", "end_lat", "end_lng", TARGET,
                ]
                frame = pd.read_parquet(PROCESSED_DIR / "trips.parquet", columns=columns).sort_values("started_at")
                if len(frame) > 500_000:
                    frame = frame.iloc[np.linspace(0, len(frame)-1, 500_000, dtype=int)]
                frame = attach_hourly_weather(frame, WEATHER_PATH)
                _, test = temporal_split(frame)
                test = test.copy()
                test["prediction"] = np.maximum(0, bundle["model"].predict(make_features(test)))
                test["error"] = test.prediction - test[TARGET]
                test["absolute_error"] = test.error.abs()
                display(test.nlargest(20, "absolute_error"))
                """
            ),
            md("## Subgroepanalyse\n\nGemiddelden kunnen een zwakke groep verbergen. We vergelijken rijderstype, fiets en weekend versus werkdag."),
            code(
                """
                test["weekpart"] = np.where(test.started_at.dt.dayofweek >= 5, "weekend", "weekday")
                subgroup = test.groupby(["member_casual", "rideable_type", "weekpart"], observed=True).agg(
                    n=("absolute_error", "size"),
                    mae=("absolute_error", "mean"),
                    median_ae=("absolute_error", "median"),
                    bias=("error", "mean"),
                ).reset_index().query("n >= 100")
                display(subgroup.sort_values("mae", ascending=False))
                """
            ),
            md("## Stabiliteit door de tijd\n\nMaandelijkse MAE signaleert seizoensdrift of een problematische laatste periode."),
            code(
                """
                test["month"] = test.started_at.dt.to_period("M").astype(str)
                monthly_error = test.groupby("month").agg(n=("absolute_error","size"), mae=("absolute_error","mean"), bias=("error","mean")).reset_index()
                display(monthly_error)
                sns.lineplot(data=monthly_error, x="month", y="mae", marker="o")
                plt.xticks(rotation=45); plt.title("Hold-out MAE door de tijd"); plt.tight_layout()
                """
            ),
            md(
                """
                ## Definitief besluit

                Accepteer de automatisch gekozen winnaar alleen als die de baseline verslaat, de winst
                niet door één kleine groep komt, bias beperkt is en de API-latency praktisch blijft.
                Beschrijf hier de concrete score, baselineverbetering, zwakste subgroup en vermoedelijke
                oorzaak (ontbrekend weer/evenement/route). Dit is de model card voor de presentatie.
                """
            ),
        ],
    ),
    "07_aws_sagemaker.ipynb": notebook(
        "07 · AWS SageMaker training",
        "Minstens één model reproduceerbaar in AWS trainen en artifacts/metrics terughalen.",
        [
            md(
                """
                ## Voorwaarden en kosten

                Dit notebook draait alleen in een geconfigureerd AWS-account. Controleer regio, IAM-rol,
                S3-bucket en budgetalarm. Gebruik eerst `ml.m5.large`, stop endpoints na evaluatie en
                commit geen credentials. De lokale Parquet-data wordt naar een eigen versleutelde
                bucket geüpload; controleer de Citi Bike data policy.
                """
            ),
            code(
                """
                # Uitvoeren in SageMaker Studio of na: pip install sagemaker boto3
                import sagemaker
                from sagemaker.sklearn.estimator import SKLearn
                from src.config import PROCESSED_DIR, WEATHER_PATH

                session = sagemaker.Session()
                role = sagemaker.get_execution_role()
                bucket = session.default_bucket()
                prefix = "green-wheels/citibike"
                session.upload_data(
                    str(PROCESSED_DIR / "trips.parquet"), bucket=bucket, key_prefix=f"{prefix}/train"
                )
                session.upload_data(
                    str(WEATHER_PATH), bucket=bucket, key_prefix=f"{prefix}/train"
                )
                train_s3 = f"s3://{bucket}/{prefix}/train"
                print(train_s3)
                """
            ),
            md("## Training job\n\nDe AWS-job gebruikt dezelfde code, temporal hold-out en metrics als lokaal; alleen de computeomgeving verandert."),
            code(
                """
                estimator = SKLearn(
                    # Root als source_dir bevat zowel aws/train_entrypoint.py als src/.
                    # Zo gebruikt SageMaker exact dezelfde feature- en modelcode als lokaal.
                    entry_point="aws/train_entrypoint.py",
                    source_dir="..",
                    role=role,
                    instance_type="ml.m5.large",
                    instance_count=1,
                    framework_version="1.2-1",
                    py_version="py3",
                    hyperparameters={"max-rows": 500000},
                    output_path=f"s3://{bucket}/{prefix}/output",
                    base_job_name="citibike-duration",
                )
                estimator.fit({"train": train_s3}, wait=True, logs=True)
                print("Model artifact:", estimator.model_data)
                """
            ),
            md("## Bewijs en vergelijking\n\nDownload `model.tar.gz`, bewaar jobnaam/configuratie en voeg de AWS-metrics als aparte rij toe aan notebook 06. Screenshots alleen als aanvulling, nooit als vervanging van code/config."),
            code(
                """
                import boto3
                description = boto3.client("sagemaker").describe_training_job(
                    TrainingJobName=estimator.latest_training_job.name
                )
                evidence = {
                    "job_name": description["TrainingJobName"],
                    "status": description["TrainingJobStatus"],
                    "instance": description["ResourceConfig"]["InstanceType"],
                    "model_artifact": description["ModelArtifacts"]["S3ModelArtifacts"],
                    "training_seconds": description.get("TrainingTimeInSeconds"),
                }
                evidence
                """
            ),
            md("## Opruimen\n\nDeze training maakt geen persistent endpoint. Verwijder tijdelijke S3-input en outputs pas nadat benodigde artifacts lokaal zijn veiliggesteld en volgens teambeleid mogen worden verwijderd."),
        ],
    ),
    "08_model_dashboard.ipynb": notebook(
        "08 · Visueel modeldashboard",
        "Alle features, technieken, patronen en modelresultaten in één presentatieklaar overzicht tonen.",
        [
            md(
                """
                ## Wat toont dit dashboard?

                Dit notebook traint niets opnieuw. Het leest de bestaande jaargegevens, het gekozen
                model en de opgeslagen metrics. Voor grafieken gebruiken we een systematische
                steekproef uit het volledige Parquet-bestand; de officiële modelmetrics blijven de
                waarden uit de chronologische hold-out van 100.000 ritten.

                De historische eindlocatie wordt hier geïnterpreteerd als proxy voor een **vooraf
                geplande bestemming**. Zonder bekende bestemming mag deze informatie niet als feature
                worden gebruikt.
                """
            ),
            code(
                """
                import json
                import joblib
                import numpy as np
                import pandas as pd
                import matplotlib.pyplot as plt
                import seaborn as sns
                from IPython.display import Markdown, display
                from sklearn.inspection import permutation_importance

                from src.config import MODEL_DIR, MODEL_FEATURES, PROCESSED_DIR, REPORT_DIR, TARGET, WEATHER_PATH
                from src.data.sample import systematic_parquet_sample
                from src.data.weather import attach_hourly_weather
                from src.features.build import make_features

                sns.set_theme(style="whitegrid", palette="crest")
                SAMPLE_ROWS = 300_000
                DATA_COLUMNS = [
                    "started_at", "rideable_type", "member_casual", "start_station_id",
                    "end_station_id", "start_lat", "start_lng", "end_lat", "end_lng", TARGET,
                ]

                trips = systematic_parquet_sample(
                    PROCESSED_DIR / "trips.parquet", max_rows=SAMPLE_ROWS, columns=DATA_COLUMNS
                ).sort_values("started_at").reset_index(drop=True)
                trips = attach_hourly_weather(trips, WEATHER_PATH)
                feature_frame = make_features(trips).reset_index(drop=True)
                dashboard = pd.concat(
                    [trips[["started_at", TARGET]].reset_index(drop=True), feature_frame], axis=1
                )
                bundle = joblib.load(MODEL_DIR / "duration_model.joblib")
                metrics = pd.read_csv(REPORT_DIR / "metrics.csv").sort_values("mae_minutes")
                audit = json.loads((PROCESSED_DIR / "cleaning_audit.json").read_text())

                holdout = dashboard.loc[
                    dashboard.started_at >= pd.Timestamp(bundle["holdout_period"][0])
                ].copy()
                if len(holdout) > 50_000:
                    holdout = holdout.sample(50_000, random_state=42)
                holdout["prediction"] = np.clip(
                    bundle["model"].predict(holdout[MODEL_FEATURES]), 0, 180
                )
                holdout["error"] = holdout.prediction - holdout[TARGET]
                holdout["absolute_error"] = holdout.error.abs()

                print(
                    f"Dashboardsteekproef: {len(dashboard):,} van {audit['output_rows']:,} ritten | "
                    f"dashboard-hold-out: {len(holdout):,} ritten"
                )
                """
            ),
            md(
                """
                ## Technieken die in het project zijn gebruikt

                | Onderdeel | Technieken |
                |---|---|
                | Dataverzameling | Geautomatiseerde maanddownloads, ZIP-validatie, SHA-256-manifest en externe uurweerdata |
                | Dataverwerking | Chunk processing, schemaharmonisatie, typeconversie, plausibiliteitsfilters, deduplicatie en Parquet |
                | Statistiek | Eenzijdige Mann–Whitney-U-toets, bootstrap-BI voor mediaanverschil en Cliff's delta |
                | Leakagepreventie | Alleen vertrek- en planningsinformatie, chronologische 80/20-split en een onaangeraakte toekomstige hold-out |
                | Feature engineering | Cyclische tijdcodering, weekend/feestdag, Haversine-afstand, geografische delta's en uurweer |
                | Preprocessing | Mediaanimputatie, meest-frequente imputatie, one-hot encoding voor Ridge en ordinal encoding voor boommodellen |
                | Modellen | Mediaanbaseline, Ridge-regressie, Histogram Gradient Boosting, Extra Trees en optioneel FLAML AutoML |
                | Targetbehandeling | `log1p` tegen de lange rechterstaart, veilige inverse transformatie en begrenzing tot het geldige doelbereik |
                | Evaluatie | MAE als primaire metric, daarnaast RMSE, MedAE, R², subgroepanalyse, residuen en permutation importance |
                | Deployment | Geserialiseerde sklearn-pipeline, FastAPI, Pydantic-validatie, webfrontend, Docker, GitHub Actions en SageMaker-entrypoint |

                **Waarom deze keuzes?** MAE is interpreteerbaar in minuten en minder gevoelig voor
                uitzonderlijk lange ritten. De tijdssplit simuleert een echte toekomstige voorspelling.
                De baseline voorkomt dat een complex model wordt gekozen zonder aantoonbare meerwaarde.
                """
            ),
            md("## Kerncijfers en cleaning\n\nDe filterredenen zijn niet exclusief: één ongeldige rij kan in meerdere categorieën meetellen."),
            code(
                """
                winner = metrics.iloc[0]
                baseline = metrics.loc[metrics.model == "median_baseline"].iloc[0]
                improvement = 100 * (baseline.mae_minutes - winner.mae_minutes) / baseline.mae_minutes

                fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
                axes[0].axis("off")
                kpi_text = (
                    f"{audit['output_rows']:,} geldige ritten\\n"
                    f"{bundle['model_name']} geselecteerd\\n"
                    f"MAE {winner.mae_minutes:.2f} minuten\\n"
                    f"R² {winner.r2:.3f}\\n"
                    f"{improvement:.1f}% lagere MAE dan baseline"
                )
                axes[0].text(0.02, 0.95, kpi_text, va="top", fontsize=17, linespacing=1.55)
                cleaning = pd.Series(
                    {"Ruw": audit["input_rows"], "Geldig": audit["output_rows"]}
                ) / 1_000_000
                cleaning.plot.bar(ax=axes[1], color=["#9aa7a0", "#4f8f67"])
                axes[1].set(title="Cleaningresultaat", ylabel="miljoen ritten", xlabel="")
                axes[1].tick_params(axis="x", rotation=0)
                plt.tight_layout()
                display(metrics.style.format({
                    "mae_minutes": "{:.2f}", "rmse_minutes": "{:.2f}",
                    "median_ae_minutes": "{:.2f}", "r2": "{:.3f}",
                }))
                """
            ),
            md("## Alle modelfeatures\n\nDe tabel maakt zichtbaar welke variabelen ruw, afgeleid of extern zijn en hoeveel waarden in de dashboardsteekproef ontbreken."),
            code(
                """
                feature_catalog = pd.DataFrame(
                    [
                        ("rideable_type", "categorisch", "bron", "Type fiets"),
                        ("member_casual", "categorisch", "bron", "Member of casual rider"),
                        ("start_station_id", "categorisch", "bron", "Gepland startstation"),
                        ("end_station_id", "categorisch", "planning", "Vooraf gekozen bestemming"),
                        ("start_lat", "numeriek", "bron", "Latitude start"),
                        ("start_lng", "numeriek", "bron", "Longitude start"),
                        ("end_lat", "numeriek", "planning", "Latitude bestemming"),
                        ("end_lng", "numeriek", "planning", "Longitude bestemming"),
                        ("direct_distance_km", "numeriek", "afgeleid", "Hemelsbrede Haversine-afstand"),
                        ("delta_lat", "numeriek", "afgeleid", "Noord-zuidverplaatsing"),
                        ("delta_lng", "numeriek", "afgeleid", "Oost-westverplaatsing"),
                        ("start_hour_sin", "cyclisch", "afgeleid", "Sinus van vertrekuur"),
                        ("start_hour_cos", "cyclisch", "afgeleid", "Cosinus van vertrekuur"),
                        ("weekday_sin", "cyclisch", "afgeleid", "Sinus van weekdag"),
                        ("weekday_cos", "cyclisch", "afgeleid", "Cosinus van weekdag"),
                        ("month_sin", "cyclisch", "afgeleid", "Sinus van maand"),
                        ("month_cos", "cyclisch", "afgeleid", "Cosinus van maand"),
                        ("is_weekend", "binair", "afgeleid", "Weekendindicator"),
                        ("is_holiday", "binair", "afgeleid", "Amerikaanse federale feestdag"),
                        ("temperature_2m", "numeriek", "extern weer", "Temperatuur in °C"),
                        ("relative_humidity_2m", "numeriek", "extern weer", "Relatieve luchtvochtigheid in %"),
                        ("precipitation", "numeriek", "extern weer", "Neerslag in mm"),
                        ("wind_speed_10m", "numeriek", "extern weer", "Windsnelheid in km/u"),
                    ],
                    columns=["feature", "type", "herkomst", "betekenis"],
                )
                missing = feature_frame.isna().mean().mul(100)
                feature_catalog["missing_pct"] = feature_catalog.feature.map(missing).round(2)
                display(feature_catalog.style.format({"missing_pct": "{:.2f}%"}))
                """
            ),
            md("## Wie rijdt en wanneer?\n\nVolume, typische ritduur en gebruikersmix vertellen elk een ander deel van het verhaal."),
            code(
                """
                plot_data = dashboard.copy()
                plot_data["month"] = plot_data.started_at.dt.month
                plot_data["hour"] = plot_data.started_at.dt.hour
                plot_data["weekday"] = plot_data.started_at.dt.day_name()
                month = plot_data.groupby("month").agg(
                    rides=(TARGET, "size"), median_minutes=(TARGET, "median"),
                    casual_share=("member_casual", lambda s: (s == "casual").mean()),
                ).reset_index()
                month["estimated_million_rides"] = (
                    month.rides * audit["output_rows"] / len(plot_data) / 1_000_000
                )

                fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
                sns.barplot(data=month, x="month", y="estimated_million_rides", ax=axes[0], color="#4f8f67")
                sns.lineplot(data=month, x="month", y="median_minutes", marker="o", ax=axes[1])
                sns.lineplot(data=month, x="month", y="casual_share", marker="o", ax=axes[2])
                axes[0].set(title="Geschat ritvolume", ylabel="miljoen ritten")
                axes[1].set(title="Mediane ritduur", ylabel="minuten")
                axes[2].set(title="Aandeel casual riders", ylabel="aandeel")
                plt.tight_layout()

                fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
                sns.countplot(data=plot_data, x="member_casual", ax=axes[0])
                sns.countplot(data=plot_data, x="rideable_type", ax=axes[1])
                axes[0].set(title="Ritten per gebruikerstype", xlabel="")
                axes[1].set(title="Ritten per fietstype", xlabel="")
                plt.tight_layout()
                """
            ),
            md("## Bestemming en afstand\n\nDe rechte lijn is geen echte fietsroute, maar vormt wel een sterke, vooraf berekenbare ondergrens voor de af te leggen afstand."),
            code(
                """
                route_plot = plot_data.sample(min(40_000, len(plot_data)), random_state=42)
                distance_bins = pd.cut(
                    plot_data.direct_distance_km,
                    bins=[0, 0.5, 1, 2, 3, 5, 8, 15, np.inf],
                    right=False,
                )
                distance_summary = plot_data.groupby(distance_bins, observed=True).agg(
                    n=(TARGET, "size"), median_minutes=(TARGET, "median")
                ).reset_index()
                distance_summary["distance_band"] = distance_summary.direct_distance_km.astype(str)

                fig, axes = plt.subplots(1, 2, figsize=(15, 5))
                axes[0].hexbin(
                    route_plot.direct_distance_km, route_plot[TARGET], gridsize=45,
                    mincnt=1, bins="log", cmap="viridis", extent=(0, 12, 0, 90),
                )
                axes[0].set(xlabel="hemelsbrede afstand (km)", ylabel="ritduur (min)", title="Afstand versus ritduur")
                sns.barplot(data=distance_summary, x="distance_band", y="median_minutes", ax=axes[1], color="#4f8f67")
                axes[1].set(xlabel="afstandsklasse (km)", ylabel="mediane minuten", title="Typische duur per afstand")
                axes[1].tick_params(axis="x", rotation=35)
                plt.tight_layout()
                display(distance_summary)
                """
            ),
            md("## Weerrelaties\n\nDeze grafieken tonen associaties binnen gerealiseerde ritten. Ze bewijzen niet dat weer een verandering in ritduur veroorzaakt."),
            code(
                """
                weather_plot = plot_data.copy()
                weather_plot["temperature_band"] = pd.cut(
                    weather_plot.temperature_2m, [-30, 0, 5, 10, 15, 20, 25, 30, 50]
                )
                weather_plot["rain"] = np.where(weather_plot.precipitation > 0, "neerslag", "droog")
                temperature = weather_plot.groupby("temperature_band", observed=True).agg(
                    n=(TARGET, "size"), median_minutes=(TARGET, "median")
                ).reset_index()
                temperature["temperature_label"] = temperature.temperature_band.astype(str)
                rain = weather_plot.groupby("rain").agg(
                    n=(TARGET, "size"), median_minutes=(TARGET, "median")
                ).reset_index()

                fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
                sns.lineplot(data=temperature, x="temperature_label", y="median_minutes", marker="o", ax=axes[0])
                sns.barplot(data=rain, x="rain", y="median_minutes", ax=axes[1])
                axes[0].set(title="Ritduur per temperatuurklasse", xlabel="temperatuur °C", ylabel="mediane minuten")
                axes[0].tick_params(axis="x", rotation=35)
                axes[1].set(title="Droog versus neerslag", xlabel="", ylabel="mediane minuten")
                plt.tight_layout()
                display(pd.concat({"temperatuur": temperature, "regen": rain}, names=["analyse"]))
                """
            ),
            md("## Modelvergelijking\n\nDe winnaar wordt uitsluitend gekozen op de laagste MAE in de toekomstige hold-outperiode."),
            code(
                """
                fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
                sns.barplot(data=metrics, y="model", x="mae_minutes", ax=axes[0], color="#4f8f67")
                sns.barplot(data=metrics, y="model", x="rmse_minutes", ax=axes[1], color="#708f80")
                sns.barplot(data=metrics, y="model", x="r2", ax=axes[2], color="#9bb7a8")
                axes[0].set(title="MAE — lager is beter", xlabel="minuten", ylabel="")
                axes[1].set(title="RMSE — lager is beter", xlabel="minuten", ylabel="")
                axes[2].set(title="R² — hoger is beter", xlabel="R²", ylabel="")
                plt.tight_layout()
                """
            ),
            md(
                """
                ## Permutation importance

                Iedere feature wordt afzonderlijk door elkaar geschud. De toename in MAE laat zien
                hoeveel voorspellende informatie het getrainde model daardoor verliest. Gecorreleerde
                features kunnen elkaars belang verdelen; dit is dus geen causale ranglijst.
                """
            ),
            code(
                """
                importance_sample = holdout.sample(min(12_000, len(holdout)), random_state=42)
                permutation = permutation_importance(
                    bundle["model"], importance_sample[MODEL_FEATURES], importance_sample[TARGET],
                    scoring="neg_mean_absolute_error", n_repeats=3, random_state=42, n_jobs=1,
                )
                importance = pd.DataFrame({
                    "feature": MODEL_FEATURES,
                    "mae_increase": permutation.importances_mean,
                    "std": permutation.importances_std,
                }).sort_values("mae_increase", ascending=False)

                top = importance.head(15).sort_values("mae_increase")
                plt.figure(figsize=(9, 6))
                plt.barh(top.feature, top.mae_increase, xerr=top["std"], color="#4f8f67")
                plt.xlabel("toename MAE na permutatie (minuten)")
                plt.title("Welke features gebruikt het model het sterkst?")
                plt.tight_layout()
                display(importance.style.format({"mae_increase": "{:.3f}", "std": "{:.3f}"}))
                """
            ),
            md("## Werkelijk versus voorspeld en residuen\n\nEen goed model ligt rond de diagonaal. Positieve residuen betekenen overschatting; negatieve residuen onderschatting."),
            code(
                """
                fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                axes[0].hexbin(
                    holdout[TARGET], holdout.prediction, gridsize=45, mincnt=1,
                    bins="log", cmap="viridis", extent=(0, 80, 0, 80),
                )
                axes[0].plot([0, 80], [0, 80], "--", color="white", linewidth=2)
                axes[0].set(xlabel="werkelijke minuten", ylabel="voorspelde minuten", title="Werkelijk versus voorspeld")
                sns.histplot(holdout.error.clip(-40, 40), bins=70, ax=axes[1], color="#4f8f67")
                axes[1].axvline(0, color="black", linestyle="--")
                axes[1].set(xlabel="voorspelling − werkelijkheid (minuten)", title="Verdeling van residuen")
                plt.tight_layout()
                """
            ),
            md("## Prestaties per subgroep\n\nDeze controle voorkomt dat een goede totaalscore een structureel zwakke gebruikersgroep verbergt."),
            code(
                """
                subgroup_frame = holdout.copy()
                subgroup_frame["weekpart"] = np.where(subgroup_frame.is_weekend == 1, "weekend", "werkdag")
                subgroup_frame["weather"] = np.where(subgroup_frame.precipitation > 0, "neerslag", "droog")
                subgroup_tables = []
                for column in ["member_casual", "rideable_type", "weekpart", "weather"]:
                    part = subgroup_frame.groupby(column, observed=True).agg(
                        n=("absolute_error", "size"),
                        mae=("absolute_error", "mean"),
                        median_ae=("absolute_error", "median"),
                        bias=("error", "mean"),
                    ).reset_index(names="groep")
                    part.insert(0, "dimensie", column)
                    subgroup_tables.append(part)
                subgroup = pd.concat(subgroup_tables, ignore_index=True)

                plt.figure(figsize=(11, 5))
                sns.barplot(data=subgroup, x="groep", y="mae", hue="dimensie")
                plt.ylabel("MAE (minuten)")
                plt.xlabel("")
                plt.title("Voorspelfout per subgroep")
                plt.xticks(rotation=25)
                plt.tight_layout()
                display(subgroup.style.format({"mae": "{:.2f}", "median_ae": "{:.2f}", "bias": "{:.2f}"}))
                """
            ),
            md("## Concrete fouten\n\nDe grootste fouten zijn productmatig vaak leerzamer dan het gemiddelde."),
            code(
                """
                example_columns = [
                    "started_at", "member_casual", "rideable_type", "direct_distance_km",
                    "temperature_2m", TARGET, "prediction", "error", "absolute_error",
                ]
                print("Grootste fouten")
                display(holdout.nlargest(15, "absolute_error")[example_columns])
                print("Typische goede voorspellingen")
                display(holdout.nsmallest(10, "absolute_error")[example_columns])
                """
            ),
            md("## Automatisch eindbesluit\n\nDeze cel vat de actuele artifacts samen; de tekst verandert dus mee na een nieuwe training."),
            code(
                """
                top_feature = importance.iloc[0]
                weakest = subgroup.loc[subgroup.mae.idxmax()]
                display(Markdown(
                    f'''
                    Het geselecteerde **{bundle['model_name']}**-model behaalt een hold-out-MAE van
                    **{winner.mae_minutes:.2f} minuten**, tegenover **{baseline.mae_minutes:.2f}** voor
                    de mediaanbaseline. Dat is een verbetering van **{improvement:.1f}%**. De sterkste
                    feature in deze permutation-analyse is **`{top_feature.feature}`**; permuteren
                    verhoogt de MAE met ongeveer **{top_feature.mae_increase:.2f} minuten**. De zwakste
                    gerapporteerde subgroep is **{weakest.groep}** met MAE **{weakest.mae:.2f} minuten**.

                    Dit blijft een schatting: de rechte-lijnafstand vervangt geen echte fietsroute en
                    verkeer, evenementen, routekeuze en actuele infrastructuur ontbreken.
                    '''
                ))
                """
            ),
        ],
    ),
}


def main() -> None:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    for filename, content in NOTEBOOKS.items():
        path = NOTEBOOK_DIR / filename
        path.write_text(json.dumps(content, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"Wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
