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
    all_cells = [intro, *cells]
    for index, cell in enumerate(all_cells):
        cell["id"] = f"cell-{index:03d}"
    return {
        "cells": all_cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
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

                Voor een snelle laptop-run gebruiken we hieronder drie maanden. Voor de definitieve
                analyse kiezen we minimaal twaalf aaneengesloten maanden om seizoenseffecten te zien.
                De periode is vastgelegd, zodat cherry-picking achteraf wordt voorkomen.
                """
            ),
            code(
                """
                from pathlib import Path
                import json
                import pandas as pd
                from src.config import RAW_DIR
                from src.data.download import download_range

                START_MONTH = "2024-01"
                END_MONTH = "2024-03"  # verander naar 2024-12 voor de eindrun
                RUN_DOWNLOAD = False    # zet bewust op True; download kan meerdere GB zijn
                if RUN_DOWNLOAD:
                    download_range(START_MONTH, END_MONTH)
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
                informatie die bij vertrek bekend is.
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
                een log1p-transformatie wegens de lange rechterstaart.
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
                from src.config import MODEL_DIR, PROCESSED_DIR, REPORT_DIR, TARGET
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
                columns = ["started_at", "rideable_type", "member_casual", "start_station_id", "start_lat", "start_lng", TARGET]
                frame = pd.read_parquet(PROCESSED_DIR / "trips.parquet", columns=columns).sort_values("started_at")
                if len(frame) > 500_000:
                    frame = frame.iloc[np.linspace(0, len(frame)-1, 500_000, dtype=int)]
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
                from src.config import PROCESSED_DIR

                session = sagemaker.Session()
                role = sagemaker.get_execution_role()
                bucket = session.default_bucket()
                prefix = "green-wheels/citibike"
                train_s3 = session.upload_data(
                    str(PROCESSED_DIR / "trips.parquet"), bucket=bucket, key_prefix=f"{prefix}/train"
                )
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
}


def main() -> None:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    for filename, content in NOTEBOOKS.items():
        path = NOTEBOOK_DIR / filename
        path.write_text(json.dumps(content, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"Wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
