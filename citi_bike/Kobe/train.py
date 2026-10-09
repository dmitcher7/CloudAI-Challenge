"""Hertrain het Citi Bike-vraagmodel en schrijf de bundle voor de API (deploy/citi_bike_demand/model.pkl).

Dit script is de "ML-afdeling" in de automatische pipeline (.github/workflows/citi_bike_demand_retrain.yml):
bij elke push die de data, de feature-code of de gekozen hyperparameters wijzigt, wordt het model
opnieuw getraind, getest en - als het niet slechter is dan het huidige model - als pull request
klaargezet. De merge van die PR deployt het nieuwe model naar de VM (deploy-api.yml).

Werkwijze (dezelfde als in de notebooks):
1. data/model_table.parquet (zone x uur, gemaakt door 04_prepare_data) en data/zones.json inlezen;
2. trainen op de train-uren, meten op de test-uren (laatste 7 dagen van elke maand);
3. kwaliteitspoort: MAE mag niet meer dan --tolerance slechter zijn dan het model dat nu gedeployed is;
4. het definitieve model opnieuw fitten op alle uren van het jaar en als bundle wegschrijven.

Gebruik:  python citi_bike/Kobe/train.py [--out pad/model.pkl] [--tolerance 0.02] [--force]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import citibike as cb  # noqa: E402

DEFAULT_OUT = HERE.parent.parent / "deploy" / "citi_bike_demand" / "model.pkl"
PARAMS_PATH = HERE / "models" / "best_params.json"

LIMITATIONS = ("Geschat gemiddelde, geen garantie. Het model kent geen evenementen, wegenwerken, "
               "stationsstoringen of het aantal beschikbare fietsen; de vraag is gemeten als gerealiseerde "
               "vertrekken (een leeg station telt als 0 vraag). Getraind op 2024.")


def build_model(params: dict) -> HistGradientBoostingRegressor:
    categorical = [cb.FEATURES.index(c) for c in cb.CATEGORICAL]
    return HistGradientBoostingRegressor(loss="poisson", categorical_features=categorical,
                                         random_state=cb.SEED, **params)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--tolerance", type=float, default=0.02, help="toegelaten relatieve verslechtering van de MAE")
    parser.add_argument("--force", action="store_true", help="kwaliteitspoort negeren")
    args = parser.parse_args()

    table = cb.load_model_table()
    zones = json.loads((cb.DATA_DIR / "zones.json").read_text(encoding="utf-8"))
    params = json.loads(PARAMS_PATH.read_text(encoding="utf-8"))["params"]

    X, y = cb.make_features(table), table[cb.TARGET]
    is_test = cb.split_labels(table["time"]) == "test"
    print(f"Data: {len(table):,} zone-uren ({(~is_test).sum():,} train / {is_test.sum():,} test), {len(zones)} zones")

    # 1. Eerlijke meting: trainen zonder de testweken.
    model = build_model(params).fit(X[~is_test], y[~is_test])
    metrics = cb.evaluate(y[is_test], model.predict(X[is_test]))
    print("Hold-out:", metrics)

    # 2. Kwaliteitspoort tegenover het model dat nu in productie staat.
    if args.out.exists() and not args.force:
        current = joblib.load(args.out)["metrics"]["mae"]
        limit = current * (1 + args.tolerance)
        print(f"Huidig model: MAE {current:.4f}; nieuw: {metrics['mae']:.4f} (grens {limit:.4f})")
        if metrics["mae"] > limit:
            print("::error::Nieuw model is slechter dan het huidige; niet weggeschreven.")
            return 1

    # 3. Definitief model op alle uren van het jaar (meer data, zelfde hyperparameters).
    final = build_model(params).fit(X, y)
    bundle = {
        "model": final,
        "model_name": "HistGradientBoosting (Poisson)",
        "target": "vertrekken per zone per uur",
        "features": cb.FEATURES,
        "params": params,
        "metrics": metrics,
        "metrics_note": "gemeten op de laatste 7 dagen van elke maand met een model zonder die dagen; "
                        "het gedeployde model is daarna op alle uren hertraind",
        "zones": zones,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "limitations": LIMITATIONS,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.out, compress=3)
    cb.save_results("deployed", {"model": bundle["model_name"], "params": params, **metrics,
                                 "trained_at_utc": bundle["trained_at_utc"]})
    print(f"Bundle geschreven naar {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
