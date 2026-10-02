"""Mushroom-model: eetbaar (0) of giftig (1). Wordt door ../app.py gekoppeld onder /mushrooms.

Het model bepaalt zelf welke kenmerken en categorieen geldig zijn; een ander
model gebruiken = het bestand vervangen (of MUSHROOMS_MODEL_PATH zetten) en herstarten.
"""
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import APIRouter, Body, HTTPException
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

HERE = Path(__file__).parent
MODEL_PATH = Path(os.environ.get("MUSHROOMS_MODEL_PATH", HERE / "stacking.pkl"))

# Leesbare namen voor de lettercodes van de Secondary Mushroom-dataset.
# De API aanvaardt zowel de code ("w") als het woord ("white").
LABELS = {
    "gill-color": {"n": "brown", "b": "buff", "g": "gray", "r": "green", "p": "pink", "u": "purple",
                   "e": "red", "w": "white", "y": "yellow", "l": "blue", "o": "orange", "k": "black",
                   "f": "none"},
    "habitat": {"g": "grasses", "l": "leaves", "m": "meadows", "p": "paths", "h": "heaths",
                "u": "urban", "w": "waste", "d": "woods"},
    "season": {"s": "spring", "u": "summer", "a": "autumn", "w": "winter"},
    "ring-type": {"c": "cobwebby", "e": "evanescent", "r": "flaring", "g": "grooved", "l": "large",
                  "p": "pendant", "s": "sheathing", "z": "zone", "y": "scaly", "m": "movable",
                  "f": "none"},
    "cap-shape": {"b": "bell", "c": "conical", "x": "convex", "f": "flat", "s": "sunken",
                  "p": "spherical", "o": "others"},
    "stem-surface": {"i": "fibrous", "g": "grooves", "y": "scaly", "s": "smooth", "h": "shiny",
                     "l": "leathery", "k": "silky", "t": "sticky", "w": "wrinkled", "e": "fleshy",
                     "f": "none"},
}
MISSING = {None, "", "unknown", "?"}

# joblib.load leest zowel gecomprimeerde joblib-bestanden als gewone pickles.
model = joblib.load(MODEL_PATH)


def walk(estimator):
    """Geeft het model en alle geneste (getrainde) onderdelen terug."""
    yield estimator
    for _, step in getattr(estimator, "steps", []):
        yield from walk(step)
    for sub in getattr(estimator, "estimators_", []):
        if hasattr(sub, "get_params"):
            yield from walk(sub)
    if isinstance(estimator, ColumnTransformer):
        for _, transformer, _ in estimator.transformers_:
            if hasattr(transformer, "get_params"):
                yield from walk(transformer)


def find(estimator, kind):
    return next((part for part in walk(estimator) if isinstance(part, kind)), None)


# Voor losse voorspellingen is 1 thread sneller dan n_jobs=-1 (geen thread-overhead).
for part in walk(model):
    if getattr(part, "n_jobs", None) not in (None, 1):
        part.n_jobs = 1

# Kolommen en toegestane categorieen rechtstreeks uit het model halen.
preprocessor = find(model, ColumnTransformer)
COLUMNS = [str(c) for c in model.feature_names_in_]
CLASSES = [int(c) for c in model.classes_]
NUMERIC, ALLOWED, NULLABLE = [], {}, set()
for _, transformer, columns in preprocessor.transformers_:
    if not hasattr(transformer, "get_params"):
        continue  # 'drop' of 'passthrough'
    encoder = find(transformer, OneHotEncoder)
    if encoder is None:
        NUMERIC += [str(c) for c in columns]
        continue
    for column, categories in zip(columns, encoder.categories_):
        ALLOWED[str(column)] = [str(c) for c in categories]
        if find(transformer, SimpleImputer) is not None:
            NULLABLE.add(str(column))  # ontbrekende waarde wordt door het model ingevuld


def normalize(name):
    return str(name).strip().lower().replace("-", "_").replace(" ", "_")


# Veldnamen: 'cap-diameter', 'cap_diameter' en (oude pagina) 'cap_diameter_cm' zijn allemaal goed.
ALIASES = {normalize(c): c for c in COLUMNS}
for column in NUMERIC:
    for unit in ("_cm", "_mm"):
        ALIASES.setdefault(normalize(column) + unit, column)
        if normalize(column).endswith(unit):
            ALIASES.setdefault(normalize(column)[: -len(unit)], column)

# Per kolom: woord -> code, alleen voor codes die dit model kent.
WORDS = {
    column: {LABELS[column][code]: code for code in codes if code in LABELS.get(column, {})}
    for column, codes in ALLOWED.items()
}

EXAMPLE = {c: 6.0 for c in NUMERIC} | {c: codes[0] for c, codes in ALLOWED.items()}


router = APIRouter(tags=["mushrooms"])


@router.get("/")
def info():
    return {
        "model": type(model).__name__,
        "model_file": MODEL_PATH.name,
        "classes": CLASSES,
        "numeric_features": NUMERIC,
        "categorical_features": ALLOWED,
        "labels": {c: {code: LABELS[c][code] for code in codes if code in LABELS.get(c, {})}
                   for c, codes in ALLOWED.items()},
    }


@router.post("/predict")
def predict(features: dict[str, float | str | None] = Body(examples=[EXAMPLE])):
    row, ignored, errors = {}, [], {}
    for key, value in features.items():
        column = ALIASES.get(normalize(key))
        if column is None:
            ignored.append(key)  # kenmerk dat dit model niet gebruikt
        else:
            row[column] = value

    for column in NUMERIC:
        value = row.get(column)
        try:
            row[column] = np.nan if value in MISSING else float(value)  # NaN -> mediaan
        except (TypeError, ValueError):
            errors[column] = f"'{value}' is geen getal"

    for column, codes in ALLOWED.items():
        value = row.get(column)
        value = value.strip().lower() if isinstance(value, str) else value
        value = WORDS[column].get(value, value)  # woord -> code
        if value in codes:
            row[column] = value
        elif value in MISSING and column in NULLABLE:
            row[column] = np.nan  # het model vult de meest voorkomende waarde in
        elif value in MISSING and "unknown" in codes:
            row[column] = "unknown"
        else:
            options = [f"{code} ({LABELS[column][code]})" if code in LABELS.get(column, {}) else code
                       for code in codes]
            errors[column] = f"'{value}' is onbekend, kies uit {options}"

    if errors:
        raise HTTPException(status_code=422, detail=errors)

    frame = pd.DataFrame([row], columns=COLUMNS)
    frame[NUMERIC] = frame[NUMERIC].astype(float)
    probabilities = model.predict_proba(frame)[0]
    best = int(probabilities.argmax())
    result = {
        "prediction": CLASSES[best],
        "probabilities": {str(c): round(float(p), 4) for c, p in zip(CLASSES, probabilities)},
    }
    if ignored:
        result["ignored_fields"] = ignored
    return result
