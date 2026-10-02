"""Controleert of app.py + het model (.pkl) correct werken, voordat ze naar de VM gaan.

Gebruik:  python check_model.py <map met app.py>   (standaard: mushrooms_rf)
Wordt uitgevoerd door .github/workflows/deploy-api.yml.
"""
import sys
from pathlib import Path

from fastapi.testclient import TestClient

app_dir = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent / "mushrooms_rf").resolve()
sys.path.insert(0, str(app_dir))

import app as api  # laadt het model; faalt als het .pkl-bestand kapot of incompatibel is

client = TestClient(api.app)


def check(name, condition, detail=""):
    print(("OK   " if condition else "FOUT ") + name + (f"  ->  {detail}" if detail else ""))
    if not condition:
        sys.exit(1)


r = client.get("/health")
check("/health", r.status_code == 200 and r.json() == {"status": "ok"}, r.text)

r = client.get("/")
info = r.json()
check("/ geeft model-info", r.status_code == 200 and info["numeric_features"] and info["categorical_features"],
      f"{info.get('model')} uit {info.get('model_file')}")
check("klassen zijn 0 en 1", sorted(info["classes"]) == [0, 1], info["classes"])

# Elke toegestane categorie moet een voorspelling opleveren.
for column, codes in info["categorical_features"].items():
    for code in codes:
        body = dict(api.EXAMPLE, **{column: code})
        r = client.post("/predict", json=body)
        if r.status_code != 200:
            check(f"/predict met {column}={code}", False, r.text)
check("/predict voor elke categorie", True)

r = client.post("/predict", json=api.EXAMPLE)
result = r.json()
probabilities = result.get("probabilities", {})
check("/predict met voorbeeld", r.status_code == 200 and result["prediction"] in (0, 1)
      and abs(sum(probabilities.values()) - 1) < 1e-3, result)

r = client.post("/predict", json={"habitat": "not-a-habitat"})
check("ongeldige waarde geeft 422", r.status_code == 422, r.status_code)

print("Model en API werken.")
