"""Controleert of app.py + beide modellen (.pkl) correct werken, voordat ze naar de VM gaan.

Gebruik:  python check_model.py [map met app.py]   (standaard: de map van dit script)
Wordt uitgevoerd door .github/workflows/deploy-api.yml.
"""
import sys
from pathlib import Path

from fastapi.testclient import TestClient

app_dir = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent).resolve()
sys.path.insert(0, str(app_dir))

import app as api  # laadt beide modellen; faalt als een .pkl-bestand kapot of incompatibel is

client = TestClient(api.app)


def check(name, condition, detail=""):
    print(("OK   " if condition else "FOUT ") + name + (f"  ->  {detail}" if detail else ""))
    if not condition:
        sys.exit(1)


r = client.get("/health")
check("/health", r.status_code == 200 and r.json().get("status") == "ok", r.text)

for page in ["/", "/mushrooms.html", "/citi_bike.html"]:
    r = client.get(page)
    check(f"pagina {page}", r.status_code == 200 and "text/html" in r.headers["content-type"], r.status_code)

# --- Mushrooms ---
mushrooms = api.mushrooms_api
r = client.get("/mushrooms/")
info = r.json()
check("/mushrooms/ geeft model-info", r.status_code == 200 and info["numeric_features"] and info["categorical_features"],
      f"{info.get('model')} uit {info.get('model_file')}")
check("mushrooms: klassen zijn 0 en 1", sorted(info["classes"]) == [0, 1], info["classes"])

# Elke toegestane categorie moet een voorspelling opleveren.
for column, codes in info["categorical_features"].items():
    for code in codes:
        body = dict(mushrooms.EXAMPLE, **{column: code})
        r = client.post("/mushrooms/predict", json=body)
        if r.status_code != 200:
            check(f"/mushrooms/predict met {column}={code}", False, r.text)
check("/mushrooms/predict voor elke categorie", True)

r = client.post("/mushrooms/predict", json=mushrooms.EXAMPLE)
result = r.json()
probabilities = result.get("probabilities", {})
check("/mushrooms/predict met voorbeeld", r.status_code == 200 and result["prediction"] in (0, 1)
      and abs(sum(probabilities.values()) - 1) < 1e-3, result)

r = client.post("/mushrooms/predict", json={"habitat": "not-a-habitat"})
check("mushrooms: ongeldige waarde geeft 422", r.status_code == 422, r.status_code)

# --- Citi Bike ---
r = client.get("/citi_bike/")
info = r.json()
check("/citi_bike/ geeft model-info", r.status_code == 200 and info.get("features"),
      f"{info.get('model')} uit {info.get('model_file')}")

trip = {"started_at": "2024-06-15T09:30:00", "rideable_type": "classic_bike", "member_casual": "member",
        "start_station_id": "HB101", "start_lat": 40.7359, "start_lng": -74.0303}
for variant in [{}, {"rideable_type": "electric_bike", "member_casual": "casual"},
                {"start_station_id": "bestaat-niet"}, {"started_at": "2024-12-28T23:45"}]:
    r = client.post("/citi_bike/predict", json=dict(trip, **variant))
    minutes = r.json().get("predicted_duration_minutes") if r.status_code == 200 else None
    check(f"/citi_bike/predict {variant or 'voorbeeld'}",
          isinstance(minutes, (int, float)) and 0 <= minutes < 24 * 60, r.text)

r = client.post("/citi_bike/predict", json=dict(trip, member_casual="iemand"))
check("citi_bike: ongeldige waarde geeft 422", r.status_code == 422, r.status_code)

print("Modellen en API werken.")
