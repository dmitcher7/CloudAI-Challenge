"""Controleert of app.py + alle modellen (.pkl) correct werken, voordat ze naar de VM gaan.

Gebruik:  python check_model.py [map met app.py]   (standaard: de map van dit script)
Wordt uitgevoerd door .github/workflows/deploy-api.yml.
"""
import sys
from pathlib import Path

from fastapi.testclient import TestClient

app_dir = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent).resolve()
sys.path.insert(0, str(app_dir))

import app as api  # laadt alle modellen; faalt als een .pkl-bestand kapot of incompatibel is

client = TestClient(api.app)


def check(name, condition, detail=""):
    print(("OK   " if condition else "FOUT ") + name + (f"  ->  {detail}" if detail else ""))
    if not condition:
        sys.exit(1)


r = client.get("/health")
check("/health", r.status_code == 200 and r.json().get("status") == "ok", r.text)

for page in ["/", "/mushrooms.html", "/citi_bike_models.html", "/citi_bike.html", "/citi_bike_demand.html"]:
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

r = client.get("/citi_bike/stations")
check("/citi_bike/stations geeft de stationslijst", r.status_code == 200 and len(r.json()) > 1000, f"{len(r.json())} stations")

# W 21 St & 6 Ave -> Broadway & E 14 St (± 0,8 km)
trip = {"started_at": "2026-06-15T08:30:00", "rideable_type": "classic_bike", "member_casual": "member",
        "start_station_id": "6140.05", "start_lat": 40.74174, "start_lng": -73.99416,
        "end_station_id": "5905.12", "end_lat": 40.73497, "end_lng": -73.99073,
        "temperature_2m": 20.0, "relative_humidity_2m": 60.0, "precipitation": 0.0, "wind_speed_10m": 10.0}
no_weather = {k: v for k, v in trip.items() if k not in ("temperature_2m", "relative_humidity_2m", "precipitation", "wind_speed_10m")}
for name, payload in [("voorbeeld", trip), ("e-bike, casual", dict(trip, rideable_type="electric_bike", member_casual="casual")),
                      ("zonder weer", no_weather), ("onbekende stations", dict(trip, start_station_id="bestaat-niet", end_station_id="ook-niet")),
                      ("feestdag laat op de avond", dict(trip, started_at="2026-07-04T23:45"))]:
    r = client.post("/citi_bike/predict", json=payload)
    minutes = r.json().get("predicted_duration_minutes") if r.status_code == 200 else None
    check(f"/citi_bike/predict {name}", isinstance(minutes, (int, float)) and 0 <= minutes <= 180, r.text)
far = client.post("/citi_bike/predict", json=dict(trip, end_station_id="7458.03", end_lat=40.7685, end_lng=-73.9815)).json()
near = client.post("/citi_bike/predict", json=trip).json()
check("citi_bike: verdere bestemming geeft een langere rit",
      far["predicted_duration_minutes"] > near["predicted_duration_minutes"],
      f"{near['predicted_duration_minutes']} min tegenover {far['predicted_duration_minutes']} min")

r = client.post("/citi_bike/predict", json=dict(trip, member_casual="iemand"))
check("citi_bike: ongeldige waarde geeft 422", r.status_code == 422, r.status_code)
r = client.post("/citi_bike/predict", json=no_weather | {"end_station_id": None})
check("citi_bike: zonder bestemming geeft 422", r.status_code == 422, r.status_code)

# --- Citi Bike-vraag (vertrekken per zone per uur) ---
r = client.get("/citi_bike_demand/")
info = r.json()
check("/citi_bike_demand/ geeft model-info", r.status_code == 200 and info.get("features") and info.get("zones"),
      f"{info.get('model')} uit {info.get('model_file')}, {len(info.get('zones', []))} zones")

hour = {"zone": 0, "time": "2024-06-11T08:00:00", "temperature_c": 20, "precipitation_mm": 0, "wind_kmh": 10}
for zone in info["zones"]:
    r = client.post("/citi_bike_demand/predict", json=dict(hour, zone=zone["zone"]))
    trips = r.json().get("predicted_trips") if r.status_code == 200 else None
    if not (isinstance(trips, (int, float)) and 0 <= trips < 5000):
        check(f"/citi_bike_demand/predict zone {zone['zone']}", False, r.text)
check("/citi_bike_demand/predict voor elke zone", True)

# Plausibiliteit: drukste zone, dinsdag 8u droog moet drukker zijn dan zondag 4u of in de stortregen.
busy = client.post("/citi_bike_demand/predict", json=hour).json()["predicted_trips"]
night = client.post("/citi_bike_demand/predict", json=dict(hour, time="2024-06-09T04:00:00")).json()["predicted_trips"]
storm = client.post("/citi_bike_demand/predict", json=dict(hour, precipitation_mm=8)).json()["predicted_trips"]
check("citi_bike_demand: spits > nacht en > stortregen", busy > night and busy > storm, f"{busy} / {night} / {storm}")

day = {"zone": 0, "date": "2024-06-11", "weather": [{"temperature_c": 20, "precipitation_mm": 0, "wind_kmh": 10}] * 24}
r = client.post("/citi_bike_demand/predict_day", json=day)
check("/citi_bike_demand/predict_day geeft 24 uren", r.status_code == 200 and len(r.json()["hourly_trips"]) == 24, r.text[:200])

r = client.post("/citi_bike_demand/predict", json=dict(hour, zone=999))
check("citi_bike_demand: onbekende zone geeft 422", r.status_code == 422, r.status_code)
r = client.post("/citi_bike_demand/predict", json=dict(hour, precipitation_mm=-1))
check("citi_bike_demand: negatieve neerslag geeft 422", r.status_code == 422, r.status_code)

print("Modellen en API werken.")
