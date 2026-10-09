"""Start de API op de VM en koppelt alle modellen.

    /                     keuzepagina (web/index.html)
    /mushrooms.html       webpagina mushroom-model
    /citi_bike_models.html  keuzepagina tussen de twee Citi Bike-modellen
    /citi_bike.html       webpagina Citi Bike-model
    /citi_bike_demand.html  webpagina Citi Bike-vraagmodel
    /mushrooms/           model-info      POST /mushrooms/predict
    /citi_bike/           model-info      POST /citi_bike/predict
    /citi_bike_demand/    model-info      POST /citi_bike_demand/predict, /citi_bike_demand/predict_day
    /health               ok zodra alle modellen geladen zijn
    /docs                 Swagger

Starten:  uvicorn app:app --host 0.0.0.0 --port 8000   (vanuit deze map)
"""
import os
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

# Elk model laadt bij het importeren; een kapot of incompatibel .pkl laat de API dus niet starten.
from citi_bike import api as citi_bike_api
from citi_bike_demand import api as citi_bike_demand_api
from mushrooms import api as mushrooms_api

HERE = Path(__file__).parent
API_KEY = os.environ.get("API_KEY")  # optioneel: indien gezet is de header X-API-Key verplicht bij POST
MODEL_PATHS = [mushrooms_api.MODEL_PATH, citi_bike_api.MODEL_PATH, citi_bike_demand_api.MODEL_PATH]  # gebruikt door de deploy-workflow


def check_api_key(request: Request, x_api_key: str | None = Header(default=None)):
    if API_KEY and request.method == "POST" and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Ongeldige of ontbrekende X-API-Key")


app = FastAPI(title="CloudAI model-API", version="3.0")

# CORS: laat webpagina's op een ander adres (of lokaal geopend) de API aanroepen.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(mushrooms_api.router, prefix="/mushrooms", dependencies=[Depends(check_api_key)])
app.include_router(citi_bike_api.router, prefix="/citi_bike", dependencies=[Depends(check_api_key)])
app.include_router(citi_bike_demand_api.router, prefix="/citi_bike_demand", dependencies=[Depends(check_api_key)])


@app.get("/health")
def health():
    return {"status": "ok", "models": {"mushrooms": mushrooms_api.MODEL_PATH.name,
                                       "citi_bike": citi_bike_api.MODEL_PATH.name,
                                       "citi_bike_demand": citi_bike_demand_api.MODEL_PATH.name}}


@app.get("/app", include_in_schema=False)
def old_page():
    # Oud adres van de mushroom-pagina.
    return RedirectResponse("/mushrooms.html")


# Als laatste: alles wat geen API-route is, komt uit web/ (index.html op /).
app.mount("/", StaticFiles(directory=HERE / "web", html=True), name="web")
