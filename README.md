# CloudAI Challenge

## Team & Auteurs
* **Teamnaam:** Git Push & Pray
* **Leden:**
  * Dmitrij Tchervonisjenko
  * Kobe Vandenberk
  * Cisse Vandeweyer
  * Brent Van Gelder

---

## 📌 Projectoverzicht
In dit project analyseren en modelleren we twee complexe datasets rondom het thema "Going Green". We hebben bewust gekozen voor sterke datakwaliteit-controles, statistische bewijsvoering en ethische beslissingen bij de evaluatie van onze Machine Learning modellen.

1. **Secondary Mushroom Dataset (`mushrooms/`)**
   * **Doel:** Binaire classificatie om te voorspellen of een paddenstoel *giftig* of *eetbaar* is.
   * **Bijzonderheden:** We ontdekten bewust geplaatste anomalieën in de data (zoals `jumbled_noise` kolommen die 1-op-1 kopieën waren) en hebben 95% lege kolommen verwijderd.
   * **Modellen & Techniek:** PCA clustering, FLAML AutoML, Random Forest en een Stacking Ensemble.
   * **Cruciale Keuze:** We hebben het best presterende XGBoost model geselecteerd, maar de kans-drempel (threshold) verlaagd naar **15%**. Omdat een *False Negative* dodelijk is, gooien we liever gezonde paddenstoelen weg dan dat we een fout maken.

2. **NYC Citi Bike System Data (`citi_bike/`)**
   * **Doel:** Zowel regressie (ritduur voorspellen) als classificatie (is de fietser een *Casual* of *Member*).
   * **Bijzonderheden:** Geautomatiseerde S3 datapijplijn geschreven om enorme datasets van 2024 te downloaden en mergen.
   * **Statistiek & Features:** Via een T-Test wiskundig bewezen dat Casuals significant langer fietsen. Geavanceerde *Feature Engineering* toegepast zoals Haversine-afstand en gemiddelde snelheid.
   * **Cruciale Keuze:** Gezien de extreme onbalans in de data (88% Members), hebben we de leugenachtige "Accuracy" genegeerd en onze modellen uitsluitend vergeleken op basis van de **ROC-curve en AUC-score**.

---

## 📁 Mappenstructuur & Werkwijze
Om efficiënt samen te werken via GitHub en merge-conflicten te voorkomen, hebben we een persoonsgebonden structuur gehanteerd:
* **Individuele mappen (`Brent/`, `Cisse/`, `Dimi/`, `Kobe/`):** Elk teamlid heeft zijn eigen omgeving om te experimenteren met EDA, opschonen, visualisaties en ML modellen. 
* **Verdeling:** Iedereen heeft eigen invalshoeken belicht. Bijvoorbeeld: de ene collega bouwde regressie voor ritduur, de ander classificatie voor gebruikerstype. 
* **Jupyter Notebooks:** De notebooks lezen als een verhaal en zijn voorzien van duidelijke (markdown) documentatie rondom gemaakte keuzes.
* **Continuous Integration (CI/CD):** Er draaien volautomatische GitHub Actions (`ml_pipeline.yml` e.a.) op de achtergrond die retrain-scenario's en testen nabootsen als we code pushen.

---

## ☁️ Deployment & Cloud

Het beste model van de Mushrooms en de CitiBike zijn geïntegreerd in een professionele FastAPI-backend die communiceert met een Vanilla HTML/JS-frontend. 

### AWS SageMaker
Voor de Cloud-vereisten hebben we het modelontwikkelingsproces ook deels gedraaid via het **AWS Vocareum Learner Lab** in SageMaker. Hiermee tonen we aan dat onze code zowel lokaal als in een schaalbare Cloud-omgeving werkt.

### Oracle VM (Productie)
De uiteindelijke deployment is klaargezet voor hosting op een (gratis) Oracle Virtual Machine.

```text
deploy/                      -> Alles wat naar de Oracle VM gaat
├── app.py                   -> De centrale FastAPI router
├── requirements.txt         -> Dependencies voor de server
├── web/                     -> De frontend interfaces (HTML/JS)
├── mushrooms/               -> Mushroom API logic + .pkl modellen
└── citi_bike/               -> Citi Bike API logic + .pkl modellen
```

| Route / Adres | Functionaliteit |
|---|---|
| `/` | Hoofd keuzepagina (Frontend) |
| `/mushrooms.html` | Gebruikersinterface voor Paddenstoelen |
| `/citi_bike.html` | Gebruikersinterface voor Citi Bike |
| `POST /mushrooms/predict` | API Endpoint: Voorspelt met 15% threshold (Eetbaar/Giftig) |
| `POST /citi_bike/predict` | API Endpoint: Voorspelt ritgedrag/type |
| `/docs` | Swagger UI voor het testen van de API |

*(Instructies voor de leraar / server host)*:
Lokaal testen kan eenvoudig door in de map te navigeren:
`cd deploy && pip install -r requirements.txt && uvicorn app:app --port 8000`
