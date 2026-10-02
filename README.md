# CloudAI Challenge

## Team & Auteurs
* **Teamnaam:** Git Push & Pray
* **Leden:**
  * Dmitrij Tchervonisjenko
  * Kobe Vandenberk
  * Cisse Vandeweyer
  * Brent Van Gelder

---

## Projectoverzicht

1. **Secondary Mushroom Dataset:**
   * Binaire classificatie (eetbaar vs. giftig/onbekend).
   * Locatie: map `mushrooms/`
2. **NYC Citi Bike System Data:**
   * Exploratory Data Analysis, hypothesevorming en voorspellende modellen.
   * Locatie: map `citi_bike/`
## Mappenstructuur & Werkwijze
Binnen beide dataset-mappen hebben we een strikte structuur gehanteerd om efficiënt samen te werken en versieconflicten te voorkomen:
* **Individuele mappen (`Brent/`, `Cisse/`, `Dimi/`, `Kobe/`):** 
  Elk teamlid heeft in zijn eigen map onafhankelijk de data verkend (EDA), opgeschoond en geëxperimenteerd met verschillende Machine Learning modellen en AutoML (PyCaret). Hierdoor konden we out-of-the-box ideeën testen en meerdere modellen onafhankelijk van elkaar trainen.
  
* **De `FINAL/` map:** 
  Zodra de individuele experimenten klaar waren, hebben we de metrics (zoals accuracy en de foutenanalyse) naast elkaar gelegd. De code met de beste Exploratory Data Analysis, de slimste aanpak voor data cleaning (zoals het oplossen van bewuste multicollineariteit/valstrikken) en het best presterende model is vervolgens geselecteerd en samengevoegd in de `FINAL/` map. Deze map bevat onze definitieve pipeline die gekoppeld is aan de web-deployment en API.