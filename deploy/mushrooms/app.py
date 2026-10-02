from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
import os

app = FastAPI(title="Mushroom Predictor API")

# Setup CORS for the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Model and Columns
MODEL_PATH = os.path.join(os.path.dirname(__file__), '../../mushrooms/Dimi/mushroom_rf_model.pkl')
COLS_PATH = os.path.join(os.path.dirname(__file__), '../../mushrooms/Dimi/model_columns.pkl')

try:
    model = joblib.load(MODEL_PATH)
    model_columns = joblib.load(COLS_PATH)
except Exception as e:
    model = None
    model_columns = None
    print(f"Warning: Model not found. Did you run the Jupyter Notebook first? Error: {e}")

class MushroomFeatures(BaseModel):
    cap_diameter: float
    stem_height: float
    stem_width: float
    gill_color: str
    habitat: str
    season: str
    ring_type: str
    cap_shape: str
    stem_surface: str

@app.get("/")
def read_root():
    return {"message": "Welcome to the Mushroom Classification API. Go to /docs to test the API."}

@app.post("/predict")
def predict_mushroom(features: MushroomFeatures):
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not trained yet. Run the Jupyter Notebook.")
        
    # Convert input to DataFrame
    input_data = pd.DataFrame([features.dict()])
    
    # Fill categorical NAs with 'Unknown' (just in case frontend sends empty string)
    for col in input_data.columns:
        if input_data[col].dtype == 'object' and input_data[col].iloc[0] == '':
            input_data[col] = 'Unknown'
            
    # Apply One-Hot Encoding like in training
    input_encoded = pd.get_dummies(input_data)
    
    # Reindex to ensure all columns from training are present (missing dummy columns will be filled with False/0)
    input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)
    
    # Predict
    prediction = model.predict(input_encoded)
    
    # 0 = edible, 1 = poisonous
    result = "Poisonous ☠️" if prediction[0] == 1 else "Edible 🍄"
    
    return {"prediction": result}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
