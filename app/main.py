from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import os

app = FastAPI(title="Housing Price Prediction API")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "random_forest.pkl")

model = joblib.load(MODEL_PATH)

class HousingInput(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float
    
@app.get("/")
def home():
    return {"message": "Housing Price Prediction API is running."}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(data: HousingInput):
    df = pd.DataFrame([data.model_dump()])
    df["Rooms_per_Occup"] = df["AveRooms"] / df["AveOccup"]
    df["Bedrooms_Ratio"] = df["AveBedrms"] / df["AveRooms"]
    df["Income_per_Room"] = df["MedInc"] / df["AveRooms"]

    prediction = model.predict(df)[0]

    return {
        "predicted_med_house_value": float(prediction)
    }

# test with the following cmd: uvicorn app.main:app --host 127.0.0.1 --port 8000 and run : http://127.0.0.1:8000/docs