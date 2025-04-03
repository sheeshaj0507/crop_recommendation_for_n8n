from fastapi import FastAPI
from models import CropFeatures
import joblib

app = FastAPI()
model = joblib.load("ml_model/crop_model.pkl")

@app.post("/predict_crop")
def predict_crop(features: CropFeatures):
    data = [[
        features.N, features.P, features.K,
        features.temperature, features.humidity,
        features.ph, features.rainfall
    ]]
    prediction = model.predict(data)
    return {"recommended_crop": prediction[0]}
