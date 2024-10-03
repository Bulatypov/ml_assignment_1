from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
import os

model_path = os.path.join(os.path.dirname(__file__), "models", "boston_model.pkl")
with open(model_path, "rb") as f:
    model = pickle.load(f)

app = FastAPI()

class InputData(BaseModel):
    features: list

@app.post("/predict/")
def predict(data: InputData):
    features = np.array(data.features).reshape(1, -1)
    prediction = model.predict(features)
    return {"prediction": prediction[0]}
