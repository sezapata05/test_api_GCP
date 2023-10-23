import fastapi
import pandas as pd
from fastapi import FastAPI
from typing import List, Dict

app = FastAPI()

# Importar la clase DelayModel del archivo modelo.py
from modelo import DelayModel

# Crear una instancia de la clase DelayModel
delay_model = DelayModel()

# Ruta para verificar el estado de la API
@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {
        "status": "OK"
    }

# Ruta para predecir demoras en vuelos
@app.post("/predict", status_code=200)
async def post_predict(data: List[Dict[str, str]]) -> List[int]:
    # Convertir los datos JSON en un DataFrame de pandas
    input_data = pd.DataFrame(data)
    
    # Preprocesar los datos para la predicción
    features = delay_model.preprocess(input_data)
    
    # Realizar la predicción
    predictions = delay_model.predict(features)
    
    return predictions






from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from typing import List
from delay_model import DelayModel  # Suponiendo que tengas la clase DelayModel en un archivo llamado delay_model.py

app = FastAPI()

# Crear una instancia del modelo
model = DelayModel()

class InputData(BaseModel):
    OPERA: str
    MES: int
    TIPOVUELO: str
    SIGLADES: str
    DIANOM: str

class PredictionResponse(BaseModel):
    predictions: List[int]

@app.post("/train")
async def train_model(data: List[InputData], target: List[int]):
    features = pd.DataFrame([d.dict() for d in data])
    target_series = pd.Series(target)
    model.fit(features, target_series)
    return {"message": "Model trained successfully"}

@app.post("/predict", response_model=PredictionResponse)
async def predict_delays(data: List[InputData]):
    features = pd.DataFrame([d.dict() for d in data])
    predictions = model.predict(features)
    return {"predictions": predictions}
