from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from typing import List
import pandas as pd

# Importa la clase DelayModel
from challenge.model import DelayModel

app = FastAPI()

# Crea una instancia de la clase DelayModel
model = DelayModel()

class FlightData(BaseModel):
    OPERA: str
    MES: int
    TIPOVUELO: str
    
class Fligt(BaseModel):
    flights : list[FlightData]
    
class PredictionResponse(BaseModel):
    predict: list

@app.get("/")
def home():
    return {"msg" : "hello"}
    
@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {
        "status": "OK"
    }

@app.post("/predict/")
async def predict_delay(data: Fligt) -> PredictionResponse:
    try:
        data_dict = [item.dict() for item in data.flights]
        data_df = pd.DataFrame(data_dict)

        # Preprocesa los datos
        features = model.preprocess(data_df)

        # Realiza predicciones
        predictions = model.predict(features)

        response_data = {"predict": predictions}
        return PredictionResponse(**response_data)

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error in prediction: {str(e)}")


# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)