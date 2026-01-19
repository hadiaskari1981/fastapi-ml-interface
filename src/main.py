import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing_extensions import Literal
import pdb
model_name = "model_binary.dat.gz"


from pathlib import Path
project_dir = Path(__file__).resolve().parent.parent
model_path = project_dir / "model" / model_name
model = joblib.load(model_path)
app = FastAPI()

def model_response(sample):

    X = pd.json_normalize(sample.__dict__)
    label = model.predict(X)[0]
    prob = model.predict_proba(X)[0]
    label = "benign" if label == "B" else "malignant"
    return {"diagnosis": label,  "prediction": {"probability-benign": round(prob[0]*100, 2), "probability-malignant": round(prob[1]*100, 2)}}

class BreastDataInputSchema(BaseModel):
    texture_mean: float = Field(..., gt=0)
    area_mean: float = Field(..., gt=0)
    concavity_mean: float = Field(..., gt=0)
    area_se: float = Field(..., gt=0)
    concavity_worst: float = Field(..., gt=0)

    class ConfigDict:
        # example data
        json_schema_extra = {
            "example": {
                "texture_mean": 10.38,
                "area_mean": 1003.5,
                "concavity_mean": 0.198,
                "area_se": 153.4,
                "concavity_worst": 0.689,
            }
        }

class BreastPredictionSchema(BaseModel):
    diagnosis: Literal["malignant", "benign"]
    prediction: dict
@app.get("/health")
async def service_health():
    """Return service health"""
    return {"ok"}

@app.post("/predict", response_model=BreastPredictionSchema)
async def model_predict(sample: BreastDataInputSchema):
    """Predict with input"""
    response = model_response(sample)
    return response