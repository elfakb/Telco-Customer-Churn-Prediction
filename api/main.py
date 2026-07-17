import sys
import os
import json
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from feature_engineering import engineer_features, prepare_model_data 

# fastapi ile deploy edilicek eğitilmiş modellerin yolu
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

app = FastAPI(
    title="Telco Churn Prediction API",
    description="Müşterilerin özelliklerine göre churn olasılığını tahmin eder.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_methods=["*"],
    allow_headers=["*"],
)

# tüm modelleri yükler 
model = joblib.load(f"{MODELS_DIR}/best_model.pkl")
with open(f"{MODELS_DIR}/metadata.json") as f:
    metadata = json.load(f)
scaler = joblib.load(f"{MODELS_DIR}/scaler.pkl") if metadata["needs_scaling"] else None

# tüm müşterilerde olan değişkenler 
class CustomerData(BaseModel):
    gender: Literal["Male", "Female"]
    SeniorCitizen: Literal[0, 1]
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(..., ge=0, le=100)
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(..., ge=0)
    TotalCharges: float = Field(..., ge=0)

    class Config:
        json_schema_extra = {
            "example": {
                "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes",
                "Dependents": "No", "tenure": 5, "PhoneService": "Yes",
                "MultipleLines": "No", "InternetService": "Fiber optic",
                "OnlineSecurity": "No", "OnlineBackup": "No",
                "DeviceProtection": "No", "TechSupport": "No",
                "StreamingTV": "No", "StreamingMovies": "No",
                "Contract": "Month-to-month", "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 85.5, "TotalCharges": 427.5,
            }
        }


class PredictionResponse(BaseModel):
    churn_probability: float
    churn_prediction: bool
    risk_level: str
    threshold_used: float
    model_name: str


def get_risk_level(prob: float) -> str:
    if prob < 0.3:
        return "Düşük"
    elif prob < 0.6:
        return "Orta"
    return "Yüksek"


def _prepare_single_row(customer: CustomerData) -> pd.DataFrame:
    raw_df = pd.DataFrame([customer.model_dump()])
    raw_df["Churn"] = "No"  # engineer_features/prepare_model_data 'Churn' kolonunu bekliyor
    engineered = engineer_features(raw_df)
    X, _ = prepare_model_data(engineered)

    X = X.reindex(columns=metadata["feature_names"], fill_value=0)
    return X


@app.get("/")
def root():
    return {
        "message": "Telco Churn Prediction API çalışıyor",
        "model": metadata["model_name"],
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerData):
    try:
        X = _prepare_single_row(customer)
        X_input = scaler.transform(X) if scaler is not None else X
        proba = float(model.predict_proba(X_input)[:, 1][0])
        threshold = metadata["threshold"]

        return PredictionResponse(
            churn_probability=round(proba, 4),
            churn_prediction=proba >= threshold,
            risk_level=get_risk_level(proba),
            threshold_used=threshold,
            model_name=metadata["model_name"],
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Çalıştırma: uvicorn api.main:app --reload --port 8000