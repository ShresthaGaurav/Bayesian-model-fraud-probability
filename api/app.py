
from fastapi import FastAPI
import mlflow.sklearn  # Changed from mlflow.pyfunc
import pandas as pd
import numpy as np
from pydantic import BaseModel
import os

app = FastAPI()

class Transaction(BaseModel):
    amount: float
    type: int
    hour: int
    day_of_week: int

def preprocess_transaction(data):
    # Dummy preprocessing (replace with actual stats from training data)
    amount_zscore = (data["amount"] - 3000) / 1500  # Example mean=3000, std=1500
    rolling_avg_amount = data["amount"]  # Simplified, use historical data in practice
    dr_cr_ratio = 0.5  # Neutral value, compute from historical data if available
    is_night = 1 if data["hour"] in [0, 1, 2, 3, 22, 23] else 0
    bayesian_fraud_prob = 0.5  # Neutral estimate, compute from Bayesian model if available
    return {
        "amount_zscore": amount_zscore,
        "rolling_avg_amount": rolling_avg_amount,
        "dr_cr_ratio": dr_cr_ratio,
        "is_night": is_night,
        "hour": data["hour"],
        "day_of_week": data["day_of_week"],
        "bayesian_fraud_prob": bayesian_fraud_prob
    }

@app.post("/predict")
async def predict(transaction: Transaction):
    run_id = os.environ.get("MLFLOW_RUN_ID")
    # Load the model with sklearn flavor to access predict_proba
    model = mlflow.sklearn.load_model(f"runs:/{run_id}/random_forest_model")
    
    # Preprocess the input transaction
    features_dict = preprocess_transaction(transaction.dict())
    features = pd.DataFrame([features_dict])
    
    # Use predict_proba to get fraud probability
    prob_ml = model.predict_proba(features)[0][1]  # Index [1] for fraud probability
    return {"fraud_probability": prob_ml}
