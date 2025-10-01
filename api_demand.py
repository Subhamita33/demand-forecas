#!/usr/bin/env python3
"""
FastAPI service for real-time Materials Demand Forecasting.
Loads 'demand_model.pkl' and serves predictions via an /predict endpoint.
"""
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse # <-- ADD FileResponse
import pandas as pd
import numpy as np
import joblib
from io import BytesIO

# Import utilities from our pipeline
from schema_validation import validate_and_standardize
from feature_engineering import apply_feature_engineering


app = FastAPI(title="Materials Demand Forecasting API")
# Configure CORS to allow any origin for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for caching the model
model = None
feature_names = None


def load_model_and_metadata():
    """Loads the model and feature names, caching them globally."""
    global model, feature_names
    if model is None:
        try:
            # NOTE: We assume 'demand_model.pkl' is in the same directory
            model_package = joblib.load('demand_model.pkl')
            model = model_package['demand_model']
            feature_names = model_package['feature_names']
            print("INFO: Model 'demand_model.pkl' loaded successfully.")
        except FileNotFoundError:
            raise RuntimeError("Model file 'demand_model.pkl' not found. Please train the model.")
        except KeyError:
            raise RuntimeError("Model package is corrupted (missing 'demand_model' or 'feature_names').")


def predict_df(df: pd.DataFrame) -> pd.DataFrame:
    """Core prediction logic: validates, engineers features, aligns, and predicts."""
    load_model_and_metadata()
    
    # 1. Validation and Standardization
    df_copy = df.copy()
    df_validated, _ = validate_and_standardize(df_copy)

    # 2. Feature Engineering
    X = df_validated.drop(columns=[c for c in ['DemandQuantity'] if c in df_validated.columns], errors='ignore')
    X_fe = apply_feature_engineering(X.copy())
    X_fe = X_fe.select_dtypes(include=np.number)

    # 3. Feature Alignment (CRITICAL for consistent prediction)
    missing_cols = set(feature_names) - set(X_fe.columns)
    for c in missing_cols:
        X_fe[c] = 0.0
        
    X_fe = X_fe.reindex(columns=feature_names, fill_value=0.0)

    # 4. Prediction
    try:
        predictions = model.predict(X_fe)
        predictions = np.clip(predictions, a_min=0, a_max=None).round().astype(int)
    except Exception as e:
        raise RuntimeError(f"Prediction failed due to feature mismatch or model error: {e}")

    # 5. Attach Prediction
    df_validated["Predicted_DemandQuantity"] = predictions
    
    return df_validated


@app.post("/predict")
async def predict_demand(file: UploadFile = File(...)):
    """
    Accepts a CSV file, generates demand forecasts, and returns the augmented data as JSON.
    """
    try:
        content = await file.read()
        df = pd.read_csv(BytesIO(content))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid CSV upload. Please ensure file is valid.")

    try:
        out_df = predict_df(df)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Data Validation Error: {e}")

    # Prepare JSON Response
    output_cols = [c for c in out_df.columns if c not in ['DemandQuantity']]
    output_cols.append('Predicted_DemandQuantity')
    
    return {
        "status": "success",
        "count": len(out_df),
        "forecasts": out_df[output_cols].to_dict(orient="records")
    }


@app.get("/", response_class=FileResponse)
async def serve_dashboard():
    """Serves the static dashboard.html file."""
    # Use an appropriate path. '.' means the current working directory.
    return FileResponse("dashboard.html", media_type="text/html")
# Vercel deployment compatibility
import os

# For Vercel serverless deployment
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
# The rest of your api_demand.py (including @app.post("/predict")) remains the same.
