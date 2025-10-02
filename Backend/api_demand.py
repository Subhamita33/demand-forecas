#!/usr/bin/env python3
"""
FastAPI service for real-time Materials Demand Forecasting.
Loads 'demand_model.pkl' and serves predictions via an /predict endpoint.
"""
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
import pandas as pd
import numpy as np
import joblib
from io import BytesIO
import os

# Import utilities from our pipeline
try:
    from schema_validation import validate_and_standardize
    from feature_engineering import apply_feature_engineering
    ML_DEPS_AVAILABLE = True
except ImportError as e:
    print(f"WARNING: ML dependencies not available: {e}")
    ML_DEPS_AVAILABLE = False

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
            if not ML_DEPS_AVAILABLE:
                raise RuntimeError("ML dependencies not available")
                
            # NOTE: We assume 'demand_model.pkl' is in the same directory
            model_package = joblib.load('demand_model.pkl')
            model = model_package['demand_model']
            feature_names = model_package['feature_names']
            print("INFO: Model 'demand_model.pkl' loaded successfully.")
        except FileNotFoundError:
            raise RuntimeError("Model file 'demand_model.pkl' not found. Please train the model.")
        except KeyError:
            raise RuntimeError("Model package is corrupted (missing 'demand_model' or 'feature_names').")
        except Exception as e:
            raise RuntimeError(f"Error loading model: {str(e)}")


def predict_df(df: pd.DataFrame) -> pd.DataFrame:
    """Core prediction logic: validates, engineers features, aligns, and predicts."""
    if not ML_DEPS_AVAILABLE:
        raise RuntimeError("ML dependencies not available")
        
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
    if not ML_DEPS_AVAILABLE:
        raise HTTPException(status_code=503, detail="ML dependencies not available. Service is in maintenance mode.")
    
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


@app.get("/")
async def root():
    """Root endpoint that returns API information"""
    return {
        "message": "Demand Forecast API is running",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "test": "/test", 
            "predict": "/predict"
        }
    }


@app.get("/index")
async def index():
    """Alternative index endpoint"""
    return {
        "status": "API is running",
        "documentation": "Use /predict endpoint for forecasts"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    try:
        if ML_DEPS_AVAILABLE:
            load_model_and_metadata()
            return {
                "status": "healthy",
                "model_loaded": model is not None,
                "ml_deps": True,
                "message": "Service is running normally"
            }
        else:
            return {
                "status": "healthy", 
                "model_loaded": False,
                "ml_deps": False,
                "message": "Service running without ML dependencies"
            }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "message": "Service has issues"
        }


@app.get("/test")
async def test_endpoint():
    """Simple test endpoint without model dependencies"""
    return {
        "status": "success", 
        "message": "API is working correctly",
        "ml_deps_available": ML_DEPS_AVAILABLE
    }


# For local development
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
