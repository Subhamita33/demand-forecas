#!/usr/bin/env python3
"""
FastAPI service for real-time Materials Demand Forecasting.
Loads 'demand_model.pkl' and serves predictions via an /predict endpoint.
"""
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse # Added for the simple front-end
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


# Add a basic endpoint for the simple front-end
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serves a basic HTML page for uploading CSV and viewing results."""
    # This HTML is served directly by the FastAPI app for simple testing
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>PowerGrid Demand Forecast Uploader</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .container { max-width: 800px; margin: auto; }
            #output { margin-top: 20px; padding: 10px; border: 1px solid #ccc; white-space: pre-wrap; background: #f9f9f9; max-height: 400px; overflow-y: scroll; }
            h1 { color: #007bff; }
            input[type="file"] { margin-bottom: 20px; }
            button { padding: 10px 20px; background-color: #28a745; color: white; border: none; cursor: pointer; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Materials Demand Forecasting API Test</h1>
            <p>Upload a CSV file containing upcoming project-month data for demand prediction.</p>
            
            <input type="file" id="csvFile" accept=".csv">
            <button onclick="uploadFile()">Get Forecasts</button>
            
            <h2 id="statusMessage"></h2>
            <div id="output">Results will appear here...</div>
        </div>

        <script>
            // NOTE: Change this URL if running the API on a different port/host
            const API_URL = "http://127.0.0.1:8000/predict";

            async function uploadFile() {
                const fileInput = document.getElementById('csvFile');
                const outputDiv = document.getElementById('output');
                const statusMsg = document.getElementById('statusMessage');
                
                outputDiv.textContent = "Processing...";
                statusMsg.textContent = "Status: Sending request...";

                if (fileInput.files.length === 0) {
                    statusMsg.textContent = "Status: Please select a file.";
                    outputDiv.textContent = "No file selected.";
                    return;
                }

                const file = fileInput.files[0];
                const formData = new FormData();
                // 'file' must match the parameter name in the FastAPI endpoint
                formData.append('file', file);

                try {
                    const response = await fetch(API_URL, {
                        method: 'POST',
                        body: formData
                    });

                    const data = await response.json();

                    if (response.ok) {
                        statusMsg.textContent = `Status: Success! Forecasted ${data.count} records.`;
                        // Display forecasts in a readable JSON format
                        outputDiv.textContent = JSON.stringify(data.forecasts, null, 2);
                    } else {
                        // Handle 4xx or 5xx errors from the API
                        statusMsg.textContent = `Status: Error (${response.status} ${response.statusText})`;
                        outputDiv.textContent = `API Error Details: ${JSON.stringify(data, null, 2)}`;
                    }

                } catch (error) {
                    statusMsg.textContent = "Status: Network Error. Ensure the Uvicorn server is running.";
                    outputDiv.textContent = `Network Error: ${error.message}`;
                }
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)