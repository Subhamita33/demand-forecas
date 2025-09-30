# ⚡️ PowerGrid Materials Demand Forecasting System (MDPS)

This repository contains the end-to-end Machine Learning pipeline for forecasting material demand (e.g., steel, cable, insulators) based on upcoming power grid project plans (Line and Substation). The system utilizes an **XGBoost Regressor** and is served via a **FastAPI** web service for real-time predictions.

## 🎯 Project Goals

The primary objective is to enhance supply chain efficiency, reduce inventory holding costs, and prevent project delays by providing accurate, phase-specific material demand forecasts.

## 🏛️ System Architecture

The system follows a standard MLOps approach:

1.  **Offline Training:** Historical data is used to train the model, which is then saved to disk (`demand_model.pkl`).
2.  **Online Serving:** The FastAPI service loads the model once and serves real-time predictions via an API endpoint.

## 🛠️ Prerequisites

To run this project, you need Python (3.9+) and the libraries listed in `requirements.txt`.

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/Himanshushr/demand-forecast.git](https://github.com/Himanshushr/demand-forecast.git)
    cd demand-forecast
    ```

2.  **Create and Activate Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    # On Windows:
    venv\Scripts\activate
    # On Linux/macOS:
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## 🚀 Pipeline Execution

The project pipeline has three main phases: Data Generation, Model Training, and Prediction/Deployment.

### Phase 1: Data Generation

This script creates the synthetic historical dataset (`raw_demand_data.csv`).

```bash
python generate_demand_data.py
```
Phase 2: Model Training
This script executes the full ML pipeline (Validation, Feature Engineering, Training, Evaluation) and saves the trained model artifact.

```bash

python train_demand_model.py
```
Phase 3: Deployment and Prediction
You can test the model using either the Command Line Interface (CLI) or the real-time API.

A. CLI Prediction (Testing)
Use this for batch predictions or debugging outside the web environment.

```bash

python predict_demand_cli.py --input raw_demand_data.csv --output final_forecasts.csv
```
B. API Deployment (Production)
Use Uvicorn to run the FastAPI service.

Run the API Server:

```bash

uvicorn api_demand:app --reload
```
Access:

Frontend Test Page: Open your browser to http://127.0.0.1:8000/ to upload a CSV file and view forecasts.

API Documentation (Swagger UI): View endpoints at http://127.0.0.1:8000/docs.

💻 Project Files
File	Description
generate_demand_data.py	Generates the synthetic raw_demand_data.csv.
schema_validation.py	Defines data schema and enforces column/type consistency.
feature_engineering.py	Transforms raw data into model features (e.g., cyclical encoding, interactions).
train_demand_model.py	Trains the XGBoost model and saves it as demand_model.pkl.
predict_demand_cli.py	CLI for batch prediction (used for maintenance/testing).
api_demand.py	FastAPI service for real-time prediction via /predict endpoint.
.gitignore	Ignores large files like .pkl models and .csv data.

Export to Sheets
⚠️ Known Issues and Next Steps
Model Accuracy: Initial model runs show very low R-squared (R 
2
 ≈−0.32). This is due to a suspected bug in the One-Hot Encoding step of feature_engineering.py, causing the model to only train on 6 features instead of the expected 50+.

Next Priority: The immediate focus for the team should be to debug and fix feature_engineering.py to achieve acceptable model accuracy before relying on forecasts for procurement decisions.

Created by Himanshushr and Collaborators.
