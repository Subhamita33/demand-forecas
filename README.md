# ⚡️ PowerGrid Materials Demand Forecasting System (MDPS)

This repository contains the end-to-end Machine Learning pipeline for forecasting material demand (e.g., steel, cable, insulators) based on upcoming power grid project plans.

The system utilizes an **XGBoost Regressor** and is deployed via a **FastAPI** web service, with a highly interactive **HTML/JavaScript dashboard** for visualization.

## 🎯 Project Goals

The primary objective is to enhance supply chain efficiency, reduce inventory holding costs, and prevent project delays by providing accurate, phase-specific material demand forecasts.

## ⚠️ CRITICAL CURRENT STATUS: Low Accuracy Bug

**The highest priority for the team is to address model accuracy.**

* **Issue:** Initial model runs showed very low $R^2$ ($\approx -0.32$). This is due to a suspected bug in the **One-Hot Encoding** step of `feature_engineering.py`.
* **Action Required:** Team members must immediately debug and fix `feature_engineering.py` to ensure the model trains on the complete set of features before relying on forecasts for procurement decisions.

## 🏛️ System Architecture and Deployment

The system follows an MLOps approach:

1.  **Offline Training:** Historical data trains the model, which is saved as `demand_model.pkl`.
2.  **Online Serving:** FastAPI loads the model and serves real-time predictions.
3.  **Visualization:** The interactive `dashboard.html` handles file uploads and visualizes results and KPIs (including hardcoded *target* accuracy metrics for demonstration).

## 🛠️ Setup and Installation

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/Himanshushr/demand-forecast.git](https://github.com/Himanshushr/demand-forecast.git)
    cd demand-forecast
    ```

2.  **Create and Activate Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    # On Linux/macOS:
    source venv/bin/activate
    # On Windows:
    venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## 🚀 Execution Guide

### Phase 1: Data and Model Generation

Run the following scripts sequentially to prepare the model artifact:

1.  **Generate Data:**
    ```bash
    python generate_demand_data.py
    ```
2.  **Train and Save Model:**
    ```bash
    python train_demand_model.py
    ```

### Phase 2: Launch Dashboard and API

The FastAPI service hosts both the prediction endpoint and the front-end dashboard.

1.  **Run the API Server:**
    ```bash
    uvicorn api_demand:app --reload
    ```
2.  **Access the Dashboard:**
    * Open your browser to: **`http://127.0.0.1:8000/`**
    * Upload a CSV file and view the interactive KPIs and forecasted data table.

---

## 💻 Repository Contents

| File | Description |
| :--- | :--- |
| **`dashboard.html`** | The complete, interactive web front-end for file upload and results visualization (KPIs, charts, tables). |
| **`api_demand.py`** | **FastAPI service** that loads the model, exposes the `/predict` endpoint, and serves the dashboard. |
| `train_demand_model.py` | Executes the ML pipeline (feature engineering, training, and model serialization). **Requires debugging.** |
| `feature_engineering.py` | Contains the feature transformation logic where the current bug is suspected. |
| `generate_demand_data.py` | Creates the synthetic historical data file. |
| `requirements.txt` | Lists all necessary Python dependencies (FastAPI, uvicorn, pandas, scikit-learn, etc.). |

---
*Created by Himanshushr and Collaborators.*
