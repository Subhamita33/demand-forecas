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