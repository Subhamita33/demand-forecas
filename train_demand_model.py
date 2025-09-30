#!/usr/bin/env python3
"""
Train the model to forecast Materials Demand Quantity.
Saves the trained model and feature metadata to 'demand_model.pkl'.
"""

import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Import utilities from our new pipeline files
from schema_validation import validate_and_standardize
from feature_engineering import apply_feature_engineering


def train_regressor(X_train, X_test, y_train, y_test, name: str):
    """
    Trains and evaluates an XGBoost Regressor model.
    FIXED: Uses 'eval_metric' in params and 'early_stopping_patience' in fit 
           for compatibility with modern XGBoost versions (>= 2.0).
    """
    # Hold out 10% of X_train for potential early stopping/validation
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

    # 1. Define Parameters (including eval_metric for model initialization)
    params = dict(
        n_estimators=700,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.5,
        random_state=42,
        tree_method="hist",
        n_jobs=-1,
        eval_metric="rmse"  # FIX 1: Pass eval_metric during initialization
    )

    print(f"\nTraining model for: {name}...")
    
    model = xgb.XGBRegressor(**params)
    
    model.fit(X_tr, y_tr,
          eval_set=[(X_val, y_val)],
          verbose=False)


    
    # 3. Evaluate on the test set
    pred = model.predict(X_test)
    
    # Clip predictions to zero since demand cannot be negative
    pred = np.clip(pred, a_min=0, a_max=None)
    
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)

    print(f"--- {name} Results ---")
    print(f"R-squared (R²): {r2:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    
    return model


def main():
    print("Starting Materials Demand Model Training Pipeline...")
    
    # 1. Load and Validate Data
    try:
        df_raw = pd.read_csv('raw_demand_data.csv')
    except FileNotFoundError:
        print("ERROR: raw_demand_data.csv not found. Please run generate_demand_data.py first.")
        return

    df, warnings = validate_and_standardize(df_raw.copy())
    if warnings:
        print("\n--- Validation Warnings ---")
        for w in warnings:
            print(f" - {w}")
    
    # 2. Prepare Features and Target
    
    # Handle NaN values in the target (only relevant if data quality is poor, usually dropped)
    df = df.dropna(subset=['DemandQuantity'])
    
    targets = ['DemandQuantity']
    X = df.drop(columns=targets)
    y = df['DemandQuantity']
    
    # 3. Split Data
    # 80% for training, 20% for final testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. Feature Engineering
    X_train_fe = apply_feature_engineering(X_train.copy())
    X_test_fe = apply_feature_engineering(X_test.copy())
    
    # Ensure all columns are numeric before training (drop any remaining objects/non-numeric columns)
    X_train_fe = X_train_fe.select_dtypes(include=np.number)
    X_test_fe = X_test_fe.select_dtypes(include=np.number)
    
    print(f"\nFeature set shape after FE: {X_train_fe.shape}")
    
    # --- Critical Step: Align Test Features with Training Features ---
    # Due to one-hot encoding, test set columns must match the train set columns exactly.
    missing_cols_in_test = set(X_train_fe.columns) - set(X_test_fe.columns)
    for c in missing_cols_in_test:
        X_test_fe[c] = 0.0 # Use 0.0 to ensure float consistency
    X_test_fe = X_test_fe[X_train_fe.columns]
    
    # 5. Train Model
    demand_model = train_regressor(X_train_fe, X_test_fe, y_train, y_test, 'DemandQuantity_Forecast')

    # 6. Save Model and Metadata
    final_model_package = {
        'demand_model': demand_model,
        'feature_names': list(X_train_fe.columns), # Save the exact feature list for prediction
        'target_name': 'DemandQuantity',
    }
    joblib.dump(final_model_package, 'demand_model.pkl')
    print("\nSuccessfully saved trained model and feature metadata to 'demand_model.pkl'")


if __name__ == '__main__':
    main()