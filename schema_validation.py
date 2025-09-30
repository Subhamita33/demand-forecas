#!/usr/bin/env python3
"""
Schema and validation utilities for Materials Demand Forecasting data.
Ensures consistency for project-material-month records.
"""

from __future__ import annotations
from typing import List, Tuple
import pandas as pd
import numpy as np


# --- Column Definitions ---

# Required columns for identifying the demand and making a prediction.
REQUIRED_COLUMNS: List[str] = [
    "ProjectID",
    "MaterialType",
    "MonthIndex",
    "ProjectType",
    "Region",
    "BudgetSegment",
]

# Optional columns (Targets for training, or highly predictive features)
OPTIONAL_COLUMNS: List[str] = [
    "DemandQuantity",            # Target variable (required for training, optional for prediction)
    "ProjectLengthMonths",
    "TowerType",
    "SubstationType",
    "StartMonth",
    "TaxRate",
    "ProjectRiskScore",
]


def validate_and_standardize(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Validate required columns and gently standardize types for the demand data.

    Returns the (possibly modified) DataFrame and a list of warnings.
    """
    warnings: List[str] = []

    # Check 1: Hard check for required columns
    missing_required = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_required:
        raise ValueError(f"Missing essential required columns: {missing_required}")

    # Check 2: Soft check for the target variable (relevant for training)
    if "DemandQuantity" not in df.columns:
        warnings.append("Target column 'DemandQuantity' not found. Data is suitable for PREDICTION only, not training.")

    # --- Standardization ---

    # Standardize ID/Categorical columns to string type
    str_candidates = REQUIRED_COLUMNS + ["TowerType", "SubstationType"]
    for col in str_candidates:
        if col in df.columns:
            # Use fillna('') before astype(str) to handle NaNs gracefully for categorical encoding later
            df[col] = df[col].fillna('').astype(str)

    # Standardize numeric columns (coercing errors to NaN)
    numeric_candidates = [
        "MonthIndex",
        "ProjectLengthMonths",
        "StartMonth",
        "TaxRate",
        "ProjectRiskScore",
        "DemandQuantity",
    ]
    for col in numeric_candidates:
        if col in df.columns:
            # Use pd.to_numeric to convert, replacing errors (like text) with NaN
            original_nans = df[col].isnull().sum()
            df[col] = pd.to_numeric(df[col], errors="coerce")
            coerced_nans = df[col].isnull().sum()
            
            if coerced_nans > original_nans:
                warnings.append(f"Column '{col}' had {coerced_nans - original_nans} non-numeric values coerced to NaN.")

    # Final cleanup: Ensure DemandQuantity (the target) is non-negative
    if "DemandQuantity" in df.columns:
        if (df['DemandQuantity'] < 0).any():
            warnings.append("Negative values found in 'DemandQuantity' and set to 0.")
            df['DemandQuantity'] = df['DemandQuantity'].clip(lower=0)
            
    return df, warnings


# If you run this script directly, it can be tested:
if __name__ == '__main__':
    # Simple test data
    test_data = pd.DataFrame({
        "ProjectID": ["P1", "P2"],
        "MaterialType": ["Steel_Tons", "Cable_Meters"],
        "MonthIndex": [1, '2A'], # '2A' will cause a coercion warning
        "ProjectType": ["Line", "Substation"],
        "Region": ["North", "South"],
        "BudgetSegment": ["High", "Low"],
        "DemandQuantity": [100, -50], # -50 will cause a cleanup warning
        "ProjectRiskScore": [0.8, '0.5'],
        "TowerType": ["A-Type", np.nan]
    })
    
    try:
        validated_df, validation_warnings = validate_and_standardize(test_data.copy())
        print("Validation Successful.")
        print("--- Warnings ---")
        for w in validation_warnings:
            print(f"| {w}")
        print("\n--- Validated DataFrame Head ---")
        print(validated_df.head())
    except ValueError as e:
        print(f"Validation FAILED: {e}")