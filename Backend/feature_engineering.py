#!/usr/bin/env python3
import pandas as pd
import numpy as np

def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply feature engineering to the Materials Demand Forecasting DataFrame.
    
    1. Handle Cyclical/Time-Series features (Seasonality, Project Progress).
    2. Create Interaction features (Scope x Material, Risk x Progress).
    3. One-hot encode all final categorical features.
    """

    # Ensure all categorical columns are treated as strings before combining/encoding
    categorical_cols = [
        'MaterialType', 'ProjectType', 'Region', 'BudgetSegment', 
        'TowerType', 'SubstationType', 
    ]
    for col in [c for c in categorical_cols if c in df.columns]:
        df[col] = df[col].astype(str)

    # --- 1. Cyclical and Time-Series Features ---
    
    # a) Project Progress Ratio
    # This feature tells the model where the project is relative to its total duration.
    if 'MonthIndex' in df.columns and 'ProjectLengthMonths' in df.columns:
        # Avoid division by zero by clipping denominator at 1
        df['ProjectProgress'] = df['MonthIndex'] / df['ProjectLengthMonths'].clip(lower=1)
        # Clip max value at 1.0 (some MonthIndex might exceed estimated length slightly)
        df['ProjectProgress'] = df['ProjectProgress'].clip(upper=1.0)

    # b) Cyclical Encoding for StartMonth (Seasonality)
    # Converts the numerical month (1-12) into continuous sin/cos features.
    if 'StartMonth' in df.columns:
        # Use StartMonth to model annual seasonality
        month_period = df['StartMonth'] - 1 # 0 to 11
        df['StartMonth_sin'] = np.sin(2 * np.pi * month_period / 12)
        df['StartMonth_cos'] = np.cos(2 * np.pi * month_period / 12)
    
    
    # --- 2. Interaction and Domain Features ---

    # a) Scope x Material Interaction
    # Captures the specific materials demand for a given tower/substation type.
    if 'TowerType' in df.columns and 'MaterialType' in df.columns:
        # Create a new categorical feature: TowerType|MaterialType
        df['Tower_Mat_Interaction'] = df['TowerType'] + '|' + df['MaterialType']
    
    if 'SubstationType' in df.columns and 'MaterialType' in df.columns:
        # Create a new categorical feature: SubstationType|MaterialType
        df['Substation_Mat_Interaction'] = df['SubstationType'] + '|' + df['MaterialType']

    # b) Risk x Progress Interaction
    # Higher risk projects may see accelerated material consumption early on.
    if 'ProjectRiskScore' in df.columns and 'ProjectProgress' in df.columns:
        df['Risk_x_Progress'] = df['ProjectRiskScore'] * (1.0 - df['ProjectProgress'])
        
    # c) Budget Segment x Region (High-level cost/scale interaction)
    if 'BudgetSegment' in df.columns and 'Region' in df.columns:
        df['Budget_x_Region'] = df['BudgetSegment'] + '|' + df['Region']
        

    # --- 3. One-Hot Encoding ---

    # Define all categorical features we need to encode, including the new interactions
    all_cat_features = [
        'MaterialType', 'ProjectType', 'Region', 'BudgetSegment', 
        'TowerType', 'SubstationType', 'Budget_x_Region', 
        'Tower_Mat_Interaction', 'Substation_Mat_Interaction'
    ]
    
    # Select only the features that actually exist in the DataFrame
    features_to_encode = [col for col in all_cat_features if col in df.columns]
    
    # Perform One-Hot Encoding
    # dummy_na=True creates a column for NaN/missing values, important for handling 'nan' string from earlier
    df_encoded = pd.get_dummies(df, columns=features_to_encode, dummy_na=True)

    # --- Final Cleanup ---
    
    # Drop original columns used only for ID or to create new features (keep Budget/Region for now)
    cols_to_drop = [
        'ProjectID', 'StartMonth', 'MonthIndex', 'ProjectLengthMonths',
    ]
    df_encoded = df_encoded.drop(columns=[c for c in cols_to_drop if c in df_encoded.columns], errors='ignore')

    return df_encoded


# A minimal FeatureEngineer class wrapper (for pipeline compatibility)
class FeatureEngineer:
    def __init__(self):
        pass

    def fit_transform(self, df):
        # In a real system, fit() would learn the categories here.
        return apply_feature_engineering(df)

    def transform(self, df):
        # Transform applies the learned steps. Here, it's stateless.
        return apply_feature_engineering(df)

if __name__ == '__main__':
    # This test is simplified; typically, you'd test against the raw_demand_data.csv
    test_data = pd.DataFrame({
        "ProjectID": ["P1", "P1", "P2"],
        "MaterialType": ["Steel_Tons", "Cable_Meters", "Steel_Tons"],
        "MonthIndex": [1, 10, 5],
        "ProjectLengthMonths": [12, 12, 24],
        "ProjectType": ["Line", "Line", "Substation"],
        "TowerType": ["A-Type", "A-Type", 'nan'], # nan comes from Substation type
        "SubstationType": ['nan', 'nan', "400kV"],
        "Region": ["North", "North", "South"],
        "StartMonth": [10, 10, 5],
        "BudgetSegment": ["High", "High", "Low"],
        "ProjectRiskScore": [0.8, 0.8, 0.3],
        "TaxRate": [0.18, 0.18, 0.05],
        "DemandQuantity": [100, 50, 200]
    })
    
    fe_df = apply_feature_engineering(test_data)
    print("Feature Engineering Successful. Final Feature Count:", len(fe_df.columns))
    print("\n--- Final Features Head ---")
    # Displaying the engineered features and some original numerical features
    print(fe_df[[c for c in fe_df.columns if 'DemandQuantity' in c or 'sin' in c or 'Progress' in c or 'Type|' in c]].head())