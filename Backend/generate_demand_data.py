#!/usr/bin/env python3
"""
Generates a synthetic dataset for Materials Demand Forecasting.
The data is structured as Material/Project/Month time-series.
"""
import numpy as np
import pandas as pd
from typing import List

def create_demand_data(n_projects: int = 150, n_months: int = 24, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    
    # 1. Define Master Entities
    project_ids = [f'PROJ-{i+1:03d}' for i in range(n_projects)]
    material_types = ['Steel_Tons', 'Cable_Meters', 'Insulator_Units', 'Conductor_KMs']
    tower_types = ['A-Type', 'B-Type', 'D-Type']
    substation_types = ['400kV', '220kV', '132kV']
    regions = ['North', 'West', 'South', 'East']
    budget_segments = ['Low', 'Medium', 'High']

    records = []
    
    # Base Quantities (BOM proxy)
    base_bom = {
        'Steel_Tons': {'A-Type': 150, 'B-Type': 100, 'D-Type': 50, '400kV': 1200, '220kV': 800, '132kV': 400},
        'Cable_Meters': {'A-Type': 2000, 'B-Type': 1500, 'D-Type': 1000, '400kV': 8000, '220kV': 5000, '132kV': 3000},
        'Insulator_Units': {'A-Type': 120, 'B-Type': 80, 'D-Type': 40, '400kV': 400, '220kV': 200, '132kV': 100},
        'Conductor_KMs': {'A-Type': 5, 'B-Type': 3, 'D-Type': 2, '400kV': 0, '220kV': 0, '132kV': 0} # Conductor tied mostly to line types
    }
    
    # Demand Timing Curve (e.g., Civil demand early, Electrical later)
    # Peak demand month for a 24-month project
    timing_profile = {
        'Steel_Tons': 6, # Foundation/Tower construction
        'Cable_Meters': 12, # Erection/Wiring
        'Insulator_Units': 18, # Final Erection/Testing
        'Conductor_KMs': 10 
    }

    # --- 2. Generate Project-Level Data (Corrected) ---

    # 1. Create the base DataFrame with non-conditional columns
    project_data = pd.DataFrame({
        'ProjectID': project_ids,
        # The ProjectType must be defined first
        'ProjectType': rng.choice(['Line', 'Substation'], n_projects, p=[0.6, 0.4]),
        'Region': rng.choice(regions, n_projects, p=[0.3, 0.25, 0.25, 0.2]),
        'ProjectLengthMonths': rng.integers(12, 36, n_projects),
        'StartMonth': rng.integers(1, 13, n_projects),
        'BudgetSegment': rng.choice(budget_segments, n_projects),
        'ProjectRiskScore': rng.uniform(0.1, 0.8, n_projects)
    })

    
    # 2. Add TowerType and SubstationType using vectorized conditional logic (np.where)
    # Note: We generate the full array of random choices first, then select based on the condition.

    # Generate random choices for all rows
    random_tower_choices = rng.choice(tower_types, n_projects)
    random_substation_choices = rng.choice(substation_types, n_projects)

    project_data['TowerType'] = np.where(
        project_data['ProjectType'] == 'Line',
        random_tower_choices,  # Assign a random tower type if ProjectType is 'Line'
        np.nan                 # Otherwise, assign NaN
    )

    project_data['SubstationType'] = np.where(
        project_data['ProjectType'] == 'Substation',
        random_substation_choices, # Assign a random substation type if ProjectType is 'Substation'
        np.nan                     # Otherwise, assign NaN
    )

    # 3. Complete the Project-Level Data (Mapping Tax and Scale)
    # This was part of the original script's logic after project_data creation

    project_data['TaxRate'] = project_data['Region'].map({
        'North': 0.18, 'West': 0.12, 'South': 0.05, 'East': 0.18
    })
    project_data['ScaleFactor'] = project_data['BudgetSegment'].map({
        'Low': 0.8, 'Medium': 1.0, 'High': 1.5
    })

    # project_data is now complete and ready for the next step (monthly material demand generation)

    
    # 3. Generate Monthly Material Demand
    for _, proj in project_data.iterrows():
        for month in range(1, proj['ProjectLengthMonths'] + 1):
            
            # Seasonality (Higher demand in Q1/Q4 due to financial year end/clear weather)
            month_of_year = (proj['StartMonth'] + month - 2) % 12 + 1
            seasonal_mult = 1.0 + np.cos((month_of_year - 1) / 12 * 2 * np.pi) * 0.2
            
            for material in material_types:
                
                # Determine base quantity from project type
                if proj['ProjectType'] == 'Line' and material in base_bom:
                    base_qty = base_bom[material].get(proj['TowerType'], 0)
                elif proj['ProjectType'] == 'Substation' and material in base_bom:
                    base_qty = base_bom[material].get(proj['SubstationType'], 0)
                else:
                    base_qty = 0

                if base_qty > 0:
                    # Time-Series Demand Distribution (Bell Curve around peak month)
                    peak_month = timing_profile[material]
                    time_diff = abs(month - peak_month)
                    time_mult = np.exp(-0.5 * (time_diff / 5)**2) # Gaussian distribution around peak
                    
                    # Risk factor: High risk pulls demand forward slightly
                    risk_mult = 1.0 + proj['ProjectRiskScore'] * 0.3 * (1 - (month / proj['ProjectLengthMonths']))
                    
                    # Calculate final demand
                    demand_qty = base_qty * time_mult * proj['ScaleFactor'] * seasonal_mult * risk_mult
                    
                    # Add noise and ensure non-negativity
                    demand_qty = demand_qty + rng.normal(0, demand_qty * 0.1)
                    
                    records.append({
                        'ProjectID': proj['ProjectID'],
                        'MaterialType': material,
                        'MonthIndex': month, # Month relative to project start
                        'ProjectLengthMonths': proj['ProjectLengthMonths'],
                        'ProjectType': proj['ProjectType'],
                        'TowerType': proj['TowerType'],
                        'SubstationType': proj['SubstationType'],
                        'Region': proj['Region'],
                        'StartMonth': proj['StartMonth'],
                        'BudgetSegment': proj['BudgetSegment'],
                        'TaxRate': proj['TaxRate'],
                        'ProjectRiskScore': proj['ProjectRiskScore'],
                        'DemandQuantity': max(0, demand_qty)
                    })

    df = pd.DataFrame(records)
    # Convert Demand to integer units for more realistic ordering (e.g., tons, units)
    df['DemandQuantity'] = df['DemandQuantity'].round().astype(int) 
    return df


def main():
    print("Generating synthetic Materials Demand Data...")
    df = create_demand_data(n_projects=200, n_months=48)
    
    # Save the raw data
    df.to_csv('raw_demand_data.csv', index=False)
    print(f"Generated and saved raw_demand_data.csv with {len(df)} records.")
    print("\nSample Data:")
    print(df.head())

if __name__ == '__main__':
    main()