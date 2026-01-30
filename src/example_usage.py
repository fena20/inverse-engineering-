#!/usr/bin/env python3
"""
Example usage of the Retrofit Decision Support System.

This script demonstrates the three main use cases:
1. UC-1: Performance Prediction
2. UC-2: Inverse Design (Optimization)
3. UC-3: Sensitivity Analysis
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np

from retrofit_dss.data.loader import DataLoader
from retrofit_dss.data.preprocessor import DataPreprocessor
from retrofit_dss.models.surrogate import SurrogateModelFactory
from retrofit_dss.optimization.engine import OptimizationEngine


def main():
    print("=" * 70)
    print("Retrofit Decision Support System - Example Usage")
    print("=" * 70)
    
    # Load data and models
    print("\n[Setup] Loading data and models...")
    
    loader = DataLoader('data')
    loader.discover_cities()
    certs_df, recs_df = loader.get_merged_data()
    
    preprocessor = DataPreprocessor()
    processed_df = preprocessor.fit_transform(certs_df)
    
    # Load trained models
    model_factory = SurrogateModelFactory('gradient_boosting')
    model_factory.create_all_models()
    model_factory.load_all('models')
    
    # Initialize optimizer
    optimizer = OptimizationEngine(model_factory)
    optimizer.load_recommendations(recs_df)
    
    # ===================================================================
    # UC-1: Performance Prediction
    # ===================================================================
    print("\n" + "=" * 70)
    print("UC-1: Performance Prediction")
    print("=" * 70)
    
    # Create a sample building profile with ALL required features
    sample_building = {
        'TOTAL_FLOOR_AREA': 90.0,
        'WALLS_ENERGY_EFF': 'Poor',
        'WALLS_ENV_EFF': 'Poor',
        'ROOF_ENERGY_EFF': 'Average',
        'ROOF_ENV_EFF': 'Average',
        'FLOOR_ENERGY_EFF': 'N/A',
        'FLOOR_ENV_EFF': 'N/A',
        'WINDOWS_ENERGY_EFF': 'Average',
        'WINDOWS_ENV_EFF': 'Average',
        'MAINHEAT_ENERGY_EFF': 'Good',
        'MAINHEAT_ENV_EFF': 'Good',
        'MAINHEATC_ENERGY_EFF': 'Good',
        'MAINHEATC_ENV_EFF': 'Good',
        'HOT_WATER_ENERGY_EFF': 'Good',
        'HOT_WATER_ENV_EFF': 'Good',
        'LIGHTING_ENERGY_EFF': 'Poor',
        'LIGHTING_ENV_EFF': 'Poor',
        'SHEATING_ENERGY_EFF': 'N/A',
        'SHEATING_ENV_EFF': 'N/A',
        'CONSTRUCTION_AGE_BAND': 'England and Wales: 1930-1949',
        'PROPERTY_TYPE': 'House',
        'BUILT_FORM': 'Semi-Detached',
        'GLAZED_TYPE': 'double glazing installed before 2002',
        'MAINS_GAS_FLAG': 'Y',
        'SOLAR_WATER_HEATING_FLAG': 'N',
        'LOW_ENERGY_LIGHTING': 20,
        'MULTI_GLAZE_PROPORTION': 100,
        'NUMBER_HABITABLE_ROOMS': 5,
        'NUMBER_HEATED_ROOMS': 5,
        'EXTENSION_COUNT': 0,
        'NUMBER_OPEN_FIREPLACES': 1,
        'PHOTO_SUPPLY': 0,
        'WIND_TURBINE_COUNT': 0,
        'CITY': 'Liverpool'
    }
    
    # Display key building characteristics
    key_chars = [
        'TOTAL_FLOOR_AREA', 'WALLS_ENERGY_EFF', 'ROOF_ENERGY_EFF',
        'WINDOWS_ENERGY_EFF', 'MAINHEAT_ENERGY_EFF', 'CONSTRUCTION_AGE_BAND',
        'PROPERTY_TYPE', 'BUILT_FORM', 'LOW_ENERGY_LIGHTING', 'CITY'
    ]
    
    print("\nBuilding Profile:")
    print("-" * 40)
    for key in key_chars:
        print(f"  {key}: {sample_building[key]}")
    
    # Add climate data for Liverpool
    sample_building['HDD'] = 2150  # Heating Degree Days
    sample_building['AVG_TEMP'] = 10.3  # Average temperature
    sample_building['SOLAR_RADIATION'] = 900  # kWh/m2/year
    
    # Preprocess and predict
    df = pd.DataFrame([sample_building])
    processed = preprocessor.transform(df)
    
    # Load the feature columns from the trained model
    feature_cols = model_factory.feature_columns
    
    # Ensure all feature columns exist with physics-appropriate defaults
    for col in feature_cols:
        if col not in processed.columns:
            # Set physics-appropriate defaults
            if 'EFF_NUM' in col:
                processed[col] = 3  # Average
            elif 'HEATING_DEMAND_PROXY' in col:
                # HDD * heat_loss_proxy / 1000
                processed[col] = 2150 * processed.get('HEAT_LOSS_PROXY', 50) / 1000
            else:
                processed[col] = 0
    
    X = processed[feature_cols]
    predictions = model_factory.predict(X)
    if not predictions or 'energy' not in predictions:
        print("\nWarning: No fitted models available for prediction. "
              "Train models before running the example.")
        return
    
    print("\nPredicted Performance:")
    print("-" * 40)
    print(f"  Energy Intensity: {predictions['energy'][0]:.1f} kWh/m²/year")
    print(f"  Carbon Intensity: {predictions['carbon'][0]:.1f} kg CO₂/m²/year")
    print(f"  Heating Cost: £{predictions['heating_cost'][0]:.0f}/year")
    print(f"  Hot Water Cost: £{predictions['hot_water_cost'][0]:.0f}/year")
    print(f"  Lighting Cost: £{predictions['lighting_cost'][0]:.0f}/year")
    print(f"  Total Annual Cost: £{predictions['total_cost'][0]:.0f}/year")
    
    # Estimate EPC grade
    energy_intensity = predictions['energy'][0]
    if energy_intensity <= 50:
        grade = 'A'
    elif energy_intensity <= 100:
        grade = 'B'
    elif energy_intensity <= 150:
        grade = 'C'
    elif energy_intensity <= 225:
        grade = 'D'
    elif energy_intensity <= 325:
        grade = 'E'
    elif energy_intensity <= 450:
        grade = 'F'
    else:
        grade = 'G'
    print(f"  Estimated EPC Grade: {grade}")
    
    # ===================================================================
    # UC-2: Inverse Design (Optimization)
    # ===================================================================
    print("\n" + "=" * 70)
    print("UC-2: Inverse Design - 50% Carbon Reduction Target")
    print("=" * 70)
    
    # Convert to Series for optimization
    profile_series = processed.iloc[0]
    
    # Get retrofit packages
    packages = optimizer.optimize(
        profile_series,
        target_type='carbon',
        target_reduction=50.0,
        max_budget=25000,
        max_measures=4
    )
    
    print(f"\nFound {len(packages)} packages meeting target:")
    print("-" * 40)
    
    for i, pkg in enumerate(packages[:5]):  # Show top 5
        print(f"\nPackage {i+1}:")
        print(f"  Measures: {', '.join(m.name for m in pkg.measures)}")
        print(f"  Cost Range: £{pkg.total_cost_min:,.0f} - £{pkg.total_cost_max:,.0f}")
        print(f"  Energy Reduction: {pkg.predicted_energy_reduction:.1f}%")
        print(f"  Carbon Reduction: {pkg.predicted_carbon_reduction:.1f}%")
        print(f"  Annual Savings: £{pkg.predicted_cost_savings:,.0f}")
        if pkg.payback_years:
            print(f"  Payback Period: {pkg.payback_years:.1f} years")
    
    # Cost-benefit summary
    print("\n" + "-" * 40)
    print("Cost-Benefit Summary Table:")
    summary_df = optimizer.get_cost_benefit_summary(packages[:5])
    print(summary_df.to_string(index=False))
    
    # ===================================================================
    # UC-3: Sensitivity Analysis
    # ===================================================================
    print("\n" + "=" * 70)
    print("UC-3: Sensitivity Analysis - Wall Insulation Impact")
    print("=" * 70)
    
    # Test wall efficiency impact
    wall_values = [1, 2, 3, 4, 5]  # Very Poor to Very Good
    wall_labels = ['Very Poor', 'Poor', 'Average', 'Good', 'Very Good']
    
    results = []
    for val in wall_values:
        test_profile = profile_series.copy()
        test_profile['WALLS_ENERGY_EFF_NUM'] = val
        # Recalculate envelope quality
        test_profile['ENVELOPE_QUALITY'] = (
            0.35 * val + 
            0.25 * test_profile.get('ROOF_ENERGY_EFF_NUM', 3) +
            0.15 * test_profile.get('FLOOR_ENERGY_EFF_NUM', 0) +
            0.25 * test_profile.get('WINDOWS_ENERGY_EFF_NUM', 3)
        )
        
        X_test = pd.DataFrame([test_profile])[feature_cols]
        pred = model_factory.predict(X_test)
        results.append({
            'Wall Rating': wall_labels[val-1],
            'Energy (kWh/m²)': pred['energy'][0],
            'Carbon (kg/m²)': pred['carbon'][0],
            'Total Cost (£)': pred['total_cost'][0]
        })
    
    results_df = pd.DataFrame(results)
    print("\nImpact of Wall Insulation Quality:")
    print("-" * 40)
    print(results_df.to_string(index=False))
    
    # Calculate sensitivity
    energy_range = results_df['Energy (kWh/m²)'].max() - results_df['Energy (kWh/m²)'].min()
    print(f"\nEnergy consumption range: {energy_range:.1f} kWh/m² across wall ratings")
    print(f"This represents a {energy_range/results_df['Energy (kWh/m²)'].max()*100:.1f}% potential reduction")
    
    # ===================================================================
    # Feature Importance
    # ===================================================================
    print("\n" + "=" * 70)
    print("Physical Interpretability - Feature Importance")
    print("=" * 70)
    
    importance = model_factory.get_all_feature_importance()
    energy_importance = importance['energy'].get_top_features(15)
    
    print("\nTop 15 Features for Energy Prediction:")
    print("-" * 40)
    for _, row in energy_importance.iterrows():
        bar_len = int(row['importance'] * 50)
        bar = '█' * bar_len
        print(f"  {row['feature'][:25]:<25} {row['importance']:.4f} {bar}")
    
    # Validate physical intuition
    validation = importance['energy'].validate_physical_intuition()
    print(f"\nPhysical Intuition Match Rate: {validation['match_rate']*100:.0f}%")
    print(f"Expected features found in top 15: {validation['expected_found']}")
    
    print("\n" + "=" * 70)
    print("Example Complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
