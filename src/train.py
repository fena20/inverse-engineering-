#!/usr/bin/env python3
"""
Training script for Retrofit DSS models.

This script:
1. Loads and merges EPC data from all cities
2. Preprocesses features with physics-based encoding
3. Trains surrogate models for energy, carbon, and costs
4. Evaluates models and validates physical consistency
5. Saves trained models for API use
"""
import os
import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

from retrofit_dss.data.loader import DataLoader
from retrofit_dss.data.preprocessor import DataPreprocessor, create_train_test_split
from retrofit_dss.models.surrogate import SurrogateModelFactory
from retrofit_dss.optimization.engine import OptimizationEngine


def main(data_dir: str = 'data', model_dir: str = 'models', sample_size: int = None):
    """
    Main training pipeline.
    
    Args:
        data_dir: Path to data directory
        model_dir: Path to save models
        sample_size: Optional sample size for faster training
    """
    print("=" * 60)
    print("Retrofit DSS - Model Training Pipeline")
    print("=" * 60)
    
    # Create model directory
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    
    # 1. Load Data
    print("\n[1/6] Loading EPC data from all cities...")
    print("-" * 40)
    
    loader = DataLoader(data_dir)
    cities = loader.discover_cities()
    print(f"Found cities: {cities}")
    
    certs_df, recs_df = loader.get_merged_data()
    
    # 2. Train/Test Split (by postcode to avoid leakage)
    print("\n[2/6] Creating train/test split (by postcode)...")
    print("-" * 40)
    
    raw_train_df, raw_test_df = create_train_test_split(
        certs_df,
        test_size=0.2,
        random_state=42
    )
    
    # 3. Preprocess Data (fit on train only)
    print("\n[3/6] Preprocessing data...")
    print("-" * 40)
    
    preprocessor = DataPreprocessor()
    preprocessor.fit(raw_train_df)
    train_df = preprocessor.transform(raw_train_df)
    test_df = preprocessor.transform(raw_test_df)
    test_df = preprocessor.ensure_feature_columns(test_df)
    
    print(f"Total records: {len(train_df) + len(test_df):,}")
    print(f"Feature columns: {len(preprocessor.get_feature_columns())}")
    
    # 4. Sample if needed
    if sample_size and sample_size < len(train_df):
        print(f"\nSampling {sample_size:,} records for training...")
        train_df = train_df.sample(sample_size, random_state=42)
    
    print(f"Training set: {len(train_df):,} records")
    print(f"Test set: {len(test_df):,} records")
    
    # Verify no postcode overlap
    train_postcodes = set(raw_train_df['POSTCODE'].dropna().unique())
    test_postcodes = set(raw_test_df['POSTCODE'].dropna().unique())
    overlap = train_postcodes.intersection(test_postcodes)
    print(f"Postcode overlap check: {len(overlap)} overlapping postcodes")
    
    # 5. Train Models
    print("\n[4/6] Training surrogate models...")
    print("-" * 40)
    
    model_factory = SurrogateModelFactory('gradient_boosting')
    model_factory.create_all_models()
    
    feature_columns = preprocessor.get_feature_columns()
    model_factory.fit_all(train_df, feature_columns)
    
    # 6. Evaluate Models
    print("\n[5/6] Evaluating models...")
    print("-" * 40)
    
    metrics = model_factory.evaluate_all(test_df)
    
    # 7. Validate Physical Consistency
    print("\n[6/6] Validating physical consistency...")
    print("-" * 40)
    
    validation = model_factory.validate_physical_consistency()
    
    for model_name, results in validation.items():
        print(f"\n{model_name}:")
        print(f"  Match rate: {results['match_rate']*100:.1f}%")
        print(f"  Physically consistent: {results['physically_consistent']}")
        print(f"  Expected features found: {results['expected_found'][:5]}...")
        if results['expected_missing']:
            print(f"  Missing expected: {results['expected_missing'][:3]}...")
    
    # 8. Feature Importance Analysis
    print("\n" + "=" * 60)
    print("Feature Importance Analysis (Energy Model)")
    print("=" * 60)
    
    importance_dict = model_factory.get_all_feature_importance()
    if 'energy' in importance_dict:
        top_features = importance_dict['energy'].get_top_features(15)
        print("\nTop 15 features:")
        for i, row in top_features.iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
    
    # 9. Save Models
    print("\n" + "=" * 60)
    print("Saving models...")
    print("=" * 60)
    
    model_factory.save_all(model_dir)
    print(f"Models saved to: {model_dir}")
    
    # 10. Generate Report
    print("\n" + "=" * 60)
    print("Training Summary Report")
    print("=" * 60)
    
    print("\nModel Performance Summary:")
    print("-" * 40)
    for name, m in metrics.items():
        print(f"{name}:")
        print(f"  R² Score: {m.r2:.4f}")
        print(f"  MAE: {m.mae:.2f}")
        print(f"  RMSE: {m.rmse:.2f}")
        if m.mape:
            print(f"  MAPE: {m.mape:.1f}%")
        print()
    
    # Save metrics to file
    metrics_data = {name: m.to_dict() for name, m in metrics.items()}
    pd.DataFrame(metrics_data).T.to_csv(Path(model_dir) / 'model_metrics.csv')
    print(f"Metrics saved to: {model_dir}/model_metrics.csv")
    
    # 11. Test Optimization Engine
    print("\n" + "=" * 60)
    print("Testing Optimization Engine")
    print("=" * 60)
    
    optimizer = OptimizationEngine(model_factory, preprocessor)
    optimizer.load_recommendations(recs_df)
    
    # Test with sample building
    sample_building = test_df.iloc[0]
    packages = optimizer.optimize(
        sample_building,
        target_type='carbon',
        target_reduction=30.0,
        max_measures=3
    )
    
    print(f"\nSample optimization (30% carbon reduction target):")
    print(f"Found {len(packages)} packages")
    
    if packages:
        best = packages[0]
        print(f"\nBest package:")
        print(f"  Measures: {[m.name for m in best.measures]}")
        print(f"  Cost: £{best.total_cost_min:,.0f} - £{best.total_cost_max:,.0f}")
        print(f"  Carbon reduction: {best.predicted_carbon_reduction:.1f}%")
        print(f"  Annual savings: £{best.predicted_cost_savings:,.0f}")
        if best.predicted_carbon_reduction < 30.0:
            print("  Target not achievable under current constraints; showing best available package.")
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Retrofit DSS models')
    parser.add_argument('--data-dir', default='data', help='Path to data directory')
    parser.add_argument('--model-dir', default='models', help='Path to save models')
    parser.add_argument('--sample-size', type=int, default=None, 
                       help='Sample size for training (optional)')
    
    args = parser.parse_args()
    
    main(args.data_dir, args.model_dir, args.sample_size)
