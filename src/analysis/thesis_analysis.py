#!/usr/bin/env python3
"""
Comprehensive Thesis Analysis Script for Retrofit DSS

This script generates all figures and tables required for the thesis:
- Chapter 3: EDA and Data Quality
- Chapter 5: Model Results and Validation
- Chapter 6: Interpretability and Physics Analysis
- Chapter 7: Optimization and Retrofit Recommendations

Author: Retrofit DSS Team
"""
import os
import sys
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 11

# Create output directory
OUTPUT_DIR = Path('outputs/thesis_figures')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_and_prepare_data():
    """Load and preprocess all city data."""
    from retrofit_dss.data.loader import DataLoader
    from retrofit_dss.data.preprocessor import DataPreprocessor, create_train_test_split
    
    print("=" * 70)
    print("Loading Data from All Cities...")
    print("=" * 70)
    
    loader = DataLoader('data')
    loader.discover_cities()
    certs_df, recs_df = loader.get_merged_data()
    
    # Create train/test split on raw data to avoid preprocessing leakage
    train_raw, test_raw = create_train_test_split(certs_df, test_size=0.2, random_state=42)

    preprocessor = DataPreprocessor()
    preprocessor.fit(train_raw)

    train_df = preprocessor.transform(train_raw)
    test_df = preprocessor.transform(test_raw)
    processed_df = preprocessor.transform(certs_df)
    
    return certs_df, recs_df, processed_df, train_df, test_df, preprocessor


# =============================================================================
# CHAPTER 3: EXPLORATORY DATA ANALYSIS
# =============================================================================

def chapter3_eda(df, output_dir):
    """Generate EDA figures for Chapter 3."""
    print("\n" + "=" * 70)
    print("CHAPTER 3: Exploratory Data Analysis")
    print("=" * 70)
    
    # 3.1 City-wise Energy Distribution (Violin Plot)
    print("\n[3.1] City-wise Energy Distribution...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Filter valid data
    energy_data = df[df['ENERGY_CONSUMPTION_CURRENT'].notna() & 
                     (df['ENERGY_CONSUMPTION_CURRENT'] > 0) &
                     (df['ENERGY_CONSUMPTION_CURRENT'] < 800)]
    
    # Violin plot
    ax1 = axes[0]
    city_order = ['Cambridge', 'Boston', 'Liverpool', 'Sheffield']
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']
    
    parts = ax1.violinplot([energy_data[energy_data['CITY'] == city]['ENERGY_CONSUMPTION_CURRENT'].dropna() 
                           for city in city_order],
                          positions=range(len(city_order)), showmeans=True, showmedians=True)
    
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)
    
    ax1.set_xticks(range(len(city_order)))
    ax1.set_xticklabels(city_order)
    ax1.set_ylabel('Primary Energy Intensity (kWh/m²/year)')
    ax1.set_title('Distribution of Energy Consumption by City')
    ax1.axhline(y=energy_data['ENERGY_CONSUMPTION_CURRENT'].median(), 
                color='gray', linestyle='--', alpha=0.5, label='Overall Median')
    ax1.legend()
    
    # Box plot for comparison
    ax2 = axes[1]
    box_data = [energy_data[energy_data['CITY'] == city]['ENERGY_CONSUMPTION_CURRENT'].dropna() 
                for city in city_order]
    bp = ax2.boxplot(box_data, labels=city_order, patch_artist=True)
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax2.set_ylabel('Primary Energy Intensity (kWh/m²/year)')
    ax2.set_title('Energy Consumption Statistics by City')
    
    # Add statistics annotation
    stats_text = ""
    for i, city in enumerate(city_order):
        city_data = energy_data[energy_data['CITY'] == city]['ENERGY_CONSUMPTION_CURRENT']
        stats_text += f"{city}: μ={city_data.mean():.0f}, σ={city_data.std():.0f}\n"
    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig3_1_city_energy_distribution.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'fig3_1_city_energy_distribution.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: fig3_1_city_energy_distribution.png/pdf")
    
    # 3.2 Building Age vs Efficiency Heatmap
    print("\n[3.2] Building Age vs Efficiency Heatmap...")
    
    # Prepare data
    age_bands = [
        'England and Wales: before 1900',
        'England and Wales: 1900-1929',
        'England and Wales: 1930-1949',
        'England and Wales: 1950-1966',
        'England and Wales: 1967-1975',
        'England and Wales: 1976-1982',
        'England and Wales: 1983-1990',
        'England and Wales: 1991-1995',
        'England and Wales: 1996-2002',
        'England and Wales: 2003-2006',
        'England and Wales: 2007 onwards'
    ]
    
    age_labels = ['<1900', '1900-29', '1930-49', '1950-66', '1967-75',
                  '1976-82', '1983-90', '1991-95', '1996-02', '2003-06', '2007+']
    
    efficiency_mapping = {'Very Poor': 1, 'Poor': 2, 'Average': 3, 'Good': 4, 'Very Good': 5}
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    for idx, (city, ax) in enumerate(zip(city_order, axes.flatten())):
        city_data = df[df['CITY'] == city].copy()
        
        # Create pivot table for walls efficiency
        pivot_data = []
        for age in age_bands:
            age_subset = city_data[city_data['CONSTRUCTION_AGE_BAND'] == age]
            if len(age_subset) > 0:
                walls_mean = age_subset['WALLS_ENERGY_EFF'].map(efficiency_mapping).mean()
                roof_mean = age_subset['ROOF_ENERGY_EFF'].map(efficiency_mapping).mean()
                pivot_data.append({
                    'Age Band': age_labels[age_bands.index(age)],
                    'Walls': walls_mean if pd.notna(walls_mean) else 0,
                    'Roof': roof_mean if pd.notna(roof_mean) else 0
                })
        
        if pivot_data:
            pivot_df = pd.DataFrame(pivot_data).set_index('Age Band')
            
            sns.heatmap(pivot_df.T, annot=True, fmt='.2f', cmap='RdYlGn',
                       vmin=1, vmax=5, ax=ax, cbar_kws={'label': 'Efficiency Rating'})
            ax.set_title(f'{city}: Envelope Efficiency by Building Age')
            ax.set_xlabel('Construction Period')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig3_2_age_efficiency_heatmap.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'fig3_2_age_efficiency_heatmap.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: fig3_2_age_efficiency_heatmap.png/pdf")
    
    # 3.3 Correlation Matrix
    print("\n[3.3] Correlation Matrix...")
    
    # Select relevant columns
    corr_cols = [
        'TOTAL_FLOOR_AREA', 'AGE_BAND_NUM',
        'WALLS_ENERGY_EFF_NUM', 'ROOF_ENERGY_EFF_NUM', 'WINDOWS_ENERGY_EFF_NUM',
        'MAINHEAT_ENERGY_EFF_NUM', 'HOT_WATER_ENERGY_EFF_NUM',
        'LOW_ENERGY_LIGHTING', 'NUMBER_OPEN_FIREPLACES',
        'ENERGY_CONSUMPTION_CURRENT', 'CO2_EMISS_CURR_PER_FLOOR_AREA',
        'HEATING_COST_CURRENT', 'HOT_WATER_COST_CURRENT', 'LIGHTING_COST_CURRENT'
    ]
    
    # Filter existing columns
    available_cols = [c for c in corr_cols if c in df.columns]
    corr_data = df[available_cols].dropna()
    
    # Rename for display
    col_names = {
        'TOTAL_FLOOR_AREA': 'Floor Area',
        'AGE_BAND_NUM': 'Building Age',
        'WALLS_ENERGY_EFF_NUM': 'Wall Eff.',
        'ROOF_ENERGY_EFF_NUM': 'Roof Eff.',
        'WINDOWS_ENERGY_EFF_NUM': 'Window Eff.',
        'MAINHEAT_ENERGY_EFF_NUM': 'Heating Eff.',
        'HOT_WATER_ENERGY_EFF_NUM': 'Hot Water Eff.',
        'LOW_ENERGY_LIGHTING': 'LED %',
        'NUMBER_OPEN_FIREPLACES': 'Fireplaces',
        'ENERGY_CONSUMPTION_CURRENT': 'Energy (kWh/m²)',
        'CO2_EMISS_CURR_PER_FLOOR_AREA': 'Carbon (kg/m²)',
        'HEATING_COST_CURRENT': 'Heating Cost',
        'HOT_WATER_COST_CURRENT': 'Hot Water Cost',
        'LIGHTING_COST_CURRENT': 'Lighting Cost'
    }
    
    corr_data = corr_data.rename(columns=col_names)
    corr_matrix = corr_data.corr()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, vmin=-1, vmax=1, ax=ax, square=True,
                cbar_kws={'label': 'Correlation Coefficient'})
    
    ax.set_title('Correlation Matrix: Building Features vs Energy Performance')
    plt.tight_layout()
    plt.savefig(output_dir / 'fig3_3_correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'fig3_3_correlation_matrix.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: fig3_3_correlation_matrix.png/pdf")
    
    # 3.4 Data Summary Table
    print("\n[3.4] Data Summary Statistics...")
    
    summary_data = []
    for city in city_order:
        city_df = df[df['CITY'] == city]
        summary_data.append({
            'City': city,
            'Records': len(city_df),
            'Mean Energy (kWh/m²)': city_df['ENERGY_CONSUMPTION_CURRENT'].mean(),
            'Std Energy': city_df['ENERGY_CONSUMPTION_CURRENT'].std(),
            'Mean Carbon (kg/m²)': city_df['CO2_EMISS_CURR_PER_FLOOR_AREA'].mean(),
            'Mean Floor Area (m²)': city_df['TOTAL_FLOOR_AREA'].mean(),
            'Mean Heating Cost (£)': city_df['HEATING_COST_CURRENT'].mean(),
            'HDD': city_df['HDD'].iloc[0] if 'HDD' in city_df.columns else 'N/A'
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_dir / 'table3_1_city_summary.csv', index=False)
    print(f"  Saved: table3_1_city_summary.csv")
    print("\n  City Data Summary:")
    print(summary_df.to_string(index=False))
    
    return summary_df


# =============================================================================
# CHAPTER 5: MODEL RESULTS
# =============================================================================

def chapter5_model_results(train_df, test_df, preprocessor, output_dir):
    """Generate model results for Chapter 5."""
    print("\n" + "=" * 70)
    print("CHAPTER 5: Model Results and Validation")
    print("=" * 70)
    
    from retrofit_dss.models.surrogate import SurrogateModelFactory
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    # Train models
    print("\n[5.1] Training Models...")
    model_factory = SurrogateModelFactory('gradient_boosting')
    model_factory.create_all_models()
    
    feature_cols = preprocessor.get_feature_columns()
    model_factory.fit_all(train_df, feature_cols)
    
    # 5.1 Model Accuracy Table by City
    print("\n[5.2] Calculating Per-City Model Accuracy...")
    
    cities = ['Cambridge', 'Boston', 'Liverpool', 'Sheffield']
    targets = ['energy', 'carbon', 'heating_cost', 'total_cost']
    target_names = {
        'energy': 'ENERGY_CONSUMPTION_CURRENT',
        'carbon': 'CO2_EMISS_CURR_PER_FLOOR_AREA',
        'heating_cost': 'HEATING_COST_CURRENT',
        'total_cost': 'TOTAL_COST_CURRENT'
    }
    
    accuracy_results = []
    
    for target in targets:
        target_col = target_names[target]
        model = model_factory.models.get(target)
        
        if model is None or not model._fitted:
            continue
        
        # Overall metrics
        test_valid = test_df[test_df[target_col].notna()]
        predictions = model.predict(test_valid[feature_cols])
        actuals = test_valid[target_col].values
        
        accuracy_results.append({
            'Target': target.replace('_', ' ').title(),
            'City': 'All Cities',
            'R²': r2_score(actuals, predictions),
            'MAE': mean_absolute_error(actuals, predictions),
            'RMSE': np.sqrt(mean_squared_error(actuals, predictions)),
            'N': len(actuals)
        })
        
        # Per-city metrics
        for city in cities:
            city_test = test_valid[test_valid['CITY'] == city]
            if len(city_test) < 50:
                continue
            
            city_pred = model.predict(city_test[feature_cols])
            city_actual = city_test[target_col].values
            
            accuracy_results.append({
                'Target': target.replace('_', ' ').title(),
                'City': city,
                'R²': r2_score(city_actual, city_pred),
                'MAE': mean_absolute_error(city_actual, city_pred),
                'RMSE': np.sqrt(mean_squared_error(city_actual, city_pred)),
                'N': len(city_actual)
            })
    
    accuracy_df = pd.DataFrame(accuracy_results)
    accuracy_df.to_csv(output_dir / 'table5_1_model_accuracy.csv', index=False)
    print(f"  Saved: table5_1_model_accuracy.csv")
    
    # Print formatted table
    print("\n  Model Accuracy Summary:")
    pivot_r2 = accuracy_df[accuracy_df['Target'] == 'Energy'].pivot(
        index='City', columns='Target', values='R²'
    )
    print(accuracy_df.to_string(index=False))
    
    # 5.2 Actual vs Predicted Scatter Plots
    print("\n[5.3] Generating Actual vs Predicted Plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    for idx, (target, ax) in enumerate(zip(targets, axes.flatten())):
        target_col = target_names[target]
        model = model_factory.models.get(target)
        
        if model is None or not model._fitted:
            continue
        
        test_valid = test_df[test_df[target_col].notna()]
        predictions = model.predict(test_valid[feature_cols])
        actuals = test_valid[target_col].values
        
        # Color by city
        colors_map = {'Cambridge': '#2ecc71', 'Boston': '#3498db', 
                     'Liverpool': '#e74c3c', 'Sheffield': '#9b59b6'}
        
        for city in cities:
            mask = test_valid['CITY'] == city
            ax.scatter(actuals[mask], predictions[mask], 
                      alpha=0.3, s=10, c=colors_map[city], label=city)
        
        # Perfect prediction line
        max_val = max(actuals.max(), predictions.max())
        min_val = min(actuals.min(), predictions.min())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Perfect Prediction')
        
        # Add R² annotation
        r2 = r2_score(actuals, predictions)
        mae = mean_absolute_error(actuals, predictions)
        ax.text(0.05, 0.95, f'R² = {r2:.3f}\nMAE = {mae:.1f}', 
                transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xlabel(f'Actual {target.replace("_", " ").title()}')
        ax.set_ylabel(f'Predicted {target.replace("_", " ").title()}')
        ax.set_title(f'{target.replace("_", " ").title()} Model Performance')
        ax.legend(loc='lower right', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig5_1_actual_vs_predicted.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'fig5_1_actual_vs_predicted.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: fig5_1_actual_vs_predicted.png/pdf")
    
    # 5.3 Residual Analysis
    print("\n[5.4] Residual Analysis...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    target = 'energy'
    target_col = 'ENERGY_CONSUMPTION_CURRENT'
    model = model_factory.models['energy']
    
    test_valid = test_df[test_df[target_col].notna()]
    predictions = model.predict(test_valid[feature_cols])
    actuals = test_valid[target_col].values
    residuals = actuals - predictions
    
    # Residuals vs Predicted
    ax1 = axes[0, 0]
    ax1.scatter(predictions, residuals, alpha=0.3, s=5)
    ax1.axhline(y=0, color='r', linestyle='--')
    ax1.set_xlabel('Predicted Energy (kWh/m²)')
    ax1.set_ylabel('Residual')
    ax1.set_title('Residuals vs Predicted Values')
    
    # Residuals by City
    ax2 = axes[0, 1]
    city_residuals = []
    for city in cities:
        mask = test_valid['CITY'] == city
        city_residuals.append(residuals[mask])
    
    bp = ax2.boxplot(city_residuals, labels=cities, patch_artist=True)
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax2.axhline(y=0, color='r', linestyle='--')
    ax2.set_ylabel('Residual (kWh/m²)')
    ax2.set_title('Residuals by City')
    
    # Residuals by Building Age
    ax3 = axes[1, 0]
    age_groups = pd.cut(test_valid['AGE_BAND_NUM'], bins=[0, 3, 6, 9, 15], 
                       labels=['<1950', '1950-75', '1976-95', '1996+'])
    test_valid_copy = test_valid.copy()
    test_valid_copy['age_group'] = age_groups
    test_valid_copy['residual'] = residuals
    
    age_residuals = [test_valid_copy[test_valid_copy['age_group'] == g]['residual'].values 
                    for g in ['<1950', '1950-75', '1976-95', '1996+']]
    bp2 = ax3.boxplot([r for r in age_residuals if len(r) > 0], 
                     labels=['<1950', '1950-75', '1976-95', '1996+'], patch_artist=True)
    ax3.axhline(y=0, color='r', linestyle='--')
    ax3.set_xlabel('Construction Period')
    ax3.set_ylabel('Residual (kWh/m²)')
    ax3.set_title('Residuals by Building Age')
    
    # Residual Distribution
    ax4 = axes[1, 1]
    ax4.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    ax4.axvline(x=0, color='r', linestyle='--')
    ax4.axvline(x=residuals.mean(), color='blue', linestyle='-', label=f'Mean={residuals.mean():.1f}')
    ax4.set_xlabel('Residual (kWh/m²)')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Residual Distribution')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig5_2_residual_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'fig5_2_residual_analysis.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: fig5_2_residual_analysis.png/pdf")
    
    return model_factory, accuracy_df


# =============================================================================
# CHAPTER 6: INTERPRETABILITY
# =============================================================================

def chapter6_interpretability(model_factory, train_df, test_df, preprocessor, output_dir):
    """Generate interpretability analysis for Chapter 6."""
    print("\n" + "=" * 70)
    print("CHAPTER 6: Model Interpretability and Physics Analysis")
    print("=" * 70)
    
    feature_cols = preprocessor.get_feature_columns()
    
    # 6.1 Feature Importance Chart
    print("\n[6.1] Feature Importance Analysis...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    targets = ['energy', 'carbon', 'heating_cost', 'total_cost']
    
    for idx, (target, ax) in enumerate(zip(targets, axes.flatten())):
        model = model_factory.models.get(target)
        if model is None or not model._fitted:
            continue
        
        importance = model.get_feature_importance()
        top_features = importance.get_top_features(15)
        
        # Create readable names
        name_map = {
            'FORM_FACTOR': 'Form Factor',
            'TOTAL_FLOOR_AREA': 'Floor Area',
            'ENVELOPE_QUALITY': 'Envelope Quality',
            'SYSTEM_EFFICIENCY': 'System Efficiency',
            'HEAT_LOSS_PROXY': 'Heat Loss Index',
            'WALLS_ENERGY_EFF_NUM': 'Wall Efficiency',
            'WALLS_ENV_EFF_NUM': 'Wall Env. Rating',
            'ROOF_ENERGY_EFF_NUM': 'Roof Efficiency',
            'ROOF_ENV_EFF_NUM': 'Roof Env. Rating',
            'WINDOWS_ENERGY_EFF_NUM': 'Window Efficiency',
            'MAINHEAT_ENERGY_EFF_NUM': 'Heating Efficiency',
            'MAINHEAT_ENV_EFF_NUM': 'Heating Env. Rating',
            'HOT_WATER_ENERGY_EFF_NUM': 'Hot Water Eff.',
            'MAINHEATC_ENERGY_EFF_NUM': 'Heat Control Eff.',
            'AGE_BAND_NUM': 'Building Age',
            'HEATING_DEMAND_PROXY': 'Heating Demand',
            'HDD': 'Heating Degree Days',
            'LOW_ENERGY_LIGHTING': 'LED Lighting %'
        }
        
        top_features['feature_name'] = top_features['feature'].map(
            lambda x: name_map.get(x, x.replace('_', ' ').title()[:20])
        )
        
        colors = ['#e74c3c' if 'Wall' in n or 'Roof' in n or 'Envelope' in n 
                  else '#3498db' for n in top_features['feature_name']]
        
        ax.barh(top_features['feature_name'][::-1], 
                top_features['importance'][::-1], color=colors[::-1])
        ax.set_xlabel('Feature Importance')
        ax.set_title(f'{target.replace("_", " ").title()} Model')
        
        # Highlight envelope features
        ax.text(0.95, 0.05, '■ Envelope  ■ Other', transform=ax.transAxes,
                fontsize=9, ha='right', color='gray')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig6_1_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'fig6_1_feature_importance.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: fig6_1_feature_importance.png/pdf")
    
    # 6.2 SHAP Analysis
    print("\n[6.2] SHAP Analysis...")
    
    try:
        import shap
        
        # Get energy model
        model = model_factory.models['energy']
        
        # Sample data for SHAP (computational efficiency)
        sample_size = min(1000, len(test_df))
        sample_df = test_df.sample(sample_size, random_state=42)
        X_sample = sample_df[feature_cols].fillna(0)
        
        # Calculate SHAP values
        explainer = shap.TreeExplainer(model.model)
        shap_values = explainer.shap_values(X_sample)
        
        # SHAP Summary Plot
        fig, ax = plt.subplots(figsize=(12, 10))
        shap.summary_plot(shap_values, X_sample, show=False, max_display=20)
        plt.title('SHAP Feature Impact on Energy Prediction')
        plt.tight_layout()
        plt.savefig(output_dir / 'fig6_2_shap_summary.png', dpi=300, bbox_inches='tight')
        plt.savefig(output_dir / 'fig6_2_shap_summary.pdf', bbox_inches='tight')
        plt.close()
        print(f"  Saved: fig6_2_shap_summary.png/pdf")
        
        # SHAP by City Analysis
        print("\n[6.3] SHAP Analysis by City...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        cities = ['Cambridge', 'Boston', 'Liverpool', 'Sheffield']
        
        for idx, (city, ax) in enumerate(zip(cities, axes.flatten())):
            city_mask = sample_df['CITY'] == city
            city_shap = shap_values[city_mask]
            city_X = X_sample[city_mask]
            
            if len(city_X) > 10:
                plt.sca(ax)
                shap.summary_plot(city_shap, city_X, show=False, max_display=10, plot_size=None)
                ax.set_title(f'{city}: SHAP Feature Impact')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'fig6_3_shap_by_city.png', dpi=300, bbox_inches='tight')
        plt.savefig(output_dir / 'fig6_3_shap_by_city.pdf', bbox_inches='tight')
        plt.close()
        print(f"  Saved: fig6_3_shap_by_city.png/pdf")
        
    except Exception as e:
        print(f"  SHAP analysis failed: {e}")
    
    # 6.4 Sensitivity Analysis
    print("\n[6.4] Sensitivity Analysis...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Get a reference building
    ref_building = test_df.iloc[0].copy()
    
    # Sensitivity to Wall Efficiency
    ax1 = axes[0, 0]
    wall_values = [1, 2, 3, 4, 5]
    wall_labels = ['Very Poor', 'Poor', 'Average', 'Good', 'Very Good']
    
    results = []
    for val in wall_values:
        test_profile = ref_building.copy()
        test_profile['WALLS_ENERGY_EFF_NUM'] = val
        test_profile['ENVELOPE_QUALITY'] = (
            0.35 * val + 
            0.25 * test_profile.get('ROOF_ENERGY_EFF_NUM', 3) +
            0.15 * test_profile.get('FLOOR_ENERGY_EFF_NUM', 0) +
            0.25 * test_profile.get('WINDOWS_ENERGY_EFF_NUM', 3)
        )
        X = pd.DataFrame([test_profile])[feature_cols].fillna(0)
        pred = model_factory.models['energy'].predict(X)[0]
        results.append(pred)
    
    ax1.plot(wall_values, results, 'o-', linewidth=2, markersize=8, color='#e74c3c')
    ax1.set_xticks(wall_values)
    ax1.set_xticklabels(wall_labels, rotation=45)
    ax1.set_xlabel('Wall Insulation Rating')
    ax1.set_ylabel('Predicted Energy (kWh/m²)')
    ax1.set_title('Sensitivity to Wall Insulation')
    ax1.grid(True, alpha=0.3)
    
    # Sensitivity to Roof Efficiency
    ax2 = axes[0, 1]
    results_roof = []
    for val in wall_values:
        test_profile = ref_building.copy()
        test_profile['ROOF_ENERGY_EFF_NUM'] = val
        test_profile['ENVELOPE_QUALITY'] = (
            0.35 * test_profile.get('WALLS_ENERGY_EFF_NUM', 3) +
            0.25 * val +
            0.15 * test_profile.get('FLOOR_ENERGY_EFF_NUM', 0) +
            0.25 * test_profile.get('WINDOWS_ENERGY_EFF_NUM', 3)
        )
        X = pd.DataFrame([test_profile])[feature_cols].fillna(0)
        pred = model_factory.models['energy'].predict(X)[0]
        results_roof.append(pred)
    
    ax2.plot(wall_values, results_roof, 'o-', linewidth=2, markersize=8, color='#3498db')
    ax2.set_xticks(wall_values)
    ax2.set_xticklabels(wall_labels, rotation=45)
    ax2.set_xlabel('Roof Insulation Rating')
    ax2.set_ylabel('Predicted Energy (kWh/m²)')
    ax2.set_title('Sensitivity to Roof Insulation')
    ax2.grid(True, alpha=0.3)
    
    # Sensitivity to Heating System
    ax3 = axes[1, 0]
    results_heat = []
    for val in wall_values:
        test_profile = ref_building.copy()
        test_profile['MAINHEAT_ENERGY_EFF_NUM'] = val
        test_profile['SYSTEM_EFFICIENCY'] = (val + 
            test_profile.get('MAINHEATC_ENERGY_EFF_NUM', 3) +
            test_profile.get('HOT_WATER_ENERGY_EFF_NUM', 3)) / 3
        X = pd.DataFrame([test_profile])[feature_cols].fillna(0)
        pred = model_factory.models['energy'].predict(X)[0]
        results_heat.append(pred)
    
    ax3.plot(wall_values, results_heat, 'o-', linewidth=2, markersize=8, color='#9b59b6')
    ax3.set_xticks(wall_values)
    ax3.set_xticklabels(wall_labels, rotation=45)
    ax3.set_xlabel('Heating System Rating')
    ax3.set_ylabel('Predicted Energy (kWh/m²)')
    ax3.set_title('Sensitivity to Heating System')
    ax3.grid(True, alpha=0.3)
    
    # Combined Sensitivity
    ax4 = axes[1, 1]
    ax4.plot(wall_values, results, 'o-', linewidth=2, label='Wall', color='#e74c3c')
    ax4.plot(wall_values, results_roof, 's-', linewidth=2, label='Roof', color='#3498db')
    ax4.plot(wall_values, results_heat, '^-', linewidth=2, label='Heating', color='#9b59b6')
    ax4.set_xticks(wall_values)
    ax4.set_xticklabels(wall_labels, rotation=45)
    ax4.set_xlabel('Component Rating')
    ax4.set_ylabel('Predicted Energy (kWh/m²)')
    ax4.set_title('Combined Sensitivity Analysis')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig6_4_sensitivity_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'fig6_4_sensitivity_analysis.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: fig6_4_sensitivity_analysis.png/pdf")


# =============================================================================
# CHAPTER 7: OPTIMIZATION
# =============================================================================

def chapter7_optimization(model_factory, test_df, recs_df, preprocessor, output_dir):
    """Generate optimization results for Chapter 7."""
    print("\n" + "=" * 70)
    print("CHAPTER 7: Optimization and Retrofit Recommendations")
    print("=" * 70)
    
    from retrofit_dss.optimization.engine import OptimizationEngine
    
    feature_cols = preprocessor.get_feature_columns()
    
    # Initialize optimizer
    optimizer = OptimizationEngine(model_factory)
    optimizer.load_recommendations(recs_df)
    
    # 7.1 Case Studies - 4 Buildings (one from each city)
    print("\n[7.1] Retrofit Case Studies...")
    
    cities = ['Cambridge', 'Boston', 'Liverpool', 'Sheffield']
    case_studies = []
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    for idx, (city, ax) in enumerate(zip(cities, axes.flatten())):
        # Find a building with poor rating
        city_df = test_df[test_df['CITY'] == city]
        poor_buildings = city_df[
            (city_df['WALLS_ENERGY_EFF_NUM'] <= 2) | 
            (city_df['ROOF_ENERGY_EFF_NUM'] <= 2)
        ]
        
        if len(poor_buildings) > 0:
            building = poor_buildings.iloc[0].copy()
        else:
            building = city_df.iloc[0].copy()
        
        # Get current predictions
        X_current = pd.DataFrame([building])[feature_cols].fillna(0)
        current_energy = model_factory.models['energy'].predict(X_current)[0]
        current_carbon = model_factory.models['carbon'].predict(X_current)[0]
        current_cost = model_factory.models['total_cost'].predict(X_current)[0]
        
        # Get optimization recommendations
        packages = optimizer.optimize(
            building,
            target_type='carbon',
            target_reduction=40.0,
            max_measures=4
        )
        
        if packages:
            best_pkg = packages[0]
            
            # Estimate post-retrofit performance
            effects = optimizer.estimate_improvement_effect(building, best_pkg.measures)
            
            new_energy = current_energy * (1 - effects['energy_reduction_pct'] / 100)
            new_carbon = current_carbon * (1 - effects['carbon_reduction_pct'] / 100)
            
            # Create before/after bar chart
            categories = ['Energy\n(kWh/m²)', 'Carbon\n(kg/m²)', 'Cost\n(£/year)']
            before = [current_energy, current_carbon, current_cost]
            after = [new_energy, new_carbon, current_cost * (1 - effects['energy_reduction_pct'] / 100)]
            
            x = np.arange(len(categories))
            width = 0.35
            
            ax.bar(x - width/2, before, width, label='Current', color='#e74c3c', alpha=0.8)
            ax.bar(x + width/2, after, width, label='After Retrofit', color='#2ecc71', alpha=0.8)
            
            ax.set_ylabel('Value')
            ax.set_title(f'{city} Case Study')
            ax.set_xticks(x)
            ax.set_xticklabels(categories)
            ax.legend()
            
            # Add reduction percentages
            for i, (b, a) in enumerate(zip(before, after)):
                reduction = (b - a) / b * 100 if b > 0 else 0
                ax.annotate(f'-{reduction:.0f}%', xy=(i + width/2, a),
                           xytext=(0, 5), textcoords='offset points',
                           ha='center', fontsize=9, color='green')
            
            # Store case study data
            case_studies.append({
                'City': city,
                'Current Energy': current_energy,
                'Current Carbon': current_carbon,
                'Current Cost': current_cost,
                'New Energy': new_energy,
                'New Carbon': new_carbon,
                'Energy Reduction %': effects['energy_reduction_pct'],
                'Carbon Reduction %': effects['carbon_reduction_pct'],
                'Retrofit Measures': ', '.join(m.name for m in best_pkg.measures),
                'Retrofit Cost': best_pkg.total_cost_avg
            })
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig7_1_case_studies.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'fig7_1_case_studies.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: fig7_1_case_studies.png/pdf")
    
    # Save case studies table
    case_df = pd.DataFrame(case_studies)
    case_df.to_csv(output_dir / 'table7_1_case_studies.csv', index=False)
    print(f"  Saved: table7_1_case_studies.csv")
    
    # 7.2 Cost-Benefit Pareto Curve
    print("\n[7.2] Cost-Benefit Pareto Analysis...")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    carbon_targets = [10, 20, 30, 40, 50, 60]
    
    for city in cities:
        city_df = test_df[test_df['CITY'] == city]
        if len(city_df) == 0:
            continue
        
        sample_building = city_df.iloc[0]
        
        costs = []
        reductions = []
        
        for target in carbon_targets:
            packages = optimizer.optimize(
                sample_building,
                target_type='carbon',
                target_reduction=target,
                max_measures=4
            )
            
            if packages:
                best = packages[0]
                costs.append(best.total_cost_avg)
                reductions.append(best.predicted_carbon_reduction)
        
        if costs:
            ax.plot(reductions, costs, 'o-', linewidth=2, markersize=8, label=city)
    
    ax.set_xlabel('Carbon Reduction (%)')
    ax.set_ylabel('Retrofit Investment Cost (£)')
    ax.set_title('Cost-Benefit Pareto Curve: Investment vs Carbon Reduction')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add reference line for 20% reduction
    ax.axvline(x=20, color='gray', linestyle='--', alpha=0.5)
    ax.annotate('20% Target', xy=(20, ax.get_ylim()[1] * 0.9), 
                fontsize=10, color='gray')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig7_2_pareto_curve.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'fig7_2_pareto_curve.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: fig7_2_pareto_curve.png/pdf")
    
    # 7.3 Recommended Measures Analysis
    print("\n[7.3] Recommended Measures Analysis...")
    
    # Analyze recommendations frequency
    rec_counts = recs_df.groupby('IMPROVEMENT_SUMMARY_TEXT').size().sort_values(ascending=False).head(15)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(rec_counts)))
    ax.barh(rec_counts.index[::-1], rec_counts.values[::-1], color=colors[::-1])
    ax.set_xlabel('Frequency in EPC Recommendations')
    ax.set_title('Top 15 Most Recommended Retrofit Measures')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig7_3_recommended_measures.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'fig7_3_recommended_measures.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: fig7_3_recommended_measures.png/pdf")
    
    # 7.4 Cost Analysis by Measure Type
    print("\n[7.4] Cost Analysis by Measure Type...")
    
    # Extract cost data
    cost_data = []
    for _, row in recs_df.drop_duplicates('IMPROVEMENT_SUMMARY_TEXT').iterrows():
        cost_str = row.get('INDICATIVE_COST', '')
        if pd.notna(cost_str) and cost_str:
            from retrofit_dss.utils.helpers import parse_cost_range
            min_cost, max_cost = parse_cost_range(str(cost_str))
            if max_cost > 0:
                cost_data.append({
                    'Measure': str(row['IMPROVEMENT_SUMMARY_TEXT'])[:40],
                    'Min Cost': min_cost,
                    'Max Cost': max_cost,
                    'Avg Cost': (min_cost + max_cost) / 2
                })
    
    if cost_data:
        cost_df = pd.DataFrame(cost_data).sort_values('Avg Cost', ascending=False).head(15)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        y_pos = np.arange(len(cost_df))
        ax.barh(y_pos, cost_df['Avg Cost'], xerr=[cost_df['Avg Cost'] - cost_df['Min Cost'],
                                                   cost_df['Max Cost'] - cost_df['Avg Cost']],
                capsize=3, color='#3498db', alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(cost_df['Measure'])
        ax.set_xlabel('Indicative Cost (£)')
        ax.set_title('Retrofit Measure Costs (with ranges)')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'fig7_4_measure_costs.png', dpi=300, bbox_inches='tight')
        plt.savefig(output_dir / 'fig7_4_measure_costs.pdf', bbox_inches='tight')
        plt.close()
        print(f"  Saved: fig7_4_measure_costs.png/pdf")
    
    return case_df


def main():
    """Main analysis pipeline."""
    print("=" * 70)
    print("THESIS ANALYSIS - Building Retrofit Decision Support System")
    print("=" * 70)
    
    # Load data
    certs_df, recs_df, processed_df, train_df, test_df, preprocessor = load_and_prepare_data()

    required_columns = {
        'CITY',
        'ENERGY_CONSUMPTION_CURRENT',
        'CONSTRUCTION_AGE_BAND'
    }
    missing_columns = sorted(required_columns - set(processed_df.columns))
    if missing_columns:
        print("\nSkipping analysis: required columns missing from sample data.")
        print(f"Missing columns: {', '.join(missing_columns)}")
        return
    
    # Chapter 3: EDA
    summary_df = chapter3_eda(processed_df, OUTPUT_DIR)
    
    # Chapter 5: Model Results
    model_factory, accuracy_df = chapter5_model_results(train_df, test_df, preprocessor, OUTPUT_DIR)
    
    # Chapter 6: Interpretability
    chapter6_interpretability(model_factory, train_df, test_df, preprocessor, OUTPUT_DIR)
    
    # Chapter 7: Optimization
    case_df = chapter7_optimization(model_factory, test_df, recs_df, preprocessor, OUTPUT_DIR)
    
    # Generate summary report
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE - Summary")
    print("=" * 70)
    
    print(f"\nOutput directory: {OUTPUT_DIR.absolute()}")
    print("\nGenerated files:")
    for f in sorted(OUTPUT_DIR.glob('*')):
        print(f"  - {f.name}")
    
    print("\n" + "=" * 70)
    print("Thesis figures and tables have been generated successfully!")
    print("=" * 70)


if __name__ == '__main__':
    main()
