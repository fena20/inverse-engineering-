#!/usr/bin/env python3
"""
Fast Thesis Analysis Script for Retrofit DSS

This is an optimized version that uses sampling for faster execution.
Generates all figures and tables required for the thesis.
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

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 11

# Create output directory
OUTPUT_DIR = Path('outputs/thesis_figures')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Sample size for faster processing
SAMPLE_SIZE = 50000


def load_data_fast():
    """Load data with sampling for speed."""
    from retrofit_dss.data.loader import DataLoader
    from retrofit_dss.data.preprocessor import DataPreprocessor, create_train_test_split
    
    print("Loading data...")
    loader = DataLoader('data')
    loader.discover_cities()
    certs_df, recs_df = loader.get_merged_data()
    
    print("Preprocessing...")
    train_raw, test_raw = create_train_test_split(certs_df, test_size=0.2, random_state=42)
    
    preprocessor = DataPreprocessor()
    train_df = preprocessor.fit_transform(train_raw)
    test_df = preprocessor.transform(test_raw)
    
    # Sample for speed
    if len(train_df) > SAMPLE_SIZE:
        train_df = train_df.sample(SAMPLE_SIZE, random_state=42)
    if len(test_df) > SAMPLE_SIZE:
        test_df = test_df.sample(SAMPLE_SIZE, random_state=42)
    
    processed_df = pd.concat([train_df, test_df], ignore_index=True)
    
    return certs_df, recs_df, processed_df, train_df, test_df, preprocessor


def fig3_1_energy_distribution(df, output_dir):
    """City-wise energy distribution."""
    print("\n[Fig 3.1] City Energy Distribution...")
    
    energy_data = df[df['ENERGY_CONSUMPTION_CURRENT'].notna() & 
                     (df['ENERGY_CONSUMPTION_CURRENT'] > 0) &
                     (df['ENERGY_CONSUMPTION_CURRENT'] < 600)]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    cities = ['Cambridge', 'Boston', 'Liverpool', 'Sheffield']
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']
    
    # Box plot
    ax = axes[0]
    box_data = [energy_data[energy_data['CITY'] == city]['ENERGY_CONSUMPTION_CURRENT'].dropna() 
                for city in cities]
    bp = ax.boxplot(box_data, labels=cities, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel('Primary Energy Intensity (kWh/m²/year)')
    ax.set_title('Energy Distribution by City')
    
    # Add means
    for i, data in enumerate(box_data):
        mean_val = data.mean()
        ax.scatter(i+1, mean_val, color='red', s=50, zorder=3, marker='D', label='Mean' if i==0 else '')
    ax.legend()
    
    # Histogram
    ax2 = axes[1]
    for city, color in zip(cities, colors):
        city_data = energy_data[energy_data['CITY'] == city]['ENERGY_CONSUMPTION_CURRENT']
        ax2.hist(city_data, bins=30, alpha=0.5, label=city, color=color)
    ax2.set_xlabel('Primary Energy Intensity (kWh/m²/year)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Energy Distribution Histogram')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig3_1_city_energy_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: fig3_1_city_energy_distribution.png")


def fig3_2_age_efficiency_heatmap(df, output_dir):
    """Building age vs efficiency heatmap."""
    print("\n[Fig 3.2] Age-Efficiency Heatmap...")
    
    age_mapping = {
        'England and Wales: before 1900': '<1900',
        'England and Wales: 1900-1929': '1900-29',
        'England and Wales: 1930-1949': '1930-49',
        'England and Wales: 1950-1966': '1950-66',
        'England and Wales: 1967-1975': '1967-75',
        'England and Wales: 1976-1982': '1976-82',
        'England and Wales: 1983-1990': '1983-90',
        'England and Wales: 1991-1995': '1991-95',
        'England and Wales: 1996-2002': '1996-02',
        'England and Wales: 2003-2006': '2003-06',
        'England and Wales: 2007 onwards': '2007+',
    }
    
    eff_mapping = {'Very Poor': 1, 'Poor': 2, 'Average': 3, 'Good': 4, 'Very Good': 5}
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    cities = ['Cambridge', 'Boston', 'Liverpool', 'Sheffield']
    
    for city, ax in zip(cities, axes.flatten()):
        city_df = df[df['CITY'] == city].copy()
        city_df['Age'] = city_df['CONSTRUCTION_AGE_BAND'].map(age_mapping)
        city_df['Wall_Eff'] = city_df['WALLS_ENERGY_EFF'].map(eff_mapping)
        city_df['Roof_Eff'] = city_df['ROOF_ENERGY_EFF'].map(eff_mapping)
        
        pivot = city_df.groupby('Age').agg({'Wall_Eff': 'mean', 'Roof_Eff': 'mean'}).T
        
        # Reorder columns
        ordered_cols = ['<1900', '1900-29', '1930-49', '1950-66', '1967-75', 
                       '1976-82', '1983-90', '1991-95', '1996-02', '2003-06', '2007+']
        pivot = pivot[[c for c in ordered_cols if c in pivot.columns]]
        
        if len(pivot.columns) > 0:
            sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn',
                       vmin=1, vmax=5, ax=ax, cbar_kws={'label': 'Rating'})
        ax.set_title(f'{city}: Envelope Efficiency by Age')
        ax.set_xlabel('Construction Period')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig3_2_age_efficiency_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: fig3_2_age_efficiency_heatmap.png")


def fig3_3_correlation_matrix(df, output_dir):
    """Correlation matrix."""
    print("\n[Fig 3.3] Correlation Matrix...")
    
    cols = ['TOTAL_FLOOR_AREA', 'WALLS_ENERGY_EFF_NUM', 'ROOF_ENERGY_EFF_NUM',
            'WINDOWS_ENERGY_EFF_NUM', 'MAINHEAT_ENERGY_EFF_NUM',
            'ENERGY_CONSUMPTION_CURRENT', 'CO2_EMISS_CURR_PER_FLOOR_AREA',
            'HEATING_COST_CURRENT']
    
    available = [c for c in cols if c in df.columns]
    data = df[available].dropna()
    if data.empty or len(available) < 2:
        print("  Skipping Fig 3.3 (insufficient data for correlation).")
        return
    
    names = {
        'TOTAL_FLOOR_AREA': 'Floor Area',
        'WALLS_ENERGY_EFF_NUM': 'Wall Eff.',
        'ROOF_ENERGY_EFF_NUM': 'Roof Eff.',
        'WINDOWS_ENERGY_EFF_NUM': 'Window Eff.',
        'MAINHEAT_ENERGY_EFF_NUM': 'Heating Eff.',
        'ENERGY_CONSUMPTION_CURRENT': 'Energy',
        'CO2_EMISS_CURR_PER_FLOOR_AREA': 'Carbon',
        'HEATING_COST_CURRENT': 'Heat Cost'
    }
    data = data.rename(columns=names)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = data.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, ax=ax, square=True)
    ax.set_title('Correlation: Building Features vs Energy Performance')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig3_3_correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: fig3_3_correlation_matrix.png")


def fig5_1_actual_vs_predicted(model_factory, test_df, feature_cols, output_dir):
    """Actual vs Predicted scatter plots."""
    print("\n[Fig 5.1] Actual vs Predicted...")
    
    from sklearn.metrics import r2_score, mean_absolute_error
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    targets = [
        ('energy', 'ENERGY_CONSUMPTION_CURRENT'),
        ('carbon', 'CO2_EMISS_CURR_PER_FLOOR_AREA'),
        ('heating_cost', 'HEATING_COST_CURRENT'),
        ('total_cost', 'TOTAL_COST_CURRENT')
    ]
    
    cities = ['Cambridge', 'Boston', 'Liverpool', 'Sheffield']
    colors = {'Cambridge': '#2ecc71', 'Boston': '#3498db', 
              'Liverpool': '#e74c3c', 'Sheffield': '#9b59b6'}
    
    for (name, col), ax in zip(targets, axes.flatten()):
        model = model_factory.models.get(name)
        if model is None or not model._fitted:
            continue
        
        valid = test_df[test_df[col].notna()]
        if len(valid) == 0:
            continue
            
        pred = model.predict(valid[feature_cols])
        actual = valid[col].values
        
        for city in cities:
            mask = valid['CITY'] == city
            if mask.sum() > 0:
                ax.scatter(actual[mask], pred[mask], alpha=0.3, s=10, 
                          c=colors[city], label=city)
        
        # Perfect line
        max_val = max(actual.max(), pred.max())
        min_val = min(actual.min(), pred.min())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2)
        
        r2 = r2_score(actual, pred)
        mae = mean_absolute_error(actual, pred)
        ax.text(0.05, 0.95, f'R² = {r2:.3f}\nMAE = {mae:.1f}',
                transform=ax.transAxes, fontsize=10, va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xlabel(f'Actual {name.replace("_", " ").title()}')
        ax.set_ylabel(f'Predicted {name.replace("_", " ").title()}')
        ax.set_title(f'{name.replace("_", " ").title()} Model')
        ax.legend(loc='lower right', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig5_1_actual_vs_predicted.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: fig5_1_actual_vs_predicted.png")


def fig5_2_residual_analysis(model_factory, test_df, feature_cols, output_dir):
    """Residual analysis."""
    print("\n[Fig 5.2] Residual Analysis...")
    
    model = model_factory.models['energy']
    valid = test_df[test_df['ENERGY_CONSUMPTION_CURRENT'].notna()]
    pred = model.predict(valid[feature_cols])
    actual = valid['ENERGY_CONSUMPTION_CURRENT'].values
    residuals = actual - pred
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Residuals vs Predicted
    ax = axes[0, 0]
    ax.scatter(pred, residuals, alpha=0.3, s=5)
    ax.axhline(y=0, color='r', linestyle='--')
    ax.set_xlabel('Predicted Energy')
    ax.set_ylabel('Residual')
    ax.set_title('Residuals vs Predicted')
    
    # By City
    ax = axes[0, 1]
    cities = ['Cambridge', 'Boston', 'Liverpool', 'Sheffield']
    city_res = [residuals[valid['CITY'] == c] for c in cities]
    bp = ax.boxplot(city_res, labels=cities, patch_artist=True)
    ax.axhline(y=0, color='r', linestyle='--')
    ax.set_ylabel('Residual')
    ax.set_title('Residuals by City')
    
    # By Age
    ax = axes[1, 0]
    age_bins = pd.cut(valid['AGE_BAND_NUM'], bins=[0, 3, 6, 9, 15],
                     labels=['<1950', '1950-75', '1976-95', '1996+'])
    valid_copy = valid.copy()
    valid_copy['age_group'] = age_bins
    valid_copy['residual'] = residuals
    
    age_res = [valid_copy[valid_copy['age_group'] == g]['residual'].values
               for g in ['<1950', '1950-75', '1976-95', '1996+']]
    ax.boxplot([r for r in age_res if len(r) > 0],
              labels=['<1950', '1950-75', '1976-95', '1996+'])
    ax.axhline(y=0, color='r', linestyle='--')
    ax.set_xlabel('Building Age')
    ax.set_ylabel('Residual')
    ax.set_title('Residuals by Age')
    
    # Distribution
    ax = axes[1, 1]
    ax.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(x=0, color='r', linestyle='--')
    ax.axvline(x=residuals.mean(), color='blue', label=f'Mean={residuals.mean():.1f}')
    ax.set_xlabel('Residual')
    ax.set_ylabel('Frequency')
    ax.set_title('Residual Distribution')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig5_2_residual_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: fig5_2_residual_analysis.png")


def fig6_1_feature_importance(model_factory, output_dir):
    """Feature importance."""
    print("\n[Fig 6.1] Feature Importance...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    targets = ['energy', 'carbon', 'heating_cost', 'total_cost']
    
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
        'HOT_WATER_ENERGY_EFF_NUM': 'Hot Water Eff.',
        'AGE_BAND_NUM': 'Building Age',
    }
    
    for target, ax in zip(targets, axes.flatten()):
        model = model_factory.models.get(target)
        if model is None or not model._fitted:
            continue
        
        imp = model.get_feature_importance()
        top = imp.get_top_features(12)
        top['name'] = top['feature'].map(lambda x: name_map.get(x, x[:15]))
        
        colors = ['#e74c3c' if 'Wall' in n or 'Roof' in n or 'Envelope' in n 
                  else '#3498db' for n in top['name']]
        
        ax.barh(top['name'][::-1], top['importance'][::-1], color=colors[::-1])
        ax.set_xlabel('Importance')
        ax.set_title(f'{target.replace("_", " ").title()} Model')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig6_1_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: fig6_1_feature_importance.png")


def fig6_2_sensitivity(model_factory, test_df, feature_cols, output_dir):
    """Sensitivity analysis."""
    print("\n[Fig 6.2] Sensitivity Analysis...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ref = test_df.iloc[0].copy()
    labels = ['Very Poor', 'Poor', 'Average', 'Good', 'Very Good']
    
    # Wall
    ax = axes[0, 0]
    results = []
    for val in [1, 2, 3, 4, 5]:
        p = ref.copy()
        p['WALLS_ENERGY_EFF_NUM'] = val
        p['ENVELOPE_QUALITY'] = 0.35*val + 0.25*p.get('ROOF_ENERGY_EFF_NUM',3) + 0.15*0 + 0.25*p.get('WINDOWS_ENERGY_EFF_NUM',3)
        X = pd.DataFrame([p])[feature_cols].fillna(0)
        results.append(model_factory.models['energy'].predict(X)[0])
    ax.plot([1,2,3,4,5], results, 'o-', lw=2, ms=8, c='#e74c3c')
    ax.set_xticks([1,2,3,4,5])
    ax.set_xticklabels(labels, rotation=45)
    ax.set_ylabel('Energy (kWh/m²)')
    ax.set_title('Wall Insulation Sensitivity')
    ax.grid(True, alpha=0.3)
    
    # Roof
    ax = axes[0, 1]
    results = []
    for val in [1, 2, 3, 4, 5]:
        p = ref.copy()
        p['ROOF_ENERGY_EFF_NUM'] = val
        p['ENVELOPE_QUALITY'] = 0.35*p.get('WALLS_ENERGY_EFF_NUM',3) + 0.25*val + 0.15*0 + 0.25*p.get('WINDOWS_ENERGY_EFF_NUM',3)
        X = pd.DataFrame([p])[feature_cols].fillna(0)
        results.append(model_factory.models['energy'].predict(X)[0])
    ax.plot([1,2,3,4,5], results, 'o-', lw=2, ms=8, c='#3498db')
    ax.set_xticks([1,2,3,4,5])
    ax.set_xticklabels(labels, rotation=45)
    ax.set_ylabel('Energy (kWh/m²)')
    ax.set_title('Roof Insulation Sensitivity')
    ax.grid(True, alpha=0.3)
    
    # Heating
    ax = axes[1, 0]
    results = []
    for val in [1, 2, 3, 4, 5]:
        p = ref.copy()
        p['MAINHEAT_ENERGY_EFF_NUM'] = val
        p['SYSTEM_EFFICIENCY'] = (val + p.get('MAINHEATC_ENERGY_EFF_NUM',3) + p.get('HOT_WATER_ENERGY_EFF_NUM',3))/3
        X = pd.DataFrame([p])[feature_cols].fillna(0)
        results.append(model_factory.models['energy'].predict(X)[0])
    ax.plot([1,2,3,4,5], results, 'o-', lw=2, ms=8, c='#9b59b6')
    ax.set_xticks([1,2,3,4,5])
    ax.set_xticklabels(labels, rotation=45)
    ax.set_ylabel('Energy (kWh/m²)')
    ax.set_title('Heating System Sensitivity')
    ax.grid(True, alpha=0.3)
    
    # Combined
    ax = axes[1, 1]
    for comp, color, label in [('WALLS_ENERGY_EFF_NUM', '#e74c3c', 'Wall'),
                                ('ROOF_ENERGY_EFF_NUM', '#3498db', 'Roof'),
                                ('MAINHEAT_ENERGY_EFF_NUM', '#9b59b6', 'Heating')]:
        results = []
        for val in [1, 2, 3, 4, 5]:
            p = ref.copy()
            p[comp] = val
            X = pd.DataFrame([p])[feature_cols].fillna(0)
            results.append(model_factory.models['energy'].predict(X)[0])
        ax.plot([1,2,3,4,5], results, 'o-', lw=2, ms=6, c=color, label=label)
    ax.set_xticks([1,2,3,4,5])
    ax.set_xticklabels(labels, rotation=45)
    ax.set_ylabel('Energy (kWh/m²)')
    ax.set_title('Combined Sensitivity')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig6_2_sensitivity_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: fig6_2_sensitivity_analysis.png")


def fig7_case_studies(model_factory, test_df, recs_df, feature_cols, output_dir):
    """Case studies and Pareto."""
    print("\n[Fig 7.1-7.4] Optimization Outputs...")
    
    from retrofit_dss.optimization.engine import OptimizationEngine
    
    optimizer = OptimizationEngine(model_factory)
    optimizer.load_recommendations(recs_df)
    
    cities = ['Cambridge', 'Boston', 'Liverpool', 'Sheffield']
    
    # Case Studies
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    case_data = []
    
    for city, ax in zip(cities, axes.flatten()):
        city_df = test_df[test_df['CITY'] == city]
        if len(city_df) == 0:
            continue
        
        building = city_df.iloc[0].copy()
        
        X = pd.DataFrame([building])[feature_cols].fillna(0)
        curr_energy = model_factory.models['energy'].predict(X)[0]
        curr_carbon = model_factory.models['carbon'].predict(X)[0]
        curr_cost = model_factory.models['total_cost'].predict(X)[0]
        
        packages = optimizer.optimize(building, 'carbon', 40.0, max_measures=3)
        
        if packages:
            best = packages[0]
            eff = optimizer.estimate_improvement_effect(building, best.measures)
            
            new_energy = curr_energy * (1 - eff['energy_reduction_pct']/100)
            new_carbon = curr_carbon * (1 - eff['carbon_reduction_pct']/100)
            new_cost = curr_cost * (1 - eff['energy_reduction_pct']/100)
            
            cats = ['Energy\n(kWh/m²)', 'Carbon\n(kg/m²)', 'Cost\n(£/yr)']
            before = [curr_energy, curr_carbon, curr_cost]
            after = [new_energy, new_carbon, new_cost]
            
            x = np.arange(3)
            ax.bar(x - 0.2, before, 0.4, label='Current', color='#e74c3c', alpha=0.8)
            ax.bar(x + 0.2, after, 0.4, label='Retrofit', color='#2ecc71', alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(cats)
            ax.set_title(f'{city} Case Study')
            ax.legend()
            
            for i, (b, a) in enumerate(zip(before, after)):
                red = (b-a)/b*100 if b > 0 else 0
                ax.annotate(f'-{red:.0f}%', (i+0.2, a), textcoords='offset points',
                           xytext=(0,5), ha='center', fontsize=9, color='green')
            
            case_data.append({
                'City': city,
                'Current Energy': curr_energy,
                'New Energy': new_energy,
                'Current Carbon': curr_carbon,
                'New Carbon': new_carbon,
                'Current Cost': curr_cost,
                'New Cost': new_cost,
                'Measures': ', '.join(m.name for m in best.measures),
                'Retrofit Cost': best.total_cost_avg
            })
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig7_1_case_studies.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: fig7_1_case_studies.png")
    
    pd.DataFrame(case_data).to_csv(output_dir / 'table7_1_case_studies.csv', index=False)
    print(f"  Saved: table7_1_case_studies.csv")
    
    # Pareto Curve
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for city in cities:
        city_df = test_df[test_df['CITY'] == city]
        if len(city_df) == 0:
            continue
        
        building = city_df.iloc[0]
        costs, reds = [], []
        
        for target in [10, 20, 30, 40, 50]:
            pkgs = optimizer.optimize(building, 'carbon', target, max_measures=4)
            if pkgs:
                costs.append(pkgs[0].total_cost_avg)
                reds.append(pkgs[0].predicted_carbon_reduction)
        
        if costs:
            ax.plot(reds, costs, 'o-', lw=2, ms=8, label=city)
    
    ax.set_xlabel('Carbon Reduction (%)')
    ax.set_ylabel('Investment Cost (£)')
    ax.set_title('Cost-Benefit Pareto Curve')
    ax.axvline(x=20, color='gray', ls='--', alpha=0.5)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig7_2_pareto_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: fig7_2_pareto_curve.png")
    
    # Top Recommendations
    rec_counts = recs_df.groupby('IMPROVEMENT_SUMMARY_TEXT').size().sort_values(ascending=False).head(15)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(rec_counts.index[::-1], rec_counts.values[::-1], color=plt.cm.viridis(np.linspace(0,1,15))[::-1])
    ax.set_xlabel('Frequency')
    ax.set_title('Top 15 Recommended Retrofit Measures')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig7_3_recommended_measures.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: fig7_3_recommended_measures.png")


def generate_tables(df, model_factory, test_df, feature_cols, output_dir):
    """Generate summary tables."""
    print("\n[Tables] Generating summary tables...")
    
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    
    # City summary
    cities = ['Cambridge', 'Boston', 'Liverpool', 'Sheffield']
    summary = []
    for city in cities:
        c = df[df['CITY'] == city]
        summary.append({
            'City': city,
            'Records': len(c),
            'Mean Energy (kWh/m²)': c['ENERGY_CONSUMPTION_CURRENT'].mean(),
            'Std Energy': c['ENERGY_CONSUMPTION_CURRENT'].std(),
            'Mean Carbon (kg/m²)': c['CO2_EMISS_CURR_PER_FLOOR_AREA'].mean(),
            'Mean Floor Area (m²)': c['TOTAL_FLOOR_AREA'].mean(),
            'HDD': c['HDD'].iloc[0] if 'HDD' in c.columns else 'N/A'
        })
    pd.DataFrame(summary).to_csv(output_dir / 'table3_1_city_summary.csv', index=False)
    print(f"  Saved: table3_1_city_summary.csv")
    
    # Model accuracy
    accuracy = []
    for target, col in [('Energy', 'ENERGY_CONSUMPTION_CURRENT'),
                        ('Carbon', 'CO2_EMISS_CURR_PER_FLOOR_AREA'),
                        ('Heating Cost', 'HEATING_COST_CURRENT'),
                        ('Total Cost', 'TOTAL_COST_CURRENT')]:
        model = model_factory.models.get(target.lower().replace(' ', '_'))
        if model is None or not model._fitted:
            continue
        
        valid = test_df[test_df[col].notna()]
        pred = model.predict(valid[feature_cols])
        actual = valid[col].values
        
        accuracy.append({
            'Target': target,
            'City': 'All',
            'R²': r2_score(actual, pred),
            'MAE': mean_absolute_error(actual, pred),
            'RMSE': np.sqrt(mean_squared_error(actual, pred)),
            'N': len(actual)
        })
        
        for city in cities:
            mask = valid['CITY'] == city
            if mask.sum() < 50:
                continue
            accuracy.append({
                'Target': target,
                'City': city,
                'R²': r2_score(actual[mask], pred[mask]),
                'MAE': mean_absolute_error(actual[mask], pred[mask]),
                'RMSE': np.sqrt(mean_squared_error(actual[mask], pred[mask])),
                'N': mask.sum()
            })
    
    pd.DataFrame(accuracy).to_csv(output_dir / 'table5_1_model_accuracy.csv', index=False)
    print(f"  Saved: table5_1_model_accuracy.csv")


def main():
    """Main analysis."""
    print("=" * 70)
    print("THESIS ANALYSIS - Fast Version")
    print("=" * 70)
    
    # Load data
    certs_df, recs_df, processed_df, train_df, test_df, preprocessor = load_data_fast()
    
    # Generate figures
    print("\nGenerating figures...")
    
    if 'ENERGY_CONSUMPTION_CURRENT' in processed_df.columns:
        fig3_1_energy_distribution(processed_df, OUTPUT_DIR)
    else:
        print("Warning: ENERGY_CONSUMPTION_CURRENT missing; skipping Fig 3.1.")
    
    required_age_cols = {'CONSTRUCTION_AGE_BAND', 'WALLS_ENERGY_EFF', 'ROOF_ENERGY_EFF'}
    if required_age_cols.issubset(processed_df.columns):
        fig3_2_age_efficiency_heatmap(processed_df, OUTPUT_DIR)
    else:
        print("Warning: Age/efficiency columns missing; skipping Fig 3.2.")
    
    fig3_3_correlation_matrix(processed_df, OUTPUT_DIR)
    
    required_targets = {
        'ENERGY_CONSUMPTION_CURRENT',
        'CO2_EMISS_CURR_PER_FLOOR_AREA',
        'HEATING_COST_CURRENT',
        'TOTAL_COST_CURRENT'
    }
    has_targets = required_targets.issubset(train_df.columns) and required_targets.issubset(test_df.columns)
    model_factory = None
    feature_cols = preprocessor.get_feature_columns()
    
    if has_targets:
        print("\nTraining models...")
        from retrofit_dss.models.surrogate import SurrogateModelFactory
        
        model_factory = SurrogateModelFactory('gradient_boosting')
        model_factory.create_all_models()
        model_factory.fit_all(train_df, feature_cols)
        
        fig5_1_actual_vs_predicted(model_factory, test_df, feature_cols, OUTPUT_DIR)
        fig5_2_residual_analysis(model_factory, test_df, feature_cols, OUTPUT_DIR)
        fig6_1_feature_importance(model_factory, OUTPUT_DIR)
        fig6_2_sensitivity(model_factory, test_df, feature_cols, OUTPUT_DIR)
        fig7_case_studies(model_factory, test_df, recs_df, feature_cols, OUTPUT_DIR)
        generate_tables(processed_df, model_factory, test_df, feature_cols, OUTPUT_DIR)
    else:
        print("Warning: Required target columns missing; skipping model-based figures.")
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nOutput: {OUTPUT_DIR.absolute()}")
    print("\nFiles generated:")
    for f in sorted(OUTPUT_DIR.glob('*')):
        print(f"  - {f.name}")


if __name__ == '__main__':
    main()
