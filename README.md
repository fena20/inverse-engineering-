# Building Retrofit Decision Support System (DSS)

A physics-based surrogate modeling system for building energy performance prediction and retrofit optimization using UK EPC (Energy Performance Certificate) data from four major English cities.

## Overview

This system addresses the challenge of quickly estimating the effect of physical changes (envelope and systems) on energy consumption and carbon emissions without requiring heavy dynamic simulations. It provides:

- **Performance Prediction**: Estimate energy consumption, carbon emissions, and costs based on building characteristics
- **Inverse Design**: Given a target (e.g., 60% carbon reduction), find the optimal combination of retrofit measures
- **Sensitivity Analysis**: Analyze the impact of changing individual parameters on building performance

## Features

### Multi-City Data Integration (FR-1)
- Integrates EPC data from Cambridge, Sheffield, Liverpool, and Boston
- Accounts for climate variations through Heating Degree Days (HDD)
- ~650,000 building records with ~2.4 million improvement recommendations

### Physics-Interpretable Models (FR-2)
- Feature importance aligned with heat transfer principles
- Envelope quality features (walls, roof, windows) ranked high
- Physical consistency validation ensures model predictions respect thermodynamic laws

### Load Disaggregation (FR-3)
- Separate prediction models for:
  - Heating costs
  - Hot water costs
  - Lighting costs
  - Total annual costs

### Retrofit Recommendation Engine (FR-4)
- Uses EPC recommendations with indicative costs
- Discrete optimization for finding cost-effective measure combinations
- Payback period calculations

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd <repository>

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Train Models

```bash
python src/train.py --data-dir data --model-dir models --sample-size 100000
```

### 2. Run Example

```bash
python src/example_usage.py
```

### 3. Start API Server

```bash
cd src
python -m retrofit_dss.api.app
```

## API Endpoints

### POST /evaluate
Evaluate building performance based on characteristics.

**Request:**
```json
{
  "building_profile": {
    "TOTAL_FLOOR_AREA": 90.0,
    "WALLS_ENERGY_EFF": "Poor",
    "ROOF_ENERGY_EFF": "Average",
    "MAINHEAT_ENERGY_EFF": "Good",
    "CONSTRUCTION_AGE_BAND": "England and Wales: 1930-1949",
    "PROPERTY_TYPE": "House",
    "BUILT_FORM": "Semi-Detached",
    "CITY": "Liverpool"
  }
}
```

**Response:**
```json
{
  "energy_intensity_kwh_m2": 297.4,
  "carbon_intensity_kg_m2": 53.3,
  "heating_cost": 871,
  "hot_water_cost": 110,
  "lighting_cost": 101,
  "total_annual_cost": 1072,
  "epc_grade_estimate": "E"
}
```

### POST /optimize
Get optimal retrofit recommendations to achieve a target.

**Request:**
```json
{
  "building_profile": { ... },
  "target_type": "carbon",
  "target_reduction": 50.0,
  "max_budget": 25000,
  "max_measures": 4
}
```

**Response:**
```json
{
  "recommendations": [
    {
      "package_rank": 1,
      "measures": [...],
      "total_cost_range": "£7,450 - £13,800",
      "energy_reduction_pct": 39.1,
      "carbon_reduction_pct": 50.2,
      "payback_years": 18.5
    }
  ]
}
```

### POST /sensitivity
Perform sensitivity analysis on a parameter.

**Request:**
```json
{
  "building_profile": { ... },
  "feature": "WALLS_ENERGY_EFF_NUM",
  "values": [1, 2, 3, 4, 5]
}
```

## Model Performance

| Model | R² | MAE | Description |
|-------|-----|-----|-------------|
| Energy | 0.42 | 37.7 kWh/m² | Primary energy intensity |
| Carbon | 0.45 | 6.4 kg/m² | Carbon emissions intensity |
| Heating Cost | 0.75 | £152 | Annual heating costs |
| Hot Water Cost | 0.66 | £38 | Annual hot water costs |
| Total Cost | 0.74 | £183 | Total annual energy costs |

## Feature Importance (Energy Model)

The model correctly identifies physics-relevant features:

1. **FORM_FACTOR** (31.7%) - Building geometry affects heat loss surface area
2. **SYSTEM_EFFICIENCY** (10.5%) - Heating system performance
3. **TOTAL_FLOOR_AREA** (10.3%) - Larger buildings have higher absolute consumption
4. **WALLS_ENERGY_EFF** (4.8%) - Wall insulation quality
5. **ENVELOPE_QUALITY** (4.7%) - Composite envelope score

## Data Structure

```
data/
├── domestic-E07000008-Cambridge/
│   ├── certificates.csv      # EPC certificates
│   ├── recommendations.csv   # Improvement recommendations
│   └── open-meteo-*.csv      # Weather data
├── domestic-E07000136-Boston/
├── domestic-E08000012-Liverpool/
└── domestic-E08000019-Sheffield/
```

## Key Input Features

### Envelope Features (Ordinal 1-5: Very Poor to Very Good)
- `WALLS_ENERGY_EFF` - Wall insulation efficiency
- `ROOF_ENERGY_EFF` - Roof insulation efficiency
- `FLOOR_ENERGY_EFF` - Floor insulation efficiency
- `WINDOWS_ENERGY_EFF` - Window thermal performance

### System Features
- `MAINHEAT_ENERGY_EFF` - Main heating system efficiency
- `HOT_WATER_ENERGY_EFF` - Hot water system efficiency
- `LOW_ENERGY_LIGHTING` - Percentage of low energy lights (0-100)

### Building Characteristics
- `TOTAL_FLOOR_AREA` - Floor area in m²
- `CONSTRUCTION_AGE_BAND` - Building age period
- `PROPERTY_TYPE` - House, Flat, etc.
- `BUILT_FORM` - Detached, Semi-Detached, Terrace, etc.

## Physical Consistency

The system validates that model predictions respect physical laws:

- **Better insulation → Lower energy consumption**: Model correctly predicts that improving wall rating from "Very Poor" to "Very Good" reduces energy by ~23%
- **Feature importance alignment**: Envelope and system features rank high, consistent with heat transfer physics
- **No data leakage**: Train/test split by postcode sector ensures geographic independence

## Project Structure

```
src/
├── retrofit_dss/
│   ├── data/
│   │   ├── loader.py         # Multi-city data loading
│   │   └── preprocessor.py   # Feature engineering
│   ├── models/
│   │   └── surrogate.py      # ML surrogate models
│   ├── optimization/
│   │   └── engine.py         # Retrofit optimization
│   └── api/
│       └── app.py            # Flask REST API
├── train.py                  # Model training script
└── example_usage.py          # Usage examples
```

## Limitations & Risks

1. **Data Missingness**: Many EPC records lack detailed envelope specifications
2. **Cost Accuracy**: Indicative costs may not reflect current market prices
3. **Regional Coverage**: Model trained on 4 cities; may not generalize to all UK regions
4. **Annual Data Only**: System uses annual aggregates, not hourly load profiles

## References

- UK EPC Open Data: https://epc.opendatacommunitites.org/
- Open-Meteo Weather API: https://open-meteo.com/
- SAP Methodology: Standard Assessment Procedure for UK dwellings

## License

See LICENCE.txt in the data directories for EPC data usage terms.
