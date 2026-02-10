
import pandas as pd
import numpy as np
import os
import random

cities = {
    'Cambridge': 'E07000008',
    'Boston': 'E07000136',
    'Liverpool': 'E08000012',
    'Sheffield': 'E08000019'
}

for city, code in cities.items():
    path = f'mock_data/domestic-{code}-{city}'
    os.makedirs(path, exist_ok=True)

    # Generate 100 records per city
    n_records = 100

    # Certificates
    data = []
    for i in range(n_records):
        energy = np.random.normal(250, 50)
        floor_area = np.random.normal(90, 20)

        data.append({
            'LMK_KEY': f'{city}_{i}',
            'POSTCODE': f'XX{i%20} {i%9}YZ',
            'CITY': city,
            'TOTAL_FLOOR_AREA': floor_area,
            'NUMBER_HABITABLE_ROOMS': np.random.randint(3, 8),
            'WALLS_ENERGY_EFF': np.random.choice(['Poor', 'Average', 'Good', 'Very Good']),
            'ROOF_ENERGY_EFF': np.random.choice(['Very Poor', 'Poor', 'Average', 'Good']),
            'WINDOWS_ENERGY_EFF': np.random.choice(['Average', 'Good', 'Very Good']),
            'MAINHEAT_ENERGY_EFF': np.random.choice(['Good', 'Very Good']),
            'MAINHEATC_ENERGY_EFF': 'Good',
            'HOT_WATER_ENERGY_EFF': 'Average',
            'LIGHTING_ENERGY_EFF': 'Very Good',
            'FLOOR_ENERGY_EFF': 'Average',
            'CONSTRUCTION_AGE_BAND': np.random.choice(['England and Wales: 1900-1929', 'England and Wales: 1950-1966', 'England and Wales: 2007 onwards']),
            'PROPERTY_TYPE': np.random.choice(['House', 'Flat', 'Bungalow']),
            'BUILT_FORM': np.random.choice(['Detached', 'Semi-Detached', 'Mid-Terrace']),
            'ENERGY_CONSUMPTION_CURRENT': energy,
            'CO2_EMISS_CURR_PER_FLOOR_AREA': energy * 0.2,
            'HEATING_COST_CURRENT': energy * floor_area * 0.05,
            'HOT_WATER_COST_CURRENT': 150,
            'LIGHTING_COST_CURRENT': 80,
            'TOTAL_COST_CURRENT': (energy * floor_area * 0.05) + 230,
            'CURRENT_ENERGY_RATING': np.random.choice(['C', 'D', 'E']),
            'POTENTIAL_ENERGY_RATING': 'B',
            'GLAZED_TYPE': 'Double glazing',
            'MAINS_GAS_FLAG': 'Y',
            'SOLAR_WATER_HEATING_FLAG': 'N',
            'LOW_ENERGY_LIGHTING': np.random.randint(0, 100),
            'MULTI_GLAZE_PROPORTION': 100,
            'NUMBER_OPEN_FIREPLACES': 0,
            'PHOTO_SUPPLY': 0,
            'WIND_TURBINE_COUNT': 0
        })

    df = pd.DataFrame(data)
    df.to_csv(f'{path}/certificates.csv', index=False)

    # Recommendations
    recs = []
    for i in range(n_records):
        # Randomly assign recommendations
        if i % 2 == 0:
            recs.append({
                'LMK_KEY': f'{city}_{i}',
                'IMPROVEMENT_ID': '34',
                'IMPROVEMENT_SUMMARY_TEXT': 'Solar PV',
                'IMPROVEMENT_DESCR_TEXT': 'Install Solar PV',
                'INDICATIVE_COST': '£3,500 - £5,500'
            })
        if i % 3 == 0:
             recs.append({
                'LMK_KEY': f'{city}_{i}',
                'IMPROVEMENT_ID': '6',
                'IMPROVEMENT_SUMMARY_TEXT': 'Wall Insulation',
                'IMPROVEMENT_DESCR_TEXT': 'Cavity Wall Insulation',
                'INDICATIVE_COST': '£500 - £1,500'
            })

    pd.DataFrame(recs).to_csv(f'{path}/recommendations.csv', index=False)

    # Weather
    with open(f'{path}/open-meteo-test.csv', 'w') as f:
        f.write("Header1\nHeader2\n")
        f.write("time,temperature_2m_mean,shortwave_radiation_sum\n")
        for d in range(365):
            f.write(f"2023-01-{d%30+1:02d},10.0,3.0\n")
