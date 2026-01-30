"""
Constants and configuration for the Retrofit DSS.
"""

# Efficiency ratings ordinal encoding (physics-guided: higher = better efficiency)
EFFICIENCY_RATINGS = {
    'Very Poor': 1,
    'Poor': 2,
    'Average': 3,
    'Good': 4,
    'Very Good': 5,
    'N/A': 0,  # Not applicable (e.g., no roof for mid-floor flat)
    'NO DATA!': 0,
    '': 0,
    None: 0
}

# EPC grades mapping (A=best, G=worst)
EPC_GRADES = {
    'A': 7,
    'B': 6,
    'C': 5,
    'D': 4,
    'E': 3,
    'F': 2,
    'G': 1
}

# Climate data for each city (Heating Degree Days base 15.5°C, annual avg)
# Derived from Open-Meteo weather data
CITY_CLIMATE_DATA = {
    'Cambridge': {
        'code': 'E07000008',
        'latitude': 52.25,
        'longitude': 0.0,
        'hdd_base_15_5': 2100,  # Approximate HDD
        'avg_temp': 10.5,
        'solar_radiation': 950  # kWh/m2/year
    },
    'Boston': {
        'code': 'E07000136',
        'latitude': 53.0,
        'longitude': 0.0,
        'hdd_base_15_5': 2250,
        'avg_temp': 10.0,
        'solar_radiation': 920
    },
    'Liverpool': {
        'code': 'E08000012',
        'latitude': 53.5,
        'longitude': -3.0,
        'hdd_base_15_5': 2150,
        'avg_temp': 10.3,
        'solar_radiation': 900
    },
    'Sheffield': {
        'code': 'E08000019',
        'latitude': 53.5,
        'longitude': -1.5,
        'hdd_base_15_5': 2300,
        'avg_temp': 9.8,
        'solar_radiation': 880
    }
}

# Local authority code to city name mapping
LA_CODE_TO_CITY = {
    'E07000008': 'Cambridge',
    'E07000136': 'Boston',
    'E08000012': 'Liverpool',
    'E08000019': 'Sheffield'
}

# Building age bands ordinal encoding (newer = better thermal standards typically)
AGE_BAND_ENCODING = {
    'England and Wales: before 1900': 1,
    'before 1900': 1,
    'England and Wales: 1900-1929': 2,
    '1900-1929': 2,
    'England and Wales: 1930-1949': 3,
    '1930-1949': 3,
    'England and Wales: 1950-1966': 4,
    '1950-1966': 4,
    'England and Wales: 1967-1975': 5,
    '1967-1975': 5,
    'England and Wales: 1976-1982': 6,
    '1976-1982': 6,
    'England and Wales: 1983-1990': 7,
    '1983-1990': 7,
    'England and Wales: 1991-1995': 8,
    '1991-1995': 8,
    'England and Wales: 1996-2002': 9,
    '1996-2002': 9,
    'England and Wales: 2003-2006': 10,
    '2003-2006': 10,
    'England and Wales: 2007 onwards': 11,
    'England and Wales: 2007-2011': 11,
    '2007 onwards': 11,
    '2007-2011': 11,
    'England and Wales: 2012 onwards': 12,
    '2012 onwards': 12,
    '2019': 13,
    '2020': 13,
    '2021': 13,
    '2022': 13,
    '2023': 13,
    'NO DATA!': 0,
    '': 0,
    None: 0
}

# Property type encoding
PROPERTY_TYPE_ENCODING = {
    'Detached': 1,  # Most exposed
    'Semi-Detached': 2,
    'End-Terrace': 3,
    'Mid-Terrace': 4,  # Least exposed
    'Enclosed End-Terrace': 3,
    'Enclosed Mid-Terrace': 4,
    'House': 2,  # Generic
    'Bungalow': 2,
    'Flat': 5,  # Often less surface area per floor
    'Maisonette': 5,
    'Park home': 1,  # Very exposed
    '': 0,
    None: 0
}

# Built form encoding (related to heat loss surface area)
BUILT_FORM_ENCODING = {
    'Detached': 1,
    'Semi-Detached': 2,
    'End-Terrace': 3,
    'Mid-Terrace': 4,
    'Enclosed End-Terrace': 3,
    'Enclosed Mid-Terrace': 4,
    '': 0,
    None: 0
}

# Glazed type encoding (thermal performance)
GLAZED_TYPE_ENCODING = {
    'single': 1,
    'Single': 1,
    'single glazing': 1,
    'double': 2,
    'Double': 2,
    'double glazing installed before 2002': 2,
    'double glazing installed during or after 2002': 3,
    'double glazing, unknown install date': 2,
    'triple': 4,
    'Triple': 4,
    'triple glazing': 4,
    'secondary glazing': 2,
    'not defined': 0,
    'NO DATA!': 0,
    '': 0,
    None: 0
}

# Main fuel type encoding (carbon intensity ranking)
FUEL_TYPE_CARBON = {
    'mains gas': 0.184,  # kgCO2/kWh
    'mains gas (not community)': 0.184,
    'Gas: mains gas': 0.184,
    'electricity': 0.233,  # UK grid average
    'electricity (not community)': 0.233,
    'Electricity: electricity, unspecified tariff': 0.233,
    'oil': 0.247,
    'Oil: oil': 0.247,
    'lpg': 0.214,
    'LPG: LPG': 0.214,
    'coal': 0.341,
    'Coal: coal': 0.341,
    'wood': 0.039,  # Biomass
    'B30K (biofuel)': 0.150,
    'biogas': 0.039,
    'district heating': 0.150,  # Varies significantly
}

# Feature columns for modeling
ENVELOPE_FEATURES = [
    'WALLS_ENERGY_EFF',
    'ROOF_ENERGY_EFF',
    'FLOOR_ENERGY_EFF',
    'WINDOWS_ENERGY_EFF',
    'MAINHEAT_ENERGY_EFF',
    'MAINHEATC_ENERGY_EFF',
    'HOT_WATER_ENERGY_EFF',
    'LIGHTING_ENERGY_EFF'
]

GEOMETRY_FEATURES = [
    'TOTAL_FLOOR_AREA',
    'NUMBER_HABITABLE_ROOMS',
    'NUMBER_HEATED_ROOMS',
    'EXTENSION_COUNT',
    'FLOOR_HEIGHT'
]

SYSTEM_FEATURES = [
    'LOW_ENERGY_LIGHTING',
    'MULTI_GLAZE_PROPORTION',
    'NUMBER_OPEN_FIREPLACES',
    'PHOTO_SUPPLY',
    'WIND_TURBINE_COUNT'
]

CATEGORICAL_FEATURES = [
    'PROPERTY_TYPE',
    'BUILT_FORM',
    'CONSTRUCTION_AGE_BAND',
    'GLAZED_TYPE',
    'MAINS_GAS_FLAG',
    'MAIN_FUEL'
]

FEATURE_COLUMNS = ENVELOPE_FEATURES + GEOMETRY_FEATURES + SYSTEM_FEATURES + CATEGORICAL_FEATURES

TARGET_COLUMNS = [
    'ENERGY_CONSUMPTION_CURRENT',  # Primary energy intensity (kWh/m2)
    'CO2_EMISS_CURR_PER_FLOOR_AREA',  # Carbon intensity (kg/m2)
    'HEATING_COST_CURRENT',
    'HOT_WATER_COST_CURRENT',
    'LIGHTING_COST_CURRENT'
]

# Improvement measures from recommendations
IMPROVEMENT_CATEGORIES = {
    'wall_insulation': [6, 7],  # Cavity wall, Internal/external wall
    'roof_insulation': [45, 5],  # Flat roof, Loft insulation
    'floor_insulation': [47, 57, 58],  # Floor, suspended, solid
    'windows': [8],  # Double glazing
    'boiler': [20, 37],  # Condensing boiler
    'heating_controls': [14, 15, 16],  # TRVs, programmers
    'lighting': [35],  # Low energy lighting
    'solar_thermal': [19],  # Solar water heating
    'solar_pv': [34],  # Photovoltaic
    'heat_pump': [22, 23],  # Heat pumps
    'draught_proofing': [10]  # Draught proofing
}
