"""
Data preprocessing module for EPC data.

Handles cleaning, feature engineering, and encoding of categorical variables
with physics-based interpretations.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.model_selection import GroupShuffleSplit

from ..utils.constants import (
    EFFICIENCY_RATINGS,
    EPC_GRADES,
    AGE_BAND_ENCODING,
    PROPERTY_TYPE_ENCODING,
    BUILT_FORM_ENCODING,
    GLAZED_TYPE_ENCODING,
    ENVELOPE_FEATURES,
    GEOMETRY_FEATURES,
    SYSTEM_FEATURES,
    TARGET_COLUMNS
)
from ..utils.helpers import (
    estimate_u_value_from_description,
    estimate_window_u_value,
    calculate_form_factor,
    get_average_cost
)


class DataPreprocessor:
    """
    Preprocesses EPC data for machine learning.
    
    Features:
    - Ordinal encoding for efficiency ratings (physics-based)
    - Age band encoding (building regulations evolution)
    - Missing value imputation
    - Feature engineering for physics-based analysis
    """
    
    def __init__(self):
        """Initialize the preprocessor."""
        self.feature_columns = []
        self.target_columns = TARGET_COLUMNS.copy()
        self._fitted = False
        self._value_counts = {}
    
    def fit(self, df: pd.DataFrame) -> 'DataPreprocessor':
        """
        Fit the preprocessor on training data.
        
        Args:
            df: Training DataFrame
        
        Returns:
            Self for chaining
        """
        # Store value distributions for imputation
        for col in ENVELOPE_FEATURES:
            if col in df.columns:
                self._value_counts[col] = df[col].value_counts()
        
        self._fitted = True
        return self
    
    def transform(self, df: pd.DataFrame, 
                 include_targets: bool = True) -> pd.DataFrame:
        """
        Transform the DataFrame with preprocessing.
        
        Args:
            df: DataFrame to transform
            include_targets: Whether to include target columns
        
        Returns:
            Preprocessed DataFrame
        """
        df = df.copy()
        
        # 1. Clean and encode efficiency ratings
        df = self._encode_efficiency_ratings(df)
        
        # 2. Encode categorical variables
        df = self._encode_categoricals(df)
        
        # 3. Handle numeric columns
        df = self._process_numerics(df)
        
        # 4. Add physics-based features
        df = self._add_physics_features(df)
        
        # 5. Handle missing values
        df = self._handle_missing(df)
        
        # Build feature columns list (only when not already set)
        if not self.feature_columns:
            self._build_feature_list(df)
        
        return df
    
    def fit_transform(self, df: pd.DataFrame, 
                     include_targets: bool = True) -> pd.DataFrame:
        """
        Fit and transform in one step.
        
        Args:
            df: DataFrame to process
            include_targets: Whether to include target columns
        
        Returns:
            Preprocessed DataFrame
        """
        self.fit(df)
        return self.transform(df, include_targets)
    
    def _encode_efficiency_ratings(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert efficiency rating strings to ordinal numeric values.
        
        Physics interpretation:
        - Very Good (5): Excellent thermal performance
        - Good (4): Above average
        - Average (3): Meets basic standards
        - Poor (2): Below average, significant heat loss
        - Very Poor (1): Major thermal bridging/losses
        - N/A (0): Not applicable (e.g., no roof for mid-floor flat)
        """
        efficiency_cols = [
            'WALLS_ENERGY_EFF', 'WALLS_ENV_EFF',
            'ROOF_ENERGY_EFF', 'ROOF_ENV_EFF',
            'FLOOR_ENERGY_EFF', 'FLOOR_ENV_EFF',
            'WINDOWS_ENERGY_EFF', 'WINDOWS_ENV_EFF',
            'MAINHEAT_ENERGY_EFF', 'MAINHEAT_ENV_EFF',
            'MAINHEATC_ENERGY_EFF', 'MAINHEATC_ENV_EFF',
            'HOT_WATER_ENERGY_EFF', 'HOT_WATER_ENV_EFF',
            'LIGHTING_ENERGY_EFF', 'LIGHTING_ENV_EFF',
            'SHEATING_ENERGY_EFF', 'SHEATING_ENV_EFF'
        ]
        
        for col in efficiency_cols:
            if col in df.columns:
                df[col + '_NUM'] = df[col].map(
                    lambda x: EFFICIENCY_RATINGS.get(x, 0) if pd.notna(x) else 0
                )
        
        # EPC grades
        for col in ['CURRENT_ENERGY_RATING', 'POTENTIAL_ENERGY_RATING']:
            if col in df.columns:
                df[col + '_NUM'] = df[col].map(
                    lambda x: EPC_GRADES.get(x, 0) if pd.notna(x) else 0
                )
        
        return df
    
    def _encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical variables with physics-meaningful ordinal values.
        """
        # Construction age - newer buildings generally have better insulation
        if 'CONSTRUCTION_AGE_BAND' in df.columns:
            df['AGE_BAND_NUM'] = df['CONSTRUCTION_AGE_BAND'].map(
                lambda x: AGE_BAND_ENCODING.get(x, 0) if pd.notna(x) else 0
            )
        
        # Property type - affects surface area to volume ratio
        if 'PROPERTY_TYPE' in df.columns:
            df['PROPERTY_TYPE_NUM'] = df['PROPERTY_TYPE'].map(
                lambda x: PROPERTY_TYPE_ENCODING.get(x, 0) if pd.notna(x) else 0
            )
        
        # Built form - affects exposed surface area
        if 'BUILT_FORM' in df.columns:
            df['BUILT_FORM_NUM'] = df['BUILT_FORM'].map(
                lambda x: BUILT_FORM_ENCODING.get(x, 0) if pd.notna(x) else 0
            )
        
        # Glazing type - affects window U-value
        if 'GLAZED_TYPE' in df.columns:
            df['GLAZED_TYPE_NUM'] = df['GLAZED_TYPE'].map(
                lambda x: GLAZED_TYPE_ENCODING.get(x, 0) if pd.notna(x) else 0
            )
        
        # Mains gas flag - binary
        if 'MAINS_GAS_FLAG' in df.columns:
            df['MAINS_GAS_NUM'] = df['MAINS_GAS_FLAG'].map(
                lambda x: 1 if x in ['Y', 'Yes', 'yes', True] else 0
            )
        
        # Solar water heating flag
        if 'SOLAR_WATER_HEATING_FLAG' in df.columns:
            df['SOLAR_WATER_HEATING_NUM'] = df['SOLAR_WATER_HEATING_FLAG'].map(
                lambda x: 1 if x in ['Y', 'Yes', 'yes', True] else 0
            )
        
        # City encoding (for climate effects)
        if 'CITY' in df.columns:
            city_map = {'Cambridge': 0, 'Boston': 1, 'Liverpool': 2, 'Sheffield': 3}
            df['CITY_NUM'] = df['CITY'].map(city_map)
        
        return df
    
    def _process_numerics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process and clean numeric columns.
        """
        numeric_cols = [
            'TOTAL_FLOOR_AREA', 'NUMBER_HABITABLE_ROOMS', 'NUMBER_HEATED_ROOMS',
            'EXTENSION_COUNT', 'FLOOR_HEIGHT', 'LOW_ENERGY_LIGHTING',
            'MULTI_GLAZE_PROPORTION', 'NUMBER_OPEN_FIREPLACES',
            'PHOTO_SUPPLY', 'WIND_TURBINE_COUNT',
            'ENERGY_CONSUMPTION_CURRENT', 'ENERGY_CONSUMPTION_POTENTIAL',
            'CO2_EMISS_CURR_PER_FLOOR_AREA', 'CO2_EMISSIONS_CURRENT',
            'HEATING_COST_CURRENT', 'HEATING_COST_POTENTIAL',
            'HOT_WATER_COST_CURRENT', 'HOT_WATER_COST_POTENTIAL',
            'LIGHTING_COST_CURRENT', 'LIGHTING_COST_POTENTIAL',
            'CURRENT_ENERGY_EFFICIENCY', 'POTENTIAL_ENERGY_EFFICIENCY',
            'HDD', 'AVG_TEMP', 'SOLAR_RADIATION'
        ]
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Ensure non-negative values for certain columns
        non_negative_cols = [
            'TOTAL_FLOOR_AREA', 'NUMBER_HABITABLE_ROOMS', 'NUMBER_HEATED_ROOMS',
            'LOW_ENERGY_LIGHTING', 'MULTI_GLAZE_PROPORTION',
            'PHOTO_SUPPLY', 'WIND_TURBINE_COUNT'
        ]
        
        for col in non_negative_cols:
            if col in df.columns:
                df[col] = df[col].clip(lower=0)
        
        # Cap unrealistic values
        if 'TOTAL_FLOOR_AREA' in df.columns:
            # Cap at reasonable residential size (500 m²)
            df['TOTAL_FLOOR_AREA'] = df['TOTAL_FLOOR_AREA'].clip(upper=500)
        
        if 'LOW_ENERGY_LIGHTING' in df.columns:
            df['LOW_ENERGY_LIGHTING'] = df['LOW_ENERGY_LIGHTING'].clip(upper=100)
        
        if 'MULTI_GLAZE_PROPORTION' in df.columns:
            df['MULTI_GLAZE_PROPORTION'] = df['MULTI_GLAZE_PROPORTION'].clip(upper=100)
        
        return df
    
    def _add_physics_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add physics-based engineered features.
        
        These features are derived from building physics principles:
        - Envelope composite scores
        - System efficiency scores
        - Form factor estimates
        - Heat loss proxies
        """
        # 1. Envelope quality composite (weighted by typical heat loss contribution)
        # Walls: ~35%, Roof: ~25%, Floor: ~15%, Windows: ~25%
        envelope_weights = {
            'WALLS_ENERGY_EFF_NUM': 0.35,
            'ROOF_ENERGY_EFF_NUM': 0.25,
            'FLOOR_ENERGY_EFF_NUM': 0.15,
            'WINDOWS_ENERGY_EFF_NUM': 0.25
        }
        
        df['ENVELOPE_QUALITY'] = sum(
            df[col].fillna(0) * weight 
            for col, weight in envelope_weights.items() 
            if col in df.columns
        )
        
        # 2. System efficiency composite
        system_cols = ['MAINHEAT_ENERGY_EFF_NUM', 'MAINHEATC_ENERGY_EFF_NUM', 
                       'HOT_WATER_ENERGY_EFF_NUM']
        existing_system_cols = [col for col in system_cols if col in df.columns]
        if existing_system_cols:
            df['SYSTEM_EFFICIENCY'] = df[existing_system_cols].mean(axis=1)
        
        # 3. Form factor estimate
        if all(col in df.columns for col in ['TOTAL_FLOOR_AREA', 'BUILT_FORM']):
            df['FORM_FACTOR'] = df.apply(
                lambda row: calculate_form_factor(
                    row.get('TOTAL_FLOOR_AREA', 80),
                    n_storeys=2,  # Default assumption
                    built_form=row.get('BUILT_FORM', 'Semi-Detached')
                ),
                axis=1
            )
        
        # 4. Heat loss coefficient proxy
        # HLC ∝ (Surface Area × U-value) + Ventilation losses
        if 'ENVELOPE_QUALITY' in df.columns and 'TOTAL_FLOOR_AREA' in df.columns:
            # Lower envelope quality = higher U-value = more heat loss
            df['HEAT_LOSS_PROXY'] = (
                df['TOTAL_FLOOR_AREA'] * (6 - df['ENVELOPE_QUALITY']) / 5
            )
        
        # 5. Heating demand proxy (considering climate)
        if 'HDD' in df.columns and 'HEAT_LOSS_PROXY' in df.columns:
            df['HEATING_DEMAND_PROXY'] = df['HDD'] * df['HEAT_LOSS_PROXY'] / 1000
        
        # 6. Ventilation heat loss proxy (open fireplaces)
        if 'NUMBER_OPEN_FIREPLACES' in df.columns:
            # Each open fireplace adds significant infiltration
            df['INFILTRATION_PROXY'] = df['NUMBER_OPEN_FIREPLACES'] * 0.5
        
        # 7. Renewable contribution
        if 'PHOTO_SUPPLY' in df.columns:
            df['RENEWABLE_FRACTION'] = df['PHOTO_SUPPLY'].fillna(0) / 100
        
        # 8. Lighting efficiency ratio
        if 'LOW_ENERGY_LIGHTING' in df.columns:
            df['LIGHTING_EFF_RATIO'] = df['LOW_ENERGY_LIGHTING'].fillna(0) / 100
        
        # 9. Glazing quality
        if 'MULTI_GLAZE_PROPORTION' in df.columns and 'GLAZED_TYPE_NUM' in df.columns:
            df['GLAZING_QUALITY'] = (
                df['MULTI_GLAZE_PROPORTION'].fillna(0) / 100 * 
                df['GLAZED_TYPE_NUM'].fillna(1)
            )
        
        # 10. Building age score (proxy for built-in efficiency)
        if 'AGE_BAND_NUM' in df.columns:
            df['BUILDING_VINTAGE'] = df['AGE_BAND_NUM'] / 13  # Normalize to 0-1
        
        # 11. Total annual cost
        cost_cols = ['HEATING_COST_CURRENT', 'HOT_WATER_COST_CURRENT', 'LIGHTING_COST_CURRENT']
        existing_cost_cols = [col for col in cost_cols if col in df.columns]
        if existing_cost_cols:
            df['TOTAL_COST_CURRENT'] = df[existing_cost_cols].sum(axis=1)
        
        # 12. Energy intensity (already provided but ensure it exists)
        if 'ENERGY_CONSUMPTION_CURRENT' in df.columns and 'TOTAL_FLOOR_AREA' in df.columns:
            # This should already be in kWh/m², but verify
            df['ENERGY_INTENSITY'] = df['ENERGY_CONSUMPTION_CURRENT']
        
        return df
    
    def _handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values with physics-informed imputation.
        """
        # For efficiency ratings, use median (most common case is average = 3)
        eff_cols = [col for col in df.columns if col.endswith('_EFF_NUM')]
        for col in eff_cols:
            df[col] = df[col].fillna(3)  # Average
        
        # Numeric features - use median
        numeric_cols = [
            'TOTAL_FLOOR_AREA', 'NUMBER_HABITABLE_ROOMS', 'NUMBER_HEATED_ROOMS',
            'EXTENSION_COUNT', 'FLOOR_HEIGHT', 'LOW_ENERGY_LIGHTING',
            'MULTI_GLAZE_PROPORTION', 'PHOTO_SUPPLY'
        ]
        
        for col in numeric_cols:
            if col in df.columns:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val if pd.notna(median_val) else 0)
        
        # Fill derived features that might have NaN
        derived_cols = [
            'ENVELOPE_QUALITY', 'SYSTEM_EFFICIENCY', 'FORM_FACTOR',
            'HEAT_LOSS_PROXY', 'HEATING_DEMAND_PROXY', 'INFILTRATION_PROXY',
            'RENEWABLE_FRACTION', 'LIGHTING_EFF_RATIO', 'GLAZING_QUALITY',
            'BUILDING_VINTAGE'
        ]
        
        for col in derived_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        
        return df
    
    def _build_feature_list(self, df: pd.DataFrame):
        """Build the list of feature columns to use for modeling."""
        self.feature_columns = []
        
        # Encoded efficiency ratings
        eff_cols = [col for col in df.columns if col.endswith('_EFF_NUM')]
        self.feature_columns.extend(eff_cols)
        
        # Encoded categoricals
        cat_cols = [
            'AGE_BAND_NUM', 'PROPERTY_TYPE_NUM', 'BUILT_FORM_NUM',
            'GLAZED_TYPE_NUM', 'MAINS_GAS_NUM', 'SOLAR_WATER_HEATING_NUM',
            'CITY_NUM'
        ]
        self.feature_columns.extend([col for col in cat_cols if col in df.columns])
        
        # Numeric features
        num_cols = [
            'TOTAL_FLOOR_AREA', 'NUMBER_HABITABLE_ROOMS', 'NUMBER_HEATED_ROOMS',
            'EXTENSION_COUNT', 'LOW_ENERGY_LIGHTING', 'MULTI_GLAZE_PROPORTION',
            'NUMBER_OPEN_FIREPLACES', 'PHOTO_SUPPLY', 'WIND_TURBINE_COUNT'
        ]
        self.feature_columns.extend([col for col in num_cols if col in df.columns])
        
        # Climate features
        climate_cols = ['HDD', 'AVG_TEMP', 'SOLAR_RADIATION']
        self.feature_columns.extend([col for col in climate_cols if col in df.columns])
        
        # Physics-derived features
        physics_cols = [
            'ENVELOPE_QUALITY', 'SYSTEM_EFFICIENCY', 'FORM_FACTOR',
            'HEAT_LOSS_PROXY', 'HEATING_DEMAND_PROXY', 'INFILTRATION_PROXY',
            'RENEWABLE_FRACTION', 'LIGHTING_EFF_RATIO', 'GLAZING_QUALITY',
            'BUILDING_VINTAGE'
        ]
        self.feature_columns.extend([col for col in physics_cols if col in df.columns])

    def ensure_feature_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure all feature columns exist in the DataFrame."""
        df = df.copy()
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0
        return df

    def recompute_derived_features(
        self,
        df: pd.DataFrame,
        overrides: Optional[Dict[str, float]] = None
    ) -> pd.DataFrame:
        """Recompute physics-derived features, with optional overrides."""
        df = df.copy()
        df = self._add_physics_features(df)
        df = self._handle_missing(df)
        if overrides:
            for col, value in overrides.items():
                df[col] = value
        return df
    
    def get_feature_columns(self) -> List[str]:
        """Get the list of feature columns."""
        return self.feature_columns
    
    def get_target_columns(self) -> List[str]:
        """Get the list of target columns."""
        return self.target_columns


def create_train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    postcode_column: str = 'POSTCODE',
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create train/test split ensuring no postcode leakage.
    
    This prevents geographic correlation from inflating model performance.
    Buildings in the same postcode sector are kept together.
    
    Args:
        df: DataFrame to split
        test_size: Proportion for test set
        postcode_column: Column containing postcodes
        random_state: Random seed for reproducibility
    
    Returns:
        Tuple of (train_df, test_df)
    """
    # Extract postcode sector for grouping
    def get_sector(postcode):
        if pd.isna(postcode):
            return 'UNKNOWN_' + str(np.random.randint(10000))
        parts = str(postcode).split(' ')
        if len(parts) >= 2:
            return parts[0] + ' ' + parts[1][0] if len(parts[1]) > 0 else parts[0]
        return parts[0] if parts else 'UNKNOWN_' + str(np.random.randint(10000))
    
    df = df.copy()
    df['_postcode_sector'] = df[postcode_column].apply(get_sector)
    
    # Use GroupShuffleSplit to keep postcode sectors together
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    
    train_idx, test_idx = next(gss.split(df, groups=df['_postcode_sector']))
    
    train_df = df.iloc[train_idx].drop(columns=['_postcode_sector'])
    test_df = df.iloc[test_idx].drop(columns=['_postcode_sector'])
    
    return train_df, test_df
