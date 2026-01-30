"""
Surrogate models for building energy performance prediction.

These models replace time-consuming dynamic simulations with fast,
physics-guided machine learning models trained on EPC data.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import joblib
from pathlib import Path

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


@dataclass
class ModelMetrics:
    """Container for model evaluation metrics."""
    mae: float
    rmse: float
    r2: float
    mape: Optional[float] = None
    n_samples: int = 0
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'mae': self.mae,
            'rmse': self.rmse,
            'r2': self.r2,
            'mape': self.mape,
            'n_samples': self.n_samples
        }


@dataclass
class FeatureImportance:
    """Container for feature importance analysis."""
    features: List[str]
    importances: List[float]
    
    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({
            'feature': self.features,
            'importance': self.importances
        }).sort_values('importance', ascending=False)
    
    def get_top_features(self, n: int = 10) -> pd.DataFrame:
        return self.to_dataframe().head(n)
    
    def validate_physical_intuition(self) -> Dict[str, Any]:
        """
        Validate that feature importance aligns with building-physics intuition.
        
        Expected high importance for:
        - Wall efficiency (major heat loss path)
        - Roof efficiency (significant heat loss)
        - Main heating efficiency
        - Building age (proxy for insulation standards)
        
        Returns:
            Dictionary with a lightweight sanity check (not a hard constraint).
        """
        df = self.to_dataframe()
        top_15 = set(df.head(15)['feature'].tolist())
        
        # Physics-expected important features
        expected_high = {
            'WALLS_ENERGY_EFF_NUM',
            'ROOF_ENERGY_EFF_NUM',
            'ENVELOPE_QUALITY',
            'HEAT_LOSS_PROXY',
            'MAINHEAT_ENERGY_EFF_NUM',
            'AGE_BAND_NUM',
            'TOTAL_FLOOR_AREA',
            'HDD'
        }
        
        found_expected = expected_high.intersection(top_15)
        
        return {
            'expected_found': list(found_expected),
            'expected_missing': list(expected_high - top_15),
            'match_rate': len(found_expected) / len(expected_high),
            'physically_consistent': len(found_expected) >= 4  # At least 4 of 8
        }


class BaseSurrogateModel:
    """
    Base class for surrogate models.
    
    Provides common functionality for training, evaluation, and persistence.
    """
    
    def __init__(self, model_type: str = 'xgboost'):
        """
        Initialize the surrogate model.
        
        Args:
            model_type: Type of model ('xgboost', 'gradient_boosting', 'random_forest', 'ridge')
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.target_column = None
        self._fitted = False
        self._metrics = None
        
        self._init_model()
    
    def _init_model(self):
        """Initialize the underlying model."""
        if self.model_type == 'xgboost' and HAS_XGBOOST:
            self.model = XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42
            )
        elif self.model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'ridge':
            self.model = Ridge(alpha=1.0)
        else:
            # Default to gradient boosting
            self.model = GradientBoostingRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
    
    def fit(self, X: pd.DataFrame, y: pd.Series, 
            feature_columns: Optional[List[str]] = None) -> 'BaseSurrogateModel':
        """
        Train the model.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            feature_columns: Columns to use as features
        
        Returns:
            Self for chaining
        """
        if feature_columns is not None:
            self.feature_columns = feature_columns
            X = X[feature_columns]
        else:
            self.feature_columns = list(X.columns)
        
        self.target_column = y.name
        
        # Handle missing values
        X = X.fillna(0)
        y = y.fillna(y.median())
        
        # Scale features for linear models
        if self.model_type == 'ridge':
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = X.values
        
        # Train
        self.model.fit(X_scaled, y)
        self._fitted = True
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Feature DataFrame
        
        Returns:
            Array of predictions
        """
        if not self._fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X = X[self.feature_columns].fillna(0)
        
        if self.model_type == 'ridge':
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X.values
        
        return self.model.predict(X_scaled)
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> ModelMetrics:
        """
        Evaluate model performance.
        
        Args:
            X: Feature DataFrame
            y: True target values
        
        Returns:
            ModelMetrics object
        """
        predictions = self.predict(X)
        y = y.fillna(y.median())
        
        mae = mean_absolute_error(y, predictions)
        rmse = np.sqrt(mean_squared_error(y, predictions))
        r2 = r2_score(y, predictions)
        
        # MAPE (avoiding division by zero)
        mask = y != 0
        if mask.sum() > 0:
            mape = np.mean(np.abs((y[mask] - predictions[mask]) / y[mask])) * 100
        else:
            mape = None
        
        self._metrics = ModelMetrics(
            mae=mae,
            rmse=rmse,
            r2=r2,
            mape=mape,
            n_samples=len(y)
        )
        
        return self._metrics
    
    def get_feature_importance(self) -> FeatureImportance:
        """
        Get feature importance from the model.
        
        Returns:
            FeatureImportance object
        """
        if not self._fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importances = np.abs(self.model.coef_)
        else:
            importances = np.zeros(len(self.feature_columns))
        
        return FeatureImportance(
            features=self.feature_columns,
            importances=list(importances)
        )
    
    def save(self, path: str):
        """Save model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'target_column': self.target_column,
            'model_type': self.model_type,
            'metrics': self._metrics
        }, path)
    
    def load(self, path: str) -> 'BaseSurrogateModel':
        """Load model from disk."""
        data = joblib.load(path)
        
        self.model = data['model']
        self.scaler = data['scaler']
        self.feature_columns = data['feature_columns']
        self.target_column = data['target_column']
        self.model_type = data['model_type']
        self._metrics = data.get('metrics')
        self._fitted = True
        
        return self


class EnergyModel(BaseSurrogateModel):
    """
    Model for predicting primary energy intensity (kWh/m²).
    
    This is the main target for building performance assessment.
    """
    
    def __init__(self, model_type: str = 'xgboost'):
        super().__init__(model_type)
        self.target_name = 'ENERGY_CONSUMPTION_CURRENT'
    
    def validate_predictions(self, X: pd.DataFrame, predictions: np.ndarray) -> Dict[str, bool]:
        """
        Validate predictions against physical constraints.
        
        Checks:
        - Energy consumption should be positive (or very small negative for net-zero)
        - Better envelope should not increase energy consumption
        """
        results = {
            'all_positive': np.all(predictions >= -20),  # Allow small negative for PV
            'reasonable_range': np.all((predictions >= -50) & (predictions <= 1000)),
        }
        
        # Check physical consistency - better walls should mean lower energy
        if 'WALLS_ENERGY_EFF_NUM' in X.columns:
            high_wall_eff = X['WALLS_ENERGY_EFF_NUM'] >= 4
            low_wall_eff = X['WALLS_ENERGY_EFF_NUM'] <= 2
            
            if high_wall_eff.sum() > 0 and low_wall_eff.sum() > 0:
                avg_high = predictions[high_wall_eff].mean()
                avg_low = predictions[low_wall_eff].mean()
                results['wall_consistent'] = avg_high < avg_low
            else:
                results['wall_consistent'] = True
        
        return results


class CarbonModel(BaseSurrogateModel):
    """
    Model for predicting carbon emissions (kg CO₂/m²).
    """
    
    def __init__(self, model_type: str = 'xgboost'):
        super().__init__(model_type)
        self.target_name = 'CO2_EMISS_CURR_PER_FLOOR_AREA'


class CostModel(BaseSurrogateModel):
    """
    Model for predicting energy costs.
    
    Can predict heating, hot water, and lighting costs separately.
    """
    
    def __init__(self, model_type: str = 'xgboost', cost_type: str = 'total'):
        """
        Args:
            model_type: Type of ML model
            cost_type: 'heating', 'hot_water', 'lighting', or 'total'
        """
        super().__init__(model_type)
        self.cost_type = cost_type
        
        target_map = {
            'heating': 'HEATING_COST_CURRENT',
            'hot_water': 'HOT_WATER_COST_CURRENT',
            'lighting': 'LIGHTING_COST_CURRENT',
            'total': 'TOTAL_COST_CURRENT'
        }
        self.target_name = target_map.get(cost_type, 'TOTAL_COST_CURRENT')


class SurrogateModelFactory:
    """
    Factory for creating and managing multiple surrogate models.
    """
    
    def __init__(self, model_type: str = 'xgboost'):
        """
        Initialize the factory.
        
        Args:
            model_type: Default model type for all models
        """
        self.model_type = model_type
        self.models: Dict[str, BaseSurrogateModel] = {}
        self.feature_columns = []
    
    def create_all_models(self) -> 'SurrogateModelFactory':
        """Create all standard models."""
        self.models = {
            'energy': EnergyModel(self.model_type),
            'carbon': CarbonModel(self.model_type),
            'heating_cost': CostModel(self.model_type, 'heating'),
            'hot_water_cost': CostModel(self.model_type, 'hot_water'),
            'lighting_cost': CostModel(self.model_type, 'lighting'),
            'total_cost': CostModel(self.model_type, 'total')
        }
        return self
    
    def fit_all(self, df: pd.DataFrame, feature_columns: List[str]) -> 'SurrogateModelFactory':
        """
        Fit all models on the provided data.
        
        Args:
            df: DataFrame with features and targets
            feature_columns: List of feature column names
        """
        self.feature_columns = feature_columns
        
        for name, model in self.models.items():
            target_col = model.target_name
            if target_col in df.columns:
                print(f"Training {name} model...")
                valid_mask = df[target_col].notna()
                model.fit(
                    df[valid_mask][feature_columns],
                    df[valid_mask][target_col],
                    feature_columns
                )
                print(f"  Trained on {valid_mask.sum():,} samples")
            else:
                print(f"Warning: Target '{target_col}' not found for {name} model")
        
        return self
    
    def evaluate_all(self, df: pd.DataFrame) -> Dict[str, ModelMetrics]:
        """
        Evaluate all models on test data.
        
        Args:
            df: Test DataFrame
        
        Returns:
            Dictionary of model names to metrics
        """
        results = {}
        
        for name, model in self.models.items():
            if model._fitted:
                target_col = model.target_name
                if target_col in df.columns:
                    valid_mask = df[target_col].notna()
                    metrics = model.evaluate(
                        df[valid_mask][self.feature_columns],
                        df[valid_mask][target_col]
                    )
                    results[name] = metrics
                    print(f"{name}:")
                    print(f"  MAE: {metrics.mae:.2f}")
                    print(f"  RMSE: {metrics.rmse:.2f}")
                    print(f"  R²: {metrics.r2:.4f}")
        
        return results
    
    def predict(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Get predictions from all models.
        
        Args:
            X: Feature DataFrame
        
        Returns:
            Dictionary of model names to predictions
        """
        predictions = {}
        
        for name, model in self.models.items():
            if model._fitted:
                predictions[name] = model.predict(X)
        
        return predictions
    
    def get_all_feature_importance(self) -> Dict[str, FeatureImportance]:
        """Get feature importance from all models."""
        return {
            name: model.get_feature_importance()
            for name, model in self.models.items()
            if model._fitted
        }
    
    def save_all(self, directory: str):
        """Save all models to a directory."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        
        for name, model in self.models.items():
            if model._fitted:
                model.save(directory / f"{name}_model.joblib")
        
        # Save feature columns
        joblib.dump(self.feature_columns, directory / "feature_columns.joblib")
    
    def load_all(self, directory: str) -> 'SurrogateModelFactory':
        """Load all models from a directory."""
        directory = Path(directory)
        
        self.feature_columns = joblib.load(directory / "feature_columns.joblib")
        
        for name, model in self.models.items():
            model_path = directory / f"{name}_model.joblib"
            if model_path.exists():
                model.load(model_path)
        
        return self
    
    def validate_physical_consistency(self) -> Dict[str, Dict]:
        """
        Validate that all models respect physical intuition.
        
        Returns:
            Validation results for each model
        """
        results = {}
        
        for name, model in self.models.items():
            if model._fitted:
                importance = model.get_feature_importance()
                results[name] = importance.validate_physical_intuition()
        
        return results
