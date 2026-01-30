"""
Retrofit Decision Support System (DSS)

A physics-guided surrogate modeling system for building energy performance
prediction and retrofit optimization using UK EPC data.

Key Features:
- Multi-city EPC data integration with climate effects
- Physics-guided feature engineering
- Surrogate models for energy intensity, carbon emissions, and costs
- Discrete optimization engine for retrofit recommendations
"""

__version__ = "1.0.0"
__author__ = "Retrofit DSS Team"

# Lazy imports to avoid circular dependencies
def __getattr__(name):
    if name == 'DataLoader':
        from .data.loader import DataLoader
        return DataLoader
    elif name == 'DataPreprocessor':
        from .data.preprocessor import DataPreprocessor
        return DataPreprocessor
    elif name == 'SurrogateModelFactory':
        from .models.surrogate import SurrogateModelFactory
        return SurrogateModelFactory
    elif name == 'OptimizationEngine':
        from .optimization.engine import OptimizationEngine
        return OptimizationEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
