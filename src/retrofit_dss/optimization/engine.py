"""
Optimization engine for retrofit recommendations.

Provides discrete optimization to find the most cost-effective
retrofit measures to achieve energy/carbon targets.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from itertools import combinations

from ..utils.constants import IMPROVEMENT_CATEGORIES
from ..utils.helpers import parse_cost_range, get_average_cost
from ..data.preprocessor import DataPreprocessor


@dataclass
class RetrofitMeasure:
    """Represents a single retrofit measure."""
    id: int
    name: str
    category: str
    description: str
    cost_min: float
    cost_max: float
    
    @property
    def cost_avg(self) -> float:
        return (self.cost_min + self.cost_max) / 2
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'name': self.name,
            'category': self.category,
            'description': self.description,
            'cost_range': f"£{self.cost_min:,.0f} - £{self.cost_max:,.0f}",
            'cost_average': self.cost_avg
        }


@dataclass
class RetrofitPackage:
    """Represents a combination of retrofit measures."""
    measures: List[RetrofitMeasure]
    total_cost_min: float = 0
    total_cost_max: float = 0
    predicted_energy_reduction: float = 0
    predicted_carbon_reduction: float = 0
    predicted_cost_savings: float = 0
    payback_years: Optional[float] = None
    target_met: bool = False
    
    def __post_init__(self):
        if self.measures:
            self.total_cost_min = sum(m.cost_min for m in self.measures)
            self.total_cost_max = sum(m.cost_max for m in self.measures)
    
    @property
    def total_cost_avg(self) -> float:
        return (self.total_cost_min + self.total_cost_max) / 2
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'measures': [m.to_dict() for m in self.measures],
            'total_cost_range': f"£{self.total_cost_min:,.0f} - £{self.total_cost_max:,.0f}",
            'total_cost_average': self.total_cost_avg,
            'predicted_energy_reduction_pct': self.predicted_energy_reduction,
            'predicted_carbon_reduction_pct': self.predicted_carbon_reduction,
            'predicted_annual_savings': self.predicted_cost_savings,
            'payback_years': self.payback_years,
            'target_met': self.target_met
        }


class RecommendationDatabase:
    """
    Database of retrofit recommendations from EPC data.
    """
    
    def __init__(self):
        self.measures: Dict[int, RetrofitMeasure] = {}
        self._categories_map = {}
        self._build_categories_map()
    
    def _build_categories_map(self):
        """Build reverse mapping from improvement ID to category."""
        for category, ids in IMPROVEMENT_CATEGORIES.items():
            for imp_id in ids:
                self._categories_map[imp_id] = category
    
    def load_from_dataframe(self, df: pd.DataFrame):
        """
        Load measures from recommendations DataFrame.
        
        Args:
            df: DataFrame with columns IMPROVEMENT_ID, IMPROVEMENT_SUMMARY_TEXT,
                IMPROVEMENT_DESCR_TEXT, INDICATIVE_COST
        """
        # Get unique measures with their most common cost range
        grouped = df.groupby('IMPROVEMENT_ID').agg({
            'IMPROVEMENT_SUMMARY_TEXT': 'first',
            'IMPROVEMENT_DESCR_TEXT': 'first',
            'INDICATIVE_COST': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else ''
        }).reset_index()
        
        for _, row in grouped.iterrows():
            imp_id = int(row['IMPROVEMENT_ID']) if pd.notna(row['IMPROVEMENT_ID']) else 0
            if imp_id == 0:
                continue
            
            cost_min, cost_max = parse_cost_range(row['INDICATIVE_COST'])
            category = self._categories_map.get(imp_id, 'other')
            
            self.measures[imp_id] = RetrofitMeasure(
                id=imp_id,
                name=str(row['IMPROVEMENT_SUMMARY_TEXT']) if pd.notna(row['IMPROVEMENT_SUMMARY_TEXT']) else '',
                category=category,
                description=str(row['IMPROVEMENT_DESCR_TEXT']) if pd.notna(row['IMPROVEMENT_DESCR_TEXT']) else '',
                cost_min=cost_min,
                cost_max=cost_max
            )
        
        print(f"Loaded {len(self.measures)} unique retrofit measures")
    
    def get_measure(self, imp_id: int) -> Optional[RetrofitMeasure]:
        """Get a measure by ID."""
        return self.measures.get(imp_id)
    
    def get_measures_by_category(self, category: str) -> List[RetrofitMeasure]:
        """Get all measures in a category."""
        return [m for m in self.measures.values() if m.category == category]
    
    def get_all_categories(self) -> List[str]:
        """Get list of all categories."""
        return list(set(m.category for m in self.measures.values()))


class OptimizationEngine:
    """
    Engine for optimizing retrofit measure selection.
    
    Features:
    - Discrete optimization over measure combinations
    - Cost-benefit analysis
    - Physics-informed measure effect estimation (Model-Based)
    - Multi-objective optimization (cost vs. reduction)
    """
    
    def __init__(self, model_factory=None):
        """
        Initialize the optimization engine.
        
        Args:
            model_factory: SurrogateModelFactory for predictions
        """
        self.model_factory = model_factory
        self.recommendation_db = RecommendationDatabase()
        
        # Effect estimates for each measure category
        # Used for defining feature updates and fallback values
        self.measure_effects = {
            'wall_insulation': {
                'energy_reduction': 0.20,
                'carbon_reduction': 0.18,
                'features_affected': {
                    'WALLS_ENERGY_EFF_NUM': 4,  # Improve to Good
                }
            },
            'roof_insulation': {
                'energy_reduction': 0.10,
                'carbon_reduction': 0.09,
                'features_affected': {
                    'ROOF_ENERGY_EFF_NUM': 4,
                }
            },
            'floor_insulation': {
                'energy_reduction': 0.05,
                'carbon_reduction': 0.04,
                'features_affected': {
                    'FLOOR_ENERGY_EFF_NUM': 4,
                }
            },
            'windows': {
                'energy_reduction': 0.10,
                'carbon_reduction': 0.09,
                'features_affected': {
                    'WINDOWS_ENERGY_EFF_NUM': 4,
                    'GLAZED_TYPE_NUM': 3,
                }
            },
            'boiler': {
                'energy_reduction': 0.15,
                'carbon_reduction': 0.12,
                'features_affected': {
                    'MAINHEAT_ENERGY_EFF_NUM': 5,
                }
            },
            'heating_controls': {
                'energy_reduction': 0.05,
                'carbon_reduction': 0.04,
                'features_affected': {
                    'MAINHEATC_ENERGY_EFF_NUM': 4,
                }
            },
            'lighting': {
                'energy_reduction': 0.03,
                'carbon_reduction': 0.03,
                'features_affected': {
                    'LIGHTING_ENERGY_EFF_NUM': 5,
                    'LOW_ENERGY_LIGHTING': 100
                }
            },
            'solar_thermal': {
                'energy_reduction': 0.05,
                'carbon_reduction': 0.05,
                'features_affected': {
                    'HOT_WATER_ENERGY_EFF_NUM': 5,
                    'SOLAR_WATER_HEATING_NUM': 1
                }
            },
            'solar_pv': {
                'energy_reduction': 0.15,
                'carbon_reduction': 0.20,
                'features_affected': {
                    'PHOTO_SUPPLY': 30,  # 30% of roof
                }
            },
            'heat_pump': {
                'energy_reduction': 0.25,
                'carbon_reduction': 0.40,
                'features_affected': {
                    'MAINHEAT_ENERGY_EFF_NUM': 5,
                }
            },
            'draught_proofing': {
                'energy_reduction': 0.03,
                'carbon_reduction': 0.02,
                'features_affected': {
                    'INFILTRATION_PROXY': -0.3
                }
            }
        }
    
    def load_recommendations(self, df: pd.DataFrame):
        """Load recommendations data."""
        self.recommendation_db.load_from_dataframe(df)
    
    def estimate_improvement_effect(
        self,
        building_profile: pd.Series,
        measures: List[RetrofitMeasure]
    ) -> Dict[str, float]:
        """
        Estimate the effect of retrofit measures on a building using Model-Based Counterfactuals.
        
        1. Modifies the building profile features based on measures.
        2. Re-calculates derived physics features.
        3. Predicts new performance using trained surrogate models.
        
        Args:
            building_profile: Series with building features
            measures: List of measures to apply
        
        Returns:
            Dictionary with estimated reductions and new values
        """
        # Start with base profile values
        current_energy = building_profile.get('ENERGY_CONSUMPTION_CURRENT', 
                        building_profile.get('ENERGY_INTENSITY', 200))
        current_carbon = building_profile.get('CO2_EMISS_CURR_PER_FLOOR_AREA', 40)
        
        # Calculate current cost
        current_cost = building_profile.get('TOTAL_COST_CURRENT')
        if not current_cost or pd.isna(current_cost):
             heating = building_profile.get('HEATING_COST_CURRENT', 500)
             hot_water = building_profile.get('HOT_WATER_COST_CURRENT', 150)
             lighting = building_profile.get('LIGHTING_COST_CURRENT', 100)
             current_cost = heating + hot_water + lighting
             if current_cost <= 0 or pd.isna(current_cost):
                 current_cost = building_profile.get('TOTAL_FLOOR_AREA', 80) * 12

        # 1. Prepare modified profile
        # Convert to dict for easier modification, then to DataFrame
        modified_profile_dict = building_profile.to_dict()
        
        applied_categories = set()
        rule_based_energy_reduction = 0
        rule_based_carbon_reduction = 0
        
        for measure in measures:
            if measure.category in applied_categories:
                continue
            
            effects = self.measure_effects.get(measure.category, {})
            
            # Apply feature updates (Counterfactual)
            if 'features_affected' in effects:
                for feature, value in effects['features_affected'].items():
                    # Special handling for proxies/deltas
                    if feature == 'INFILTRATION_PROXY':
                        # This will be handled after physics feature recalculation
                        pass
                    elif feature in ['ENVELOPE_QUALITY', 'SYSTEM_EFFICIENCY', 'RENEWABLE_FRACTION']:
                        # Ignore these, they will be recalculated from raw inputs
                        pass
                    elif feature == 'PHOTO_SUPPLY' and modified_profile_dict.get(feature, 0) > value:
                        # Don't downgrade existing solar
                        pass
                    else:
                        # Update raw feature (e.g., WALLS_ENERGY_EFF_NUM = 4)
                        modified_profile_dict[feature] = value
            
            # Track rule-based for fallback
            energy_red = effects.get('energy_reduction', 0.05)
            carbon_red = effects.get('carbon_reduction', 0.05)
            remaining_energy = 1 - rule_based_energy_reduction
            remaining_carbon = 1 - rule_based_carbon_reduction
            rule_based_energy_reduction += energy_red * remaining_energy
            rule_based_carbon_reduction += carbon_red * remaining_carbon
            
            applied_categories.add(measure.category)

        # 2. Re-calculate derived physics features
        temp_df = pd.DataFrame([modified_profile_dict])

        # Use DataPreprocessor to update derived features (ENVELOPE_QUALITY, etc.)
        pp = DataPreprocessor()
        temp_df = pp.add_physics_features(temp_df)
        
        # 3. Apply manual overrides for non-raw features (e.g. draught proofing)
        for measure in measures:
             effects = self.measure_effects.get(measure.category, {})
             if 'features_affected' in effects:
                  for feature, value in effects['features_affected'].items():
                       if feature == 'INFILTRATION_PROXY':
                            # Additive effect
                            current_val = temp_df[feature].iloc[0]
                            temp_df[feature] = current_val + value

        # 4. Model Prediction
        if self.model_factory:
             try:
                 # Use correct feature columns
                 X = temp_df[self.model_factory.feature_columns].fillna(0)

                 new_energy = self.model_factory.models['energy'].predict(X)[0]
                 new_carbon = self.model_factory.models['carbon'].predict(X)[0]
                 new_cost = self.model_factory.models['total_cost'].predict(X)[0]

                 # Sanity check
                 new_energy = max(10, new_energy) # Min 10 kWh/m2
                 new_carbon = max(1, new_carbon)
                 new_cost = max(50, new_cost)

                 # Calculate reductions
                 energy_red_pct = max(0, (current_energy - new_energy) / current_energy * 100)
                 carbon_red_pct = max(0, (current_carbon - new_carbon) / current_carbon * 100)
                 cost_savings = max(0, current_cost - new_cost)

                 return {
                     'energy_reduction_pct': energy_red_pct,
                     'carbon_reduction_pct': carbon_red_pct,
                     'new_energy_intensity': new_energy,
                     'new_carbon_intensity': new_carbon,
                     'annual_cost_savings': cost_savings
                 }
             except Exception as e:
                 # print(f"Prediction failed: {e}")
                 pass
        
        # Fallback to rule-based
        rule_based_energy_reduction = min(rule_based_energy_reduction, 0.70)
        rule_based_carbon_reduction = min(rule_based_carbon_reduction, 0.80)
        
        return {
            'energy_reduction_pct': rule_based_energy_reduction * 100,
            'carbon_reduction_pct': rule_based_carbon_reduction * 100,
            'new_energy_intensity': current_energy * (1 - rule_based_energy_reduction),
            'new_carbon_intensity': current_carbon * (1 - rule_based_carbon_reduction),
            'annual_cost_savings': current_cost * rule_based_energy_reduction
        }
    
    def get_applicable_measures(
        self,
        building_profile: pd.Series,
        recommendations_df: pd.DataFrame
    ) -> List[RetrofitMeasure]:
        """
        Get measures applicable to a specific building.
        
        Args:
            building_profile: Series with building features
            recommendations_df: DataFrame with property-specific recommendations
        
        Returns:
            List of applicable measures
        """
        lmk_key = building_profile.get('LMK_KEY')
        
        if lmk_key is None or recommendations_df is None:
            # Return generic measures based on building condition
            return self._get_generic_measures(building_profile)
        
        # Get property-specific recommendations
        property_recs = recommendations_df[recommendations_df['LMK_KEY'] == lmk_key]
        
        measures = []
        for _, rec in property_recs.iterrows():
            imp_id = int(rec['IMPROVEMENT_ID']) if pd.notna(rec['IMPROVEMENT_ID']) else 0
            measure = self.recommendation_db.get_measure(imp_id)
            
            if measure is None and imp_id > 0:
                # Create from recommendation data
                cost_min, cost_max = parse_cost_range(rec.get('INDICATIVE_COST', ''))
                measure = RetrofitMeasure(
                    id=imp_id,
                    name=str(rec.get('IMPROVEMENT_SUMMARY_TEXT', '')),
                    category=self.recommendation_db._categories_map.get(imp_id, 'other'),
                    description=str(rec.get('IMPROVEMENT_DESCR_TEXT', '')),
                    cost_min=cost_min,
                    cost_max=cost_max
                )
            
            if measure:
                measures.append(measure)
        
        return measures
    
    def _get_generic_measures(self, building_profile: pd.Series) -> List[RetrofitMeasure]:
        """Get generic measures based on building condition."""
        measures = []
        
        # Check wall condition
        wall_eff = building_profile.get('WALLS_ENERGY_EFF_NUM', 3)
        if wall_eff < 4:
            for imp_id in IMPROVEMENT_CATEGORIES.get('wall_insulation', []):
                if m := self.recommendation_db.get_measure(imp_id):
                    measures.append(m)
                    break
        
        # Check roof condition
        roof_eff = building_profile.get('ROOF_ENERGY_EFF_NUM', 3)
        if roof_eff < 4:
            for imp_id in IMPROVEMENT_CATEGORIES.get('roof_insulation', []):
                if m := self.recommendation_db.get_measure(imp_id):
                    measures.append(m)
                    break
        
        # Check heating system
        heat_eff = building_profile.get('MAINHEAT_ENERGY_EFF_NUM', 3)
        if heat_eff < 4:
            for imp_id in IMPROVEMENT_CATEGORIES.get('boiler', []):
                if m := self.recommendation_db.get_measure(imp_id):
                    measures.append(m)
                    break
        
        # Check lighting
        lighting_pct = building_profile.get('LOW_ENERGY_LIGHTING', 50)
        if lighting_pct < 80:
            for imp_id in IMPROVEMENT_CATEGORIES.get('lighting', []):
                if m := self.recommendation_db.get_measure(imp_id):
                    measures.append(m)
                    break
        
        # Always consider renewables
        for imp_id in IMPROVEMENT_CATEGORIES.get('solar_pv', []):
            if m := self.recommendation_db.get_measure(imp_id):
                measures.append(m)
                break
        
        return measures
    
    def optimize(
        self,
        building_profile: pd.Series,
        target_type: str = 'carbon',
        target_reduction: float = 50.0,
        max_budget: Optional[float] = None,
        max_measures: int = 5
    ) -> List[RetrofitPackage]:
        """
        Find optimal retrofit packages to achieve target.
        
        Uses discrete optimization to find combinations of measures
        that achieve the target at minimum cost.
        
        Args:
            building_profile: Series with building features
            target_type: 'energy' or 'carbon'
            target_reduction: Target percentage reduction
            max_budget: Maximum budget (£)
            max_measures: Maximum number of measures to combine
        
        Returns:
            List of recommended packages, sorted by cost-effectiveness
        """
        # Get applicable measures
        applicable_measures = list(self.recommendation_db.measures.values())
        
        # Filter by budget if specified
        if max_budget:
            applicable_measures = [
                m for m in applicable_measures 
                if m.cost_min <= max_budget
            ]
        
        packages = []
        
        # Try all combinations up to max_measures
        # Limit total combinations to avoid combinatorial explosion
        # If too many measures, limit n
        max_n = max_measures
        if len(applicable_measures) > 15:
             max_n = min(max_measures, 3)

        for n in range(1, min(max_n + 1, len(applicable_measures) + 1)):
            for measure_combo in combinations(applicable_measures, n):
                # Skip if over budget
                total_cost = sum(m.cost_avg for m in measure_combo)
                if max_budget and total_cost > max_budget:
                    continue
                
                # Estimate effect (Model-Based)
                effects = self.estimate_improvement_effect(
                    building_profile,
                    list(measure_combo)
                )
                
                reduction = effects.get(f'{target_type}_reduction_pct', 0)
                cost_savings = effects.get('annual_cost_savings', 0)
                
                # Create package
                package = RetrofitPackage(
                    measures=list(measure_combo),
                    predicted_energy_reduction=effects.get('energy_reduction_pct', 0),
                    predicted_carbon_reduction=effects.get('carbon_reduction_pct', 0),
                    predicted_cost_savings=cost_savings
                )
                
                # Check if target met
                target_key = 'predicted_carbon_reduction_pct' if target_type == 'carbon' else 'predicted_energy_reduction_pct'
                package.target_met = (package.to_dict()[target_key] >= target_reduction)

                # Calculate payback
                if cost_savings > 0:
                    package.payback_years = package.total_cost_avg / cost_savings
                
                packages.append(package)
        
        # Filter packages that meet target
        valid_packages = [
            p for p in packages 
            if (p.predicted_carbon_reduction if target_type == 'carbon' 
                else p.predicted_energy_reduction) >= target_reduction
        ]
        
        # Sort by cost (ascending)
        valid_packages.sort(key=lambda p: p.total_cost_avg)
        
        # If no packages meet target, return best available (closest to target)
        if not valid_packages:
            # Sort by reduction descending
            packages.sort(
                key=lambda p: (
                    p.predicted_carbon_reduction if target_type == 'carbon' 
                    else p.predicted_energy_reduction
                ),
                reverse=True
            )
            return packages[:5]
        
        return valid_packages[:10]
    
    def sensitivity_analysis(
        self,
        building_profile: pd.Series,
        feature: str,
        values: List[Any]
    ) -> pd.DataFrame:
        """
        Perform sensitivity analysis on a single feature.
        
        Args:
            building_profile: Base building profile
            feature: Feature to vary
            values: Values to test
        
        Returns:
            DataFrame with predictions for each value
        """
        if self.model_factory is None:
            raise ValueError("Model factory required for sensitivity analysis")
        
        results = []
        
        for value in values:
            # Create modified profile
            profile = building_profile.copy()
            profile[feature] = value
            
            # Recalculate physics features! (Important for consistency)
            temp_df = pd.DataFrame([profile])
            pp = DataPreprocessor()
            temp_df = pp.add_physics_features(temp_df)

            # Get predictions
            X = temp_df[self.model_factory.feature_columns].fillna(0)
            predictions = self.model_factory.predict(X)
            
            result = {'feature_value': value}
            for name, pred in predictions.items():
                result[f'predicted_{name}'] = pred[0]
            results.append(result)
        
        return pd.DataFrame(results)
    
    def get_cost_benefit_summary(
        self,
        packages: List[RetrofitPackage]
    ) -> pd.DataFrame:
        """
        Create cost-benefit summary table.
        
        Args:
            packages: List of retrofit packages
        
        Returns:
            DataFrame with cost-benefit analysis
        """
        data = []
        
        for i, pkg in enumerate(packages):
            data.append({
                'Package': i + 1,
                'Measures': ', '.join(m.name for m in pkg.measures),
                'Cost (min)': f"£{pkg.total_cost_min:,.0f}",
                'Cost (max)': f"£{pkg.total_cost_max:,.0f}",
                'Energy Reduction': f"{pkg.predicted_energy_reduction:.1f}%",
                'Carbon Reduction': f"{pkg.predicted_carbon_reduction:.1f}%",
                'Annual Savings': f"£{pkg.predicted_cost_savings:,.0f}",
                'Payback (years)': f"{pkg.payback_years:.1f}" if pkg.payback_years else "N/A",
                'Target Met': 'Yes' if pkg.target_met else 'No'
            })
        
        return pd.DataFrame(data)
