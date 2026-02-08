"""
Optimization engine for retrofit recommendations.

Provides discrete optimization to find the most cost-effective
retrofit measures to achieve energy/carbon targets.
"""
import numpy as np
import pandas as pd
import os
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from itertools import combinations

from ..utils.constants import IMPROVEMENT_CATEGORIES
from ..utils.helpers import parse_cost_range, get_average_cost


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
            'payback_years': self.payback_years
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
    - Physics-informed measure effect estimation
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
        self._baseline_cache: Dict[str, Dict[str, float]] = {}
        
        # Effect estimates for each measure category (% reduction in energy)
        # These are physics-based estimates from typical improvements
        self.measure_effects = {
            'wall_insulation': {
                'energy_reduction': 0.20,  # 20% reduction in envelope losses
                'carbon_reduction': 0.18,
                'features_affected': {
                    'WALLS_ENERGY_EFF_NUM': 4,  # Improve to Good
                    'ENVELOPE_QUALITY': 0.3  # Improvement
                }
            },
            'roof_insulation': {
                'energy_reduction': 0.10,
                'carbon_reduction': 0.09,
                'features_affected': {
                    'ROOF_ENERGY_EFF_NUM': 4,
                    'ENVELOPE_QUALITY': 0.2
                }
            },
            'floor_insulation': {
                'energy_reduction': 0.05,
                'carbon_reduction': 0.04,
                'features_affected': {
                    'FLOOR_ENERGY_EFF_NUM': 4,
                    'ENVELOPE_QUALITY': 0.1
                }
            },
            'windows': {
                'energy_reduction': 0.10,
                'carbon_reduction': 0.09,
                'features_affected': {
                    'WINDOWS_ENERGY_EFF_NUM': 4,
                    'GLAZED_TYPE_NUM': 3,
                    'ENVELOPE_QUALITY': 0.15
                }
            },
            'boiler': {
                'energy_reduction': 0.15,
                'carbon_reduction': 0.12,
                'features_affected': {
                    'MAINHEAT_ENERGY_EFF_NUM': 5,
                    'SYSTEM_EFFICIENCY': 0.3
                }
            },
            'heating_controls': {
                'energy_reduction': 0.05,
                'carbon_reduction': 0.04,
                'features_affected': {
                    'MAINHEATC_ENERGY_EFF_NUM': 4,
                    'SYSTEM_EFFICIENCY': 0.1
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
                    'RENEWABLE_FRACTION': 0.3
                }
            },
            'heat_pump': {
                'energy_reduction': 0.25,
                'carbon_reduction': 0.40,
                'features_affected': {
                    'MAINHEAT_ENERGY_EFF_NUM': 5,
                    'SYSTEM_EFFICIENCY': 0.5
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

    def _get_env_int(self, key: str, default: int) -> int:
        """Read integer environment variable with fallback."""
        raw = os.getenv(key)
        if raw is None:
            return default
        try:
            return int(raw)
        except ValueError:
            return default

    def _building_cache_key(self, building_profile: pd.Series) -> str:
        """Create stable cache key for baseline prediction of a building."""
        lmk_key = building_profile.get('LMK_KEY')
        if pd.notna(lmk_key):
            return f"lmk::{lmk_key}"

        safe_values = []
        for k in sorted(building_profile.index):
            v = building_profile.get(k)
            if pd.isna(v):
                v = 'nan'
            safe_values.append(f"{k}={v}")
        return 'hash::' + '|'.join(safe_values)

    def _get_baseline_state(self, building_profile: pd.Series) -> Dict[str, float]:
        """Get cached baseline/current state for a building."""
        cache_key = self._building_cache_key(building_profile)
        cached = self._baseline_cache.get(cache_key)
        if cached is not None:
            return cached

        current_energy = building_profile.get(
            'ENERGY_CONSUMPTION_CURRENT',
            building_profile.get('ENERGY_INTENSITY', 200)
        )
        current_carbon = building_profile.get('CO2_EMISS_CURR_PER_FLOOR_AREA', 40)

        heating = building_profile.get('HEATING_COST_CURRENT', 500)
        hot_water = building_profile.get('HOT_WATER_COST_CURRENT', 150)
        lighting = building_profile.get('LIGHTING_COST_CURRENT', 100)
        current_cost = building_profile.get('TOTAL_COST_CURRENT', heating + hot_water + lighting)

        if current_cost <= 0:
            floor_area = building_profile.get('TOTAL_FLOOR_AREA', 80)
            current_cost = floor_area * 12

        baseline = {
            'current_energy': float(current_energy),
            'current_carbon': float(current_carbon),
            'current_cost': float(current_cost)
        }
        self._baseline_cache[cache_key] = baseline
        return baseline

    def _prescreen_measures(
        self,
        building_profile: pd.Series,
        applicable_measures: List[RetrofitMeasure],
        top_k: int,
        target_type: str
    ) -> List[RetrofitMeasure]:
        """Reduce measure search-space to top-k by single-measure marginal gain."""
        if top_k <= 0 or len(applicable_measures) <= top_k:
            return applicable_measures

        scored = []
        for measure in applicable_measures:
            effects = self.estimate_improvement_effect(building_profile, [measure])
            target_reduction = effects.get(f'{target_type}_reduction_pct', 0.0)
            value_score = target_reduction / max(measure.cost_avg, 1.0)
            scored.append((value_score, target_reduction, measure))

        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return [m for _, _, m in scored[:top_k]]

    def generate_packages(
        self,
        building_profile: pd.Series,
        target_type: str = 'carbon',
        max_budget: Optional[float] = None,
        max_measures: Optional[int] = None,
        topk_measures: Optional[int] = None
    ) -> List[RetrofitPackage]:
        """Generate all candidate packages once, for query-many target filtering."""
        t0 = time.perf_counter()
        effective_max_measures = max_measures or self._get_env_int('MAX_MEASURES', 5)
        effective_topk = topk_measures or self._get_env_int('TOPK_MEASURES', 15)

        applicable_measures = list(self.recommendation_db.measures.values())
        if max_budget:
            applicable_measures = [m for m in applicable_measures if m.cost_min <= max_budget]

        t1 = time.perf_counter()
        reduced_measures = self._prescreen_measures(
            building_profile=building_profile,
            applicable_measures=applicable_measures,
            top_k=effective_topk,
            target_type=target_type
        )
        t2 = time.perf_counter()

        packages = []
        evaluated = 0
        max_n = min(effective_max_measures + 1, len(reduced_measures) + 1)
        for n in range(1, max_n):
            for measure_combo in combinations(reduced_measures, n):
                total_cost = sum(m.cost_avg for m in measure_combo)
                if max_budget and total_cost > max_budget:
                    continue

                effects = self.estimate_improvement_effect(building_profile, list(measure_combo))
                cost_savings = effects.get('annual_cost_savings', 0)

                package = RetrofitPackage(
                    measures=list(measure_combo),
                    predicted_energy_reduction=effects.get('energy_reduction_pct', 0),
                    predicted_carbon_reduction=effects.get('carbon_reduction_pct', 0),
                    predicted_cost_savings=cost_savings
                )
                if cost_savings > 0:
                    package.payback_years = package.total_cost_avg / cost_savings
                packages.append(package)
                evaluated += 1

        t3 = time.perf_counter()
        print(
            "[OptimizationEngine.generate_packages] "
            f"initial_measures={len(applicable_measures)} prescreened={len(reduced_measures)} "
            f"combinations_evaluated={evaluated} load_s={(t1-t0):.3f} "
            f"prescreen_s={(t2-t1):.3f} search_s={(t3-t2):.3f} total_s={(t3-t0):.3f}"
        )
        return packages
    
    def load_recommendations(self, df: pd.DataFrame):
        """Load recommendations data."""
        self.recommendation_db.load_from_dataframe(df)
    
    def estimate_improvement_effect(
        self,
        building_profile: pd.Series,
        measures: List[RetrofitMeasure]
    ) -> Dict[str, float]:
        """
        Estimate the effect of retrofit measures on a building.
        
        Uses physics-based estimates combined with model predictions
        when available.
        
        Args:
            building_profile: Series with building features
            measures: List of measures to apply
        
        Returns:
            Dictionary with estimated reductions
        """
        baseline = self._get_baseline_state(building_profile)
        current_energy = baseline['current_energy']
        current_carbon = baseline['current_carbon']
        current_cost = baseline['current_cost']
        
        # Calculate cumulative effect (not strictly additive due to diminishing returns)
        total_energy_reduction = 0
        total_carbon_reduction = 0
        
        applied_categories = set()
        
        for measure in measures:
            if measure.category in applied_categories:
                continue  # Don't double-count same category
            
            effects = self.measure_effects.get(measure.category, {})
            energy_red = effects.get('energy_reduction', 0.05)
            carbon_red = effects.get('carbon_reduction', 0.05)
            
            # Apply diminishing returns
            remaining_energy = 1 - total_energy_reduction
            remaining_carbon = 1 - total_carbon_reduction
            
            total_energy_reduction += energy_red * remaining_energy
            total_carbon_reduction += carbon_red * remaining_carbon
            
            applied_categories.add(measure.category)
        
        # Cap at realistic maximum
        total_energy_reduction = min(total_energy_reduction, 0.70)
        total_carbon_reduction = min(total_carbon_reduction, 0.80)
        
        # Estimate cost savings (proportional to energy reduction)
        cost_savings = current_cost * total_energy_reduction
        
        return {
            'energy_reduction_pct': total_energy_reduction * 100,
            'carbon_reduction_pct': total_carbon_reduction * 100,
            'new_energy_intensity': current_energy * (1 - total_energy_reduction),
            'new_carbon_intensity': current_carbon * (1 - total_carbon_reduction),
            'annual_cost_savings': cost_savings
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
        max_measures: Optional[int] = None
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
        effective_max_measures = max_measures or self._get_env_int('MAX_MEASURES', 5)

        packages = self.generate_packages(
            building_profile=building_profile,
            target_type=target_type,
            max_budget=max_budget,
            max_measures=effective_max_measures,
            topk_measures=self._get_env_int('TOPK_MEASURES', 15)
        )
        
        # Filter packages that meet target
        valid_packages = [
            p for p in packages 
            if (p.predicted_carbon_reduction if target_type == 'carbon' 
                else p.predicted_energy_reduction) >= target_reduction
        ]
        
        # Sort by cost (ascending)
        valid_packages.sort(key=lambda p: p.total_cost_avg)
        
        # If no packages meet target, return best available
        if not valid_packages:
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
            
            # Get predictions
            X = pd.DataFrame([profile])[self.model_factory.feature_columns]
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
                'Payback (years)': f"{pkg.payback_years:.1f}" if pkg.payback_years else "N/A"
            })
        
        return pd.DataFrame(data)
