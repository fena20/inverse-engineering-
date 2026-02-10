"""
Optimization engine for retrofit recommendations.

Provides discrete optimization to find cost-effective retrofit packages,
plus scenario-oriented planning helpers for cost/carbon trade-off analysis.
"""
import os
import time
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Dict, List, Optional

import pandas as pd

from ..utils.constants import IMPROVEMENT_CATEGORIES
from ..utils.helpers import parse_cost_range


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
            "id": self.id,
            "name": self.name,
            "category": self.category,
            "description": self.description,
            "cost_range": f"£{self.cost_min:,.0f} - £{self.cost_max:,.0f}",
            "cost_average": self.cost_avg,
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
            "measures": [m.to_dict() for m in self.measures],
            "total_cost_range": f"£{self.total_cost_min:,.0f} - £{self.total_cost_max:,.0f}",
            "total_cost_average": self.total_cost_avg,
            "predicted_energy_reduction_pct": self.predicted_energy_reduction,
            "predicted_carbon_reduction_pct": self.predicted_carbon_reduction,
            "predicted_annual_savings": self.predicted_cost_savings,
            "payback_years": self.payback_years,
        }


class RecommendationDatabase:
    """Database of retrofit recommendations from EPC data."""

    def __init__(self):
        self.measures: Dict[int, RetrofitMeasure] = {}
        self._categories_map: Dict[int, str] = {}
        self._build_categories_map()

    def _build_categories_map(self):
        for category, ids in IMPROVEMENT_CATEGORIES.items():
            for imp_id in ids:
                self._categories_map[imp_id] = category

    def load_from_dataframe(self, df: pd.DataFrame):
        grouped = (
            df.groupby("IMPROVEMENT_ID")
            .agg(
                {
                    "IMPROVEMENT_SUMMARY_TEXT": "first",
                    "IMPROVEMENT_DESCR_TEXT": "first",
                    "INDICATIVE_COST": lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else "",
                }
            )
            .reset_index()
        )

        for _, row in grouped.iterrows():
            imp_id = int(row["IMPROVEMENT_ID"]) if pd.notna(row["IMPROVEMENT_ID"]) else 0
            if imp_id <= 0:
                continue

            cost_min, cost_max = parse_cost_range(row["INDICATIVE_COST"])
            category = self._categories_map.get(imp_id, "other")
            self.measures[imp_id] = RetrofitMeasure(
                id=imp_id,
                name=str(row["IMPROVEMENT_SUMMARY_TEXT"]) if pd.notna(row["IMPROVEMENT_SUMMARY_TEXT"]) else "",
                category=category,
                description=str(row["IMPROVEMENT_DESCR_TEXT"]) if pd.notna(row["IMPROVEMENT_DESCR_TEXT"]) else "",
                cost_min=cost_min,
                cost_max=cost_max,
            )

        print(f"Loaded {len(self.measures)} unique retrofit measures")

    def get_measure(self, imp_id: int) -> Optional[RetrofitMeasure]:
        return self.measures.get(imp_id)


class OptimizationEngine:
    """Discrete optimizer with category-aware pruning and scenario presets."""

    SCENARIO_PRESETS: Dict[str, Dict[str, Any]] = {
        "fast_upgrade_epc_c": {
            "target_type": "carbon",
            "target_reduction": 15.0,
            "max_budget": 2500,
            "max_measures": 2,
            "focus_categories": ["lighting", "heating_controls", "draught_proofing"],
        },
        "solid_wall_first": {
            "target_type": "carbon",
            "target_reduction": 35.0,
            "max_budget": 18000,
            "max_measures": 3,
            "focus_categories": ["wall_insulation", "roof_insulation", "windows"],
        },
        "electrification_ashp": {
            "target_type": "carbon",
            "target_reduction": 65.0,
            "max_budget": 10000,
            "max_measures": 3,
            "focus_categories": ["heat_pump", "heating_controls", "solar_thermal"],
        },
        "hybrid_path": {
            "target_type": "carbon",
            "target_reduction": 50.0,
            "max_budget": 12000,
            "max_measures": 4,
            "focus_categories": ["boiler", "heating_controls", "wall_insulation", "solar_pv"],
        },
        "nzeb": {
            "target_type": "carbon",
            "target_reduction": 85.0,
            "max_budget": 80000,
            "max_measures": 6,
            "focus_categories": ["wall_insulation", "roof_insulation", "windows", "heat_pump", "solar_pv"],
        },
        "prosumer": {
            "target_type": "carbon",
            "target_reduction": 100.0,
            "max_budget": 18000,
            "max_measures": 5,
            "focus_categories": ["solar_pv", "heat_pump", "lighting", "heating_controls"],
        },
    }

    def __init__(self, model_factory=None):
        self.model_factory = model_factory
        self.recommendation_db = RecommendationDatabase()
        self._baseline_cache: Dict[str, Dict[str, float]] = {}

        self.measure_effects = {
            "wall_insulation": {"energy_reduction": 0.20, "carbon_reduction": 0.18},
            "roof_insulation": {"energy_reduction": 0.10, "carbon_reduction": 0.09},
            "floor_insulation": {"energy_reduction": 0.05, "carbon_reduction": 0.04},
            "windows": {"energy_reduction": 0.10, "carbon_reduction": 0.09},
            "boiler": {"energy_reduction": 0.15, "carbon_reduction": 0.12},
            "heating_controls": {"energy_reduction": 0.05, "carbon_reduction": 0.04},
            "lighting": {"energy_reduction": 0.03, "carbon_reduction": 0.03},
            "solar_thermal": {"energy_reduction": 0.05, "carbon_reduction": 0.05},
            "solar_pv": {"energy_reduction": 0.15, "carbon_reduction": 0.20},
            "heat_pump": {"energy_reduction": 0.25, "carbon_reduction": 0.40},
            "draught_proofing": {"energy_reduction": 0.03, "carbon_reduction": 0.02},
        }

    def _get_env_int(self, key: str, default: int) -> int:
        raw = os.getenv(key)
        if raw is None:
            return default
        try:
            return int(raw)
        except ValueError:
            return default

    def _building_cache_key(self, building_profile: pd.Series) -> str:
        lmk_key = building_profile.get("LMK_KEY")
        if pd.notna(lmk_key):
            return f"lmk::{lmk_key}"

        safe_values = []
        for k in sorted(building_profile.index):
            v = building_profile.get(k)
            safe_values.append(f"{k}={'nan' if pd.isna(v) else v}")
        return "hash::" + "|".join(safe_values)

    def _get_baseline_state(self, building_profile: pd.Series) -> Dict[str, float]:
        cache_key = self._building_cache_key(building_profile)
        cached = self._baseline_cache.get(cache_key)
        if cached is not None:
            return cached

        current_energy = building_profile.get("ENERGY_CONSUMPTION_CURRENT", building_profile.get("ENERGY_INTENSITY", 200))
        current_carbon = building_profile.get("CO2_EMISS_CURR_PER_FLOOR_AREA", 40)
        heating = building_profile.get("HEATING_COST_CURRENT", 500)
        hot_water = building_profile.get("HOT_WATER_COST_CURRENT", 150)
        lighting = building_profile.get("LIGHTING_COST_CURRENT", 100)
        current_cost = building_profile.get("TOTAL_COST_CURRENT", heating + hot_water + lighting)

        if current_cost <= 0:
            current_cost = building_profile.get("TOTAL_FLOOR_AREA", 80) * 12

        baseline = {
            "current_energy": float(current_energy),
            "current_carbon": float(current_carbon),
            "current_cost": float(current_cost),
        }
        self._baseline_cache[cache_key] = baseline
        return baseline

    def _measure_gain(self, building_profile: pd.Series, measure: RetrofitMeasure, target_type: str) -> float:
        effects = self.estimate_improvement_effect(building_profile, [measure])
        return effects.get(f"{target_type}_reduction_pct", 0.0)

    def _prescreen_measures(
        self,
        building_profile: pd.Series,
        applicable_measures: List[RetrofitMeasure],
        top_k: int,
        target_type: str,
        focus_categories: Optional[List[str]] = None,
    ) -> List[RetrofitMeasure]:
        if top_k <= 0 or len(applicable_measures) <= top_k:
            return applicable_measures

        scored: List[Dict[str, Any]] = []
        focus_set = set(focus_categories or [])

        for measure in applicable_measures:
            gain = self._measure_gain(building_profile, measure, target_type)
            value_score = gain / max(measure.cost_avg, 1.0)
            focus_boost = 1.2 if measure.category in focus_set else 1.0
            scored.append(
                {
                    "measure": measure,
                    "category": measure.category,
                    "gain": gain,
                    "score": value_score * focus_boost,
                }
            )

        scored.sort(key=lambda x: (x["score"], x["gain"]), reverse=True)

        # category-aware selection: keep best per category first, then fill remainder
        selected: List[RetrofitMeasure] = []
        seen_categories = set()

        for item in scored:
            if item["category"] not in seen_categories:
                selected.append(item["measure"])
                seen_categories.add(item["category"])
            if len(selected) >= top_k:
                return selected

        for item in scored:
            if item["measure"] in selected:
                continue
            selected.append(item["measure"])
            if len(selected) >= top_k:
                break

        return selected

    def _is_dominated(self, candidate: RetrofitPackage, accepted: List[RetrofitPackage], target_type: str) -> bool:
        cand_target = candidate.predicted_carbon_reduction if target_type == "carbon" else candidate.predicted_energy_reduction
        for pkg in accepted:
            ref_target = pkg.predicted_carbon_reduction if target_type == "carbon" else pkg.predicted_energy_reduction
            if pkg.total_cost_avg <= candidate.total_cost_avg and ref_target >= cand_target:
                if pkg.total_cost_avg < candidate.total_cost_avg or ref_target > cand_target:
                    return True
        return False

    def pareto_front(self, packages: List[RetrofitPackage], target_type: str = "carbon") -> List[RetrofitPackage]:
        ordered = sorted(packages, key=lambda p: (p.total_cost_avg, -p.predicted_carbon_reduction, -p.predicted_energy_reduction))
        front: List[RetrofitPackage] = []
        for pkg in ordered:
            if not self._is_dominated(pkg, front, target_type):
                front.append(pkg)
        return front

    def generate_packages(
        self,
        building_profile: pd.Series,
        target_type: str = "carbon",
        max_budget: Optional[float] = None,
        max_measures: Optional[int] = None,
        topk_measures: Optional[int] = None,
        focus_categories: Optional[List[str]] = None,
    ) -> List[RetrofitPackage]:
        t0 = time.perf_counter()
        effective_max_measures = max_measures or self._get_env_int("MAX_MEASURES", 5)
        effective_topk = topk_measures or self._get_env_int("TOPK_MEASURES", 15)

        applicable_measures = list(self.recommendation_db.measures.values())
        if max_budget:
            applicable_measures = [m for m in applicable_measures if m.cost_min <= max_budget]

        reduced_measures = self._prescreen_measures(
            building_profile=building_profile,
            applicable_measures=applicable_measures,
            top_k=effective_topk,
            target_type=target_type,
            focus_categories=focus_categories,
        )

        # one measure per category prevents redundant combinations and speeds search
        by_category: Dict[str, List[RetrofitMeasure]] = {}
        for m in reduced_measures:
            by_category.setdefault(m.category, []).append(m)

        packages: List[RetrofitPackage] = []
        evaluated = 0
        categories = list(by_category.keys())
        max_n = min(effective_max_measures, len(categories))

        for n in range(1, max_n + 1):
            for selected_categories in combinations(categories, n):
                pools = [by_category[c] for c in selected_categories]

                def _search(idx: int, picked: List[RetrofitMeasure]):
                    nonlocal evaluated
                    if idx == len(pools):
                        total_cost = sum(m.cost_avg for m in picked)
                        if max_budget and total_cost > max_budget:
                            return
                        effects = self.estimate_improvement_effect(building_profile, picked)
                        pkg = RetrofitPackage(
                            measures=list(picked),
                            predicted_energy_reduction=effects.get("energy_reduction_pct", 0),
                            predicted_carbon_reduction=effects.get("carbon_reduction_pct", 0),
                            predicted_cost_savings=effects.get("annual_cost_savings", 0),
                        )
                        if pkg.predicted_cost_savings > 0:
                            pkg.payback_years = pkg.total_cost_avg / pkg.predicted_cost_savings
                        packages.append(pkg)
                        evaluated += 1
                        return

                    for m in pools[idx]:
                        picked.append(m)
                        _search(idx + 1, picked)
                        picked.pop()

                _search(0, [])

        pruned = self.pareto_front(packages, target_type=target_type)
        t1 = time.perf_counter()
        print(
            "[OptimizationEngine.generate_packages] "
            f"initial_measures={len(applicable_measures)} prescreened={len(reduced_measures)} "
            f"combinations_evaluated={evaluated} pareto={len(pruned)} total_s={(t1 - t0):.3f}"
        )
        return pruned

    def load_recommendations(self, df: pd.DataFrame):
        self.recommendation_db.load_from_dataframe(df)

    def estimate_improvement_effect(self, building_profile: pd.Series, measures: List[RetrofitMeasure]) -> Dict[str, float]:
        baseline = self._get_baseline_state(building_profile)
        total_energy_reduction = 0.0
        total_carbon_reduction = 0.0
        applied_categories = set()

        for measure in measures:
            if measure.category in applied_categories:
                continue
            effects = self.measure_effects.get(measure.category, {})
            energy_red = effects.get("energy_reduction", 0.05)
            carbon_red = effects.get("carbon_reduction", 0.05)

            total_energy_reduction += energy_red * (1 - total_energy_reduction)
            total_carbon_reduction += carbon_red * (1 - total_carbon_reduction)
            applied_categories.add(measure.category)

        total_energy_reduction = min(total_energy_reduction, 0.70)
        total_carbon_reduction = min(total_carbon_reduction, 0.95)
        cost_savings = baseline["current_cost"] * total_energy_reduction

        return {
            "energy_reduction_pct": total_energy_reduction * 100,
            "carbon_reduction_pct": total_carbon_reduction * 100,
            "new_energy_intensity": baseline["current_energy"] * (1 - total_energy_reduction),
            "new_carbon_intensity": baseline["current_carbon"] * (1 - total_carbon_reduction),
            "annual_cost_savings": cost_savings,
        }

    def get_applicable_measures(self, building_profile: pd.Series, recommendations_df: pd.DataFrame) -> List[RetrofitMeasure]:
        lmk_key = building_profile.get("LMK_KEY")
        if lmk_key is None or recommendations_df is None:
            return self._get_generic_measures(building_profile)

        property_recs = recommendations_df[recommendations_df["LMK_KEY"] == lmk_key]
        measures: List[RetrofitMeasure] = []
        for _, rec in property_recs.iterrows():
            imp_id = int(rec["IMPROVEMENT_ID"]) if pd.notna(rec["IMPROVEMENT_ID"]) else 0
            measure = self.recommendation_db.get_measure(imp_id)
            if measure is None and imp_id > 0:
                cost_min, cost_max = parse_cost_range(rec.get("INDICATIVE_COST", ""))
                measure = RetrofitMeasure(
                    id=imp_id,
                    name=str(rec.get("IMPROVEMENT_SUMMARY_TEXT", "")),
                    category=self.recommendation_db._categories_map.get(imp_id, "other"),
                    description=str(rec.get("IMPROVEMENT_DESCR_TEXT", "")),
                    cost_min=cost_min,
                    cost_max=cost_max,
                )
            if measure:
                measures.append(measure)
        return measures

    def _get_generic_measures(self, building_profile: pd.Series) -> List[RetrofitMeasure]:
        measures = []

        wall_eff = building_profile.get("WALLS_ENERGY_EFF_NUM", 3)
        if wall_eff < 4:
            for imp_id in IMPROVEMENT_CATEGORIES.get("wall_insulation", []):
                m = self.recommendation_db.get_measure(imp_id)
                if m:
                    measures.append(m)
                    break

        roof_eff = building_profile.get("ROOF_ENERGY_EFF_NUM", 3)
        if roof_eff < 4:
            for imp_id in IMPROVEMENT_CATEGORIES.get("roof_insulation", []):
                m = self.recommendation_db.get_measure(imp_id)
                if m:
                    measures.append(m)
                    break

        heat_eff = building_profile.get("MAINHEAT_ENERGY_EFF_NUM", 3)
        if heat_eff < 4:
            for imp_id in IMPROVEMENT_CATEGORIES.get("boiler", []):
                m = self.recommendation_db.get_measure(imp_id)
                if m:
                    measures.append(m)
                    break

        lighting_pct = building_profile.get("LOW_ENERGY_LIGHTING", 50)
        if lighting_pct < 80:
            for imp_id in IMPROVEMENT_CATEGORIES.get("lighting", []):
                m = self.recommendation_db.get_measure(imp_id)
                if m:
                    measures.append(m)
                    break

        for imp_id in IMPROVEMENT_CATEGORIES.get("solar_pv", []):
            m = self.recommendation_db.get_measure(imp_id)
            if m:
                measures.append(m)
                break

        return measures

    def optimize(
        self,
        building_profile: pd.Series,
        target_type: str = "carbon",
        target_reduction: float = 50.0,
        max_budget: Optional[float] = None,
        max_measures: Optional[int] = None,
    ) -> List[RetrofitPackage]:
        packages = self.generate_packages(
            building_profile=building_profile,
            target_type=target_type,
            max_budget=max_budget,
            max_measures=max_measures or self._get_env_int("MAX_MEASURES", 5),
            topk_measures=self._get_env_int("TOPK_MEASURES", 15),
        )

        valid_packages = [
            p
            for p in packages
            if (p.predicted_carbon_reduction if target_type == "carbon" else p.predicted_energy_reduction) >= target_reduction
        ]
        valid_packages.sort(key=lambda p: p.total_cost_avg)

        if not valid_packages:
            packages.sort(
                key=lambda p: (p.predicted_carbon_reduction if target_type == "carbon" else p.predicted_energy_reduction),
                reverse=True,
            )
            return packages[:5]

        return valid_packages[:10]

    def optimize_with_preset(self, building_profile: pd.Series, preset_name: str) -> List[RetrofitPackage]:
        preset = self.SCENARIO_PRESETS.get(preset_name)
        if preset is None:
            raise ValueError(f"Unknown preset: {preset_name}")

        packages = self.generate_packages(
            building_profile=building_profile,
            target_type=preset["target_type"],
            max_budget=preset.get("max_budget"),
            max_measures=preset.get("max_measures"),
            topk_measures=self._get_env_int("TOPK_MEASURES", 15),
            focus_categories=preset.get("focus_categories"),
        )
        valid = [
            p for p in packages
            if (p.predicted_carbon_reduction if preset["target_type"] == "carbon" else p.predicted_energy_reduction)
            >= preset["target_reduction"]
        ]
        valid.sort(key=lambda p: p.total_cost_avg)
        return valid[:10] if valid else packages[:5]

    def build_scenario_summary(self, building_profile: pd.Series, scenario_map: Optional[Dict[str, Dict[str, Any]]] = None) -> pd.DataFrame:
        scenarios = scenario_map or self.SCENARIO_PRESETS
        rows = []
        for name, cfg in scenarios.items():
            packages = self.generate_packages(
                building_profile=building_profile,
                target_type=cfg["target_type"],
                max_budget=cfg.get("max_budget"),
                max_measures=cfg.get("max_measures"),
                topk_measures=self._get_env_int("TOPK_MEASURES", 15),
                focus_categories=cfg.get("focus_categories"),
            )
            valid = [
                p
                for p in packages
                if (p.predicted_carbon_reduction if cfg["target_type"] == "carbon" else p.predicted_energy_reduction)
                >= cfg["target_reduction"]
            ]
            best = min(valid, key=lambda p: p.total_cost_avg) if valid else (packages[0] if packages else None)
            rows.append(
                {
                    "scenario": name,
                    "target_type": cfg["target_type"],
                    "target_reduction_pct": cfg["target_reduction"],
                    "max_budget": cfg.get("max_budget"),
                    "max_measures": cfg.get("max_measures"),
                    "feasible": bool(valid),
                    "best_cost_avg": best.total_cost_avg if best else None,
                    "best_energy_reduction_pct": best.predicted_energy_reduction if best else None,
                    "best_carbon_reduction_pct": best.predicted_carbon_reduction if best else None,
                    "best_payback_years": best.payback_years if best else None,
                    "best_measures": ", ".join(m.name for m in best.measures) if best else "",
                }
            )
        return pd.DataFrame(rows)

    def sensitivity_analysis(self, building_profile: pd.Series, feature: str, values: List[Any]) -> pd.DataFrame:
        if self.model_factory is None:
            raise ValueError("Model factory required for sensitivity analysis")

        results = []
        for value in values:
            profile = building_profile.copy()
            profile[feature] = value
            X = pd.DataFrame([profile])[self.model_factory.feature_columns]
            predictions = self.model_factory.predict(X)
            result = {"feature_value": value}
            for name, pred in predictions.items():
                result[f"predicted_{name}"] = pred[0]
            results.append(result)

        return pd.DataFrame(results)

    def get_cost_benefit_summary(self, packages: List[RetrofitPackage]) -> pd.DataFrame:
        data = []
        for i, pkg in enumerate(packages):
            data.append(
                {
                    "Package": i + 1,
                    "Measures": ", ".join(m.name for m in pkg.measures),
                    "Cost (min)": f"£{pkg.total_cost_min:,.0f}",
                    "Cost (max)": f"£{pkg.total_cost_max:,.0f}",
                    "Energy Reduction": f"{pkg.predicted_energy_reduction:.1f}%",
                    "Carbon Reduction": f"{pkg.predicted_carbon_reduction:.1f}%",
                    "Annual Savings": f"£{pkg.predicted_cost_savings:,.0f}",
                    "Payback (years)": f"{pkg.payback_years:.1f}" if pkg.payback_years else "N/A",
                }
            )
        return pd.DataFrame(data)
