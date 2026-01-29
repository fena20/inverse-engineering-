"""
Helper functions for the Retrofit DSS.
"""
import re
import numpy as np
from typing import Tuple, Optional, Union


def parse_cost_range(cost_string: str) -> Tuple[float, float]:
    """
    Parse indicative cost string to min/max values.
    
    Args:
        cost_string: Cost string like "£1,500 - £3,500" or "£38"
    
    Returns:
        Tuple of (min_cost, max_cost)
    """
    if not cost_string or not isinstance(cost_string, str):
        return (0.0, 0.0)
    
    # Remove currency symbol and commas
    cleaned = cost_string.replace('£', '').replace(',', '').strip()
    
    # Check for range
    if ' - ' in cleaned:
        parts = cleaned.split(' - ')
        try:
            min_cost = float(parts[0].strip())
            max_cost = float(parts[1].strip())
            return (min_cost, max_cost)
        except (ValueError, IndexError):
            return (0.0, 0.0)
    else:
        # Single value
        try:
            value = float(cleaned)
            return (value, value)
        except ValueError:
            return (0.0, 0.0)


def get_average_cost(cost_string: str) -> float:
    """Get average of cost range."""
    min_cost, max_cost = parse_cost_range(cost_string)
    return (min_cost + max_cost) / 2


def calculate_hdd(temperatures: np.ndarray, base_temp: float = 15.5) -> float:
    """
    Calculate Heating Degree Days from temperature array.
    
    Args:
        temperatures: Array of daily average temperatures (°C)
        base_temp: Base temperature for HDD calculation
    
    Returns:
        Total HDD for the period
    """
    diff = base_temp - temperatures
    return float(np.sum(np.maximum(diff, 0)))


def estimate_u_value_from_description(description: str) -> Optional[float]:
    """
    Estimate U-value from wall/roof description.
    
    Physics-based estimation based on typical construction types.
    
    Args:
        description: Description string from EPC
    
    Returns:
        Estimated U-value (W/m²K) or None
    """
    if not description or not isinstance(description, str):
        return None
    
    desc_lower = description.lower()
    
    # Wall U-values (W/m²K)
    if 'wall' in desc_lower or 'brick' in desc_lower or 'cavity' in desc_lower:
        if 'external insulation' in desc_lower:
            return 0.25
        elif 'internal insulation' in desc_lower:
            return 0.30
        elif 'filled cavity' in desc_lower:
            return 0.50
        elif 'insulated' in desc_lower:
            return 0.45
        elif 'partial insulation' in desc_lower:
            return 0.70
        elif 'no insulation' in desc_lower or 'as built' in desc_lower:
            if 'solid' in desc_lower:
                return 2.10  # Solid brick uninsulated
            else:
                return 1.50  # Cavity uninsulated
        elif 'solid brick' in desc_lower:
            return 2.10
        else:
            return 1.00  # Default moderate
    
    # Roof U-values
    if 'roof' in desc_lower or 'loft' in desc_lower or 'pitched' in desc_lower:
        # Extract insulation thickness if mentioned
        mm_match = re.search(r'(\d+)\s*mm', desc_lower)
        if mm_match:
            thickness_mm = int(mm_match.group(1))
            # Approximate U-value from loft insulation thickness
            # Assuming lambda = 0.040 W/mK for mineral wool
            if thickness_mm >= 300:
                return 0.11
            elif thickness_mm >= 250:
                return 0.14
            elif thickness_mm >= 200:
                return 0.16
            elif thickness_mm >= 150:
                return 0.19
            elif thickness_mm >= 100:
                return 0.25
            elif thickness_mm >= 50:
                return 0.40
            else:
                return 0.50
        elif 'no insulation' in desc_lower:
            return 2.30
        elif 'limited' in desc_lower:
            return 0.50
        elif 'very good' in desc_lower:
            return 0.11
        elif 'good' in desc_lower:
            return 0.16
        else:
            return 0.25  # Default moderate
    
    # Floor U-values
    if 'floor' in desc_lower or 'suspended' in desc_lower:
        if 'insulated' in desc_lower and 'no insulation' not in desc_lower:
            return 0.25
        elif 'no insulation' in desc_lower:
            return 0.70
        else:
            return 0.45
    
    return None


def estimate_window_u_value(glazed_type: str, installation_age: str = None) -> float:
    """
    Estimate window U-value from glazing type.
    
    Args:
        glazed_type: Type of glazing
        installation_age: When installed (before/after 2002)
    
    Returns:
        Estimated U-value (W/m²K)
    """
    if not glazed_type:
        return 2.8  # Default single
    
    gt_lower = str(glazed_type).lower()
    
    if 'triple' in gt_lower:
        return 1.0
    elif 'high performance' in gt_lower:
        return 1.2
    elif 'double' in gt_lower:
        if 'after 2002' in gt_lower or 'during or after' in gt_lower:
            return 1.6  # Low-E double
        else:
            return 2.2  # Older double
    elif 'secondary' in gt_lower:
        return 2.5
    elif 'single' in gt_lower:
        return 5.0
    else:
        return 2.8  # Default


def calculate_form_factor(floor_area: float, n_storeys: float = 2, 
                         property_type: str = 'House',
                         built_form: str = 'Semi-Detached') -> float:
    """
    Estimate building form factor (surface area / volume ratio).
    
    This is a simplified geometric estimation based on typical UK housing.
    
    Args:
        floor_area: Total floor area (m²)
        n_storeys: Number of storeys
        property_type: Type of property
        built_form: Built form (detached, semi, etc.)
    
    Returns:
        Estimated form factor (m⁻¹)
    """
    # Estimate footprint
    footprint = floor_area / max(n_storeys, 1)
    
    # Assume roughly square footprint
    side_length = np.sqrt(footprint)
    height = n_storeys * 2.5  # Average storey height
    
    # Calculate exposed surfaces based on built form
    form_lower = str(built_form).lower() if built_form else ''
    
    if 'detached' in form_lower:
        # All 4 walls exposed
        wall_area = 4 * side_length * height
    elif 'semi' in form_lower:
        # 3 walls exposed
        wall_area = 3 * side_length * height
    elif 'end' in form_lower:
        # 3 walls exposed
        wall_area = 3 * side_length * height
    elif 'mid' in form_lower:
        # 2 walls exposed (front and back)
        wall_area = 2 * side_length * height
    else:
        wall_area = 3 * side_length * height  # Default semi
    
    # Roof area (assuming pitched roof adds ~10%)
    roof_area = footprint * 1.1
    
    # Floor area (ground floor for heat loss)
    ground_floor = footprint
    
    # Total surface area
    total_surface = wall_area + roof_area + ground_floor
    
    # Volume
    volume = floor_area * 2.5
    
    return total_surface / volume if volume > 0 else 0.5


def efficiency_to_numeric(rating: str) -> int:
    """
    Convert efficiency rating string to numeric value.
    
    Args:
        rating: Rating string ('Very Poor' to 'Very Good')
    
    Returns:
        Numeric rating (1-5, 0 for N/A)
    """
    from .constants import EFFICIENCY_RATINGS
    return EFFICIENCY_RATINGS.get(rating, 0)


def epc_grade_to_numeric(grade: str) -> int:
    """
    Convert EPC grade to numeric value.
    
    Args:
        grade: Grade string ('A' to 'G')
    
    Returns:
        Numeric grade (1-7)
    """
    from .constants import EPC_GRADES
    return EPC_GRADES.get(grade, 0)


def validate_physical_consistency(
    wall_eff: int, 
    energy_consumption: float,
    wall_eff_improved: int,
    energy_consumption_predicted: float
) -> bool:
    """
    Validate that improving wall efficiency reduces energy consumption.
    
    This is a physics sanity check - better insulation should reduce heat loss.
    
    Args:
        wall_eff: Original wall efficiency rating
        energy_consumption: Original energy consumption
        wall_eff_improved: Improved wall efficiency rating
        energy_consumption_predicted: Predicted energy consumption after improvement
    
    Returns:
        True if physically consistent, False otherwise
    """
    if wall_eff_improved > wall_eff:
        # Improvement made, energy should decrease or stay same
        return energy_consumption_predicted <= energy_consumption * 1.05  # 5% tolerance
    return True
