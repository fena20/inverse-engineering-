# Scientific & Methodological Review of Retrofit DSS

This document provides a detailed scientific and methodological review of the "Retrofit Decision Support System" repository. The review evaluates the data handling, modeling approach, optimization strategy, and overall code quality from a research and engineering perspective.

## 1. Data Methodology

### 1.1. Data Loading & Integration (`loader.py`)
-   **Multi-City Support**: The `DataLoader` class is designed to handle data from multiple UK cities (Cambridge, Boston, Liverpool, Sheffield), which improves the generalizability of the models.
-   **Climate Integration**: Integrating weather data (HDD, solar radiation) from Open-Meteo is a strong methodological step, allowing the model to distinguish between building efficiency and climate-driven energy consumption.
-   **Data Merging**: Certificates are correctly merged with recommendations, providing a ground truth for retrofit measures.

### 1.2. Preprocessing & Feature Engineering (`preprocessor.py`)
-   **Physics-Informed Encoding**: The `DataPreprocessor` class uses domain knowledge to encode categorical variables. For example, efficiency ratings (Very Good, Good, etc.) are mapped to ordinal values (5, 4, etc.) rather than one-hot encoding. This preserves the ordinal nature of the data and helps the tree-based models.
-   **Derived Physics Features**: The addition of features like `ENVELOPE_QUALITY`, `HEAT_LOSS_PROXY`, `FORM_FACTOR`, and `SYSTEM_EFFICIENCY` is excellent. These features bridge the gap between raw data and physical reality, likely improving model performance and interpretability.
-   **Missing Value Imputation**: Missing values are largely handled by filling with median values. While simple and robust, this might introduce bias, particularly for physical characteristics where "missing" might imply "not present" or "unknown/average". A more sophisticated imputation (e.g., KNN or iterative imputer) could be explored.
-   **Train/Test Split**: The use of `GroupShuffleSplit` on postcode sectors is a critical methodological success. This prevents **spatial data leakage**, where models might learn location-specific proxies (like neighborhood wealth or specific housing developments) rather than building physics.

## 2. Modeling Methodology (`surrogate.py`)

### 2.1. Surrogate Model Architecture
-   **Algorithm Choice**: The use of Gradient Boosting (XGBoost/sklearn) is appropriate for this type of tabular, heterogeneous data. Tree-based models handle non-linear interactions and missing values well.
-   **Modular Design**: The `SurrogateModelFactory` allows for easy switching between model types (Random Forest, XGBoost, Ridge), facilitating comparison.
-   **Target Variables**: Separate models for Energy (`kWh/m²`), Carbon (`kgCO₂/m²`), and Cost allow for multi-objective optimization, which is valuable for decision support.

### 2.2. Validation & Interpretability
-   **Metrics**: Standard regression metrics (MAE, RMSE, R²) are used.
-   **Physical Consistency**: The `validate_physical_intuition` method checks if the most important features align with building physics (e.g., Wall U-value should be important). This is a great "sanity check" for the "black box" model.
-   **Constraint Validation**: The `validate_predictions` method ensures predictions are physically possible (e.g., positive energy consumption, better walls = lower energy). This builds trust in the surrogate model.

## 3. Optimization Methodology (`engine.py`)

### 3.1. Optimization Strategy
-   **Approach**: The system uses a **Discrete Optimization** approach (enumerating combinations) rather than continuous optimization. This is appropriate given that retrofit measures are discrete choices (e.g., "Install Wall Insulation" vs "Do Nothing").
-   **Heuristic vs. Model-Based**: **[CRITICAL FINDING]** The optimization engine relies on a hardcoded dictionary (`measure_effects`) to estimate the impact of retrofits (e.g., "Wall Insulation reduces energy by 20%").
    -   *Limitation*: It **does not** fully utilize the trained surrogate model to predict the post-retrofit performance in the optimization loop. Instead, it applies a rule-based reduction factor.
    -   *Recommendation*: The methodology should be upgraded to "Inverse Design" or true "Surrogate-Based Optimization". The engine should modify the feature vector (e.g., change `WALLS_ENERGY_EFF_NUM` from 2 to 4) and query the surrogate model to predict the new `ENERGY_CONSUMPTION`. This would capture non-linear interactions specific to that building (e.g., wall insulation might be less effective if the roof is terrible) that the hardcoded percentages miss.

### 3.2. Cost-Benefit Analysis
-   **Diminishing Returns**: The engine correctly implements a logic for diminishing returns (e.g., `total_reduction += new_reduction * remaining_energy`), avoiding the trap of simply adding percentages (20% + 20% ≠ 40%).
-   **Payback Calculation**: Simple payback period is calculated. Integrating Net Present Value (NPV) would add more economic rigor.

## 4. Code Quality & Reproducibility

-   **Structure**: The project is well-structured with clear separation of concerns (`data`, `models`, `optimization`).
-   **Reproducibility**: Random seeds are set in training and splitting, ensuring results can be replicated.
-   **Documentation**: The code is well-commented with docstrings explaining the purpose of classes and methods.
-   **Dependencies**: `requirements.txt` is provided, though exact versions are pinned with `>=` which is good for compatibility but might lead to minor differences over time.

## 5. Summary & Recommendations

### Strengths
1.  **Physics-Informed ML**: Strong integration of domain knowledge into feature engineering.
2.  **Rigorous Validation**: Spatially aware cross-validation prevents leakage.
3.  **Physical Consistency Checks**: Ensures the model behaves rationally.

### Areas for Improvement
1.  **Optimization Logic**: Move from rule-based estimation (`measure_effects`) to true model-based prediction (modifying features and querying the surrogate).
2.  **Imputation**: Explore more advanced imputation for missing physical data.
3.  **Uncertainty Quantification**: The current models provide point estimates. Adding prediction intervals (e.g., Quantile Regression) would be valuable for decision support.
