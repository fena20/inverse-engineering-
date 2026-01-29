"""
Flask API for Retrofit Decision Support System.

Provides REST endpoints for:
- /evaluate: Building performance evaluation
- /optimize: Retrofit optimization recommendations
- /sensitivity: Sensitivity analysis for engineering decisions
"""
import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

from ..data.loader import DataLoader
from ..data.preprocessor import DataPreprocessor
from ..models.surrogate import SurrogateModelFactory
from ..optimization.engine import OptimizationEngine
from ..utils.constants import EFFICIENCY_RATINGS, EPC_GRADES


def create_app(config: Optional[Dict] = None) -> Flask:
    """
    Create and configure the Flask application.
    
    Args:
        config: Optional configuration dictionary
    
    Returns:
        Configured Flask application
    """
    app = Flask(__name__)
    CORS(app)
    
    # Default configuration
    app.config['DATA_DIR'] = config.get('data_dir', 'data') if config else 'data'
    app.config['MODEL_DIR'] = config.get('model_dir', 'models') if config else 'models'
    
    # Initialize components (lazy loading)
    app.dss_components = {
        'loader': None,
        'preprocessor': None,
        'model_factory': None,
        'optimizer': None,
        'recommendations_df': None,
        'initialized': False
    }
    
    def initialize_components():
        """Lazy initialization of DSS components."""
        if app.dss_components['initialized']:
            return True
        
        try:
            # Load data
            loader = DataLoader(app.config['DATA_DIR'])
            loader.discover_cities()
            certs_df, recs_df = loader.get_merged_data()
            
            # Preprocess
            preprocessor = DataPreprocessor()
            processed_df = preprocessor.fit_transform(certs_df)
            
            # Initialize models
            model_factory = SurrogateModelFactory('gradient_boosting')
            model_factory.create_all_models()
            
            # Check if models exist, otherwise train
            model_path = os.path.join(app.config['MODEL_DIR'], 'energy_model.joblib')
            if os.path.exists(model_path):
                model_factory.load_all(app.config['MODEL_DIR'])
            else:
                # Train models on subset for speed
                train_df = processed_df.sample(min(50000, len(processed_df)), random_state=42)
                model_factory.fit_all(train_df, preprocessor.get_feature_columns())
            
            # Initialize optimizer
            optimizer = OptimizationEngine(model_factory)
            optimizer.load_recommendations(recs_df)
            
            # Store components
            app.dss_components['loader'] = loader
            app.dss_components['preprocessor'] = preprocessor
            app.dss_components['model_factory'] = model_factory
            app.dss_components['optimizer'] = optimizer
            app.dss_components['recommendations_df'] = recs_df
            app.dss_components['initialized'] = True
            
            return True
        
        except Exception as e:
            print(f"Error initializing components: {e}")
            return False
    
    @app.route('/health', methods=['GET'])
    def health_check():
        """Health check endpoint."""
        return jsonify({
            'status': 'healthy',
            'initialized': app.dss_components['initialized']
        })
    
    @app.route('/initialize', methods=['POST'])
    def initialize():
        """
        Initialize the DSS components.
        
        This endpoint triggers loading of data and models.
        """
        success = initialize_components()
        
        if success:
            return jsonify({
                'status': 'success',
                'message': 'DSS components initialized',
                'cities': app.dss_components['loader'].cities
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Failed to initialize components'
            }), 500
    
    @app.route('/evaluate', methods=['POST'])
    def evaluate():
        """
        Evaluate building performance.
        
        POST /evaluate
        
        Request body:
        {
            "building_profile": {
                "TOTAL_FLOOR_AREA": 100.0,
                "WALLS_ENERGY_EFF": "Poor",
                "ROOF_ENERGY_EFF": "Average",
                "WINDOWS_ENERGY_EFF": "Average",
                "MAINHEAT_ENERGY_EFF": "Good",
                "CONSTRUCTION_AGE_BAND": "1930-1949",
                "PROPERTY_TYPE": "House",
                "BUILT_FORM": "Semi-Detached",
                "CITY": "Cambridge"
                ...
            }
        }
        
        Response:
        {
            "energy_intensity_kwh_m2": 180.5,
            "carbon_intensity_kg_m2": 35.2,
            "heating_cost": 850.0,
            "hot_water_cost": 200.0,
            "lighting_cost": 120.0,
            "total_annual_cost": 1170.0,
            "epc_grade_estimate": "D",
            "comparison": {
                "city_average": 195.0,
                "percentile": 35
            }
        }
        """
        if not app.dss_components['initialized']:
            if not initialize_components():
                return jsonify({'error': 'DSS not initialized'}), 500
        
        try:
            data = request.get_json()
            if not data or 'building_profile' not in data:
                return jsonify({'error': 'Missing building_profile in request'}), 400
            
            profile = data['building_profile']
            
            # Convert to DataFrame and preprocess
            df = pd.DataFrame([profile])
            
            # Apply encoding
            preprocessor = app.dss_components['preprocessor']
            processed = preprocessor.transform(df)
            
            # Get predictions
            model_factory = app.dss_components['model_factory']
            feature_cols = preprocessor.get_feature_columns()
            
            # Ensure all feature columns exist
            for col in feature_cols:
                if col not in processed.columns:
                    processed[col] = 0
            
            X = processed[feature_cols]
            predictions = model_factory.predict(X)
            
            # Calculate total cost
            heating = predictions.get('heating_cost', [0])[0]
            hot_water = predictions.get('hot_water_cost', [0])[0]
            lighting = predictions.get('lighting_cost', [0])[0]
            total_cost = heating + hot_water + lighting
            
            # Estimate EPC grade from energy efficiency score
            energy_intensity = predictions.get('energy', [200])[0]
            epc_grade = estimate_epc_grade(energy_intensity)
            
            response = {
                'energy_intensity_kwh_m2': round(float(predictions.get('energy', [0])[0]), 1),
                'carbon_intensity_kg_m2': round(float(predictions.get('carbon', [0])[0]), 1),
                'heating_cost': round(float(heating), 2),
                'hot_water_cost': round(float(hot_water), 2),
                'lighting_cost': round(float(lighting), 2),
                'total_annual_cost': round(float(total_cost), 2),
                'epc_grade_estimate': epc_grade,
                'breakdown': {
                    'heating_pct': round(heating / total_cost * 100 if total_cost > 0 else 0, 1),
                    'hot_water_pct': round(hot_water / total_cost * 100 if total_cost > 0 else 0, 1),
                    'lighting_pct': round(lighting / total_cost * 100 if total_cost > 0 else 0, 1)
                }
            }
            
            return jsonify(response)
        
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/optimize', methods=['POST'])
    def optimize():
        """
        Get optimal retrofit recommendations.
        
        POST /optimize
        
        Request body:
        {
            "building_profile": { ... },
            "target_type": "carbon",  // or "energy"
            "target_reduction": 50.0,  // percentage
            "max_budget": 20000,  // optional
            "max_measures": 5  // optional
        }
        
        Response:
        {
            "recommendations": [
                {
                    "package_rank": 1,
                    "measures": [...],
                    "total_cost_range": "£5,000 - £10,000",
                    "energy_reduction_pct": 35.5,
                    "carbon_reduction_pct": 52.3,
                    "annual_savings": 420.0,
                    "payback_years": 18.5
                },
                ...
            ],
            "summary": {
                "target_achievable": true,
                "lowest_cost_package": 1
            }
        }
        """
        if not app.dss_components['initialized']:
            if not initialize_components():
                return jsonify({'error': 'DSS not initialized'}), 500
        
        try:
            data = request.get_json()
            if not data or 'building_profile' not in data:
                return jsonify({'error': 'Missing building_profile'}), 400
            
            profile = data['building_profile']
            target_type = data.get('target_type', 'carbon')
            target_reduction = data.get('target_reduction', 50.0)
            max_budget = data.get('max_budget')
            max_measures = data.get('max_measures', 5)
            
            # Convert profile to Series
            profile_series = pd.Series(profile)
            
            # Preprocess for numeric values
            preprocessor = app.dss_components['preprocessor']
            df = preprocessor.transform(pd.DataFrame([profile]))
            profile_series = df.iloc[0]
            
            # Get recommendations
            optimizer = app.dss_components['optimizer']
            packages = optimizer.optimize(
                profile_series,
                target_type=target_type,
                target_reduction=target_reduction,
                max_budget=max_budget,
                max_measures=max_measures
            )
            
            # Format response
            recommendations = []
            for i, pkg in enumerate(packages):
                rec = {
                    'package_rank': i + 1,
                    'measures': [m.to_dict() for m in pkg.measures],
                    'total_cost_range': f"£{pkg.total_cost_min:,.0f} - £{pkg.total_cost_max:,.0f}",
                    'total_cost_average': round(pkg.total_cost_avg, 0),
                    'energy_reduction_pct': round(pkg.predicted_energy_reduction, 1),
                    'carbon_reduction_pct': round(pkg.predicted_carbon_reduction, 1),
                    'annual_savings': round(pkg.predicted_cost_savings, 2),
                    'payback_years': round(pkg.payback_years, 1) if pkg.payback_years else None
                }
                recommendations.append(rec)
            
            # Check if target is achievable
            target_achieved = any(
                (pkg.predicted_carbon_reduction if target_type == 'carbon' 
                 else pkg.predicted_energy_reduction) >= target_reduction
                for pkg in packages
            )
            
            response = {
                'recommendations': recommendations,
                'summary': {
                    'target_achievable': target_achieved,
                    'target_type': target_type,
                    'target_reduction': target_reduction,
                    'packages_found': len(packages),
                    'lowest_cost_package': 1 if packages else None
                }
            }
            
            return jsonify(response)
        
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/sensitivity', methods=['POST'])
    def sensitivity():
        """
        Perform sensitivity analysis.
        
        POST /sensitivity
        
        Request body:
        {
            "building_profile": { ... },
            "feature": "WALLS_ENERGY_EFF_NUM",
            "values": [1, 2, 3, 4, 5]
        }
        
        Response:
        {
            "analysis": [
                {"feature_value": 1, "predicted_energy": 250, ...},
                ...
            ],
            "impact_summary": {
                "energy_range": [180, 250],
                "sensitivity": "high"
            }
        }
        """
        if not app.dss_components['initialized']:
            if not initialize_components():
                return jsonify({'error': 'DSS not initialized'}), 500
        
        try:
            data = request.get_json()
            
            profile = data.get('building_profile', {})
            feature = data.get('feature')
            values = data.get('values', [])
            
            if not feature or not values:
                return jsonify({'error': 'Missing feature or values'}), 400
            
            # Preprocess profile
            preprocessor = app.dss_components['preprocessor']
            df = preprocessor.transform(pd.DataFrame([profile]))
            profile_series = df.iloc[0]
            
            # Run sensitivity analysis
            optimizer = app.dss_components['optimizer']
            results_df = optimizer.sensitivity_analysis(
                profile_series,
                feature,
                values
            )
            
            # Format response
            analysis = results_df.to_dict(orient='records')
            
            # Calculate summary
            if 'predicted_energy' in results_df.columns:
                energy_range = [
                    float(results_df['predicted_energy'].min()),
                    float(results_df['predicted_energy'].max())
                ]
                sensitivity = 'high' if (energy_range[1] - energy_range[0]) > 50 else 'moderate'
            else:
                energy_range = None
                sensitivity = 'unknown'
            
            response = {
                'analysis': analysis,
                'impact_summary': {
                    'feature': feature,
                    'energy_range': energy_range,
                    'sensitivity': sensitivity
                }
            }
            
            return jsonify(response)
        
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/feature-importance', methods=['GET'])
    def feature_importance():
        """
        Get feature importance from trained models.
        
        GET /feature-importance?model=energy
        """
        if not app.dss_components['initialized']:
            if not initialize_components():
                return jsonify({'error': 'DSS not initialized'}), 500
        
        try:
            model_name = request.args.get('model', 'energy')
            
            model_factory = app.dss_components['model_factory']
            importance_dict = model_factory.get_all_feature_importance()
            
            if model_name in importance_dict:
                importance = importance_dict[model_name]
                df = importance.to_dataframe()
                
                # Validate physical intuition
                validation = importance.validate_physical_intuition()
                
                response = {
                    'model': model_name,
                    'features': df.to_dict(orient='records'),
                    'physical_validation': validation
                }
            else:
                response = {
                    'available_models': list(importance_dict.keys())
                }
            
            return jsonify(response)
        
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/model-metrics', methods=['GET'])
    def model_metrics():
        """
        Get model performance metrics.
        
        GET /model-metrics
        """
        if not app.dss_components['initialized']:
            return jsonify({'error': 'DSS not initialized'}), 500
        
        try:
            model_factory = app.dss_components['model_factory']
            
            metrics = {}
            for name, model in model_factory.models.items():
                if model._fitted and model._metrics:
                    metrics[name] = model._metrics.to_dict()
            
            return jsonify({'metrics': metrics})
        
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return app


def estimate_epc_grade(energy_intensity: float) -> str:
    """
    Estimate EPC grade from energy intensity.
    
    Based on SAP scoring bands.
    """
    if energy_intensity <= 0:
        return 'A'
    elif energy_intensity <= 50:
        return 'A'
    elif energy_intensity <= 100:
        return 'B'
    elif energy_intensity <= 150:
        return 'C'
    elif energy_intensity <= 225:
        return 'D'
    elif energy_intensity <= 325:
        return 'E'
    elif energy_intensity <= 450:
        return 'F'
    else:
        return 'G'


# Standalone app creation for direct running
app = create_app()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
