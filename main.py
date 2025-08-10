"""
Solavix AI Backend Service
Free AI models for predictive maintenance
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestRegressor, RandomForestClassifier
from sklearn.svm import OneClassSVM, SVC
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_squared_error
import xgboost as xgb
from prophet import Prophet
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
import joblib
import os
from datetime import datetime, timedelta
import logging
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Solavix AI Service", version="1.0.0")

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for your domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data models
class SensorReading(BaseModel):
    timestamp: str
    temperature: float
    voltage: float
    current: float
    power: Optional[float] = None
    efficiency: Optional[float] = None
    asset_id: str

class AnomalyResult(BaseModel):
    is_anomaly: bool
    anomaly_score: float
    risk_level: str
    confidence: float
    recommendations: List[str]

class HealthScore(BaseModel):
    health_score: float
    trend: str
    risk_factors: List[str]
    next_maintenance: str
    recommendations: List[str]

class MaintenancePrediction(BaseModel):
    maintenance_needed: bool
    days_until_maintenance: int
    confidence: float
    priority: str
    estimated_cost: float
    recommendations: List[str]

class SolavixAI:
    def __init__(self):
        # Anomaly Detection Models
        self.isolation_forest = None
        self.one_class_svm = None
        self.dbscan = None
        self.autoencoder = None
        
        # Classification Models
        self.random_forest_classifier = None
        self.xgb_classifier = None
        self.svm_classifier = None
        
        # Regression Models
        self.random_forest_regressor = None
        self.xgb_regressor = None
        
        # Time Series Models
        self.prophet_model = None
        self.lstm_model = None
        
        # Preprocessing
        self.scaler = StandardScaler()
        self.minmax_scaler = MinMaxScaler()
        
        self.models_trained = False
        
        # Initialize all models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize all AI models with synthetic industrial data"""
        logger.info("Initializing comprehensive AI model suite...")
        
        # Generate synthetic training data
        synthetic_data = self._generate_synthetic_data(20000)  # More data for better training
        
        # Prepare features and targets
        feature_columns = ['temperature', 'voltage', 'current', 'power', 'age_days']
        features = synthetic_data[feature_columns].values
        health_scores = synthetic_data['health_score'].values
        failure_labels = (synthetic_data['health_score'] < 40).astype(int)  # Binary failure classification
        
        # Fit scalers
        self.scaler.fit(features)
        self.minmax_scaler.fit(features)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        features_minmax = self.minmax_scaler.transform(features)
        
        # Split data
        X_train, X_test, y_health_train, y_health_test, y_failure_train, y_failure_test = train_test_split(
            features_scaled, health_scores, failure_labels, test_size=0.2, random_state=42
        )
        
        # 1. ANOMALY DETECTION MODELS
        logger.info("Training anomaly detection models...")
        
        # Isolation Forest
        self.isolation_forest = IsolationForest(
            contamination=0.1,
            random_state=42,
            n_estimators=200
        )
        self.isolation_forest.fit(X_train)
        
        # One-Class SVM
        self.one_class_svm = OneClassSVM(
            kernel='rbf',
            gamma='scale',
            nu=0.1
        )
        self.one_class_svm.fit(X_train)
        
        # DBSCAN for clustering-based anomaly detection
        self.dbscan = DBSCAN(eps=0.5, min_samples=5)
        self.dbscan.fit(X_train)
        
        # Autoencoder for neural network-based anomaly detection
        self.autoencoder = self._build_autoencoder(features_scaled.shape[1])
        self.autoencoder.fit(
            X_train, X_train,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            verbose=0
        )
        
        # 2. CLASSIFICATION MODELS
        logger.info("Training classification models...")
        
        # Random Forest Classifier
        self.random_forest_classifier = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            random_state=42
        )
        self.random_forest_classifier.fit(X_train, y_failure_train)
        
        # XGBoost Classifier
        self.xgb_classifier = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        self.xgb_classifier.fit(X_train, y_failure_train)
        
        # SVM Classifier
        self.svm_classifier = SVC(
            kernel='rbf',
            probability=True,
            random_state=42
        )
        self.svm_classifier.fit(X_train, y_failure_train)
        
        # 3. REGRESSION MODELS
        logger.info("Training regression models...")
        
        # Random Forest Regressor
        self.random_forest_regressor = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            random_state=42
        )
        self.random_forest_regressor.fit(X_train, y_health_train)
        
        # XGBoost Regressor
        self.xgb_regressor = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        self.xgb_regressor.fit(X_train, y_health_train)
        
        # 4. TIME SERIES MODELS
        logger.info("Training time series models...")
        
        # LSTM Model
        self.lstm_model = self._build_lstm_model(5, 1)  # 5 features, 1 output
        
        # Prepare LSTM training data
        lstm_X, lstm_y = self._prepare_lstm_data(features_minmax, health_scores)
        if lstm_X.shape[0] > 0:
            self.lstm_model.fit(
                lstm_X, lstm_y,
                epochs=50,
                batch_size=32,
                validation_split=0.2,
                verbose=0
            )
        
        self.models_trained = True
        logger.info("All AI models initialized successfully!")
        
        # Log model performance
        self._evaluate_models(X_test, y_health_test, y_failure_test)
    
    def _generate_synthetic_data(self, n_samples=10000):
        """Generate realistic synthetic data for training"""
        np.random.seed(42)
        
        # Normal operating ranges for solar/wind equipment
        data = []
        
        for i in range(n_samples):
            # Base parameters
            age_days = np.random.uniform(0, 3650)  # 0-10 years
            
            # Normal operating conditions (80% of data)
            if i < n_samples * 0.8:
                temperature = np.random.normal(45, 8)  # 45°C ± 8°C
                voltage = np.random.normal(12.5, 1.2)  # 12.5V ± 1.2V
                current = np.random.normal(8.0, 1.5)   # 8A ± 1.5A
                health_score = np.random.uniform(70, 95)
            
            # Degraded conditions (15% of data)
            elif i < n_samples * 0.95:
                temperature = np.random.normal(55, 10)  # Higher temp
                voltage = np.random.normal(11.0, 1.5)   # Lower voltage
                current = np.random.normal(6.5, 2.0)    # Lower current
                health_score = np.random.uniform(40, 70)
            
            # Failure conditions (5% of data)
            else:
                temperature = np.random.normal(70, 15)  # Very high temp
                voltage = np.random.normal(9.0, 2.0)    # Very low voltage
                current = np.random.normal(4.0, 2.5)    # Very low current
                health_score = np.random.uniform(10, 40)
            
            power = voltage * current
            
            data.append({
                'temperature': max(0, temperature),
                'voltage': max(0, voltage),
                'current': max(0, current),
                'power': max(0, power),
                'age_days': age_days,
                'health_score': max(0, min(100, health_score))
            })
        
        return pd.DataFrame(data)
    
    def _build_autoencoder(self, input_dim):
        """Build autoencoder for anomaly detection"""
        model = Sequential([
            Input(shape=(input_dim,)),
            Dense(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(8, activation='relu'),
            Dense(16, activation='relu'),
            Dense(32, activation='relu'),
            Dense(input_dim, activation='linear')
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model
    
    def _build_lstm_model(self, n_features, n_outputs):
        """Build LSTM model for time series prediction"""
        model = Sequential([
            Input(shape=(10, n_features)),  # 10 time steps
            LSTM(50, return_sequences=True),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25, activation='relu'),
            Dense(n_outputs, activation='linear')
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        return model
    
    def _prepare_lstm_data(self, features, targets, time_steps=10):
        """Prepare data for LSTM training"""
        X, y = [], []
        
        for i in range(time_steps, len(features)):
            X.append(features[i-time_steps:i])
            y.append(targets[i])
        
        return np.array(X), np.array(y)
    
    def _evaluate_models(self, X_test, y_health_test, y_failure_test):
        """Evaluate model performance"""
        try:
            # Classification accuracy
            rf_pred = self.random_forest_classifier.predict(X_test)
            xgb_pred = self.xgb_classifier.predict(X_test)
            
            # Regression accuracy
            rf_health_pred = self.random_forest_regressor.predict(X_test)
            xgb_health_pred = self.xgb_regressor.predict(X_test)
            
            logger.info(f"Random Forest Classification Accuracy: {np.mean(rf_pred == y_failure_test):.3f}")
            logger.info(f"XGBoost Classification Accuracy: {np.mean(xgb_pred == y_failure_test):.3f}")
            logger.info(f"Random Forest Health Score RMSE: {np.sqrt(mean_squared_error(y_health_test, rf_health_pred)):.3f}")
            logger.info(f"XGBoost Health Score RMSE: {np.sqrt(mean_squared_error(y_health_test, xgb_health_pred)):.3f}")
            
        except Exception as e:
            logger.warning(f"Model evaluation failed: {e}")
    
    def detect_anomaly(self, sensor_data: SensorReading) -> AnomalyResult:
        """Advanced multi-model anomaly detection"""
        if not self.models_trained:
            raise HTTPException(status_code=500, detail="Models not initialized")
        
        # Prepare features
        features = np.array([[
            sensor_data.temperature,
            sensor_data.voltage,
            sensor_data.current,
            sensor_data.power or (sensor_data.voltage * sensor_data.current),
            365  # Default age in days
        ]])
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        features_minmax = self.minmax_scaler.transform(features)
        
        # Multi-model anomaly detection
        anomaly_scores = {}
        
        # 1. Isolation Forest
        iso_score = self.isolation_forest.decision_function(features_scaled)[0]
        iso_anomaly = self.isolation_forest.predict(features_scaled)[0] == -1
        anomaly_scores['isolation_forest'] = {'score': abs(iso_score), 'is_anomaly': iso_anomaly}
        
        # 2. One-Class SVM
        svm_score = self.one_class_svm.decision_function(features_scaled)[0]
        svm_anomaly = self.one_class_svm.predict(features_scaled)[0] == -1
        anomaly_scores['one_class_svm'] = {'score': abs(svm_score), 'is_anomaly': svm_anomaly}
        
        # 3. Autoencoder
        try:
            reconstructed = self.autoencoder.predict(features_scaled, verbose=0)
            reconstruction_error = np.mean(np.square(features_scaled - reconstructed))
            autoencoder_anomaly = reconstruction_error > 0.1  # Threshold
            anomaly_scores['autoencoder'] = {'score': reconstruction_error, 'is_anomaly': autoencoder_anomaly}
        except Exception as e:
            logger.warning(f"Autoencoder prediction failed: {e}")
            anomaly_scores['autoencoder'] = {'score': 0.0, 'is_anomaly': False}
        
        # Ensemble decision
        anomaly_votes = sum([scores['is_anomaly'] for scores in anomaly_scores.values()])
        is_anomaly = anomaly_votes >= 2  # Majority vote
        
        # Calculate combined confidence
        avg_score = np.mean([scores['score'] for scores in anomaly_scores.values()])
        confidence = min(0.95, avg_score + 0.3)
        
        # Determine risk level based on sensor values and anomaly scores
        risk_level = self._calculate_risk_level(sensor_data, avg_score)
        
        # Generate intelligent recommendations
        recommendations = self._generate_anomaly_recommendations(sensor_data, anomaly_scores)
        
        return AnomalyResult(
            is_anomaly=is_anomaly,
            anomaly_score=float(avg_score),
            risk_level=risk_level,
            confidence=confidence,
            recommendations=recommendations
        )
    
    def _calculate_risk_level(self, sensor_data: SensorReading, anomaly_score: float) -> str:
        """Calculate risk level based on sensor data and anomaly score"""
        risk_factors = 0
        
        if sensor_data.temperature > 65:
            risk_factors += 3
        elif sensor_data.temperature > 55:
            risk_factors += 2
        elif sensor_data.temperature > 50:
            risk_factors += 1
            
        if sensor_data.voltage < 9:
            risk_factors += 3
        elif sensor_data.voltage < 10.5:
            risk_factors += 2
        elif sensor_data.voltage < 11.5:
            risk_factors += 1
            
        if sensor_data.current < 4:
            risk_factors += 3
        elif sensor_data.current < 6:
            risk_factors += 2
        elif sensor_data.current < 7:
            risk_factors += 1
        
        # Combine with anomaly score
        total_risk = risk_factors + (anomaly_score * 5)
        
        if total_risk > 8:
            return "CRITICAL"
        elif total_risk > 5:
            return "HIGH"
        elif total_risk > 2:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _generate_anomaly_recommendations(self, sensor_data: SensorReading, anomaly_scores: dict) -> List[str]:
        """Generate intelligent recommendations based on anomaly detection"""
        recommendations = []
        
        # Temperature-based recommendations
        if sensor_data.temperature > 65:
            recommendations.append("CRITICAL: Temperature exceeds safe limits - immediate shutdown recommended")
        elif sensor_data.temperature > 55:
            recommendations.append("HIGH: Check cooling system and reduce load immediately")
        elif sensor_data.temperature > 50:
            recommendations.append("MEDIUM: Monitor temperature closely and improve ventilation")
        
        # Voltage-based recommendations
        if sensor_data.voltage < 9:
            recommendations.append("CRITICAL: Voltage critically low - check electrical connections")
        elif sensor_data.voltage < 10.5:
            recommendations.append("HIGH: Low voltage detected - inspect wiring and connections")
        elif sensor_data.voltage < 11.5:
            recommendations.append("MEDIUM: Voltage below optimal - schedule electrical inspection")
        
        # Current-based recommendations
        if sensor_data.current < 4:
            recommendations.append("CRITICAL: Very low current - check for equipment failure")
        elif sensor_data.current < 6:
            recommendations.append("HIGH: Low current output - inspect for obstructions or damage")
        elif sensor_data.current < 7:
            recommendations.append("MEDIUM: Current below expected - monitor performance")
        
        # Model-specific recommendations
        if anomaly_scores.get('autoencoder', {}).get('is_anomaly', False):
            recommendations.append("AI: Neural network detected unusual pattern - detailed inspection recommended")
        
        if anomaly_scores.get('isolation_forest', {}).get('is_anomaly', False):
            recommendations.append("AI: Statistical anomaly detected - compare with historical data")
        
        if not recommendations:
            recommendations.append("All parameters within normal operating range")
        
        return recommendations[:3]  # Limit to top 3 recommendations

    def calculate_health_score(self, sensor_data: SensorReading, asset_age_days: int = 365) -> HealthScore:
        """Advanced multi-model health score calculation"""
        if not self.models_trained:
            raise HTTPException(status_code=500, detail="Models not initialized")
        
        # Prepare features
        features = np.array([[
            sensor_data.temperature,
            sensor_data.voltage,
            sensor_data.current,
            sensor_data.power or (sensor_data.voltage * sensor_data.current),
            asset_age_days
        ]])
        
        features_scaled = self.scaler.transform(features)
        
        # Multi-model health prediction
        health_predictions = {}
        
        # 1. Random Forest Regressor
        rf_health = self.random_forest_regressor.predict(features_scaled)[0]
        health_predictions['random_forest'] = max(0, min(100, rf_health))
        
        # 2. XGBoost Regressor
        xgb_health = self.xgb_regressor.predict(features_scaled)[0]
        health_predictions['xgboost'] = max(0, min(100, xgb_health))
        
        # 3. Failure probability from classifiers
        rf_failure_prob = self.random_forest_classifier.predict_proba(features_scaled)[0][1]
        xgb_failure_prob = self.xgb_classifier.predict_proba(features_scaled)[0][1]
        svm_failure_prob = self.svm_classifier.predict_proba(features_scaled)[0][1]
        
        # Convert failure probability to health score
        avg_failure_prob = (rf_failure_prob + xgb_failure_prob + svm_failure_prob) / 3
        failure_based_health = (1 - avg_failure_prob) * 100
        health_predictions['failure_based'] = failure_based_health
        
        # 4. LSTM prediction (if available)
        try:
            # Create sequence for LSTM (repeat current reading 10 times as placeholder)
            lstm_input = np.array([features_scaled[0]] * 10).reshape(1, 10, -1)
            lstm_health = self.lstm_model.predict(lstm_input, verbose=0)[0][0]
            health_predictions['lstm'] = max(0, min(100, lstm_health))
        except Exception as e:
            logger.warning(f"LSTM prediction failed: {e}")
            health_predictions['lstm'] = np.mean(list(health_predictions.values()))
        
        # Ensemble health score (weighted average)
        weights = {
            'random_forest': 0.25,
            'xgboost': 0.25,
            'failure_based': 0.3,
            'lstm': 0.2
        }
        
        health_score = sum(health_predictions[model] * weights[model] for model in health_predictions)
        health_score = max(0, min(100, health_score))
        
        # Advanced trend analysis
        trend = self._calculate_health_trend(health_score, sensor_data, asset_age_days)
        
        # Comprehensive risk factor analysis
        risk_factors = self._identify_risk_factors(sensor_data, asset_age_days, health_predictions)
        
        # Intelligent maintenance scheduling
        next_maintenance = self._calculate_next_maintenance(health_score, risk_factors, asset_age_days)
        
        # Advanced recommendations
        recommendations = self._generate_health_recommendations(health_score, sensor_data, risk_factors)
        
        return HealthScore(
            health_score=float(health_score),
            trend=trend,
            risk_factors=risk_factors,
            next_maintenance=next_maintenance,
            recommendations=recommendations
        )
    
    def predict_maintenance(self, sensor_readings: List[SensorReading]) -> MaintenancePrediction:
        """Predict maintenance needs based on sensor trends"""
        if not sensor_readings:
            raise HTTPException(status_code=400, detail="No sensor data provided")
        
        # Calculate average conditions
        avg_temp = np.mean([r.temperature for r in sensor_readings])
        avg_voltage = np.mean([r.voltage for r in sensor_readings])
        avg_current = np.mean([r.current for r in sensor_readings])
        
        # Simple rule-based prediction (can be enhanced with ML)
        maintenance_score = 0
        
        if avg_temp > 55:
            maintenance_score += 30
        if avg_voltage < 11:
            maintenance_score += 25
        if avg_current < 6:
            maintenance_score += 20
        
        # Temperature trend
        if len(sensor_readings) > 1:
            temp_trend = sensor_readings[-1].temperature - sensor_readings[0].temperature
            if temp_trend > 5:  # Rising temperature
                maintenance_score += 15
        
        maintenance_needed = maintenance_score > 40
        
        if maintenance_score > 70:
            priority = "CRITICAL"
            days_until = 3
            estimated_cost = 25000
        elif maintenance_score > 50:
            priority = "HIGH"
            days_until = 7
            estimated_cost = 15000
        elif maintenance_score > 30:
            priority = "MEDIUM"
            days_until = 21
            estimated_cost = 8000
        else:
            priority = "LOW"
            days_until = 60
            estimated_cost = 5000
        
        confidence = min(0.9, maintenance_score / 100 + 0.3)
        
        recommendations = []
        if maintenance_needed:
            recommendations.append(f"Schedule {priority.lower()} maintenance within {days_until} days")
            if avg_temp > 55:
                recommendations.append("Focus on cooling system inspection")
            if avg_voltage < 11:
                recommendations.append("Check electrical connections and components")
        else:
            recommendations.append("Continue monitoring - no immediate maintenance required")
        
        return MaintenancePrediction(
            maintenance_needed=maintenance_needed,
            days_until_maintenance=days_until,
            confidence=confidence,
            priority=priority,
            estimated_cost=estimated_cost,
            recommendations=recommendations
        )
    
    def _calculate_health_trend(self, health_score: float, sensor_data: SensorReading, asset_age_days: int) -> str:
        """Calculate health trend based on multiple factors"""
        if health_score > 85 and sensor_data.temperature < 50 and asset_age_days < 1825:  # < 5 years
            return "stable"
        elif health_score > 70 and sensor_data.temperature < 60:
            return "stable"
        elif health_score > 50:
            return "declining"
        else:
            return "critical"
    
    def _identify_risk_factors(self, sensor_data: SensorReading, asset_age_days: int, health_predictions: dict) -> List[str]:
        """Identify comprehensive risk factors"""
        risk_factors = []
        
        # Temperature risks
        if sensor_data.temperature > 65:
            risk_factors.append("Critical temperature levels")
        elif sensor_data.temperature > 55:
            risk_factors.append("High operating temperature")
        elif sensor_data.temperature > 50:
            risk_factors.append("Elevated temperature")
        
        # Electrical risks
        if sensor_data.voltage < 9:
            risk_factors.append("Critical voltage drop")
        elif sensor_data.voltage < 10.5:
            risk_factors.append("Low voltage output")
        
        if sensor_data.current < 4:
            risk_factors.append("Critical current loss")
        elif sensor_data.current < 6:
            risk_factors.append("Low current output")
        
        # Age-related risks
        if asset_age_days > 3650:  # > 10 years
            risk_factors.append("Equipment end-of-life approaching")
        elif asset_age_days > 2555:  # > 7 years
            risk_factors.append("Equipment aging effects")
        elif asset_age_days > 1825:  # > 5 years
            risk_factors.append("Mid-life maintenance required")
        
        # Model disagreement risk
        health_values = list(health_predictions.values())
        if max(health_values) - min(health_values) > 20:
            risk_factors.append("Inconsistent health indicators")
        
        # Performance risks
        power = sensor_data.power or (sensor_data.voltage * sensor_data.current)
        if power < 50:  # Assuming normal power > 50W
            risk_factors.append("Low power output")
        
        return risk_factors[:4]  # Limit to top 4 risk factors
    
    def _calculate_next_maintenance(self, health_score: float, risk_factors: List[str], asset_age_days: int) -> str:
        """Calculate intelligent maintenance scheduling"""
        base_days = 60  # Default 60 days
        
        # Adjust based on health score
        if health_score < 40:
            base_days = 3
        elif health_score < 60:
            base_days = 7
        elif health_score < 75:
            base_days = 21
        elif health_score < 85:
            base_days = 45
        
        # Adjust based on risk factors
        critical_risks = [r for r in risk_factors if 'Critical' in r or 'end-of-life' in r]
        if critical_risks:
            base_days = min(base_days, 3)
        
        high_risks = [r for r in risk_factors if 'High' in r or 'aging' in r]
        if high_risks:
            base_days = min(base_days, 14)
        
        # Adjust based on age
        if asset_age_days > 3650:  # > 10 years
            base_days = min(base_days, 30)
        
        next_date = datetime.now() + timedelta(days=base_days)
        return next_date.strftime("%Y-%m-%d")
    
    def _generate_health_recommendations(self, health_score: float, sensor_data: SensorReading, risk_factors: List[str]) -> List[str]:
        """Generate intelligent health-based recommendations"""
        recommendations = []
        
        # Health score based recommendations
        if health_score < 40:
            recommendations.append("URGENT: Schedule immediate maintenance inspection")
        elif health_score < 60:
            recommendations.append("Schedule maintenance within 1 week")
        elif health_score < 75:
            recommendations.append("Plan maintenance within 3 weeks")
        elif health_score < 85:
            recommendations.append("Schedule routine maintenance within 6 weeks")
        else:
            recommendations.append("Asset performing excellently - continue monitoring")
        
        # Risk-specific recommendations
        if "Critical temperature" in ' '.join(risk_factors):
            recommendations.append("Immediate cooling system intervention required")
        elif "High operating temperature" in ' '.join(risk_factors):
            recommendations.append("Improve cooling system or reduce operational load")
        
        if "Critical voltage" in ' '.join(risk_factors):
            recommendations.append("Emergency electrical system inspection needed")
        elif "Low voltage" in ' '.join(risk_factors):
            recommendations.append("Check electrical connections and components")
        
        if "end-of-life" in ' '.join(risk_factors):
            recommendations.append("Plan equipment replacement within 6 months")
        elif "aging" in ' '.join(risk_factors):
            recommendations.append("Increase maintenance frequency for aging equipment")
        
        # Performance recommendations
        if sensor_data.efficiency and sensor_data.efficiency < 80:
            recommendations.append("Performance optimization needed - check for obstructions")
        
        return recommendations[:3]  # Limit to top 3 recommendations

# Initialize AI service
ai_service = SolavixAI()

# API Endpoints
@app.get("/")
async def root():
    return {"message": "Solavix AI Service", "status": "running", "models_ready": ai_service.models_trained}

@app.post("/api/ai/anomaly-detection", response_model=AnomalyResult)
async def detect_anomaly(sensor_data: SensorReading):
    """Detect anomalies in sensor data"""
    try:
        result = ai_service.detect_anomaly(sensor_data)
        logger.info(f"Anomaly detection completed for asset {sensor_data.asset_id}")
        return result
    except Exception as e:
        logger.error(f"Anomaly detection failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ai/health-score", response_model=HealthScore)
async def calculate_health_score(sensor_data: SensorReading, asset_age_days: int = 365):
    """Calculate asset health score"""
    try:
        result = ai_service.calculate_health_score(sensor_data, asset_age_days)
        logger.info(f"Health score calculated for asset {sensor_data.asset_id}: {result.health_score}")
        return result
    except Exception as e:
        logger.error(f"Health score calculation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ai/maintenance-prediction", response_model=MaintenancePrediction)
async def predict_maintenance(sensor_readings: List[SensorReading]):
    """Predict maintenance needs"""
    try:
        result = ai_service.predict_maintenance(sensor_readings)
        logger.info(f"Maintenance prediction completed: {result.priority} priority")
        return result
    except Exception as e:
        logger.error(f"Maintenance prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/ai/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_trained": ai_service.models_trained,
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)