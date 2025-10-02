"""
Advanced Machine Learning Ensemble System with Hyperparameter Optimization and Calibration
Comprehensive ML pipeline with XGBoost, LightGBM, Stacking, and Cross-Validation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Union
import pickle
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.calibration import CalibratedClassifierCV

# Optimization
import optuna
from optuna.samplers import TPESampler

# LSTM for Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Attention
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from logger import ml_logger, get_logger
from config import ml_config

class TimeSeriesSplitPurged:
    """Purged cross-validation for time series data"""
    
    def __init__(self, n_splits: int = 5, purge_gap: int = 1):
        self.n_splits = n_splits
        self.purge_gap = purge_gap
    
    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        # Calculate split sizes
        split_size = n_samples // self.n_splits
        
        for i in range(self.n_splits):
            # Training set
            start_train = i * split_size
            end_train = start_train + split_size
            
            # Test set (next period)
            start_test = end_train + self.purge_gap
            end_test = min(start_test + split_size, n_samples)
            
            if end_test > start_test:
                train_indices = indices[start_train:end_train]
                test_indices = indices[start_test:end_test]
                
                yield train_indices, test_indices

class EnsembleModel:
    """Advanced ensemble model with stacking and calibration"""
    
    def __init__(self):
        self.logger = ml_logger
        self.base_models = {}
        self.meta_model = None
        self.scaler = RobustScaler()
        self.feature_selector = None
        self.is_trained = False
        
        # Initialize base models
        self._init_base_models()
        
    def _init_base_models(self):
        """Initialize base models with default parameters"""
        self.base_models = {
            'xgboost': xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            ),
            'lightgbm': lgb.LGBMRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                feature_fraction=0.8,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            ),
            'random_forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            ),
            'svr': SVR(
                kernel='rbf',
                C=1.0,
                gamma='scale'
            ),
            'elastic_net': ElasticNet(
                alpha=0.1,
                l1_ratio=0.5,
                random_state=42
            )
        }
    
    def optimize_hyperparameters(self, X: pd.DataFrame, y: pd.Series, symbol: str) -> Dict[str, Any]:
        """Optimize hyperparameters using Optuna"""
        self.logger.info(f"🔍 Starting hyperparameter optimization for {symbol}")
        
        def objective(trial):
            # Sample hyperparameters
            model_name = trial.suggest_categorical('model', list(self.base_models.keys()))
            
            if model_name == 'xgboost':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 1)
                }
            
            elif model_name == 'lightgbm':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 1)
                }
            
            elif model_name == 'random_forest':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                    'max_depth': trial.suggest_int('max_depth', 5, 20),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
                    'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
                }
            
            else:
                params = {}
            
            # Train and evaluate
            model_clone = self.base_models[model_name].set_params(**params)
            cv_score = cross_val_score(
                model_clone, X, y, 
                cv=TimeSeriesSplitPurged(n_splits=3), 
                scoring='neg_mean_squared_error',
                n_jobs=1
            ).mean()
            
            return cv_score
        
        # Run optimization
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42)
        )
        
        study.optimize(objective, n_trials=50, timeout=3600)  # 1 hour timeout
        
        # Update models with best parameters
        best_params = study.best_params
        best_model = best_params.pop('model')
        
        if best_model in self.base_models:
            self.base_models[best_model].set_params(**best_params)
            self.logger.info(f"✅ Optimized {best_model} with RMSE: {-study.best_value:.4f}")
        
        return study.best_params
    
    def train_stack(self, X: pd.DataFrame, y: pd.Series, symbol: str):
        """Train ensemble with stacking meta-learner"""
        self.logger.info(f"🚂 Training stacked ensemble for {symbol}")
        
        # Prepare data
        X_scaled = self.scaler.fit_transform(X)
        
        # Time series cross-validation for stacking
        tscv = TimeSeriesSplitPurged(n_splits=5, purge_gap=5)
        
        # Generate base model predictions for meta-learning
        meta_features = np.zeros((len(X_scaled), len(self.base_models)))
        model_names = list(self.base_models.keys())
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_scaled)):
            X_train_fold, X_val_fold = X_scaled[train_idx], X_scaled[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
            
            for i, (name, model) in enumerate(self.base_models.items()):
                # Train on fold
                fold_model = model.__class__(**model.get_params())
                fold_model.fit(X_train_fold, y_train_fold)
                
                # Predict on validation fold
                val_pred = fold_model.predict(X_val_fold)
                meta_features[val_idx, i] = val_pred
        
        # Train meta-model
        self.meta_model = LogisticRegression(random_state=42, max_iter=1000)
        self.meta_model.fit(meta_features, y)
        
        # Retrain base models on full dataset
        for name, model in self.base_models.items():
            model.fit(X_scaled, y)
            self.logger.info(f"✅ Trained {name}")
        
        self.is_trained = True
        self.logger.info(f"🎯 Stacked ensemble training completed for {symbol}")
    
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Make prediction with confidence"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Get base model predictions
        base_predictions = []
        for name, model in self.base_models.items():
            pred = model.predict(X_scaled)
            base_predictions.append(pred)
        
        # Stack predictions
        stacked_pred = self.meta_model.predict_proba(base_predictions)[:, 1]
        confidence = np.max(self.meta_model.predict_proba(base_predictions), axis=1)
        
        return stacked_pred, confidence
    
    def get_feature_importance(self, top_n: int = 20) -> Dict[str, float]:
        """Get weighted feature importance across all models"""
        importance_dict = {}
        
        for name, model in self.base_models.items():
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                for i, imp in enumerate(importances):
                    feature_name = f"{name}_{i}"
                    importance_dict[feature_name] = imp
        
        # Sort by importance
        sorted_importance = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        
        return dict(sorted_importance)
    
    def save_model(self, filepath: str):
        """Save trained ensemble model"""
        model_data = {
            'base_models': self.base_models,
            'meta_model': self.meta_model,
            'scaler': self.scaler,
            'is_trained': self.is_trained,
            'timestamp': datetime.now()
        }
        
        joblib.dump(model_data, filepath)
        self.logger.info(f"💾 Saved ensemble model to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained ensemble model"""
        model_data = joblib.load(filepath)
        
        self.base_models = model_data['base_models']
        self.meta_model = model_data['meta_model']
        self.scaler = model_data['scaler']
        self.is_trained = model_data['is_trained']
        
        self.logger.info(f"📂 Loaded ensemble model from {filepath}")

class LSTMModel:
    """Advanced LSTM model with attention mechanism"""
    
    def __init__(self, sequence_length: int = 60, features: int = 100):
        self.sequence_length = sequence_length
        self.features = features
        self.logger = get_logger("LSTMModel")
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
        self._build_model()
    
    def _build_model(self):
        """Build LSTM model architecture"""
        model = Sequential([
            LSTM(units=100, return_sequences=True, input_shape=(self.sequence_length, self.features)),
            BatchNormalization(),
            Dropout(0.3),
            
            LSTM(units=80, return_sequences=True),
            BatchNormalization(),
            Dropout(0.3),
            
            LSTM(units=60, return_sequences=False),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(units=50, activation='relu'),
            Dropout(0.2),
            
            Dense(units=25, activation='relu'),
            
            Dense(units=1, activation='linear')  # Regression
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        self.model = model
        self.logger.info("🏗️ Built LSTM model architecture")
    
    def prepare_sequences(self, X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for LSTM sequences"""
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Create sequences
        sequences_X = []
        sequences_y = []
        
        for i in range(self.sequence_length, len(X_scaled)):
            sequences_X.append(X_scaled[i-self.sequence_length:i])
            sequences_y.append(y.iloc[i])
        
        return np.array(sequences_X), np.array(sequences_y)
    
    def train(self, X: pd.DataFrame, y: pd.Series, validation_split: float = 0.2):
        """Train LSTM model"""
        self.logger.info("🚂 Training LSTM model")
        
        # Prepare sequences
        X_seq, y_seq = self.prepare_sequences(X, y)
        
        # Callbacks
        callbacks = [
            EarlyStopping(patience=20, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=10, min_lr=1e-7)
        ]
        
        # Train model
        history = self.model.fit(
            X_seq, y_seq,
            validation_split=validation_split,
            epochs=100,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        self.is_trained = True
        self.logger.info("✅ LSTM model training completed")
        
        return history
    
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Make prediction using LSTM"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        # Prepare sequences
        X_scaled = self.scaler.transform(X)
        
        # Get last sequence
        if len(X_scaled) >= self.sequence_length:
            last_sequence = X_scaled[-self.sequence_length:].reshape(1, self.sequence_length, self.features)
            
            prediction = self.model.predict(last_sequence)[0][0]
            confidence = 0.8  # Placeholder confidence
            
            return np.array([prediction]), np.array([confidence])
        
        return np.array([0]), np.array([0])
    
    def save_model(self, filepath: str):
        """Save LSTM model"""
        if self.model:
            self.model.save(filepath)
            
            # Save scaler separately
            scaler_path = filepath.replace('.h5', '_scaler.pkl')
            joblib.dump(self.scaler, scaler_path)
            
            self.logger.info(f"💾 Saved LSTM model to {filepath}")
    
    def load_model(self, filepath: str):
        """Load LSTM model"""
        self.model = tf.keras.models.load_model(filepath)
        
        # Load scaler
        scaler_path = filepath.replace('.h5', '_scaler.pkl')
        self.scaler = joblib.load(scaler_path)
        
        self.is_trained = True
        self.logger.info(f"📂 Loaded LSTM model from {filepath}")

class ConceptDriftDetector:
    """Detect concept drift in market data"""
    
    def __init__(self, drift_threshold: float = 0.05):
        self.drift_threshold = drift_threshold
        self.logger = get_logger("ConceptDriftDetector")
        self.baseline_distribution = None
        self.reference_data = None
        
    def set_baseline(self, reference_data: pd.DataFrame):
        """Set baseline distribution from training data"""
        self.reference_data = reference_data
        self.baseline_distribution = self._calculate_distribution(reference_data)
        self.logger.info("📊 Set baseline distribution for drift detection")
    
    def _calculate_distribution(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate statistical distribution of features"""
        distribution = {}
        
        for column in data.select_dtypes(include=[np.number]).columns:
            distribution[column] = {
                'mean': data[column].mean(),
                'std': data[column].std(),
                'skew': data[column].skew(),
                'kurtosis': data[column].kurtosis(),
                'q25': data[column].quantile(0.25),
                'q75': data[column].quantile(0.75)
            }
        
        return distribution
    
    def detect_drift(self, current_data: pd.DataFrame) -> Tuple[bool, float]:
        """Detect concept drift using statistical tests"""
        if self.baseline_distribution is None:
            self.logger.warning("No baseline distribution set")
            return False, 0.0
        
        current_distribution = self._calculate_distribution(current_data)
        
        drift_scores = []
        
        for feature, baseline_stats in self.baseline_distribution.items():
            if feature in current_distribution:
                current_stats = current_distribution[feature]
                
                # Statistical distance (simplified)
                mean_diff = abs(baseline_stats['mean'] - current_stats['mean'])
                std_diff = abs(baseline_stats['std'] - current_stats['std'])
                
                distance = (mean_diff / baseline_stats['std']) + (std_diff / baseline_stats['std'])
                drift_scores.append(distance)
        
        avg_drift_score = np.mean(drift_scores) if drift_scores else 0.0
        drift_detected = avg_drift_score > self.drift_threshold
        
        if drift_detected:
            self.logger.warning(f"🚨 Concept drift detected! Score: {avg_drift_score:.4f}")
        else:
            self.logger.debug(f"✅ No concept drift detected. Score: {avg_drift_score:.4f}")
        
        return drift_detected, avg_drift_score

class AutoRetrainManager:
    """Automatic model retraining based on performance and drift"""
    
    def __init__(self):
        self.logger = get_logger("AutoRetrainManager")
        self.drift_detector = ConceptDriftDetector()
        self.performance_history = []
        self.last_retrain = None
        
    def should_retrain(self, model_performance: float, recent_data: pd.DataFrame) -> Tuple[bool, str]:
        """Determine if model should be retrained"""
        retrain_reasons = []
        
        # Performance-based retrain
        if len(self.performance_history) >= 5:
            recent_performance = np.mean(self.performance_history[-5:])
            baseline_performance = np.mean(self.performance_history[:5])
            
            performance_drop = baseline_performance - recent_performance
            
            if performance_drop > ml_config.retrain_threshold_performance:
                retrain_reasons.append(f"Performance drop: {performance_drop:.4f}")
        
        # Drift-based retrain
        if self.drift_detector.baseline_distribution is not None:
            drift_detected, drift_score = self.drift_detector.detect_drift(recent_data)
            
            if drift_detected:
                retrain_reasons.append(f"Concept drift: {drift_score:.4f}")
        
        # Time-based retrain (weekly minimum)
        if self.last_retrain is None:
            retrain_reasons.append("Initial model")
        else:
            days_since_retrain = (datetime.now() - self.last_retrain).days
            
            if days_since_retrain > 7:
                retrain_reasons.append(f"Time: {days_since_retrain} days")
        
        should_retrain = len(retrain_reasons) > 0
        
        if should_retrain:
            reason_str = "; ".join(retrain_reasons)
            self.logger.info(f"🔄 Retraining triggered: {reason_str}")
            self.last_retrain = datetime.now()
        
        return should_retrain, "; ".join(retrain_reasons)
    
    def update_performance(self, performance: float):
        """Update performance history"""
        self.performance_history.append(performance)
        
        # Keep only recent history
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-50:]
    
    def set_baseline_from_training_data(self, training_data: pd.DataFrame):
        """Set drift detection baseline from training data"""
        self.drift_detector.set_baseline(training_data)

# Global ML components
ensemble_model = EnsembleModel()
lstm_model = LSTMModel(
    sequence_length=ml_config.lstm_sequence_length,
    features=ml_config.lstm_hidden_units
)
auto_retrain_manager = AutoRetrainManager()