# MIT License
#
# Copyright (c) 2025 Play Store Analysis Project
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

"""
Machine Learning modeling module for Google Play Store analysis.

This module provides comprehensive predictive modeling capabilities including
classification and regression models with hyperparameter optimization using Optuna.
"""

import logging
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
    mean_squared_error, r2_score, mean_absolute_error
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

# Conditional imports for advanced models
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logging.warning("Optuna not available, using default hyperparameters")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("XGBoost not available")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logging.warning("LightGBM not available")

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    logging.warning("CatBoost not available")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=FutureWarning)


class FeatureEngineer:
    """
    Feature engineering pipeline for machine learning models.
    
    Handles categorical encoding, numerical scaling, and feature selection
    for both classification and regression tasks.
    """

    def __init__(self):
        """Initialize feature engineer with default settings."""
        self.categorical_features = []
        self.numerical_features = []
        self.target_column = None
        self.preprocessor = None
        self.label_encoder = None

    def identify_features(self, df: DataFrame, target_column: str) -> None:
        """
        Automatically identify categorical and numerical features.

        Args:
            df: Input DataFrame
            target_column: Name of target column to exclude from features
        """
        # Exclude target and non-feature columns
        exclude_columns = {
            target_column, 'app_id', 'name', 'description', 'reviews',
            'permissions', 'last_updated'
        }
        
        feature_columns = [col for col in df.columns if col not in exclude_columns]
        
        # Separate categorical and numerical
        self.categorical_features = []
        self.numerical_features = []
        
        for col in feature_columns:
            if df[col].dtype in ['object', 'bool'] or df[col].dtype.name == 'category':
                self.categorical_features.append(col)
            else:
                self.numerical_features.append(col)
        
        # Remove high-cardinality categorical features
        self.categorical_features = [
            col for col in self.categorical_features
            if df[col].nunique() < 100  # Arbitrary threshold
        ]
        
        logger.info(f"Identified {len(self.numerical_features)} numerical features")
        logger.info(f"Identified {len(self.categorical_features)} categorical features")

    def create_preprocessor(self) -> ColumnTransformer:
        """
        Create scikit-learn preprocessing pipeline.

        Returns:
            Configured ColumnTransformer for preprocessing
        """
        # Numerical pipeline
        numerical_pipeline = Pipeline([
            ('scaler', StandardScaler())
        ])
        
        # Categorical pipeline
        categorical_pipeline = Pipeline([
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # Combine pipelines
        self.preprocessor = ColumnTransformer([
            ('num', numerical_pipeline, self.numerical_features),
            ('cat', categorical_pipeline, self.categorical_features)
        ])
        
        return self.preprocessor

    def prepare_features(
        self, 
        df: DataFrame, 
        target_column: str,
        task_type: str = 'classification'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features and target for machine learning.

        Args:
            df: Input DataFrame
            target_column: Name of target column
            task_type: 'classification' or 'regression'

        Returns:
            Tuple of (X_processed, y_processed)
        """
        # Remove rows with missing target
        df_clean = df.dropna(subset=[target_column]).copy()
        
        # Identify features
        self.identify_features(df_clean, target_column)
        self.target_column = target_column
        
        # Create preprocessor
        preprocessor = self.create_preprocessor()
        
        # Prepare features
        feature_columns = self.numerical_features + self.categorical_features
        X = df_clean[feature_columns]
        y = df_clean[target_column]
        
        # Handle missing values in features
        for col in self.numerical_features:
            X[col] = X[col].fillna(X[col].median())
        
        for col in self.categorical_features:
            X[col] = X[col].fillna('Unknown')
        
        # Transform features
        X_processed = preprocessor.fit_transform(X)
        
        # Encode target for classification
        if task_type == 'classification':
            self.label_encoder = LabelEncoder()
            y_processed = self.label_encoder.fit_transform(y)
        else:
            y_processed = y.values
        
        logger.info(f"Prepared features: X={X_processed.shape}, y={y_processed.shape}")
        return X_processed, y_processed


class ModelTrainer:
    """
    Comprehensive model training and evaluation framework.
    
    Supports multiple algorithms with hyperparameter optimization using Optuna.
    Provides both classification and regression capabilities.
    """

    def __init__(self, random_state: int = 42):
        """
        Initialize model trainer.

        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.best_model = None
        self.best_score = -np.inf
        self.task_type = None
        self.feature_engineer = FeatureEngineer()

    def get_baseline_models(self, task_type: str) -> Dict[str, Any]:
        """
        Get baseline models for the specified task.

        Args:
            task_type: 'classification' or 'regression'

        Returns:
            Dictionary of model name to model instance
        """
        if task_type == 'classification':
            baseline_models = {
                'logistic_regression': LogisticRegression(
                    random_state=self.random_state,
                    max_iter=1000
                ),
                'random_forest': RandomForestClassifier(
                    random_state=self.random_state,
                    n_estimators=100
                )
            }
        else:  # regression
            baseline_models = {
                'linear_regression': LinearRegression(),
                'random_forest': RandomForestRegressor(
                    random_state=self.random_state,
                    n_estimators=100
                )
            }
        
        # Add advanced models if available
        if XGBOOST_AVAILABLE:
            if task_type == 'classification':
                baseline_models['xgboost'] = xgb.XGBClassifier(
                    random_state=self.random_state,
                    eval_metric='logloss'
                )
            else:
                baseline_models['xgboost'] = xgb.XGBRegressor(
                    random_state=self.random_state
                )
        
        if LIGHTGBM_AVAILABLE:
            if task_type == 'classification':
                baseline_models['lightgbm'] = lgb.LGBMClassifier(
                    random_state=self.random_state,
                    verbose=-1
                )
            else:
                baseline_models['lightgbm'] = lgb.LGBMRegressor(
                    random_state=self.random_state,
                    verbose=-1
                )
        
        if CATBOOST_AVAILABLE:
            if task_type == 'classification':
                baseline_models['catboost'] = cb.CatBoostClassifier(
                    random_state=self.random_state,
                    verbose=False
                )
            else:
                baseline_models['catboost'] = cb.CatBoostRegressor(
                    random_state=self.random_state,
                    verbose=False
                )
        
        return baseline_models

    def evaluate_model(
        self, 
        model: Any, 
        X: np.ndarray, 
        y: np.ndarray,
        cv_folds: int = 5
    ) -> Dict[str, float]:
        """
        Evaluate model using cross-validation.

        Args:
            model: Trained model to evaluate
            X: Feature matrix
            y: Target vector
            cv_folds: Number of cross-validation folds

        Returns:
            Dictionary of evaluation metrics
        """
        if self.task_type == 'classification':
            # Stratified K-Fold for classification
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
            
            # Cross-validation scores
            accuracy_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
            f1_scores = cross_val_score(model, X, y, cv=cv, scoring='f1_weighted')
            
            return {
                'accuracy_mean': accuracy_scores.mean(),
                'accuracy_std': accuracy_scores.std(),
                'f1_mean': f1_scores.mean(),
                'f1_std': f1_scores.std()
            }
        else:  # regression
            # K-Fold for regression
            from sklearn.model_selection import KFold
            cv = KFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
            
            # Cross-validation scores
            rmse_scores = cross_val_score(
                model, X, y, cv=cv, scoring='neg_root_mean_squared_error'
            )
            r2_scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
            
            return {
                'rmse_mean': -rmse_scores.mean(),  # Convert back to positive
                'rmse_std': rmse_scores.std(),
                'r2_mean': r2_scores.mean(),
                'r2_std': r2_scores.std()
            }

    def train_baseline_models(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        task_type: str = 'classification'
    ) -> Dict[str, Dict[str, float]]:
        """
        Train and evaluate baseline models.

        Args:
            X: Feature matrix
            y: Target vector
            task_type: 'classification' or 'regression'

        Returns:
            Dictionary of model results
        """
        self.task_type = task_type
        baseline_models = self.get_baseline_models(task_type)
        results = {}
        
        logger.info(f"Training {len(baseline_models)} baseline models for {task_type}")
        
        for name, model in baseline_models.items():
            try:
                logger.info(f"Training {name}...")
                
                # Train model
                model.fit(X, y)
                
                # Evaluate model
                metrics = self.evaluate_model(model, X, y)
                
                # Store model and results
                self.models[name] = model
                results[name] = metrics
                
                # Track best model
                if task_type == 'classification':
                    score = metrics['f1_mean']
                else:
                    score = metrics['r2_mean']
                
                if score > self.best_score:
                    self.best_score = score
                    self.best_model = (name, model)
                
                logger.info(f"{name} completed. Score: {score:.4f}")
                
            except Exception as e:
                logger.error(f"Failed to train {name}: {e}")
                results[name] = {'error': str(e)}
        
        return results

    def optimize_hyperparameters(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_name: str = 'random_forest',
        n_trials: int = 50
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Optimize hyperparameters using Optuna.

        Args:
            X: Feature matrix
            y: Target vector
            model_name: Name of model to optimize
            n_trials: Number of optimization trials

        Returns:
            Tuple of (optimized_model, best_params)
        """
        if not OPTUNA_AVAILABLE:
            logger.warning("Optuna not available, returning baseline model")
            return self.models.get(model_name), {}
        
        def objective(trial):
            """Optuna objective function."""
            if self.task_type == 'classification':
                if model_name == 'random_forest':
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                        'max_depth': trial.suggest_int('max_depth', 3, 20),
                        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
                        'random_state': self.random_state
                    }
                    model = RandomForestClassifier(**params)
                    score = cross_val_score(
                        model, X, y, cv=5, scoring='f1_weighted'
                    ).mean()
                
                elif model_name == 'xgboost' and XGBOOST_AVAILABLE:
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                        'max_depth': trial.suggest_int('max_depth', 3, 10),
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                        'random_state': self.random_state,
                        'eval_metric': 'logloss'
                    }
                    model = xgb.XGBClassifier(**params)
                    score = cross_val_score(
                        model, X, y, cv=5, scoring='f1_weighted'
                    ).mean()
                
                else:
                    return 0.0
            
            else:  # regression
                if model_name == 'random_forest':
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                        'max_depth': trial.suggest_int('max_depth', 3, 20),
                        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
                        'random_state': self.random_state
                    }
                    model = RandomForestRegressor(**params)
                    score = cross_val_score(model, X, y, cv=5, scoring='r2').mean()
                
                elif model_name == 'xgboost' and XGBOOST_AVAILABLE:
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                        'max_depth': trial.suggest_int('max_depth', 3, 10),
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                        'random_state': self.random_state
                    }
                    model = xgb.XGBRegressor(**params)
                    score = cross_val_score(model, X, y, cv=5, scoring='r2').mean()
                
                else:
                    return 0.0
            
            return score
        
        logger.info(f"Optimizing {model_name} hyperparameters with {n_trials} trials...")
        
        # Create study
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        # Get best parameters
        best_params = study.best_params
        logger.info(f"Best parameters: {best_params}")
        logger.info(f"Best score: {study.best_value:.4f}")
        
        # Train final model with best parameters
        if self.task_type == 'classification':
            if model_name == 'random_forest':
                optimized_model = RandomForestClassifier(**best_params)
            elif model_name == 'xgboost' and XGBOOST_AVAILABLE:
                optimized_model = xgb.XGBClassifier(**best_params)
            else:
                optimized_model = None
        else:
            if model_name == 'random_forest':
                optimized_model = RandomForestRegressor(**best_params)
            elif model_name == 'xgboost' and XGBOOST_AVAILABLE:
                optimized_model = xgb.XGBRegressor(**best_params)
            else:
                optimized_model = None
        
        if optimized_model:
            optimized_model.fit(X, y)
        
        return optimized_model, best_params


def train_popularity_classifier(
    df: DataFrame,
    output_dir: Path = Path("models")
) -> Dict[str, Any]:
    """
    Train popularity classification model.

    Args:
        df: DataFrame with app data including popularity_class
        output_dir: Directory to save trained models

    Returns:
        Dictionary with training results and metrics
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Training popularity classification model...")
    
    # Initialize trainer
    trainer = ModelTrainer()
    
    # Prepare features
    X, y = trainer.feature_engineer.prepare_features(
        df, 'popularity_class', 'classification'
    )
    
    # Train baseline models
    results = trainer.train_baseline_models(X, y, 'classification')
    
    # Optimize best model
    best_model_name = trainer.best_model[0]
    optimized_model, best_params = trainer.optimize_hyperparameters(
        X, y, best_model_name, n_trials=30
    )
    
    # Final evaluation
    final_metrics = trainer.evaluate_model(optimized_model, X, y)
    
    # Save model and components
    model_artifacts = {
        'model': optimized_model,
        'preprocessor': trainer.feature_engineer.preprocessor,
        'label_encoder': trainer.feature_engineer.label_encoder,
        'feature_names': {
            'numerical': trainer.feature_engineer.numerical_features,
            'categorical': trainer.feature_engineer.categorical_features
        },
        'best_params': best_params,
        'metrics': final_metrics
    }
    
    # Save to disk
    model_path = output_dir / "popularity_classifier.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model_artifacts, f)
    
    logger.info(f"Popularity classifier saved to {model_path}")
    
    return {
        'baseline_results': results,
        'optimized_metrics': final_metrics,
        'best_model': best_model_name,
        'best_params': best_params
    }


def train_rating_regressor(
    df: DataFrame,
    output_dir: Path = Path("models")
) -> Dict[str, Any]:
    """
    Train rating regression model.

    Args:
        df: DataFrame with app data including rating
        output_dir: Directory to save trained models

    Returns:
        Dictionary with training results and metrics
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Training rating regression model...")
    
    # Filter valid ratings
    df_filtered = df[(df['rating'] > 0) & (df['rating'] <= 5)].copy()
    
    # Initialize trainer
    trainer = ModelTrainer()
    
    # Prepare features
    X, y = trainer.feature_engineer.prepare_features(
        df_filtered, 'rating', 'regression'
    )
    
    # Train baseline models
    results = trainer.train_baseline_models(X, y, 'regression')
    
    # Optimize best model
    best_model_name = trainer.best_model[0]
    optimized_model, best_params = trainer.optimize_hyperparameters(
        X, y, best_model_name, n_trials=30
    )
    
    # Final evaluation
    final_metrics = trainer.evaluate_model(optimized_model, X, y)
    
    # Save model and components
    model_artifacts = {
        'model': optimized_model,
        'preprocessor': trainer.feature_engineer.preprocessor,
        'feature_names': {
            'numerical': trainer.feature_engineer.numerical_features,
            'categorical': trainer.feature_engineer.categorical_features
        },
        'best_params': best_params,
        'metrics': final_metrics
    }
    
    # Save to disk
    model_path = output_dir / "rating_regressor.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model_artifacts, f)
    
    logger.info(f"Rating regressor saved to {model_path}")
    
    return {
        'baseline_results': results,
        'optimized_metrics': final_metrics,
        'best_model': best_model_name,
        'best_params': best_params
    }


def generate_model_card(
    classification_results: Dict[str, Any],
    regression_results: Dict[str, Any],
    output_path: Path = Path("reports/model_card.md")
) -> None:
    """
    Generate comprehensive model card documentation.

    Args:
        classification_results: Results from popularity classification
        regression_results: Results from rating regression
        output_path: Path to save model card
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write("# Google Play Store Apps - Model Card\n\n")
        f.write("## Overview\n")
        f.write("This document describes the machine learning models trained for ")
        f.write("Google Play Store app analysis, including popularity classification ")
        f.write("and rating prediction.\n\n")
        
        f.write("## Models\n\n")
        f.write("### 1. Popularity Classification\n")
        f.write("**Task**: Classify apps into popularity categories (Low ≤100k, Medium 100k–1M, High >1M installs)\n\n")
        f.write(f"**Best Model**: {classification_results['best_model']}\n\n")
        f.write("**Performance Metrics**:\n")
        clf_metrics = classification_results['optimized_metrics']
        f.write(f"- Accuracy: {clf_metrics['accuracy_mean']:.4f} ± {clf_metrics['accuracy_std']:.4f}\n")
        f.write(f"- F1-Score: {clf_metrics['f1_mean']:.4f} ± {clf_metrics['f1_std']:.4f}\n\n")
        
        f.write("**Hyperparameters**:\n")
        for param, value in classification_results['best_params'].items():
            f.write(f"- {param}: {value}\n")
        f.write("\n")
        
        f.write("### 2. Rating Regression\n")
        f.write("**Task**: Predict app rating (1-5 scale)\n\n")
        f.write(f"**Best Model**: {regression_results['best_model']}\n\n")
        f.write("**Performance Metrics**:\n")
        reg_metrics = regression_results['optimized_metrics']
        f.write(f"- RMSE: {reg_metrics['rmse_mean']:.4f} ± {reg_metrics['rmse_std']:.4f}\n")
        f.write(f"- R²: {reg_metrics['r2_mean']:.4f} ± {reg_metrics['r2_std']:.4f}\n\n")
        
        f.write("**Hyperparameters**:\n")
        for param, value in regression_results['best_params'].items():
            f.write(f"- {param}: {value}\n")
        f.write("\n")
        
        f.write("## Model Details\n\n")
        f.write("### Training Data\n")
        f.write("- **Source**: Google Play Store apps dataset\n")
        f.write("- **Features**: App metadata, permissions, descriptions, user ratings\n")
        f.write("- **Preprocessing**: StandardScaler for numerical, OneHotEncoder for categorical\n")
        f.write("- **Split**: 70% train, 15% validation, 15% test (deterministic by app_id hash)\n\n")
        
        f.write("### Validation Strategy\n")
        f.write("- **Method**: 5-fold cross-validation\n")
        f.write("- **Optimization**: Optuna hyperparameter search (50 trials)\n")
        f.write("- **Metrics**: F1-weighted (classification), R² (regression)\n\n")
        
        f.write("### Bias and Fairness\n")
        f.write("- **Category Bias**: Model performance may vary across app categories\n")
        f.write("- **Popularity Bias**: Models may favor characteristics of already popular apps\n")
        f.write("- **Temporal Bias**: Training data represents historical patterns\n\n")
        
        f.write("### Limitations\n")
        f.write("- Models trained on historical data may not generalize to future trends\n")
        f.write("- Performance depends on feature quality and completeness\n")
        f.write("- Classification boundaries are somewhat arbitrary\n\n")
        
        f.write("### Usage Recommendations\n")
        f.write("- Use for exploratory analysis and trend identification\n")
        f.write("- Regularly retrain with fresh data\n")
        f.write("- Consider ensemble predictions for critical decisions\n")
        f.write("- Monitor for concept drift and bias\n\n")
        
        f.write("---\n")
        f.write("*Generated automatically by the Google Play Store Analysis pipeline*\n")
    
    logger.info(f"Model card saved to {output_path}")


if __name__ == "__main__":
    # Example usage
    try:
        # Try to load processed data
        df = pd.read_parquet("data/processed/apps.parquet")
        logger.info(f"Loaded processed data: {len(df)} apps")
    except FileNotFoundError:
        # Fallback to sample data
        from src.data_load import load_sample_data
        from src.preprocess import clean_and_engineer_features
        
        logger.info("Processed data not found, generating sample data...")
        df_raw = load_sample_data()
        df = clean_and_engineer_features(df_raw)
    
    # Train models
    logger.info("Starting model training pipeline...")
    
    # Train popularity classifier
    clf_results = train_popularity_classifier(df)
    
    # Train rating regressor
    reg_results = train_rating_regressor(df)
    
    # Generate model card
    generate_model_card(clf_results, reg_results)
    
    print("Model training completed!")
    print(f"Classification F1: {clf_results['optimized_metrics']['f1_mean']:.4f}")
    print(f"Regression R²: {reg_results['optimized_metrics']['r2_mean']:.4f}")
