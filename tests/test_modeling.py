"""
Tests for modeling module.

This module contains unit tests for the machine learning modeling functionality,
including classification, regression, hyperparameter optimization, and model evaluation.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

from src.modeling import (
    create_classification_pipeline,
    create_regression_pipeline,
    evaluate_model,
    generate_model_card,
    optimize_hyperparameters,
    train_and_evaluate_models,
)


class TestModelingFunctions:
    """Test cases for modeling functions."""

    @pytest.fixture
    def classification_data(self):
        """Create sample classification data."""
        X, y = make_classification(
            n_samples=100,
            n_features=10,
            n_classes=2,
            random_state=42
        )
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        df = pd.DataFrame(X, columns=feature_names)
        df["target"] = y
        return df

    @pytest.fixture
    def regression_data(self):
        """Create sample regression data."""
        X, y = make_regression(
            n_samples=100,
            n_features=10,
            noise=0.1,
            random_state=42
        )
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        df = pd.DataFrame(X, columns=feature_names)
        df["target"] = y
        return df

    def test_create_classification_pipeline(self) -> None:
        """Test classification pipeline creation."""
        pipeline = create_classification_pipeline("random_forest")
        
        # Check pipeline has required steps
        assert hasattr(pipeline, "steps")
        assert len(pipeline.steps) >= 2  # At least preprocessor and classifier
        
        # Check pipeline can be fitted
        X, y = make_classification(n_samples=50, n_features=5, random_state=42)
        pipeline.fit(X, y)
        predictions = pipeline.predict(X)
        assert len(predictions) == len(y)

    def test_create_regression_pipeline(self) -> None:
        """Test regression pipeline creation."""
        pipeline = create_regression_pipeline("random_forest")
        
        # Check pipeline has required steps
        assert hasattr(pipeline, "steps")
        assert len(pipeline.steps) >= 2  # At least preprocessor and regressor
        
        # Check pipeline can be fitted
        X, y = make_regression(n_samples=50, n_features=5, random_state=42)
        pipeline.fit(X, y)
        predictions = pipeline.predict(X)
        assert len(predictions) == len(y)

    def test_evaluate_model(self, classification_data: pd.DataFrame) -> None:
        """Test model evaluation."""
        X = classification_data.drop("target", axis=1)
        y = classification_data["target"]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        pipeline = create_classification_pipeline("random_forest")
        pipeline.fit(X_train, y_train)
        
        metrics = evaluate_model(pipeline, X_test, y_test, task_type="classification")
        
        # Check that metrics are returned
        assert isinstance(metrics, dict)
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics

    def test_evaluate_regression_model(self, regression_data: pd.DataFrame) -> None:
        """Test regression model evaluation."""
        X = regression_data.drop("target", axis=1)
        y = regression_data["target"]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        pipeline = create_regression_pipeline("random_forest")
        pipeline.fit(X_train, y_train)
        
        metrics = evaluate_model(pipeline, X_test, y_test, task_type="regression")
        
        # Check that metrics are returned
        assert isinstance(metrics, dict)
        assert "mse" in metrics
        assert "mae" in metrics
        assert "r2" in metrics

    @patch('src.modeling.optuna.create_study')
    def test_optimize_hyperparameters(
        self,
        mock_create_study: Mock,
        classification_data: pd.DataFrame
    ) -> None:
        """Test hyperparameter optimization."""
        # Mock optuna study
        mock_study = Mock()
        mock_study.best_params = {"n_estimators": 100, "max_depth": 5}
        mock_study.best_value = 0.85
        mock_create_study.return_value = mock_study
        
        X = classification_data.drop("target", axis=1)
        y = classification_data["target"]
        
        best_params = optimize_hyperparameters(
            X, y, "classification", "random_forest", n_trials=5
        )
        
        # Check that best parameters are returned
        assert isinstance(best_params, dict)
        assert "n_estimators" in best_params
        mock_create_study.assert_called_once()

    def test_generate_model_card(self, classification_data: pd.DataFrame) -> None:
        """Test model card generation."""
        X = classification_data.drop("target", axis=1)
        y = classification_data["target"]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        pipeline = create_classification_pipeline("random_forest")
        pipeline.fit(X_train, y_train)
        
        metrics = evaluate_model(pipeline, X_test, y_test, task_type="classification")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "model_card.md"
            
            generate_model_card(
                model_name="Test Random Forest",
                model_type="Classification",
                metrics=metrics,
                feature_names=list(X.columns),
                target_name="target",
                output_path=str(output_path)
            )
            
            # Check file was created
            assert output_path.exists()
            
            # Check content
            content = output_path.read_text()
            assert "Test Random Forest" in content
            assert "Classification" in content
            assert "accuracy" in content

    def test_train_and_evaluate_models(self, classification_data: pd.DataFrame) -> None:
        """Test complete model training and evaluation pipeline."""
        feature_columns = [col for col in classification_data.columns if col != "target"]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            
            results = train_and_evaluate_models(
                classification_data,
                target_column="target",
                feature_columns=feature_columns,
                task_type="classification",
                output_dir=str(output_dir)
            )
            
            # Check that results are returned
            assert isinstance(results, dict)
            assert len(results) > 0
            
            # Check that each model has metrics
            for model_name, model_data in results.items():
                assert "metrics" in model_data
                assert "model" in model_data

    def test_invalid_model_type(self) -> None:
        """Test handling of invalid model type."""
        with pytest.raises(ValueError):
            create_classification_pipeline("invalid_model")
        
        with pytest.raises(ValueError):
            create_regression_pipeline("invalid_model")

    def test_invalid_task_type(self, classification_data: pd.DataFrame) -> None:
        """Test handling of invalid task type."""
        X = classification_data.drop("target", axis=1)
        y = classification_data["target"]
        
        pipeline = create_classification_pipeline("random_forest")
        pipeline.fit(X, y)
        
        with pytest.raises(ValueError):
            evaluate_model(pipeline, X, y, task_type="invalid_task")

    def test_empty_data_handling(self) -> None:
        """Test handling of empty datasets."""
        empty_df = pd.DataFrame()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            
            with pytest.raises((ValueError, KeyError)):
                train_and_evaluate_models(
                    empty_df,
                    target_column="target",
                    feature_columns=[],
                    task_type="classification",
                    output_dir=str(output_dir)
                )
