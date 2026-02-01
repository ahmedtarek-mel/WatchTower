"""Tests for XGBoost training module."""

import pytest
import pandas as pd
import numpy as np

from src.training.train import XGBoostTrainer, train_model


@pytest.fixture
def training_data():
    """Create sample training data."""
    np.random.seed(42)
    n_train = 500
    n_val = 100
    n_features = 10
    
    X_train = pd.DataFrame(
        np.random.randn(n_train, n_features),
        columns=[f"feature_{i}" for i in range(n_features)]
    )
    y_train = pd.Series(np.random.randint(0, 3, n_train))
    
    X_val = pd.DataFrame(
        np.random.randn(n_val, n_features),
        columns=[f"feature_{i}" for i in range(n_features)]
    )
    y_val = pd.Series(np.random.randint(0, 3, n_val))
    
    return X_train, y_train, X_val, y_val


class TestXGBoostTrainer:
    """Tests for XGBoostTrainer class."""
    
    def test_train_returns_model(self, training_data):
        """Test that training returns a model."""
        X_train, y_train, X_val, y_val = training_data
        
        trainer = XGBoostTrainer(n_estimators=10, max_depth=3)
        model = trainer.train(X_train, y_train, X_val, y_val)
        
        assert model is not None
        assert trainer.model is not None
    
    def test_predict_after_train(self, training_data):
        """Test predictions after training."""
        X_train, y_train, X_val, y_val = training_data
        
        trainer = XGBoostTrainer(n_estimators=10, max_depth=3)
        trainer.train(X_train, y_train)
        
        predictions = trainer.model.predict(X_val)
        
        assert len(predictions) == len(X_val)
        assert all(p in [0, 1, 2] for p in predictions)
    
    def test_evaluate_returns_metrics(self, training_data):
        """Test that evaluate returns metrics dict."""
        X_train, y_train, X_val, y_val = training_data
        
        trainer = XGBoostTrainer(n_estimators=10, max_depth=3)
        trainer.train(X_train, y_train)
        metrics = trainer.evaluate(X_val, y_val)
        
        assert "accuracy" in metrics
        assert "f1_macro" in metrics
        assert 0 <= metrics["accuracy"] <= 1
    
    def test_feature_importance(self, training_data):
        """Test feature importance extraction."""
        X_train, y_train, X_val, y_val = training_data
        
        trainer = XGBoostTrainer(n_estimators=10, max_depth=3)
        trainer.train(X_train, y_train)
        
        importance = trainer.get_feature_importance(5)
        
        assert len(importance) == 5
        assert "feature" in importance.columns
        assert "importance" in importance.columns
    
    def test_hyperparameters_property(self, training_data):
        """Test hyperparameters property."""
        trainer = XGBoostTrainer(
            n_estimators=50,
            max_depth=5,
            learning_rate=0.05,
        )
        
        params = trainer.hyperparameters
        
        assert params["n_estimators"] == 50
        assert params["max_depth"] == 5
        assert params["learning_rate"] == 0.05


class TestTrainModelFunction:
    """Tests for train_model convenience function."""
    
    def test_function_returns_model_and_metrics(self, training_data):
        """Test that function returns model and metrics."""
        X_train, y_train, X_val, y_val = training_data
        
        model, metrics = train_model(
            X_train, y_train, X_val, y_val,
            n_estimators=10, max_depth=3
        )
        
        assert model is not None
        assert "f1_macro" in metrics
