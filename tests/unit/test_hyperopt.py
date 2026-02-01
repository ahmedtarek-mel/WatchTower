"""Tests for hyperparameter optimization module."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from src.training.hyperopt import OptunaOptimizer


class TestOptunaOptimizer:
    """Tests for OptunaOptimizer class."""
    
    @pytest.fixture
    def small_data(self):
        """Create small dataset for testing."""
        np.random.seed(42)
        n = 100
        n_features = 5
        
        X_train = pd.DataFrame(
            np.random.randn(n, n_features),
            columns=[f"feature_{i}" for i in range(n_features)]
        )
        y_train = pd.Series(np.random.randint(0, 3, n))
        
        X_val = pd.DataFrame(
            np.random.randn(n // 4, n_features),
            columns=[f"feature_{i}" for i in range(n_features)]
        )
        y_val = pd.Series(np.random.randint(0, 3, n // 4))
        
        return X_train, y_train, X_val, y_val
    
    def test_optimizer_initialization(self):
        """Test optimizer can be initialized."""
        optimizer = OptunaOptimizer(n_trials=5)
        
        assert optimizer.n_trials == 5
        assert optimizer.study is None
        assert optimizer.best_params == {}
    
    def test_optimizer_direction(self):
        """Test optimizer direction setting."""
        optimizer = OptunaOptimizer(direction="maximize")
        assert optimizer.direction == "maximize"
        
        optimizer_min = OptunaOptimizer(direction="minimize")
        assert optimizer_min.direction == "minimize"
    
    def test_optimizer_metric(self):
        """Test optimizer metric setting."""
        optimizer = OptunaOptimizer(metric="accuracy")
        assert optimizer.metric == "accuracy"
    
    def test_optimize_returns_params(self, small_data):
        """Test optimize returns best parameters."""
        X_train, y_train, X_val, y_val = small_data
        
        optimizer = OptunaOptimizer(n_trials=2)  # Very few trials for speed
        best_params = optimizer.optimize(X_train, y_train, X_val, y_val)
        
        assert isinstance(best_params, dict)
        assert len(best_params) > 0
    
    def test_optimize_sets_best_score(self, small_data):
        """Test optimize sets best_score."""
        X_train, y_train, X_val, y_val = small_data
        
        optimizer = OptunaOptimizer(n_trials=2)
        optimizer.optimize(X_train, y_train, X_val, y_val)
        
        assert optimizer.best_score >= 0
    
    def test_optimize_creates_study(self, small_data):
        """Test optimize creates Optuna study."""
        X_train, y_train, X_val, y_val = small_data
        
        optimizer = OptunaOptimizer(n_trials=2)
        optimizer.optimize(X_train, y_train, X_val, y_val)
        
        assert optimizer.study is not None
    
    def test_get_best_trainer_raises_before_optimize(self):
        """Test get_best_trainer raises error before optimization."""
        optimizer = OptunaOptimizer()
        
        with pytest.raises(RuntimeError, match="No optimization"):
            optimizer.get_best_trainer()
    
    def test_get_best_trainer_after_optimize(self, small_data):
        """Test get_best_trainer returns trainer after optimization."""
        X_train, y_train, X_val, y_val = small_data
        
        optimizer = OptunaOptimizer(n_trials=2)
        optimizer.optimize(X_train, y_train, X_val, y_val)
        
        trainer = optimizer.get_best_trainer()
        assert trainer is not None
    
    def test_get_optimization_history_empty_before(self):
        """Test history is empty before optimization."""
        optimizer = OptunaOptimizer()
        
        history = optimizer.get_optimization_history()
        
        assert isinstance(history, pd.DataFrame)
        assert len(history) == 0
    
    def test_get_optimization_history_after(self, small_data):
        """Test history is populated after optimization."""
        X_train, y_train, X_val, y_val = small_data
        
        optimizer = OptunaOptimizer(n_trials=2)
        optimizer.optimize(X_train, y_train, X_val, y_val)
        
        history = optimizer.get_optimization_history()
        
        assert len(history) == 2  # 2 trials
        assert "number" in history.columns
        assert "value" in history.columns
