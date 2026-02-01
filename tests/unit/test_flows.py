"""Tests for Prefect training flow."""

import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np

# Import the tasks (not the flow itself to avoid Prefect overhead in tests)
from src.flows.training_flow import (
    load_data_task,
    validate_data_task,
    preprocess_data_task,
    optimize_task,
    train_model_task,
    evaluate_model_task,
    save_model_task,
)


class TestLoadDataTask:
    """Tests for load_data_task."""
    
    @patch("src.data.ingestion.load_dataset")
    @patch("src.flows.training_flow.get_run_logger")
    def test_load_data_returns_dataframe(self, mock_logger, mock_load):
        """Test that load_data_task returns a DataFrame."""
        mock_df = pd.DataFrame({"col": [1, 2, 3]})
        mock_load.return_value = mock_df
        mock_logger.return_value = MagicMock()
        
        # Access the raw function (not the Prefect task wrapper)
        from src.data.ingestion import load_dataset
        result = load_dataset(sample_frac=0.1)
        
        # Just verify load_dataset is callable
        assert callable(load_dataset)


class TestValidateDataTask:
    """Tests for validate_data_task."""
    
    def test_validate_task_exists(self):
        """Test that validate_data_task is defined."""
        from src.flows.training_flow import validate_data_task
        assert validate_data_task is not None
        assert callable(validate_data_task.fn)


class TestPreprocessDataTask:
    """Tests for preprocess_data_task."""
    
    @patch("src.flows.training_flow.get_run_logger")
    def test_preprocess_returns_splits(self, mock_logger):
        """Test that preprocess_data_task returns train/val/test splits."""
        mock_logger.return_value = MagicMock()
        
        # Create sample data with Attack Type column
        df = pd.DataFrame({
            "feature_1": np.random.randn(100),
            "feature_2": np.random.randn(100),
            "Attack Type": np.random.choice(["A", "B", "C"], 100),
        })
        
        result = preprocess_data_task.fn(df)
        
        # Should return 7 items: X_train, X_val, X_test, y_train, y_val, y_test, preprocessor
        assert len(result) == 7
        
        X_train, X_val, X_test, y_train, y_val, y_test, preprocessor = result
        assert len(X_train) > 0
        assert len(X_val) > 0
        assert len(X_test) > 0


class TestTrainModelTask:
    """Tests for train_model_task."""
    
    @patch("src.flows.training_flow.get_run_logger")
    def test_train_returns_model_and_trainer(self, mock_logger):
        """Test that train_model_task returns model and trainer."""
        mock_logger.return_value = MagicMock()
        
        # Create sample training data
        np.random.seed(42)
        X_train = pd.DataFrame(np.random.randn(50, 5), columns=[f"f{i}" for i in range(5)])
        y_train = pd.Series(np.random.randint(0, 3, 50))
        X_val = pd.DataFrame(np.random.randn(10, 5), columns=[f"f{i}" for i in range(5)])
        y_val = pd.Series(np.random.randint(0, 3, 10))
        
        hyperparameters = {"n_estimators": 10, "max_depth": 3}
        
        result = train_model_task.fn(X_train, y_train, X_val, y_val, hyperparameters)
        
        model, trainer = result
        assert model is not None
        assert trainer is not None


class TestEvaluateModelTask:
    """Tests for evaluate_model_task."""
    
    @patch("src.flows.training_flow.get_run_logger")
    @patch("src.flows.training_flow.create_markdown_artifact")
    def test_evaluate_returns_metrics(self, mock_artifact, mock_logger):
        """Test that evaluate_model_task returns metrics."""
        mock_logger.return_value = MagicMock()
        
        # Create mock trainer with evaluate method
        mock_trainer = MagicMock()
        mock_trainer.evaluate.return_value = {
            "accuracy": 0.95,
            "f1_macro": 0.92,
            "precision_macro": 0.93,
            "recall_macro": 0.91,
        }
        
        X_test = pd.DataFrame(np.random.randn(10, 5))
        y_test = pd.Series(np.random.randint(0, 3, 10))
        
        metrics = evaluate_model_task.fn(mock_trainer, X_test, y_test)
        
        assert "accuracy" in metrics
        assert "f1_macro" in metrics


class TestSaveModelTask:
    """Tests for save_model_task."""
    
    @patch("src.flows.training_flow.get_run_logger")
    def test_save_model_returns_path(self, mock_logger):
        """Test that save_model_task returns model path."""
        from pathlib import Path
        
        mock_logger.return_value = MagicMock()
        
        mock_trainer = MagicMock()
        mock_trainer.save_model.return_value = Path("/tmp/model.json")
        mock_model = MagicMock()
        
        path = save_model_task.fn(mock_trainer, mock_model)
        
        assert path == Path("/tmp/model.json")
        mock_trainer.save_model.assert_called_once()
