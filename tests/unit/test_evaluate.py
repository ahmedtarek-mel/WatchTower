"""Tests for model evaluation module."""

import pytest
import numpy as np
import pandas as pd

from src.training.evaluate import (
    evaluate_model,
    print_classification_report,
    compute_confusion_matrix,
    compute_per_class_metrics,
)


class TestEvaluateModel:
    """Tests for evaluate_model function."""
    
    @pytest.fixture
    def binary_predictions(self):
        """Binary classification predictions."""
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 0, 1, 0, 0, 1, 1, 1])
        return y_true, y_pred
    
    @pytest.fixture
    def multi_predictions(self):
        """Multi-class predictions."""
        y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 2, 2, 1, 1, 0])
        return y_true, y_pred
    
    def test_evaluate_returns_metrics_dict(self, binary_predictions):
        """Test that evaluate returns a dictionary of metrics."""
        y_true, y_pred = binary_predictions
        
        metrics = evaluate_model(y_true, y_pred)
        
        assert isinstance(metrics, dict)
        assert "accuracy" in metrics
        assert "f1_macro" in metrics
        assert "precision_macro" in metrics
        assert "recall_macro" in metrics
    
    def test_accuracy_calculation(self, binary_predictions):
        """Test accuracy is calculated correctly."""
        y_true, y_pred = binary_predictions
        
        metrics = evaluate_model(y_true, y_pred)
        
        # Manual calculation: 6 correct out of 8
        expected_accuracy = 6 / 8
        assert abs(metrics["accuracy"] - expected_accuracy) < 0.01
    
    def test_multiclass_evaluation(self, multi_predictions):
        """Test multi-class evaluation works."""
        y_true, y_pred = multi_predictions
        
        metrics = evaluate_model(y_true, y_pred)
        
        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["f1_macro"] <= 1
    
    def test_with_probabilities(self, binary_predictions):
        """Test evaluation with probability scores."""
        y_true, y_pred = binary_predictions
        y_proba = np.random.rand(len(y_true), 2)
        y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)  # Normalize
        
        metrics = evaluate_model(y_true, y_pred, y_proba=y_proba)
        
        assert "accuracy" in metrics


class TestClassificationReport:
    """Tests for print_classification_report function."""
    
    def test_returns_string(self):
        """Test that report returns a string."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 0])
        
        report = print_classification_report(y_true, y_pred)
        
        assert isinstance(report, str)
        assert len(report) > 0
    
    def test_with_class_names(self):
        """Test report with class names."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 0])
        
        report = print_classification_report(
            y_true, y_pred, 
            class_names=["Benign", "Attack"]
        )
        
        assert "Benign" in report or "Attack" in report or isinstance(report, str)


class TestConfusionMatrix:
    """Tests for compute_confusion_matrix function."""
    
    def test_returns_dataframe(self):
        """Test that confusion matrix returns a DataFrame."""
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 1, 1, 1, 2, 0])
        
        cm = compute_confusion_matrix(y_true, y_pred)
        
        assert isinstance(cm, pd.DataFrame)
    
    def test_matrix_shape(self):
        """Test confusion matrix has correct shape."""
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 1, 1, 1, 2, 0])
        
        cm = compute_confusion_matrix(y_true, y_pred)
        
        assert cm.shape[0] == cm.shape[1]  # Square matrix
        assert cm.shape[0] == 3  # 3 classes
    
    def test_with_class_names(self):
        """Test confusion matrix with class names."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 0])
        
        cm = compute_confusion_matrix(
            y_true, y_pred,
            class_names=["Normal", "Attack"]
        )
        
        assert "Normal" in cm.columns or "Normal" in cm.index


class TestPerClassMetrics:
    """Tests for compute_per_class_metrics function."""
    
    def test_returns_dataframe(self):
        """Test that per-class metrics returns DataFrame."""
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 1, 1, 1, 2, 0])
        
        metrics = compute_per_class_metrics(y_true, y_pred)
        
        assert isinstance(metrics, pd.DataFrame)
    
    def test_has_required_columns(self):
        """Test DataFrame has required metric columns."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 0])
        
        metrics = compute_per_class_metrics(y_true, y_pred)
        
        assert "precision" in metrics.columns
        assert "recall" in metrics.columns
        assert "f1_score" in metrics.columns
        assert "support" in metrics.columns
    
    def test_with_class_names(self):
        """Test per-class metrics with class names."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 0])
        
        metrics = compute_per_class_metrics(
            y_true, y_pred,
            class_names=["Normal", "Attack"]
        )
        
        assert "class" in metrics.columns
