"""Tests for data preprocessing module."""

import pytest
import pandas as pd
import numpy as np

from src.data.preprocessing import DataPreprocessor, create_train_val_test_split


@pytest.fixture
def sample_data():
    """Create sample dataset for testing."""
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        "feature_1": np.random.randn(n_samples),
        "feature_2": np.random.randn(n_samples) * 10,
        "feature_3": np.random.uniform(0, 100, n_samples),
        "Attack Type": np.random.choice(["Benign", "Attack", "Scan"], n_samples),
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def preprocessor():
    """Create preprocessor instance."""
    return DataPreprocessor()


class TestDataPreprocessor:
    """Tests for DataPreprocessor class."""
    
    def test_fit_transform_returns_correct_shapes(self, sample_data, preprocessor):
        """Test that fit_transform returns correct shapes."""
        X, y = preprocessor.fit_transform(sample_data)
        
        assert len(X) == len(sample_data)
        assert len(y) == len(sample_data)
        assert X.shape[1] == len(sample_data.columns) - 1  # Exclude Attack Type
    
    def test_labels_are_encoded(self, sample_data, preprocessor):
        """Test that labels are numerically encoded."""
        X, y = preprocessor.fit_transform(sample_data)
        
        assert y.dtype in [np.int32, np.int64]
        assert y.min() >= 0
        assert y.max() < preprocessor.n_classes
    
    def test_features_are_scaled(self, sample_data, preprocessor):
        """Test that features are standardized."""
        X, y = preprocessor.fit_transform(sample_data)
        
        # After scaling, mean should be ~0 and std ~1
        assert abs(X.mean().mean()) < 0.1
        assert abs(X.std().mean() - 1.0) < 0.2
    
    def test_transform_uses_fitted_scaler(self, sample_data, preprocessor):
        """Test that transform uses the fitted scaler."""
        X, y = preprocessor.fit_transform(sample_data)
        
        # Transform same data again
        X_transformed = preprocessor.transform(sample_data.drop(columns=["Attack Type"]))
        
        # Should produce same result
        pd.testing.assert_frame_equal(X, X_transformed)
    
    def test_missing_values_handled(self, sample_data, preprocessor):
        """Test that missing values are handled."""
        # Add some missing values
        sample_data.loc[0:10, "feature_1"] = np.nan
        
        X, y = preprocessor.fit_transform(sample_data)
        
        assert not X.isnull().any().any()
    
    def test_class_names_property(self, sample_data, preprocessor):
        """Test class_names property."""
        X, y = preprocessor.fit_transform(sample_data)
        
        assert len(preprocessor.class_names) == 3
        assert set(preprocessor.class_names) == {"Benign", "Attack", "Scan"}


class TestTrainValTestSplit:
    """Tests for train/val/test split function."""
    
    def test_split_proportions(self, sample_data, preprocessor):
        """Test that splits have correct proportions."""
        X, y = preprocessor.fit_transform(sample_data)
        
        X_train, X_val, X_test, y_train, y_val, y_test = create_train_val_test_split(
            X, y, test_size=0.2, val_size=0.1
        )
        
        total = len(X)
        assert abs(len(X_test) / total - 0.2) < 0.05
        assert abs(len(X_val) / total - 0.1) < 0.05
    
    def test_no_data_leakage(self, sample_data, preprocessor):
        """Test that there's no overlap between splits."""
        X, y = preprocessor.fit_transform(sample_data)
        
        X_train, X_val, X_test, y_train, y_val, y_test = create_train_val_test_split(X, y)
        
        train_idx = set(X_train.index)
        val_idx = set(X_val.index)
        test_idx = set(X_test.index)
        
        assert len(train_idx & val_idx) == 0
        assert len(train_idx & test_idx) == 0
        assert len(val_idx & test_idx) == 0
    
    def test_stratification(self, sample_data, preprocessor):
        """Test that splits maintain class distribution."""
        X, y = preprocessor.fit_transform(sample_data)
        
        X_train, X_val, X_test, y_train, y_val, y_test = create_train_val_test_split(X, y)
        
        # Check class distribution is similar across splits
        train_dist = pd.Series(y_train).value_counts(normalize=True)
        test_dist = pd.Series(y_test).value_counts(normalize=True)
        
        for cls in train_dist.index:
            assert abs(train_dist[cls] - test_dist[cls]) < 0.1
