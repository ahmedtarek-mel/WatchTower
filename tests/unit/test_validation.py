"""Tests for data validation module."""

import pytest
import pandas as pd
import numpy as np

from src.data.validation import DataValidator, validate_dataset


@pytest.fixture
def valid_data():
    """Create valid sample dataset."""
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        "feature_1": np.random.randn(n_samples),
        "feature_2": np.random.randn(n_samples) * 10,
        "feature_3": np.random.uniform(0, 100, n_samples),
        "feature_4": np.random.exponential(5, n_samples),
        "feature_5": np.random.normal(50, 10, n_samples),
        "feature_6": np.random.uniform(-10, 10, n_samples),
        "feature_7": np.random.randn(n_samples),
        "feature_8": np.random.randn(n_samples),
        "feature_9": np.random.randn(n_samples),
        "feature_10": np.random.randn(n_samples),
        "Attack Type": np.random.choice(["Benign", "Attack", "Scan"], n_samples),
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def invalid_data_missing():
    """Create dataset with missing values."""
    np.random.seed(42)
    n_samples = 100
    
    data = {
        "feature_1": np.random.randn(n_samples),
        "feature_2": np.random.randn(n_samples),
        "Attack Type": np.random.choice(["A", "B"], n_samples),
    }
    df = pd.DataFrame(data)
    df.loc[0:10, "feature_1"] = np.nan
    
    return df


class TestDataValidator:
    """Tests for DataValidator class."""
    
    def test_validate_valid_data(self, valid_data):
        """Test validation on valid data."""
        validator = DataValidator()
        results = validator.validate(valid_data)
        
        assert results["checks_passed"] > 0
        assert "details" in results
    
    def test_detects_missing_values(self, invalid_data_missing):
        """Test that missing values are detected."""
        validator = DataValidator()
        results = validator.validate(invalid_data_missing)
        
        # Should fail the no missing values check
        assert results["details"]["No Missing Values"]["passed"] == False
    
    def test_detects_target_column(self, valid_data):
        """Test that target column is detected."""
        validator = DataValidator()
        results = validator.validate(valid_data)
        
        assert results["details"]["Target Column Exists"]["passed"] == True
    
    def test_detects_missing_target(self, valid_data):
        """Test that missing target column is flagged."""
        validator = DataValidator(target_column="NonExistent")
        results = validator.validate(valid_data)
        
        assert results["details"]["Target Column Exists"]["passed"] == False
    
    def test_is_valid_property(self, valid_data):
        """Test is_valid property."""
        validator = DataValidator()
        results = validator.validate(valid_data)
        
        # Should be valid if no checks failed
        if results["checks_failed"] == 0:
            assert validator.is_valid == True


class TestValidateDatasetFunction:
    """Tests for validate_dataset convenience function."""
    
    def test_function_returns_results(self, valid_data):
        """Test that function returns validation results."""
        results = validate_dataset(valid_data)
        
        assert "total_rows" in results
        assert "total_columns" in results
        assert "checks_passed" in results
        assert "checks_failed" in results
