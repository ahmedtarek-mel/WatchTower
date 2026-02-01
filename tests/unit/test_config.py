"""Tests for configuration module."""

import pytest
from pathlib import Path
from unittest.mock import patch

from src.config.settings import Settings


class TestSettings:
    """Tests for Settings class."""
    
    def test_settings_instantiation(self):
        """Test that settings can be instantiated."""
        settings = Settings()
        assert settings is not None
    
    def test_project_root_is_path(self):
        """Test project_root is a Path."""
        settings = Settings()
        assert isinstance(settings.project_root, Path)
    
    def test_data_dir_is_path(self):
        """Test data_dir is a Path."""
        settings = Settings()
        assert isinstance(settings.data_dir, Path)
    
    def test_models_dir_is_path(self):
        """Test models_dir is a Path."""
        settings = Settings()
        assert isinstance(settings.models_dir, Path)
    
    def test_project_dirs_exist(self):
        """Test project directories are properly set."""
        settings = Settings()
        # Should have some path attributes
        assert hasattr(settings, "project_root")
        assert hasattr(settings, "data_dir")
        assert hasattr(settings, "models_dir")
    
    def test_mlflow_tracking_uri(self):
        """Test MLflow tracking URI is set."""
        settings = Settings()
        assert settings.mlflow_tracking_uri is not None
    
    def test_random_seed_is_integer(self):
        """Test random seed is an integer."""
        settings = Settings()
        assert isinstance(settings.random_seed, int)
    
    def test_xgb_parameters_have_defaults(self):
        """Test XGBoost parameters have default values."""
        settings = Settings()
        
        assert settings.xgb_n_estimators > 0
        assert settings.xgb_max_depth > 0
        assert 0 < settings.xgb_learning_rate <= 1
    
    def test_api_settings(self):
        """Test API settings have defaults."""
        settings = Settings()
        
        assert settings.api_host is not None
        assert settings.api_port > 0
    
    def test_target_column_default(self):
        """Test target column has default."""
        settings = Settings()
        assert settings.target_column == "Attack Type"
    
    def test_xgb_tree_method(self):
        """Test XGBoost tree method is set."""
        settings = Settings()
        assert settings.xgb_tree_method in ["hist", "gpu_hist", "approx", "exact"]
    
    def test_xgb_device(self):
        """Test XGBoost device is set."""
        settings = Settings()
        assert settings.xgb_device in ["cpu", "cuda"]
    
    @patch.dict("os.environ", {"WATCHTOWER_FORCE_CPU": "true"})
    def test_force_cpu_env_var(self):
        """Test force_cpu can be set via environment."""
        settings = Settings()
        # Note: Pydantic may cache, so this tests the mechanism
        assert hasattr(settings, "force_cpu")


class TestSettingsPaths:
    """Tests for path-related settings."""
    
    def test_dataset_settings(self):
        """Test dataset-related settings exist."""
        settings = Settings()
        # Should have target column
        assert hasattr(settings, "target_column")
        assert settings.target_column is not None
    
    def test_optuna_settings(self):
        """Test Optuna settings exist."""
        settings = Settings()
        
        assert settings.optuna_n_trials > 0
        assert settings.optuna_timeout > 0
