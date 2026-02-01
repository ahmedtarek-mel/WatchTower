"""
WatchTower Configuration Module.

Centralized configuration using Pydantic Settings with automatic GPU detection.
All settings can be overridden via environment variables.
"""

import os
from pathlib import Path
from functools import lru_cache
from typing import Literal

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="WATCHTOWER_",
        extra="ignore",
    )
    
    # =========================================================================
    # Project Paths
    # =========================================================================
    project_root: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent.parent,
        description="Root directory of the project"
    )
    
    @computed_field
    @property
    def data_dir(self) -> Path:
        """Directory for all data files."""
        return self.project_root / "data"
    
    @computed_field
    @property
    def raw_data_dir(self) -> Path:
        """Directory for raw data files."""
        return self.data_dir / "raw"
    
    @computed_field
    @property
    def processed_data_dir(self) -> Path:
        """Directory for processed data files."""
        return self.data_dir / "processed"
    
    @computed_field
    @property
    def dataset_path(self) -> Path:
        """Path to the main dataset file."""
        # Check if dataset exists in dataset/ folder first (current location)
        dataset_folder = self.project_root / "dataset" / "cicids2017_cleaned.csv"
        if dataset_folder.exists():
            return dataset_folder
        return self.raw_data_dir / "cicids2017_cleaned.csv"
    
    @computed_field
    @property
    def models_dir(self) -> Path:
        """Directory for saved models."""
        return self.project_root / "models"
    
    # =========================================================================
    # MLflow Configuration
    # =========================================================================
    mlflow_tracking_uri: str = Field(
        default="http://localhost:5000",
        description="MLflow tracking server URI"
    )
    mlflow_experiment_name: str = Field(
        default="watchtower-intrusion-detection",
        description="MLflow experiment name"
    )
    
    # =========================================================================
    # Training Configuration
    # =========================================================================
    random_seed: int = Field(default=42, description="Random seed for reproducibility")
    test_size: float = Field(default=0.2, ge=0.1, le=0.4, description="Test set proportion")
    val_size: float = Field(default=0.1, ge=0.05, le=0.3, description="Validation set proportion")
    
    # XGBoost defaults
    xgb_n_estimators: int = Field(default=100, ge=10, le=1000)
    xgb_max_depth: int = Field(default=6, ge=1, le=15)
    xgb_learning_rate: float = Field(default=0.1, ge=0.001, le=1.0)
    xgb_early_stopping_rounds: int = Field(default=10, ge=5, le=50)
    
    # Optuna hyperparameter tuning
    optuna_n_trials: int = Field(default=50, ge=10, le=200)
    optuna_timeout: int = Field(default=3600, description="Timeout in seconds")
    
    # =========================================================================
    # GPU Configuration
    # =========================================================================
    force_cpu: bool = Field(default=False, description="Force CPU even if GPU available")
    
    @computed_field
    @property
    def gpu_available(self) -> bool:
        """Check if CUDA GPU is available for XGBoost."""
        if self.force_cpu:
            return False
        try:
            import xgboost as xgb
            # Try to create a small GPU-enabled booster
            params = {"tree_method": "gpu_hist", "device": "cuda"}
            xgb.XGBClassifier(**params)
            return True
        except Exception:
            return False
    
    @computed_field
    @property
    def xgb_tree_method(self) -> str:
        """XGBoost tree method - always hist for modern XGBoost."""
        return "hist"  # Modern XGBoost uses device param for GPU, not tree_method
    
    @computed_field
    @property
    def xgb_device(self) -> str:
        """XGBoost device setting."""
        return "cuda" if self.gpu_available else "cpu"
    
    # =========================================================================
    # API Configuration
    # =========================================================================
    api_host: str = Field(default="0.0.0.0", description="API host")
    api_port: int = Field(default=8000, ge=1024, le=65535, description="API port")
    api_reload: bool = Field(default=True, description="Auto-reload in development")
    
    # =========================================================================
    # Logging Configuration
    # =========================================================================
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Logging level"
    )
    log_to_file: bool = Field(default=True, description="Enable file logging")
    
    @computed_field
    @property
    def log_file(self) -> Path:
        """Path to log file."""
        logs_dir = self.project_root / "logs"
        logs_dir.mkdir(exist_ok=True)
        return logs_dir / "watchtower.log"
    
    # =========================================================================
    # Dataset Configuration
    # =========================================================================
    target_column: str = Field(default="Attack Type", description="Target column name")
    attack_classes: list[str] = Field(
        default=["Benign", "DoS", "PortScan", "Brute Force", "Web Attack", "Bot"],
        description="Attack class labels"
    )
    
    def print_config(self) -> None:
        """Print current configuration using Rich."""
        import os
        import sys
        
        # Force UTF-8 output on Windows
        if sys.platform == 'win32':
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        
        from rich.console import Console
        from rich.table import Table
        
        console = Console(force_terminal=True)
        table = Table(title="WatchTower Configuration", show_header=True)
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Project Root", str(self.project_root))
        table.add_row("Dataset Path", str(self.dataset_path))
        table.add_row("GPU Available", "[green]Yes[/green]" if self.gpu_available else "[red]No[/red]")
        table.add_row("XGBoost Device", self.xgb_device)
        table.add_row("Tree Method", self.xgb_tree_method)
        table.add_row("MLflow URI", self.mlflow_tracking_uri)
        table.add_row("Log Level", self.log_level)
        table.add_row("Random Seed", str(self.random_seed))
        
        console.print(table)


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Global settings instance
settings = get_settings()


if __name__ == "__main__":
    # Test configuration
    settings.print_config()
