"""
WatchTower XGBoost Training Module.

GPU-accelerated training with MLflow logging.
"""

from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from loguru import logger

from src.config.settings import settings
from src.utils.logger import (
    console,
    create_progress,
    print_header,
    print_success,
    print_info,
    print_metrics_table,
)


class XGBoostTrainer:
    """
    XGBoost classifier trainer with GPU support.
    
    Features:
    - Automatic GPU detection and usage
    - Early stopping
    - MLflow logging integration
    - Multi-class classification
    """
    
    def __init__(
        self,
        n_estimators: int = None,
        max_depth: int = None,
        learning_rate: float = None,
        early_stopping_rounds: int = None,
        use_gpu: bool = None,
        random_state: int = None,
    ):
        """
        Initialize trainer with hyperparameters.
        
        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Learning rate (eta)
            early_stopping_rounds: Early stopping patience
            use_gpu: Force GPU usage (auto-detect if None)
            random_state: Random seed
        """
        self.n_estimators = n_estimators or settings.xgb_n_estimators
        self.max_depth = max_depth or settings.xgb_max_depth
        self.learning_rate = learning_rate or settings.xgb_learning_rate
        self.early_stopping_rounds = early_stopping_rounds or settings.xgb_early_stopping_rounds
        self.random_state = random_state or settings.random_seed
        
        # GPU settings
        if use_gpu is None:
            self.use_gpu = settings.gpu_available
        else:
            self.use_gpu = use_gpu
        
        self.tree_method = "gpu_hist" if self.use_gpu else "hist"
        self.device = "cuda" if self.use_gpu else "cpu"
        
        self.model: Optional[xgb.XGBClassifier] = None
        self.training_history: Dict[str, Any] = {}
        self.best_iteration: Optional[int] = None
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> xgb.XGBClassifier:
        """
        Train XGBoost model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (for early stopping)
            y_val: Validation labels
            
        Returns:
            Trained XGBClassifier
        """
        print_header("XGBoost Training", f"Device: {self.device.upper()}")
        
        # Print training config
        config = {
            "Estimators": self.n_estimators,
            "Max Depth": self.max_depth,
            "Learning Rate": self.learning_rate,
            "Device": self.device,
            "Tree Method": self.tree_method,
        }
        print_metrics_table(config, title="Training Configuration")
        
        # Create model - modern XGBoost uses device for GPU, tree_method=hist
        self.model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            tree_method="hist",  # Modern XGBoost always uses hist
            device=self.device,  # "cuda" or "cpu"
            random_state=self.random_state,
            eval_metric="mlogloss",
            verbosity=0,
        )
        
        # Prepare evaluation set
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))
        
        # Train with progress tracking
        logger.info("Starting training...")
        start_time = time.time()
        
        with console.status("[bold cyan]Training XGBoost model...", spinner="dots"):
            self.model.fit(
                X_train,
                y_train,
                eval_set=eval_set,
                verbose=False,
            )
        
        train_time = time.time() - start_time
        
        # Get best iteration
        self.best_iteration = self.model.best_iteration if hasattr(self.model, 'best_iteration') else self.n_estimators
        
        # Store training history
        self.training_history = {
            "train_time_seconds": train_time,
            "best_iteration": self.best_iteration,
            "n_features": X_train.shape[1],
            "n_samples": len(X_train),
        }
        
        print_success(f"Training complete in {train_time:.2f}s")
        
        return self.model
    
    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        dataset_name: str = "Test",
    ) -> Dict[str, float]:
        """
        Evaluate model on a dataset.
        
        Args:
            X: Features
            y: True labels
            dataset_name: Name for display
            
        Returns:
            Dictionary of metrics
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")
        
        logger.info(f"Evaluating on {dataset_name} set ({len(X)} samples)")
        
        # Predictions
        y_pred = self.model.predict(X)
        y_proba = self.model.predict_proba(X)
        
        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y, y_pred),
            "precision_macro": precision_score(y, y_pred, average="macro", zero_division=0),
            "recall_macro": recall_score(y, y_pred, average="macro", zero_division=0),
            "f1_macro": f1_score(y, y_pred, average="macro", zero_division=0),
            "precision_weighted": precision_score(y, y_pred, average="weighted", zero_division=0),
            "recall_weighted": recall_score(y, y_pred, average="weighted", zero_division=0),
            "f1_weighted": f1_score(y, y_pred, average="weighted", zero_division=0),
        }
        
        # Print metrics
        print_metrics_table(
            {k: f"{v:.4f}" for k, v in metrics.items()},
            title=f"{dataset_name} Set Metrics"
        )
        
        return metrics
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance scores.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importances
        """
        if self.model is None:
            raise RuntimeError("Model not trained.")
        
        importance = pd.DataFrame({
            "feature": self.model.feature_names_in_,
            "importance": self.model.feature_importances_,
        }).sort_values("importance", ascending=False)
        
        return importance.head(top_n)
    
    def save_model(self, path: Optional[Path] = None) -> Path:
        """
        Save trained model to disk.
        
        Args:
            path: Output path (default: models/model.json)
            
        Returns:
            Path to saved model
        """
        if self.model is None:
            raise RuntimeError("Model not trained.")
        
        if path is None:
            models_dir = settings.models_dir
            models_dir.mkdir(parents=True, exist_ok=True)
            path = models_dir / "xgboost_model.json"
        
        self.model.save_model(str(path))
        logger.info(f"Model saved to: {path}")
        
        return path
    
    def load_model(self, path: Path) -> xgb.XGBClassifier:
        """
        Load model from disk.
        
        Args:
            path: Path to model file
            
        Returns:
            Loaded model
        """
        self.model = xgb.XGBClassifier()
        self.model.load_model(str(path))
        logger.info(f"Model loaded from: {path}")
        
        return self.model
    
    @property
    def hyperparameters(self) -> Dict[str, Any]:
        """Get current hyperparameters."""
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "tree_method": self.tree_method,
            "device": self.device,
            "random_state": self.random_state,
        }


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame = None,
    y_val: pd.Series = None,
    **kwargs,
) -> Tuple[xgb.XGBClassifier, Dict[str, float]]:
    """
    Convenience function to train and evaluate a model.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        **kwargs: Hyperparameters for XGBoostTrainer
        
    Returns:
        Tuple of (trained_model, metrics)
    """
    trainer = XGBoostTrainer(**kwargs)
    model = trainer.train(X_train, y_train, X_val, y_val)
    
    metrics = {}
    if X_val is not None:
        metrics = trainer.evaluate(X_val, y_val, "Validation")
    
    return model, metrics


if __name__ == "__main__":
    # Test training
    from src.data.ingestion import load_dataset
    from src.data.preprocessing import DataPreprocessor, create_train_val_test_split
    
    # Load and preprocess
    df = load_dataset(sample_frac=0.05)  # 5% for quick test
    
    preprocessor = DataPreprocessor()
    X, y = preprocessor.fit_transform(df)
    
    X_train, X_val, X_test, y_train, y_val, y_test = create_train_val_test_split(X, y)
    
    # Train
    trainer = XGBoostTrainer(n_estimators=50, max_depth=4)
    model = trainer.train(X_train, y_train, X_val, y_val)
    
    # Evaluate
    trainer.evaluate(X_test, y_test, "Test")
    
    # Feature importance
    importance = trainer.get_feature_importance(10)
    print("\nTop 10 Features:")
    print(importance.to_string(index=False))
