"""
WatchTower Prefect Training Flow.

Orchestrated training pipeline with Prefect.
"""

from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

import pandas as pd
from prefect import flow, task, get_run_logger
from prefect.artifacts import create_markdown_artifact

from src.config.settings import settings


@task(name="load-dataset", retries=2, retry_delay_seconds=10)
def load_data_task(sample_frac: Optional[float] = None) -> pd.DataFrame:
    """Load the CICIDS2017 dataset."""
    from src.data.ingestion import load_dataset
    
    logger = get_run_logger()
    logger.info("Loading dataset...")
    
    df = load_dataset(sample_frac=sample_frac)
    logger.info(f"Loaded {len(df)} samples")
    
    return df


@task(name="validate-data")
def validate_data_task(df: pd.DataFrame) -> Dict[str, Any]:
    """Validate dataset quality."""
    from src.data.validation import validate_dataset
    
    logger = get_run_logger()
    logger.info("Running data validation...")
    
    results = validate_dataset(df)
    
    if results["checks_failed"] > 0:
        logger.warning(f"{results['checks_failed']} validation checks failed")
    else:
        logger.info("All validation checks passed")
    
    return results


@task(name="preprocess-data")
def preprocess_data_task(df: pd.DataFrame) -> tuple:
    """Preprocess and split data."""
    from src.data.preprocessing import DataPreprocessor, create_train_val_test_split
    
    logger = get_run_logger()
    logger.info("Preprocessing data...")
    
    preprocessor = DataPreprocessor()
    X, y = preprocessor.fit_transform(df)
    
    X_train, X_val, X_test, y_train, y_val, y_test = create_train_val_test_split(X, y)
    
    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, preprocessor


@task(name="optimize-hyperparameters")
def optimize_task(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_trials: int = 20,
) -> Dict[str, Any]:
    """Run hyperparameter optimization."""
    from src.training.hyperopt import OptunaOptimizer
    
    logger = get_run_logger()
    logger.info(f"Running {n_trials} optimization trials...")
    
    optimizer = OptunaOptimizer(n_trials=n_trials)
    best_params = optimizer.optimize(X_train, y_train, X_val, y_val)
    
    logger.info(f"Best F1: {optimizer.best_score:.4f}")
    
    return best_params


@task(name="train-model")
def train_model_task(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    hyperparameters: Optional[Dict[str, Any]] = None,
) -> tuple:
    """Train XGBoost model."""
    from src.training.train import XGBoostTrainer
    
    logger = get_run_logger()
    logger.info("Training model...")
    
    # Use provided hyperparameters or defaults
    params = hyperparameters or {}
    
    trainer = XGBoostTrainer(
        n_estimators=params.get("n_estimators", 100),
        max_depth=params.get("max_depth", 6),
        learning_rate=params.get("learning_rate", 0.1),
    )
    
    model = trainer.train(X_train, y_train, X_val, y_val)
    
    logger.info(f"Training complete on {trainer.device}")
    
    return model, trainer


@task(name="evaluate-model")
def evaluate_model_task(
    trainer,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, float]:
    """Evaluate trained model."""
    logger = get_run_logger()
    logger.info("Evaluating model...")
    
    metrics = trainer.evaluate(X_test, y_test, "Test")
    
    # Create artifact with results
    markdown = f"""
# Model Evaluation Results

| Metric | Value |
|--------|-------|
| Accuracy | {metrics['accuracy']:.4f} |
| F1 (Macro) | {metrics['f1_macro']:.4f} |
| Precision | {metrics['precision_macro']:.4f} |
| Recall | {metrics['recall_macro']:.4f} |

**Evaluated on {len(X_test)} samples**
"""
    create_markdown_artifact(
        key="model-metrics",
        markdown=markdown,
        description="Model evaluation metrics",
    )
    
    return metrics


@task(name="save-model")
def save_model_task(trainer, model) -> Path:
    """Save trained model."""
    logger = get_run_logger()
    
    model_path = trainer.save_model()
    logger.info(f"Model saved to {model_path}")
    
    return model_path


@flow(name="watchtower-training-pipeline")
def training_flow(
    sample_frac: Optional[float] = None,
    run_optimization: bool = True,
    n_trials: int = 20,
) -> Dict[str, Any]:
    """
    Complete training pipeline.
    
    Args:
        sample_frac: Fraction of data to use (None for full dataset)
        run_optimization: Whether to run hyperparameter optimization
        n_trials: Number of Optuna trials if optimizing
        
    Returns:
        Dictionary with training results
    """
    logger = get_run_logger()
    logger.info("Starting WatchTower training pipeline")
    
    # Load data
    df = load_data_task(sample_frac=sample_frac)
    
    # Validate
    validation_results = validate_data_task(df)
    
    # Preprocess
    X_train, X_val, X_test, y_train, y_val, y_test, preprocessor = preprocess_data_task(df)
    
    # Optimize hyperparameters (optional)
    hyperparameters = None
    if run_optimization:
        hyperparameters = optimize_task(X_train, y_train, X_val, y_val, n_trials)
    
    # Train
    model, trainer = train_model_task(X_train, y_train, X_val, y_val, hyperparameters)
    
    # Evaluate
    metrics = evaluate_model_task(trainer, X_test, y_test)
    
    # Save model
    model_path = save_model_task(trainer, model)
    
    logger.info("Training pipeline complete!")
    
    return {
        "metrics": metrics,
        "model_path": str(model_path),
        "hyperparameters": hyperparameters,
        "validation_results": validation_results,
    }


if __name__ == "__main__":
    # Run the flow locally
    result = training_flow(sample_frac=0.05, run_optimization=False)
    print(f"\nPipeline complete!")
    print(f"Metrics: {result['metrics']}")
    print(f"Model: {result['model_path']}")
