"""
WatchTower Hyperparameter Optimization Module.

Optuna-based hyperparameter tuning for XGBoost.
"""

from typing import Dict, Any, Optional, Callable
import time

import numpy as np
import pandas as pd
import optuna
from optuna.trial import Trial
import xgboost as xgb
from sklearn.metrics import f1_score
from loguru import logger

from src.config.settings import settings
from src.utils.logger import (
    console,
    print_header,
    print_success,
    print_info,
    print_metrics_table,
)
from src.training.train import XGBoostTrainer


class OptunaOptimizer:
    """
    Optuna-based hyperparameter optimization for XGBoost.
    
    Optimizes:
    - n_estimators
    - max_depth
    - learning_rate
    - min_child_weight
    - subsample
    - colsample_bytree
    - gamma
    - reg_alpha
    - reg_lambda
    """
    
    def __init__(
        self,
        n_trials: int = None,
        timeout: int = None,
        metric: str = "f1_macro",
        direction: str = "maximize",
        random_state: int = None,
    ):
        """
        Initialize optimizer.
        
        Args:
            n_trials: Number of Optuna trials
            timeout: Timeout in seconds
            metric: Optimization metric
            direction: 'maximize' or 'minimize'
            random_state: Random seed
        """
        self.n_trials = n_trials or settings.optuna_n_trials
        self.timeout = timeout or settings.optuna_timeout
        self.metric = metric
        self.direction = direction
        self.random_state = random_state or settings.random_seed
        
        self.study: Optional[optuna.Study] = None
        self.best_params: Dict[str, Any] = {}
        self.best_score: float = 0.0
    
    def optimize(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> Dict[str, Any]:
        """
        Run hyperparameter optimization.
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            
        Returns:
            Best hyperparameters
        """
        print_header("Hyperparameter Optimization", f"Optuna - {self.n_trials} trials")
        
        # Create objective function
        def objective(trial: Trial) -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "gamma": trial.suggest_float("gamma", 0, 5),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            }
            
            # Train model with trial parameters
            model = xgb.XGBClassifier(
                **params,
                tree_method="hist",  # Modern XGBoost uses hist
                device=settings.xgb_device,  # cuda or cpu
                random_state=self.random_state,
                eval_metric="mlogloss",
                verbosity=0,
                early_stopping_rounds=10,
            )
            
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
            
            # Evaluate
            y_pred = model.predict(X_val)
            score = f1_score(y_val, y_pred, average="macro", zero_division=0)
            
            return score
        
        # Create study with progress callback
        self.study = optuna.create_study(
            direction=self.direction,
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
        )
        
        # Custom callback for progress
        class ProgressCallback:
            def __init__(self, n_trials: int):
                self.n_trials = n_trials
                self.start_time = time.time()
            
            def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial):
                elapsed = time.time() - self.start_time
                best = study.best_value
                console.print(
                    f"  Trial {trial.number + 1}/{self.n_trials}: "
                    f"Score={trial.value:.4f}, Best={best:.4f}, "
                    f"Time={elapsed:.1f}s",
                    style="dim"
                )
        
        # Run optimization
        logger.info(f"Starting {self.n_trials} optimization trials...")
        
        with console.status("[bold cyan]Optimizing hyperparameters...", spinner="dots"):
            self.study.optimize(
                objective,
                n_trials=self.n_trials,
                timeout=self.timeout,
                callbacks=[ProgressCallback(self.n_trials)],
                show_progress_bar=False,
            )
        
        # Store best results
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value
        
        # Print results
        print_success(f"Optimization complete! Best {self.metric}: {self.best_score:.4f}")
        print_metrics_table(self.best_params, title="Best Hyperparameters")
        
        return self.best_params
    
    def get_best_trainer(self) -> XGBoostTrainer:
        """
        Get XGBoostTrainer with best hyperparameters.
        
        Returns:
            Configured XGBoostTrainer
        """
        if not self.best_params:
            raise RuntimeError("No optimization run yet. Call optimize() first.")
        
        return XGBoostTrainer(
            n_estimators=self.best_params.get("n_estimators", 100),
            max_depth=self.best_params.get("max_depth", 6),
            learning_rate=self.best_params.get("learning_rate", 0.1),
        )
    
    def get_optimization_history(self) -> pd.DataFrame:
        """
        Get optimization history as DataFrame.
        
        Returns:
            DataFrame with trial history
        """
        if self.study is None:
            return pd.DataFrame()
        
        trials = []
        for trial in self.study.trials:
            trial_data = {
                "number": trial.number,
                "value": trial.value,
                "state": trial.state.name,
            }
            trial_data.update(trial.params)
            trials.append(trial_data)
        
        return pd.DataFrame(trials)


def optimize_hyperparameters(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_trials: int = None,
) -> Dict[str, Any]:
    """
    Convenience function for hyperparameter optimization.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        n_trials: Number of trials
        
    Returns:
        Best hyperparameters
    """
    optimizer = OptunaOptimizer(n_trials=n_trials)
    return optimizer.optimize(X_train, y_train, X_val, y_val)


if __name__ == "__main__":
    # Test optimization (with small sample for speed)
    from src.data.ingestion import load_dataset
    from src.data.preprocessing import DataPreprocessor, create_train_val_test_split
    
    # Load small sample
    df = load_dataset(sample_frac=0.02)  # 2% for quick test
    
    preprocessor = DataPreprocessor()
    X, y = preprocessor.fit_transform(df)
    
    X_train, X_val, X_test, y_train, y_val, y_test = create_train_val_test_split(X, y)
    
    # Optimize with few trials
    optimizer = OptunaOptimizer(n_trials=5)
    best_params = optimizer.optimize(X_train, y_train, X_val, y_val)
    
    print(f"\nBest params: {best_params}")
