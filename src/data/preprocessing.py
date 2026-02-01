"""
WatchTower Data Preprocessing Module.

Feature engineering and data transformation pipeline.
"""

from pathlib import Path
from typing import Optional, Tuple, List
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from loguru import logger

from src.config.settings import settings
from src.utils.logger import (
    console,
    create_progress,
    print_header,
    print_success,
    print_info,
    print_warning,
    print_metrics_table,
)


class DataPreprocessor:
    """
    Data preprocessing pipeline for CICIDS2017 dataset.
    
    Handles:
    - Label encoding
    - Feature scaling
    - Train/val/test splitting
    - Saving processed data
    """
    
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.feature_columns: List[str] = []
        self.target_column = settings.target_column
        self._is_fitted = False
    
    def fit_transform(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Fit the preprocessor and transform the data.
        
        Args:
            df: Input DataFrame
            target_column: Target column name (defaults to settings)
            
        Returns:
            Tuple of (X_transformed, y_encoded)
        """
        print_header("Data Preprocessing", "Feature Engineering Pipeline")
        
        target = target_column or self.target_column
        
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found")
        
        # Separate features and target
        X = df.drop(columns=[target])
        y = df[target]
        
        # Store feature columns
        self.feature_columns = X.columns.tolist()
        
        logger.info(f"Processing {len(X)} samples with {len(self.feature_columns)} features")
        
        with create_progress() as progress:
            task = progress.add_task("[cyan]Preprocessing data...", total=4)
            
            # Step 1: Handle missing values
            progress.update(task, description="[cyan]Handling missing values...")
            X = self._handle_missing_values(X)
            progress.advance(task)
            
            # Step 2: Handle infinite values
            progress.update(task, description="[cyan]Handling infinite values...")
            X = self._handle_infinite_values(X)
            progress.advance(task)
            
            # Step 3: Encode labels
            progress.update(task, description="[cyan]Encoding labels...")
            y_encoded = self.label_encoder.fit_transform(y)
            progress.advance(task)
            
            # Step 4: Scale features
            progress.update(task, description="[cyan]Scaling features...")
            X_scaled = pd.DataFrame(
                self.scaler.fit_transform(X),
                columns=self.feature_columns,
                index=X.index,
            )
            progress.advance(task)
        
        self._is_fitted = True
        
        # Print class distribution
        self._print_class_distribution(y)
        
        print_success(f"Preprocessing complete: {X_scaled.shape}")
        
        return X_scaled, pd.Series(y_encoded, index=y.index, name=target)
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using fitted preprocessor.
        
        Args:
            df: Input DataFrame (features only)
            
        Returns:
            Transformed DataFrame
        """
        if not self._is_fitted:
            raise RuntimeError("Preprocessor not fitted. Call fit_transform first.")
        
        # Ensure same columns
        missing_cols = set(self.feature_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")
        
        # Select and order columns
        X = df[self.feature_columns].copy()
        
        # Handle missing/infinite values
        X = self._handle_missing_values(X)
        X = self._handle_infinite_values(X)
        
        # Scale
        X_scaled = pd.DataFrame(
            self.scaler.transform(X),
            columns=self.feature_columns,
            index=X.index,
        )
        
        return X_scaled
    
    def encode_labels(self, y: pd.Series) -> np.ndarray:
        """Encode labels using fitted encoder."""
        return self.label_encoder.transform(y)
    
    def decode_labels(self, y_encoded: np.ndarray) -> np.ndarray:
        """Decode encoded labels back to original."""
        return self.label_encoder.inverse_transform(y_encoded)
    
    def _handle_missing_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """Replace missing values with column median."""
        missing_count = X.isnull().sum().sum()
        if missing_count > 0:
            print_warning(f"Found {missing_count} missing values - filling with median")
            X = X.fillna(X.median())
        return X
    
    def _handle_infinite_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """Replace infinite values with column max/min."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            inf_mask = np.isinf(X.values)
            inf_count = inf_mask.sum()
            
        if inf_count > 0:
            print_warning(f"Found {inf_count} infinite values - replacing with finite bounds")
            X = X.replace([np.inf, -np.inf], np.nan)
            X = X.fillna(X.median())
        return X
    
    def _print_class_distribution(self, y: pd.Series) -> None:
        """Print class distribution table."""
        from rich.table import Table
        
        dist = y.value_counts()
        total = len(y)
        
        table = Table(title="Class Distribution", show_header=True)
        table.add_column("Class", style="cyan")
        table.add_column("Count", style="green", justify="right")
        table.add_column("Percentage", style="yellow", justify="right")
        
        for label, count in dist.items():
            pct = (count / total) * 100
            table.add_row(str(label), f"{count:,}", f"{pct:.2f}%")
        
        console.print(table)
    
    @property
    def class_names(self) -> List[str]:
        """Get ordered class names."""
        return list(self.label_encoder.classes_)
    
    @property
    def n_classes(self) -> int:
        """Get number of classes."""
        return len(self.label_encoder.classes_)
    
    @property
    def n_features(self) -> int:
        """Get number of features."""
        return len(self.feature_columns)


def create_train_val_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = None,
    val_size: float = None,
    random_state: int = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Create train/validation/test split.
    
    Args:
        X: Features DataFrame
        y: Target Series
        test_size: Test set proportion (default from settings)
        val_size: Validation set proportion (default from settings)
        random_state: Random seed (default from settings)
        
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    test_size = test_size or settings.test_size
    val_size = val_size or settings.val_size
    random_state = random_state or settings.random_seed
    
    print_info(f"Creating train/val/test split ({1-test_size-val_size:.0%}/{val_size:.0%}/{test_size:.0%})")
    
    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    
    # Second split: train vs val
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_ratio,
        random_state=random_state,
        stratify=y_temp,
    )
    
    # Print split info
    splits = {
        "Train": len(X_train),
        "Validation": len(X_val),
        "Test": len(X_test),
    }
    print_metrics_table(splits, title="Data Split")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def save_processed_data(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
    output_dir: Optional[Path] = None,
) -> None:
    """
    Save processed datasets to parquet files.
    
    Args:
        X_train, X_val, X_test: Feature DataFrames
        y_train, y_val, y_test: Target Series
        output_dir: Output directory (default: settings.processed_data_dir)
    """
    output_dir = output_dir or settings.processed_data_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving processed data to: {output_dir}")
    
    with create_progress() as progress:
        task = progress.add_task("[cyan]Saving datasets...", total=6)
        
        # Combine X and y for each split
        for name, X, y in [
            ("train", X_train, y_train),
            ("val", X_val, y_val),
            ("test", X_test, y_test),
        ]:
            df = X.copy()
            df["target"] = y.values
            df.to_parquet(output_dir / f"{name}.parquet", index=False)
            progress.advance(task)
            progress.advance(task)
    
    print_success(f"Saved processed data to {output_dir}")


if __name__ == "__main__":
    # Test preprocessing
    from src.data.ingestion import load_dataset
    
    df = load_dataset(sample_frac=0.01)
    
    preprocessor = DataPreprocessor()
    X, y = preprocessor.fit_transform(df)
    
    X_train, X_val, X_test, y_train, y_val, y_test = create_train_val_test_split(X, y)
    
    print(f"\nPreprocessor info:")
    print(f"  Classes: {preprocessor.class_names}")
    print(f"  Features: {preprocessor.n_features}")
