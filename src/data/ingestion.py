"""
WatchTower Data Ingestion Module.

Load and prepare the CICIDS2017 dataset with progress visualization.
"""

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from loguru import logger

from src.config.settings import settings
from src.utils.logger import (
    console,
    create_progress,
    print_header,
    print_success,
    print_dataset_info,
    print_error,
)


def load_dataset(
    path: Optional[Path] = None,
    sample_frac: Optional[float] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Load the CICIDS2017 dataset with progress tracking.
    
    Args:
        path: Path to CSV file. Defaults to settings.dataset_path
        sample_frac: Optional fraction to sample (for quick testing)
        random_state: Random state for sampling
        
    Returns:
        DataFrame with loaded data
    """
    print_header("🏰 WatchTower Data Ingestion", "Loading CICIDS2017 Dataset")
    
    data_path = path or settings.dataset_path
    
    if not data_path.exists():
        print_error(f"Dataset not found at: {data_path}")
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    
    logger.info(f"📂 Loading dataset from: {data_path}")
    
    # Get file size for progress indication
    file_size_mb = data_path.stat().st_size / (1024 * 1024)
    console.print(f"[dim]File size: {file_size_mb:.1f} MB[/dim]")
    
    with create_progress() as progress:
        task = progress.add_task(
            "[cyan]Reading CSV...",
            total=100,
        )
        
        # Read in chunks for progress tracking
        chunks = []
        chunk_size = 100_000
        
        # Count total rows first (fast estimate from file size)
        estimated_rows = int(file_size_mb * 4000)  # ~4000 rows per MB estimate
        
        reader = pd.read_csv(data_path, chunksize=chunk_size, low_memory=False)
        rows_read = 0
        
        for chunk in reader:
            chunks.append(chunk)
            rows_read += len(chunk)
            pct = min((rows_read / estimated_rows) * 100, 99)
            progress.update(task, completed=pct)
        
        progress.update(task, completed=100)
    
    df = pd.concat(chunks, ignore_index=True)
    
    # Sample if requested
    if sample_frac is not None and sample_frac < 1.0:
        original_len = len(df)
        df = df.sample(frac=sample_frac, random_state=random_state)
        logger.info(f"📊 Sampled {len(df):,} rows from {original_len:,} ({sample_frac*100:.0f}%)")
    
    print_success(f"Loaded {len(df):,} samples with {len(df.columns)} features")
    
    # Print dataset info
    if settings.target_column in df.columns:
        class_dist = df[settings.target_column].value_counts().to_dict()
        print_dataset_info(
            n_samples=len(df),
            n_features=len(df.columns) - 1,  # Exclude target
            class_distribution=class_dist,
        )
    else:
        print_dataset_info(n_samples=len(df), n_features=len(df.columns))
    
    return df


def get_feature_target_split(
    df: pd.DataFrame,
    target_column: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split dataframe into features and target.
    
    Args:
        df: Input DataFrame
        target_column: Target column name. Defaults to settings.target_column
        
    Returns:
        Tuple of (X, y)
    """
    target = target_column or settings.target_column
    
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in DataFrame")
    
    X = df.drop(columns=[target])
    y = df[target]
    
    logger.info(f"✂️ Split: X shape = {X.shape}, y shape = {y.shape}")
    
    return X, y


def validate_dataset(df: pd.DataFrame) -> dict:
    """
    Run basic validation checks on the dataset.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        Dictionary with validation results
    """
    from src.utils.logger import print_metrics_table
    
    logger.info("🔍 Running dataset validation...")
    
    results = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "missing_values": df.isnull().sum().sum(),
        "duplicate_rows": df.duplicated().sum(),
        "memory_usage_mb": df.memory_usage(deep=True).sum() / (1024 * 1024),
    }
    
    # Check for infinities
    numeric_cols = df.select_dtypes(include=["number"]).columns
    inf_count = 0
    for col in numeric_cols:
        inf_count += (~df[col].apply(lambda x: pd.isna(x) or abs(x) != float('inf'))).sum()
    results["infinite_values"] = inf_count
    
    print_metrics_table(results, title="Dataset Validation")
    
    # Validation status
    is_valid = (
        results["missing_values"] == 0 and
        results["infinite_values"] == 0
    )
    
    if is_valid:
        print_success("Dataset validation passed!")
    else:
        from src.utils.logger import print_warning
        print_warning("Dataset has quality issues - check missing/infinite values")
    
    return results


if __name__ == "__main__":
    # Quick test
    df = load_dataset(sample_frac=0.01)  # Load 1% for testing
    validate_dataset(df)
