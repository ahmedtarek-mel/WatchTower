"""
WatchTower Data Validation Module.

Great Expectations-based data quality validation.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional

import pandas as pd
from loguru import logger

from src.config.settings import settings
from src.utils.logger import (
    console,
    create_progress,
    print_header,
    print_success,
    print_error,
    print_warning,
    print_metrics_table,
)


class DataValidator:
    """
    Data validation using custom expectations.
    
    Validates:
    - Column presence
    - Data types
    - Value ranges
    - Missing values
    - Class distribution
    """
    
    def __init__(self, target_column: str = None):
        self.target_column = target_column or settings.target_column
        self.expected_columns: List[str] = []
        self.validation_results: Dict[str, Any] = {}
    
    def validate(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Run all validation checks on the dataset.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dictionary with validation results
        """
        print_header("Data Validation", "Running Quality Checks")
        
        results = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "checks_passed": 0,
            "checks_failed": 0,
            "details": {},
        }
        
        checks = [
            ("Column Count", self._check_column_count, df),
            ("Target Column Exists", self._check_target_exists, df),
            ("No Missing Values", self._check_no_missing, df),
            ("No Infinite Values", self._check_no_infinite, df),
            ("No Duplicate Rows", self._check_no_duplicates, df),
            ("Numeric Features", self._check_numeric_features, df),
            ("Target Has Multiple Classes", self._check_class_distribution, df),
            ("Feature Value Ranges", self._check_value_ranges, df),
        ]
        
        with create_progress() as progress:
            task = progress.add_task(
                "[cyan]Running validation checks...",
                total=len(checks)
            )
            
            for check_name, check_func, data in checks:
                progress.update(task, description=f"[cyan]Checking: {check_name}...")
                
                try:
                    passed, details = check_func(data)
                    results["details"][check_name] = {
                        "passed": passed,
                        "details": details,
                    }
                    
                    if passed:
                        results["checks_passed"] += 1
                    else:
                        results["checks_failed"] += 1
                        
                except Exception as e:
                    results["details"][check_name] = {
                        "passed": False,
                        "details": f"Error: {str(e)}",
                    }
                    results["checks_failed"] += 1
                
                progress.advance(task)
        
        # Print results
        self._print_validation_results(results)
        
        self.validation_results = results
        return results
    
    def _check_column_count(self, df: pd.DataFrame) -> tuple[bool, str]:
        """Check minimum column count."""
        min_columns = 10  # At least 10 columns expected
        passed = len(df.columns) >= min_columns
        details = f"Found {len(df.columns)} columns (min: {min_columns})"
        return passed, details
    
    def _check_target_exists(self, df: pd.DataFrame) -> tuple[bool, str]:
        """Check if target column exists."""
        passed = self.target_column in df.columns
        details = f"Target '{self.target_column}' {'found' if passed else 'NOT FOUND'}"
        return passed, details
    
    def _check_no_missing(self, df: pd.DataFrame) -> tuple[bool, str]:
        """Check for missing values."""
        missing_count = df.isnull().sum().sum()
        passed = missing_count == 0
        details = f"Missing values: {missing_count:,}"
        return passed, details
    
    def _check_no_infinite(self, df: pd.DataFrame) -> tuple[bool, str]:
        """Check for infinite values in numeric columns."""
        import numpy as np
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        inf_count = 0
        
        for col in numeric_cols:
            inf_count += np.isinf(df[col]).sum()
        
        passed = inf_count == 0
        details = f"Infinite values: {inf_count:,}"
        return passed, details
    
    def _check_no_duplicates(self, df: pd.DataFrame) -> tuple[bool, str]:
        """Check for duplicate rows."""
        dup_count = df.duplicated().sum()
        pct = (dup_count / len(df)) * 100
        # Allow up to 5% duplicates (network data often has similar patterns)
        passed = pct <= 5.0
        details = f"Duplicates: {dup_count:,} ({pct:.2f}%)"
        return passed, details
    
    def _check_numeric_features(self, df: pd.DataFrame) -> tuple[bool, str]:
        """Check that features are numeric."""
        import numpy as np
        
        # Exclude target column
        feature_cols = [c for c in df.columns if c != self.target_column]
        numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns
        
        numeric_ratio = len(numeric_cols) / len(feature_cols) if feature_cols else 0
        passed = numeric_ratio >= 0.9  # At least 90% numeric
        details = f"Numeric features: {len(numeric_cols)}/{len(feature_cols)} ({numeric_ratio:.0%})"
        return passed, details
    
    def _check_class_distribution(self, df: pd.DataFrame) -> tuple[bool, str]:
        """Check target class distribution."""
        if self.target_column not in df.columns:
            return False, "Target column not found"
        
        n_classes = df[self.target_column].nunique()
        passed = n_classes >= 2  # At least 2 classes
        
        # Check for extreme class imbalance
        class_counts = df[self.target_column].value_counts()
        min_class_pct = (class_counts.min() / len(df)) * 100
        
        if min_class_pct < 1.0:
            details = f"{n_classes} classes (warning: minority class < 1%)"
        else:
            details = f"{n_classes} classes found"
        
        return passed, details
    
    def _check_value_ranges(self, df: pd.DataFrame) -> tuple[bool, str]:
        """Check for extreme outliers in numeric features."""
        import numpy as np
        
        feature_cols = [c for c in df.columns if c != self.target_column]
        numeric_df = df[feature_cols].select_dtypes(include=[np.number])
        
        # Check for extremely large values
        max_val = numeric_df.max().max()
        min_val = numeric_df.min().min()
        
        # Flag if values exceed reasonable bounds
        extreme_threshold = 1e10
        has_extreme = max_val > extreme_threshold or min_val < -extreme_threshold
        
        passed = not has_extreme
        details = f"Value range: [{min_val:.2e}, {max_val:.2e}]"
        
        return passed, details
    
    def _print_validation_results(self, results: Dict[str, Any]) -> None:
        """Print validation results in a table."""
        from rich.table import Table
        
        table = Table(title="Validation Results", show_header=True)
        table.add_column("Check", style="cyan")
        table.add_column("Status", justify="center")
        table.add_column("Details", style="dim")
        
        for check_name, check_result in results["details"].items():
            status = "[green]PASS[/green]" if check_result["passed"] else "[red]FAIL[/red]"
            table.add_row(check_name, status, check_result["details"])
        
        console.print(table)
        
        # Summary
        total = results["checks_passed"] + results["checks_failed"]
        if results["checks_failed"] == 0:
            print_success(f"All {total} validation checks passed!")
        else:
            print_warning(
                f"Validation: {results['checks_passed']}/{total} checks passed, "
                f"{results['checks_failed']} failed"
            )
    
    @property
    def is_valid(self) -> bool:
        """Check if last validation passed all checks."""
        return self.validation_results.get("checks_failed", 1) == 0


def validate_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Convenience function to validate a dataset.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        Validation results dictionary
    """
    validator = DataValidator()
    return validator.validate(df)


if __name__ == "__main__":
    # Test validation
    from src.data.ingestion import load_dataset
    
    df = load_dataset(sample_frac=0.01)
    results = validate_dataset(df)
    
    print(f"\nValidation passed: {results['checks_failed'] == 0}")
