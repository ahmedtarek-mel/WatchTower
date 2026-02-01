"""
WatchTower Model Evaluation Module.

Comprehensive model evaluation metrics and reports.
"""

from typing import Dict, Any, List, Optional
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from loguru import logger

from src.utils.logger import console, print_header, print_metrics_table


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
    dataset_name: str = "Test",
) -> Dict[str, float]:
    """
    Comprehensive model evaluation.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (optional, for AUC)
        class_names: Class label names
        dataset_name: Name for display
        
    Returns:
        Dictionary of metrics
    """
    print_header(f"{dataset_name} Evaluation", "Computing Metrics")
    
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }
    
    # ROC AUC if probabilities available
    if y_proba is not None:
        try:
            if len(np.unique(y_true)) == 2:
                metrics["roc_auc"] = roc_auc_score(y_true, y_proba[:, 1])
            else:
                metrics["roc_auc_ovr"] = roc_auc_score(
                    y_true, y_proba, multi_class="ovr", average="weighted"
                )
        except Exception as e:
            logger.warning(f"Could not compute ROC AUC: {e}")
    
    # Print metrics table
    print_metrics_table(
        {k: f"{v:.4f}" for k, v in metrics.items()},
        title=f"{dataset_name} Metrics"
    )
    
    return metrics


def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> str:
    """
    Print detailed classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Class label names
        
    Returns:
        Classification report string
    """
    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        zero_division=0,
    )
    
    console.print("\n[bold]Classification Report:[/bold]")
    console.print(report)
    
    return report


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Compute and format confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Class label names
        
    Returns:
        Confusion matrix as DataFrame
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if class_names:
        cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    else:
        cm_df = pd.DataFrame(cm)
    
    return cm_df


def compute_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Compute metrics for each class.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Class label names
        
    Returns:
        DataFrame with per-class metrics
    """
    from sklearn.metrics import precision_recall_fscore_support
    
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, zero_division=0
    )
    
    if class_names is None:
        class_names = [f"Class {i}" for i in range(len(precision))]
    
    return pd.DataFrame({
        "class": class_names,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "support": support.astype(int),
    })
