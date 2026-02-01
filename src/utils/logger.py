"""
WatchTower Logging Module.

Rich-powered logging with colored output, progress bars, and emoji indicators.
All process visibility flows through this module.
"""

import sys
from pathlib import Path
from typing import Optional, Any
from functools import lru_cache

from loguru import logger
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
)
from rich.table import Table
from rich.panel import Panel
from rich.theme import Theme

# Force UTF-8 on Windows
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# Custom theme for WatchTower
WATCHTOWER_THEME = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "bold red",
    "success": "bold green",
    "title": "bold magenta",
    "metric": "blue",
})

# Global console instance
console = Console(theme=WATCHTOWER_THEME, force_terminal=True)


def setup_logger(
    log_level: str = "INFO",
    log_file: Optional[Path] = None,
    rotation: str = "10 MB",
    retention: str = "1 week",
) -> None:
    """
    Configure loguru logger with Rich formatting.
    
    Args:
        log_level: Minimum log level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional path for file logging
        rotation: Log rotation size
        retention: Log retention period
    """
    # Remove default handler
    logger.remove()
    
    # Custom format with emojis
    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )
    
    # Console handler with colors
    logger.add(
        sys.stderr,
        format=log_format,
        level=log_level,
        colorize=True,
        backtrace=True,
        diagnose=True,
    )
    
    # File handler (optional)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            str(log_file),
            format=log_format,
            level=log_level,
            rotation=rotation,
            retention=retention,
            compression="zip",
        )


@lru_cache
def get_logger(name: str = "watchtower"):
    """Get a logger instance with the given name."""
    return logger.bind(name=name)


def create_progress() -> Progress:
    """Create a Rich progress bar for long-running operations."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    )


def print_header(title: str, subtitle: Optional[str] = None) -> None:
    """Print a styled header."""
    content = f"[title]{title}[/title]"
    if subtitle:
        content += f"\n[dim]{subtitle}[/dim]"
    console.print(Panel(content, border_style="cyan", padding=(1, 2)))


def print_success(message: str) -> None:
    """Print a success message with checkmark."""
    console.print(f"[success]✅ {message}[/success]")


def print_warning(message: str) -> None:
    """Print a warning message."""
    console.print(f"[warning]⚠️  {message}[/warning]")


def print_error(message: str) -> None:
    """Print an error message."""
    console.print(f"[error]❌ {message}[/error]")


def print_info(message: str) -> None:
    """Print an info message."""
    console.print(f"[info]ℹ️  {message}[/info]")


def print_step(step: int, total: int, message: str) -> None:
    """Print a step indicator."""
    console.print(f"[cyan]📍 Step {step}/{total}:[/cyan] {message}")


def print_metrics_table(metrics: dict[str, Any], title: str = "Model Metrics") -> None:
    """Print metrics in a formatted table."""
    table = Table(title=f"📊 {title}", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green", justify="right")
    
    for name, value in metrics.items():
        if isinstance(value, float):
            table.add_row(name, f"{value:.4f}")
        else:
            table.add_row(name, str(value))
    
    console.print(table)


def print_dataset_info(
    n_samples: int,
    n_features: int,
    class_distribution: Optional[dict[str, int]] = None,
) -> None:
    """Print dataset information in a formatted table."""
    table = Table(title="📊 Dataset Information", show_header=True, header_style="bold cyan")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green", justify="right")
    
    table.add_row("Total Samples", f"{n_samples:,}")
    table.add_row("Features", f"{n_features:,}")
    
    if class_distribution:
        table.add_section()
        table.add_row("[bold]Class Distribution[/bold]", "")
        for class_name, count in class_distribution.items():
            pct = (count / n_samples) * 100
            table.add_row(f"  {class_name}", f"{count:,} ({pct:.1f}%)")
    
    console.print(table)


# Initialize logger on import
def _init_logger():
    """Initialize logger with default settings."""
    try:
        from src.config.settings import settings
        setup_logger(
            log_level=settings.log_level,
            log_file=settings.log_file if settings.log_to_file else None,
        )
    except ImportError:
        # Fallback if settings not available
        setup_logger(log_level="INFO")


_init_logger()
