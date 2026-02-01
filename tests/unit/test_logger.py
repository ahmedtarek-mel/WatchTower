"""Tests for logger utilities."""

import pytest
from unittest.mock import patch, MagicMock
from io import StringIO

from src.utils.logger import (
    console,
    create_progress,
    print_header,
    print_success,
    print_error,
    print_warning,
    print_info,
    print_metrics_table,
)


class TestConsole:
    """Tests for Rich console."""
    
    def test_console_exists(self):
        """Test console is available."""
        assert console is not None
    
    def test_console_can_print(self):
        """Test console can print without error."""
        # Just verify it doesn't crash
        console.print("Test message", style="dim")


class TestProgressBar:
    """Tests for progress bar creation."""
    
    def test_create_progress_returns_progress(self):
        """Test create_progress returns a Progress object."""
        progress = create_progress()
        assert progress is not None
    
    def test_progress_can_be_used_as_context(self):
        """Test progress can be used as context manager."""
        with create_progress() as progress:
            task = progress.add_task("Test", total=10)
            progress.update(task, advance=5)
            # Should complete without error


class TestPrintFunctions:
    """Tests for print utility functions."""
    
    def test_print_header_no_error(self):
        """Test print_header runs without error."""
        print_header("Test Title", "Test Subtitle")
    
    def test_print_success_no_error(self):
        """Test print_success runs without error."""
        print_success("Success message")
    
    def test_print_error_no_error(self):
        """Test print_error runs without error."""
        print_error("Error message")
    
    def test_print_warning_no_error(self):
        """Test print_warning runs without error."""
        print_warning("Warning message")
    
    def test_print_info_no_error(self):
        """Test print_info runs without error."""
        print_info("Info message")


class TestMetricsTable:
    """Tests for metrics table printing."""
    
    def test_print_metrics_table_dict(self):
        """Test printing metrics from dict."""
        metrics = {
            "accuracy": "0.95",
            "f1_score": "0.92",
        }
        print_metrics_table(metrics, title="Test Metrics")
    
    def test_print_metrics_table_numeric(self):
        """Test printing numeric metrics."""
        metrics = {
            "accuracy": 0.95,
            "f1_score": 0.92,
        }
        print_metrics_table(metrics, title="Numeric Metrics")
    
    def test_print_metrics_table_empty(self):
        """Test printing empty metrics."""
        print_metrics_table({}, title="Empty")
