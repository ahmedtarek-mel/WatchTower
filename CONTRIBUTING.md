# Contributing to WatchTower

Thank you for your interest in contributing to WatchTower!

## Development Setup

```bash
# Clone the repository
git clone https://github.com/ahmedtarek-mel/WatchTower.git
cd WatchTower

# Install with dev dependencies
pip install -e .[dev]

# Run tests
pytest tests/ -v
```

## Code Style

- **Formatter**: Ruff
- **Linter**: Ruff
- **Type Checker**: MyPy

```bash
# Format code
ruff format src/ tests/

# Lint code
ruff check src/ tests/

# Type check
mypy src/ --ignore-missing-imports
```

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/unit/test_training.py -v
```

## Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and linting
5. Commit with a descriptive message
6. Push to your fork
7. Open a Pull Request

## Reporting Issues

Please include:
- Python version
- OS
- Steps to reproduce
- Expected vs actual behavior

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
