# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-01-14

### Added
- **Data Pipeline**
  - CICIDS2017 dataset ingestion with progress tracking
  - Feature engineering and preprocessing
  - Data validation with quality checks
  - Train/validation/test splitting with stratification

- **Training Pipeline**
  - GPU-accelerated XGBoost training
  - Optuna hyperparameter optimization
  - Prefect workflow orchestration
  - MLflow experiment tracking
  - Comprehensive evaluation metrics

- **Serving Layer**
  - FastAPI inference API
  - Single and batch prediction endpoints
  - Prometheus metrics endpoint
  - Health checks and model info

- **Infrastructure**
  - Docker support (train and serve)
  - Docker Compose for full MLOps stack
  - GitHub Actions CI/CD workflows
  - Makefile for common tasks

- **Observability**
  - Rich console output with progress bars
  - Loguru structured logging
  - MLflow UI for experiment tracking
  - Prefect UI for workflow monitoring

### Performance
- Achieved 99.877% accuracy on CICIDS2017
- F1-macro score of 95.9%
- GPU training in ~9 seconds (377 estimators)
- Inference latency <5ms per prediction
