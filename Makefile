# WatchTower Makefile
# Common commands for development

.PHONY: help install dev test lint train serve docker-up docker-down clean

help:
	@echo "WatchTower Commands:"
	@echo "  install     - Install production dependencies"
	@echo "  dev         - Install development dependencies"
	@echo "  test        - Run tests with coverage"
	@echo "  lint        - Run linters (ruff)"
	@echo "  train       - Run training pipeline"
	@echo "  serve       - Start API server"
	@echo "  docker-up   - Start all Docker services"
	@echo "  docker-down - Stop all Docker services"
	@echo "  clean       - Remove build artifacts"

install:
	pip install -e .

dev:
	pip install -e ".[dev,notebooks]"

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

lint:
	ruff check src/ tests/
	ruff format src/ tests/ --check

train:
	python -m src.training.train

serve:
	uvicorn src.serving.app:app --reload --host 0.0.0.0 --port 8000

docker-up:
	docker-compose -f docker/docker-compose.yml up -d

docker-down:
	docker-compose -f docker/docker-compose.yml down

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .coverage htmlcov dist build *.egg-info
