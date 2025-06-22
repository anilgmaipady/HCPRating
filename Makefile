# RD Rating System Makefile

.PHONY: help install test clean start stop build docker-build docker-run docker-stop lint format

# Default target
help:
	@echo "RD Rating System - Available Commands:"
	@echo ""
	@echo "Installation:"
	@echo "  install     - Install dependencies"
	@echo "  install-dev - Install development dependencies"
	@echo ""
	@echo "Development:"
	@echo "  test        - Run tests"
	@echo "  lint        - Run linting"
	@echo "  format      - Format code"
	@echo "  clean       - Clean cache and temporary files"
	@echo ""
	@echo "Running:"
	@echo "  start       - Start all services"
	@echo "  start-vllm  - Start vLLM server only"
	@echo "  start-api   - Start API server only"
	@echo "  start-web   - Start web interface only"
	@echo "  stop        - Stop all services"
	@echo ""
	@echo "Docker:"
	@echo "  docker-build - Build Docker image"
	@echo "  docker-run   - Run with Docker Compose"
	@echo "  docker-stop  - Stop Docker containers"
	@echo ""
	@echo "Data:"
	@echo "  sample-data - Create sample data"
	@echo "  train       - Run fine-tuning"

# Installation
install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install -e .[dev]

# Development
test:
	python -m pytest tests/ -v

lint:
	flake8 src/ tests/
	black --check src/ tests/

format:
	black src/ tests/
	isort src/ tests/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +

# Running services
start:
	python start.py all

start-vllm:
	python start.py vllm

start-api:
	python start.py api

start-web:
	python start.py streamlit

stop:
	@echo "Stopping services..."
	@pkill -f "vllm" || true
	@pkill -f "uvicorn" || true
	@pkill -f "streamlit" || true

# Docker
docker-build:
	docker build -t rd-rating-system .

docker-run:
	docker-compose up -d

docker-stop:
	docker-compose down

# Data and training
sample-data:
	python src/training/fine_tune.py --create_sample

train:
	@echo "Please provide data path: make train DATA_PATH=path/to/data.jsonl"
	@if [ -z "$(DATA_PATH)" ]; then \
		echo "Error: DATA_PATH is required"; \
		echo "Usage: make train DATA_PATH=path/to/data.jsonl"; \
		exit 1; \
	fi
	python src/training/fine_tune.py --data_path $(DATA_PATH)

# Health checks
health:
	@echo "Checking service health..."
	@curl -f http://localhost:8001/health || echo "API server not responding"
	@curl -f http://localhost:8000/v1/models || echo "vLLM server not responding"

# Setup development environment
setup-dev: install-dev
	@echo "Setting up development environment..."
	@mkdir -p logs exports models data
	@cp env.example .env
	@echo "Development environment setup complete!"
	@echo "Please edit .env file with your configuration"

# Production setup
setup-prod: install
	@echo "Setting up production environment..."
	@mkdir -p logs exports models data
	@cp env.example .env
	@echo "Production environment setup complete!"
	@echo "Please edit .env file with your configuration"

# Backup and restore
backup:
	@echo "Creating backup..."
	@tar -czf backup_$(shell date +%Y%m%d_%H%M%S).tar.gz data/ models/ logs/ exports/ --exclude='*.pyc' --exclude='__pycache__'

restore:
	@if [ -z "$(BACKUP_FILE)" ]; then \
		echo "Error: BACKUP_FILE is required"; \
		echo "Usage: make restore BACKUP_FILE=backup_20240101_120000.tar.gz"; \
		exit 1; \
	fi
	@echo "Restoring from backup: $(BACKUP_FILE)"
	@tar -xzf $(BACKUP_FILE) 