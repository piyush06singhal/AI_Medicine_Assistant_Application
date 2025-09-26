# AI Medical Assistant - Makefile for common tasks

.PHONY: help install install-dev test lint format clean run setup

# Default target
help:
	@echo "AI Medical Assistant - Available Commands:"
	@echo ""
	@echo "  setup       - Set up the development environment"
	@echo "  install     - Install production dependencies"
	@echo "  install-dev - Install development dependencies"
	@echo "  test        - Run tests"
	@echo "  lint        - Run linting checks"
	@echo "  format      - Format code with black and isort"
	@echo "  clean       - Clean up temporary files"
	@echo "  run         - Run the Streamlit application"
	@echo "  docs        - Build documentation"
	@echo ""

# Setup development environment
setup:
	@echo "Setting up development environment..."
	python -m venv venv
	@echo "Activate the virtual environment and run 'make install-dev'"

# Install production dependencies
install:
	pip install -r requirements.txt

# Install development dependencies
install-dev:
	pip install -r requirements.txt
	pip install -e .[dev]

# Run tests
test:
	pytest tests/ -v --cov=. --cov-report=html

# Run linting
lint:
	flake8 .
	mypy .
	black --check .
	isort --check-only .

# Format code
format:
	black .
	isort .

# Clean up
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/

# Run the application
run:
	streamlit run web_app/app.py

# Build documentation
docs:
	cd docs && make html

# Install pre-commit hooks
pre-commit-install:
	pre-commit install

# Update dependencies
update-deps:
	pip-compile requirements.in
	pip-compile requirements-dev.in
