.PHONY: install test format lint clean demo help

help:		## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:	## Install the package in development mode
	pip install -e .
	pip install -r requirements.txt

install-dev:	## Install development dependencies
	pip install -e ".[dev]"

test:		## Run unit tests
	pytest tests/ -v --cov=src/promptframe --cov-report=html --cov-report=term

test-fast:	## Run tests without coverage
	pytest tests/ -v

format:		## Format code with black and ruff
	black src/ tests/
	ruff format src/ tests/

lint:		## Lint code with ruff and mypy
	ruff check src/ tests/
	mypy src/promptframe/

fix:		## Fix linting issues automatically
	ruff check --fix src/ tests/
	black src/ tests/

clean:		## Clean build artifacts and cache
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

demo:		## Run the demo script
	cd src && python promptframe/demo_quickstart.py

build:		## Build the package
	python -m build

publish:	## Publish to PyPI (requires credentials)
	python -m twine upload dist/*

docs:		## Generate documentation (placeholder)
	@echo "Documentation generation not implemented yet"

check:		## Run all checks (lint, format, test)
	$(MAKE) format
	$(MAKE) lint
	$(MAKE) test

dev-setup:	## Set up development environment
	$(MAKE) install-dev
	pre-commit install