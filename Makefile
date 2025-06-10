# GraphBit Development Makefile
# This Makefile provides convenient targets for development workflows

.PHONY: help install clean test test-rust test-python lint lint-rust lint-python format format-rust format-python build docs dev-setup all-checks ci secrets secrets-audit secrets-baseline secrets-update build-perf install-perf test-perf benchmark-perf

# Default target
help: ## Show this help message
	@echo "GraphBit Development Commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Environment variables:"
	@echo "  PYTHON_ENV: Python environment to activate (default: conda activate graphbit)"

# Environment setup
PYTHON_ENV ?= conda activate graphbit
# OPENAI_API_KEY should be set as environment variable

# Setup and Installation
create-conda-env: ## Create conda environment if it doesn't exist
	@echo "Checking if conda environment 'graphbit' exists..."
	@conda info --envs | grep -q "^graphbit " || { \
		echo "Creating conda environment 'graphbit' with Python 3.11.0..."; \
		conda create -n graphbit python=3.11.0 -y; \
		echo "Conda environment 'graphbit' created successfully!"; \
	} && echo "Conda environment 'graphbit' is ready!"

dev-setup: create-conda-env ## Set up development environment
	@echo "Setting up development environment..."
	$(PYTHON_ENV) && poetry install --with dev,benchmarks
	cargo build --workspace
	$(PYTHON_ENV) && pre-commit install
	$(PYTHON_ENV) && pre-commit install --hook-type commit-msg
	$(PYTHON_ENV) && pre-commit install --hook-type pre-push
	@echo "Development environment ready!"

install: ## Install all dependencies
	@echo "Installing Python dependencies..."
	$(PYTHON_ENV) && poetry install --with dev,benchmarks
	@echo "Installing Rust dependencies..."
	cargo fetch
	@echo "Dependencies installed!"

# Cleaning
clean: ## Clean build artifacts
	@echo "Cleaning Rust artifacts..."
	cargo clean
	@echo "Cleaning Python artifacts..."
	$(PYTHON_ENV) && find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	$(PYTHON_ENV) && find . -type f -name "*.pyc" -delete 2>/dev/null || true
	$(PYTHON_ENV) && find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@echo "Clean complete!"

# Testing
test: test-rust test-python ## Run all tests (Rust + Python)

test-rust: ## Run Rust tests
	@echo "Running Rust tests..."
	cargo test --workspace --all-features

test-python: ## Run Python tests
	@echo "Running Python tests..."
	@if [ -z "$$OPENAI_API_KEY" ]; then echo "ERROR: OPENAI_API_KEY environment variable is required"; exit 1; fi
	$(PYTHON_ENV) && pytest -v

test-integration: ## Run integration tests
	@echo "Running integration tests..."
	@if [ -z "$$OPENAI_API_KEY" ]; then echo "ERROR: OPENAI_API_KEY environment variable is required"; exit 1; fi
	cargo test --test integration_tests

test-coverage: ## Run tests with coverage
	@echo "Running Rust tests with coverage..."
	cargo tarpaulin --workspace --out Html --output-dir target/coverage
	@echo "Running Python tests with coverage..."
	@if [ -z "$$OPENAI_API_KEY" ]; then echo "ERROR: OPENAI_API_KEY environment variable is required"; exit 1; fi
	$(PYTHON_ENV) && pytest --cov=graphbit --cov-report=html:target/coverage/python

# Linting
lint: lint-rust lint-python ## Run all linting (Rust + Python)

lint-rust: ## Run Rust linting (clippy)
	@echo "Running Rust linting..."
	cargo clippy --workspace --all-targets --all-features -- -D warnings

lint-python: ## Run Python linting (flake8, mypy)
	@echo "Running Python linting..."
	$(PYTHON_ENV) && flake8 graphbit/ tests/ benchmarks/
	$(PYTHON_ENV) && mypy graphbit/ --ignore-missing-imports

lint-fix: ## Fix linting issues automatically
	@echo "Fixing Rust linting issues..."
	cargo clippy --workspace --all-targets --all-features --fix --allow-staged --allow-dirty
	@echo "Fixing Python import sorting..."
	$(PYTHON_ENV) && isort graphbit/ tests/ benchmarks/

# Formatting
format: format-rust format-python ## Format all code (Rust + Python)

format-rust: ## Format Rust code
	@echo "Formatting Rust code..."
	cargo fmt --all

format-python: ## Format Python code
	@echo "Formatting Python code..."
	$(PYTHON_ENV) && black graphbit/ tests/ benchmarks/
	$(PYTHON_ENV) && isort graphbit/ tests/ benchmarks/

format-check: ## Check if code is formatted correctly
	@echo "Checking Rust formatting..."
	cargo fmt --all -- --check
	@echo "Checking Python formatting..."
	$(PYTHON_ENV) && black --check graphbit/ tests/ benchmarks/
	$(PYTHON_ENV) && isort --check-only graphbit/ tests/ benchmarks/

# Building
build: ## Build Rust workspace and Python package
	@echo "Building Rust workspace..."
	cargo build --workspace --release
	@echo "Building Python package..."
	$(PYTHON_ENV) && poetry build

build-dev: ## Build in development mode
	@echo "Building Rust workspace (debug)..."
	cargo build --workspace
	@echo "Installing Python package in development mode..."
	$(PYTHON_ENV) && pip install -e .

# Documentation
docs: ## Build documentation
	@echo "Building Rust documentation..."
	cargo doc --workspace --no-deps --open
	@echo "Building Python documentation..."
	$(PYTHON_ENV) && cd docs && make html

docs-serve: ## Serve documentation locally
	@echo "Serving documentation..."
	$(PYTHON_ENV) && cd docs && python -m http.server 8000

# Benchmarks
bench: ## Run benchmarks
	@echo "Running Rust benchmarks..."
	cargo bench
	@echo "Running Python benchmarks..."
	@if [ -z "$$OPENAI_API_KEY" ]; then echo "ERROR: OPENAI_API_KEY environment variable is required"; exit 1; fi
	$(PYTHON_ENV) && python -m benchmarks.run_benchmarks

# Security and Quality
security: ## Run security checks
	@echo "Running Rust security audit..."
	cargo audit
	@echo "Running Python security checks..."
	$(PYTHON_ENV) && safety check
	$(PYTHON_ENV) && bandit -r graphbit/
	@echo "Running secret detection..."
	$(MAKE) secrets

secrets: ## Detect secrets in codebase using detect-secrets
	@echo "Scanning for secrets with detect-secrets..."
	$(PYTHON_ENV) && detect-secrets scan --baseline .secrets.baseline

secrets-audit: ## Run comprehensive secret audit including baseline check
	@echo "Running comprehensive secret audit with detect-secrets..."
	@echo "Scanning current files..."
	$(PYTHON_ENV) && detect-secrets scan --baseline .secrets.baseline
	@echo "Auditing baseline file..."
	$(PYTHON_ENV) && detect-secrets audit .secrets.baseline

secrets-baseline: ## Create or update detect-secrets baseline
	@echo "Creating/updating detect-secrets baseline..."
	$(PYTHON_ENV) && detect-secrets scan > .secrets.baseline
	@echo "Detect-secrets baseline updated at .secrets.baseline"

secrets-update: ## Update detect-secrets baseline
	@echo "Edit .secrets.baseline to update detect-secrets configuration"
	@echo "See: https://github.com/Yelp/detect-secrets"

typos: ## Check for typos
	@echo "Checking for typos..."
	typos

# All-in-one checks
all-checks: format-check lint test secrets ## Run all checks (format, lint, test, secrets)
	@echo "All checks completed successfully!"

ci: clean all-checks ## Run CI pipeline locally
	@echo "CI pipeline completed successfully!"

# Release preparation
release-check: all-checks docs ## Check if ready for release
	@echo "Checking if ready for release..."
	cargo publish --dry-run
	$(PYTHON_ENV) && poetry check
	@echo "Release check completed!"

# Quick development commands
quick: format-rust lint-rust test-rust ## Quick Rust development cycle
	@echo "Quick development cycle completed!"

quick-python: format-python lint-python test-python ## Quick Python development cycle
	@echo "Quick Python development cycle completed!"

# Pre-commit hooks
pre-commit-install: ## Install pre-commit hooks
	@echo "Installing pre-commit hooks..."
	$(PYTHON_ENV) && pre-commit install
	$(PYTHON_ENV) && pre-commit install --hook-type commit-msg
	$(PYTHON_ENV) && pre-commit install --hook-type pre-push
	@echo "Pre-commit hooks installed!"

pre-commit-run: ## Run all pre-commit hooks on all files
	@echo "Running pre-commit hooks on all files..."
	$(PYTHON_ENV) && pre-commit run --all-files

pre-commit-update: ## Update pre-commit hook versions
	@echo "Updating pre-commit hooks..."
	$(PYTHON_ENV) && pre-commit autoupdate

pre-commit-clean: ## Clean pre-commit cache
	@echo "Cleaning pre-commit cache..."
	$(PYTHON_ENV) && pre-commit clean

# Examples and demos
examples: ## Run example scripts
	@echo "Running examples..."
	export OPENAI_API_KEY=$(OPENAI_API_KEY) && $(PYTHON_ENV) && python examples/basic_workflow.py

# Watch mode for development
watch-test: ## Watch for changes and run tests
	@echo "Watching for changes and running tests..."
	cargo watch -x "test --workspace"

watch-check: ## Watch for changes and run checks
	@echo "Watching for changes and running checks..."
	cargo watch -x "check --workspace" -x "clippy --workspace"

# Performance build targets
build-perf:
	@echo "🚀 Building GraphBit with performance optimizations..."
	@conda run -n graphbit cargo build --release --features performance
	@conda run -n graphbit maturin develop --release

# Install with performance optimizations
install-perf: build-perf
	@echo "📦 Installing GraphBit with performance optimizations..."
	@conda run -n graphbit pip install -e python/

# Run performance tests
test-perf: build-perf
	@echo "⚡ Running performance tests..."
	@conda run -n graphbit python performance_test.py

# Run comprehensive benchmarks
benchmark-perf: build-perf
	@echo "📊 Running comprehensive benchmarks..."
	@conda run -n graphbit python benchmarks/run_comprehensive_benchmark.py
