# RNSR Makefile
# Run `make help` to see available commands

# Use virtual environment if it exists, otherwise system python
VENV := .venv
ifeq ($(wildcard $(VENV)/bin/python),)
    PYTHON := $(shell command -v python3 2>/dev/null || command -v python 2>/dev/null)
else
    PYTHON := $(VENV)/bin/python
endif

.PHONY: help demo test test-fast test-cov lint format install clean venv

# Default target
help:
	@echo "RNSR - Recursive Neural-Symbolic Retriever"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Demo & Benchmarks:"
	@echo "  demo              Launch the Gradio web demo (http://localhost:7860)"
	@echo "  demo-office       Run presentation-ready office demo (local only)"
	@echo "  benchmark         Run performance benchmarks"
	@echo "  benchmark-compare Compare RNSR vs Naive RAG vs Long Context"
	@echo ""
	@echo "Testing:"
	@echo "  test          Run all tests"
	@echo "  test-fast     Run tests without slow integration tests"
	@echo "  test-cov      Run tests with coverage report"
	@echo ""
	@echo "Development:"
	@echo "  lint          Run linter (ruff)"
	@echo "  format        Format code (ruff)"
	@echo "  venv          Create virtual environment"
	@echo "  install       Install dependencies"
	@echo "  install-dev   Install dev dependencies"
	@echo "  clean         Clean build artifacts"
	@echo ""
	@echo "Using Python: $(PYTHON)"
	@echo ""

# Create virtual environment
venv:
	@echo "🐍 Creating virtual environment..."
	python3 -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	@echo ""
	@echo "✅ Virtual environment created at $(VENV)"
	@echo "   Run 'make install-dev' to install dependencies"

# Launch the Gradio web demo
demo:
	@echo "🚀 Starting RNSR Demo..."
	@echo "   Open http://localhost:7860 in your browser"
	@echo ""
	$(PYTHON) demo.py

# Run presentation-ready office demo
demo-office:
	@echo "🎬 Starting Office Demo..."
	@echo ""
	$(PYTHON) scripts/office_demo.py

# Run office demo with a specific PDF
demo-pdf:
	@echo "Usage: make demo-pdf PDF=path/to/document.pdf"
	@test -n "$(PDF)" || (echo "Error: PDF is required" && exit 1)
	$(PYTHON) scripts/office_demo.py --pdf $(PDF)

# Run benchmarks
benchmark:
	@echo "📊 Running Benchmarks..."
	@echo ""
	$(PYTHON) scripts/benchmark_demo.py

# Run benchmark with specific PDF
benchmark-pdf:
	@echo "Usage: make benchmark-pdf PDF=path/to/document.pdf"
	@test -n "$(PDF)" || (echo "Error: PDF is required" && exit 1)
	$(PYTHON) scripts/benchmark_demo.py --pdf $(PDF)

# Compare RNSR vs Naive RAG vs Long Context
benchmark-compare:
	@echo "📊 Running Comparison Benchmark..."
	@echo "   Comparing: RNSR vs Naive RAG vs Long Context LLM"
	@echo ""
	$(PYTHON) scripts/compare_benchmarks.py --quick

# Compare with your own PDF
benchmark-compare-pdf:
	@echo "Usage: make benchmark-compare-pdf PDF=path/to/document.pdf"
	@test -n "$(PDF)" || (echo "Error: PDF is required" && exit 1)
	$(PYTHON) scripts/compare_benchmarks.py --pdf $(PDF)

# Run all tests
test:
	@echo "🧪 Running all tests..."
	$(PYTHON) -m pytest tests/ -v --tb=short

# Run tests without slow tests (e.g., e2e that need LLM)
test-fast:
	@echo "🧪 Running fast tests..."
	$(PYTHON) -m pytest tests/ -v --tb=short -m "not slow" --ignore=tests/test_e2e_workflow.py

# Run tests with coverage
test-cov:
	@echo "🧪 Running tests with coverage..."
	$(PYTHON) -m pytest tests/ -v --tb=short --cov=rnsr --cov-report=html --cov-report=term-missing
	@echo ""
	@echo "📊 Coverage report: htmlcov/index.html"

# Run specific test file
test-file:
	@echo "Usage: make test-file FILE=tests/test_xyz.py"
	@test -n "$(FILE)" || (echo "Error: FILE is required" && exit 1)
	$(PYTHON) -m pytest $(FILE) -v --tb=short

# Lint code
lint:
	@echo "🔍 Linting code..."
	$(PYTHON) -m ruff check rnsr/ tests/ demo.py

# Format code
format:
	@echo "✨ Formatting code..."
	$(PYTHON) -m ruff format rnsr/ tests/ demo.py
	$(PYTHON) -m ruff check --fix rnsr/ tests/ demo.py

# Type check
typecheck:
	@echo "🔎 Type checking..."
	$(PYTHON) -m mypy rnsr/ --ignore-missing-imports

# Install dependencies
install:
	@echo "📦 Installing dependencies..."
	$(PYTHON) -m pip install -e .

# Install dev dependencies
install-dev:
	@echo "📦 Installing dev dependencies..."
	$(PYTHON) -m pip install -e ".[dev,openai,anthropic,gemini,demo]"

# Install all dependencies including benchmarks
install-all:
	@echo "📦 Installing all dependencies..."
	$(PYTHON) -m pip install -e ".[all,dev,benchmarks,demo]"

# Clean build artifacts
clean:
	@echo "🧹 Cleaning build artifacts..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "Done!"

# Build package
build:
	@echo "📦 Building package..."
	$(PYTHON) -m build

# Publish to PyPI (requires twine)
publish:
	@echo "📤 Publishing to PyPI..."
	@echo "Make sure you have a PyPI account and API token configured."
	twine upload dist/*

# Publish to TestPyPI first
publish-test:
	@echo "📤 Publishing to TestPyPI..."
	twine upload --repository testpypi dist/*

# Check if environment is set up correctly
check-env:
	@echo "🔍 Checking environment..."
	@echo ""
	@echo "Python path: $(PYTHON)"
	@echo "Python version: $$($(PYTHON) --version)"
	@echo "Pip version: $$($(PYTHON) -m pip --version)"
	@echo ""
	@echo "API Keys:"
	@test -n "$$GOOGLE_API_KEY" && echo "  ✅ GOOGLE_API_KEY is set" || echo "  ❌ GOOGLE_API_KEY not set"
	@test -n "$$OPENAI_API_KEY" && echo "  ✅ OPENAI_API_KEY is set" || echo "  ❌ OPENAI_API_KEY not set"
	@test -n "$$ANTHROPIC_API_KEY" && echo "  ✅ ANTHROPIC_API_KEY is set" || echo "  ❌ ANTHROPIC_API_KEY not set"
	@echo ""
