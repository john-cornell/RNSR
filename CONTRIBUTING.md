# Contributing to RNSR

Thank you for your interest in contributing to RNSR! This document provides guidelines and information for contributors.

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- Git
- A GitHub account

### Development Setup

1. **Fork the repository** on GitHub

2. **Clone your fork**:
   ```bash
   git clone https://github.com/YOUR-USERNAME/rnsr.git
   cd rnsr
   ```

3. **Create a virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

4. **Install in development mode**:
   ```bash
   pip install -e ".[all,dev]"
   ```

5. **Set up your API keys** (for testing):
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

## 📋 Development Workflow

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

### 2. Make Your Changes

- Write clean, readable code
- Follow the existing code style
- Add docstrings to functions and classes
- Update tests if needed

### 3. Run Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=rnsr

# Run specific tests
pytest tests/test_ingestion.py -v
```

### 4. Run Linting

```bash
# Check code style
ruff check .

# Auto-fix issues
ruff check . --fix

# Type checking
mypy rnsr/
```

### 5. Commit Your Changes

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```bash
git commit -m "feat: add support for DOCX files"
git commit -m "fix: handle empty PDF pages correctly"
git commit -m "docs: update API reference"
git commit -m "test: add tests for font histogram"
```

### 6. Push and Create a Pull Request

```bash
git push origin feature/your-feature-name
```

Then open a Pull Request on GitHub.

## 🏗️ Project Structure

```
rnsr/
├── rnsr/
│   ├── __init__.py       # Package exports
│   ├── __main__.py       # CLI entry point
│   ├── models.py         # Data models
│   ├── exceptions.py     # Custom exceptions
│   ├── llm.py            # LLM provider abstraction
│   ├── ingestion/        # Document ingestion
│   │   ├── __init__.py
│   │   ├── pdf.py        # PDF parsing
│   │   └── hierarchy.py  # Font histogram algorithm
│   ├── indexing/         # Skeleton index
│   │   ├── __init__.py
│   │   ├── skeleton.py   # Index construction
│   │   └── kv_store.py   # Key-value storage
│   ├── agent/            # Navigator agent
│   │   ├── __init__.py
│   │   ├── graph.py      # LangGraph definition
│   │   └── actions.py    # Agent actions
│   └── benchmarks/       # Benchmarking suite
│       ├── __init__.py
│       ├── evaluation_suite.py
│       └── standard_benchmarks.py
├── tests/                # Test suite
├── pyproject.toml        # Project configuration
├── README.md             # Main documentation
└── LICENSE               # MIT License
```

## 📝 Code Style Guidelines

### Python Style

- Use type hints for all function signatures
- Maximum line length: 100 characters
- Use f-strings for string formatting
- Prefer dataclasses for data containers

### Documentation

- All public functions need docstrings
- Use Google-style docstrings:

```python
def process_document(file_path: str, strategy: str = "auto") -> DocumentTree:
    """Process a document and extract its hierarchy.
    
    Args:
        file_path: Path to the document file.
        strategy: Extraction strategy ("auto", "font", "semantic").
        
    Returns:
        A DocumentTree representing the document structure.
        
    Raises:
        FileNotFoundError: If the file doesn't exist.
        UnsupportedFormatError: If the file format isn't supported.
    """
```

### Testing

- Write tests for new functionality
- Use descriptive test names: `test_font_histogram_detects_three_heading_levels`
- Use fixtures for common setup
- Mock external services (LLMs, APIs)

## 🐛 Bug Reports

When filing a bug report, please include:

1. **Python version**: `python --version`
2. **RNSR version**: `pip show rnsr`
3. **Operating system**
4. **Steps to reproduce**
5. **Expected behavior**
6. **Actual behavior**
7. **Error messages/tracebacks**

## 💡 Feature Requests

Feature requests are welcome! Please:

1. Check if the feature is already requested
2. Describe the use case
3. Explain the expected behavior
4. Consider implementation complexity

## 🔒 Security

If you discover a security vulnerability, please:

1. **Do NOT** open a public issue
2. Email the maintainers directly
3. Include detailed information about the vulnerability

## 📜 License

By contributing to RNSR, you agree that your contributions will be licensed under the MIT License.

## 🙏 Thank You!

Every contribution, no matter how small, helps make RNSR better. Thank you for being part of this project!
