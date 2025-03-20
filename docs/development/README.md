# Development Guide

## Setup

### Prerequisites
- Python 3.8+
- Docker (optional)
- Supabase account

### Installation
```bash
# Clone repository
git clone https://github.com/yourusername/fraud_detection.git
cd fraud_detection

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"
```

### Configuration
1. Copy `.env.example` to `.env`
2. Update configuration values
3. Choose environment (dev/prod) in config files

## Development Workflow

### Code Style
- Black for formatting
- isort for import sorting
- mypy for type checking
- flake8 for linting

### Testing
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=fraud_detection

# Run specific test file
pytest tests/unit/test_model.py
```

### Documentation
- Keep docstrings up to date
- Follow Google style for docstrings
- Update README.md for major changes

### Git Workflow
1. Create feature branch
2. Make changes
3. Run tests
4. Submit PR

## Deployment

### Local Development
```bash
uvicorn fraud_detection.api.app:app --reload
```

### Docker Deployment
```bash
docker build -t fraud-detection .
docker run -p 8000:8000 fraud-detection
```

## Troubleshooting
See [Common Issues](./troubleshooting.md)
