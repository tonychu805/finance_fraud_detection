# Fraud Detection System

*** This project was developed with the assistance of Claude 3.7 - Sonnet. 

A comprehensive solution for detecting financial fraud using machine learning.

## Project Structure

```
fraud_detection/
├── fraud_detection/          # Main package
│   ├── core/                # Core functionality
│   │   └── data_processor.py
│   ├── models/             # Model-related code
│   │   ├── model_manager.py
│   │   ├── model_trainer.py
│   │   └── ml_tracker.py
│   ├── database/           # Database operations
│   │   └── supabase_client.py
│   └── api/                # API-related code
│       └── app.py
├── tests/                  # Test suite
│   ├── unit/
│   └── integration/
├── config/                 # Configuration files
│   ├── dev/
│   └── prod/
├── docs/                   # Documentation
│   ├── api/
│   ├── user_guide/
│   ├── development/
│   └── deployment/
├── notebooks/             # Jupyter notebooks
├── scripts/              # Utility scripts
└── data/                 # Data directory
    ├── raw/
    └── processed/
```

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/fraud_detection.git
cd fraud_detection
```

2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -e ".[dev]"  # Installs package in development mode with dev dependencies
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

## Development

### Running Tests
```bash
pytest
```

### Code Quality
```bash
black .
isort .
mypy .
flake8 .
```

### Running the API
```bash
uvicorn fraud_detection.api.app:app --reload
```

## Documentation

Detailed documentation is available in the `docs/` directory:
- API Documentation: `docs/api/`
- User Guide: `docs/user_guide/`
- Development Guide: `docs/development/`
- Deployment Guide: `docs/deployment/`

## License

MIT License

## Support

For support and questions, please [open an issue](https://github.com/yourusername/finance_fraud_detection/issues) or contact the maintainers.

---
