# Finance Fraud Detection System

**This project is partly assisted by Anthropic Claude 3.7 Sonnet**

A comprehensive machine learning system for detecting fraudulent financial transactions using ensemble methods. The system employs LightGBM and Random Forest models with probability calibration to provide high-accuracy fraud detection with explainable predictions.

## Features

### Core Functionality
- fraud detection with probability-based scoring
- Batch processing capabilities for large transaction volumes
- Probability calibration for optimal decision thresholds
- Feature importance analysis and visualization
- Comprehensive model evaluation metrics
- Automated model training and evaluation pipeline

### Technical Features
- Modular architecture with clear separation of concerns
- Comprehensive data processing pipeline
- Feature engineering
- Cross-validation support
- Model versioning and tracking
- Detailed logging and monitoring
- Production-ready deployment options

### Analysis & Visualization
- EDA (Exploratory Data Analysis) tools
- Feature importance visualization
- ROC and Precision-Recall curves
- Confusion matrix analysis
- Model performance metrics
- Cost-benefit analysis

## Project Structure
```
fraud_detection/
├── api/            # API endpoints and web services
├── cli/            # Command-line interface tools
├── core/           # Core functionality
│   ├── data_processor.py    # Data processing pipeline
│   ├── eda.py              # Exploratory data analysis
│   └── features/           # Feature engineering
├── models/         # Model implementations
│   ├── core/      # Base model implementations
│   ├── training/  # Training scripts
│   ├── evaluation/# Model evaluation tools
│   └── deployment/# Deployment configurations
├── scripts/        # Utility scripts
├── tests/          # Test suite
└── utils/          # Utility functions
```

## Dataset
- Issues: [GitHub Issues](https://github.com/your-username/finance-fraud-detection/issues)
- Contributors and maintainers

### Features
- `step`: Time unit (1 step = 1 hour)
- `type`: Transaction type
- `amount`: Transaction amount
- `nameOrig`: Customer initiating the transaction
- `oldbalanceOrg`: Initial balance of sender
- `newbalanceOrig`: Final balance of sender
- `nameDest`: Transaction recipient
- `oldbalanceDest`: Initial balance of recipient
- `newbalanceDest`: Final balance of recipient
- `isFraud`: Fraud label (1 = fraud, 0 = legitimate)

## Installation

### Prerequisites
- Python 3.8+
- pip or conda package manager
- Git

### Setup
```bash
# Clone repository
git clone https://github.com/your-username/finance-fraud-detection.git
cd finance-fraud-detection

# Create virtual environment
python -m venv .venv
source .venv/bin/activate 

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Data Preparation and EDA
```bash
# Run Exploratory Data Analysis
python -m fraud_detection.core.eda run_eda

# This will:
# 1. Load and analyze the raw data
# 2. Generate EDA visualizations
# 3. Perform feature engineering
# 4. Split data into train/validation/test sets
# 5. Save processed datasets
```

### Training
```bash
# Train the ensemble model
python -m fraud_detection.models.training.trainer --model-type ensemble

# Train individual models
python -m fraud_detection.models.training.trainer --model-type lightgbm
python -m fraud_detection.models.training.trainer --model-type random_forest
```

### Evaluation
```bash
# Analyze model performance
python -m fraud_detection.models.evaluation.analyzer --model-version v2025.03.28_ensemble --model-type ensemble
```

## Model Performance
Current model performance metrics (v2025.03.28):

### Ensemble Model
- F1 Score: 0.7567
- ROC AUC: 0.9971
- Precision: 0.8234
- Recall: 0.6989

### Individual Models
- **LightGBM**:
  - F1 Score: 0.7292
  - ROC AUC: 0.9961
- **Random Forest**:
  - F1 Score: 0.7456
  - ROC AUC: 0.9968


## Documentation
For detailed documentation, please visit our [Documentation Hub](docs/README.md):

- [Getting Started Guide](docs/guides/getting_started.md)
- [API Documentation](docs/api/overview.md)
- [Model Details](docs/technical/MODEL.md)
- [Development Guide](docs/technical/DEVELOPMENT.md)
- [Operations Guide](docs/technical/OPERATIONS.md)

## Development Status
- Current Version: v1.0.0
- Next Milestone: v1.1.0 (Q2 2024)
- See our [Roadmap](docs/reference/ROADMAP.md) for future plans
- Track changes in our [Changelog](docs/reference/CHANGELOG.md)

## Contributing
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License
This project is licensed under the MIT License - see the LICENSE file for details.

The dataset used in this project is licensed under the Creative Commons CC0 1.0 Universal License ([CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/)), which means it is in the public domain and can be used for any purpose without attribution.

## Support
- Documentation: [docs/](docs/)
- Issues: [GitHub Issues](https://github.com/your-username/finance-fraud-detection/issues)
- Email: [Add Contact Email]


---
