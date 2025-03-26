# Fraud Detection System

# This project is partly assisted by Anthropic Claude 3.7 Sonnet

## Overview
A machine learning system for detecting fraudulent financial transactions using ensemble methods. The system employs LightGBM and Random Forest models with probability calibration to provide high-accuracy fraud detection with explainable predictions.

## Features
- Real-time fraud detection
- Batch processing capabilities
- Probability calibration for better decision thresholds
- Comprehensive API with webhook support
- Detailed monitoring and logging
- Production-ready deployment options

## Dataset

- step: represents a unit of time where 1 step equals 1 hour
- type: type of online transaction
- amount: the amount of the transaction
- nameOrig: customer starting the transaction
- oldbalanceOrg: balance before the transaction
- newbalanceOrig: balance after the transaction
- nameDest: recipient of the transaction
- oldbalanceDest: initial balance of recipient before the transaction
- newbalanceDest: the new balance of recipient after the transaction
- isFraud: fraud transaction

## Quick Start

### Installation
```bash
# Clone repository
git clone https://github.com/your-username/fraud-detection.git
cd fraud-detection

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage
```python
from fraud_detection import FraudDetector

# Initialize detector
detector = FraudDetector()

# Check a transaction
result = detector.check_transaction({
    "amount": 1000.00,
    "type": "online",
    "merchant": "retail",
    "zip": "10001"
})

print(f"Prediction: {result.prediction}")
print(f"Confidence: {result.confidence}")
```

## Model Performance
Current model performance metrics (v1.0.0):

- **LightGBM**:
  - F1 Score: 0.7292 (threshold: 0.2071)
  - ROC AUC: 0.9961

- **Random Forest**:
  - F1 Score: 0.7567 (threshold: 0.0994)
  - ROC AUC: 0.9971

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
We welcome contributions! Please read our [Development Guide](docs/technical/DEVELOPMENT.md#contributing) for guidelines.

## License
[Add License Information]

## Contact
- Team: Fraud Detection
- Email: [Add Contact Email]
- Support: [Documentation](docs/guides/troubleshooting.md)
- Community: [Discord](https://discord.gg/fraud-detection)

---
