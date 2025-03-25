# Getting Started Guide

## Quick Start

### 1. Installation
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. API Key Setup
1. Register at [fraud-detection.com](https://fraud-detection.com)
2. Generate an API key from your dashboard
3. Store the key securely in your environment:
   ```bash
   export FRAUD_DETECTION_API_KEY="your_api_key"
   ```

### 3. Basic Usage

#### Single Transaction Check
```python
from fraud_detection import FraudDetector

# Initialize detector
detector = FraudDetector(api_key=os.getenv("FRAUD_DETECTION_API_KEY"))

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

#### Batch Processing
```python
# Check multiple transactions
results = detector.check_batch([
    {
        "amount": 1000.00,
        "type": "online",
        "merchant": "retail",
        "zip": "10001"
    },
    {
        "amount": 500.00,
        "type": "in-store",
        "merchant": "grocery",
        "zip": "90210"
    }
])

for result in results:
    print(f"Transaction {result.id}: {result.prediction}")
```

### 4. Configuration Options

```python
detector = FraudDetector(
    api_key=os.getenv("FRAUD_DETECTION_API_KEY"),
    environment="production",  # or "development"
    timeout=30,  # seconds
    retry_attempts=3
)
```

## Next Steps

1. Read the [API Documentation](../api/README.md) for detailed endpoint information
2. Check out [Integration Examples](../api/integration.md) for your language
3. Review [Security Best Practices](../api/integration.md#security-best-practices)
4. Set up [Monitoring](../OPERATIONS.md#monitoring) for your integration

## Common Issues

### API Connection
- Verify API key is correct
- Check network connectivity
- Ensure proper endpoint URLs

### Data Format
- Validate transaction data structure
- Check required fields
- Verify data types

## Support

If you encounter any issues:
1. Check our [Troubleshooting Guide](troubleshooting.md)
2. Visit our [Documentation](https://docs.fraud-detection.com)
3. Contact [Support](mailto:support@fraud-detection.com) 