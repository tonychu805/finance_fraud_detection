# API Documentation

## Overview
The Fraud Detection API provides endpoints for real-time fraud detection and model management.

## Endpoints

### Prediction API
- `POST /api/v1/predict`
  - Input: Transaction data
  - Output: Fraud probability and classification

### Model Management
- `GET /api/v1/model/status`
  - Get current model version and status
- `POST /api/v1/model/update`
  - Trigger model update process

## Authentication
API authentication is handled via API keys. See [Authentication Guide](../development/authentication.md)

## Rate Limiting
- Standard tier: 100 requests/minute
- Enterprise tier: Custom limits

## Error Handling
Standard error responses follow RFC 7807 Problem Details format.

## Examples
```python
import requests

# Example prediction request
response = requests.post(
    "https://api.frauddetection.com/v1/predict",
    json={
        "amount": 1000.00,
        "type": "TRANSFER",
        "sender_id": "user123",
        "recipient_id": "merchant456"
    },
    headers={"X-API-Key": "your-api-key"}
)
```
