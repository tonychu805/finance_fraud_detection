# API Documentation

## Version: 0.1.0

This document provides comprehensive information about the Finance Fraud Detection API endpoints, usage, and integration.

## API Overview

### Base URL
- Production: `https://api.fraud-detection.com/v1`
- Development: `http://localhost:8000/v1`

### Authentication
```bash
Authorization: Bearer YOUR_API_KEY
```

## Endpoints

### 1. Fraud Detection Prediction

#### Request
```http
POST /predict
Content-Type: application/json
Authorization: Bearer YOUR_API_KEY

{
    "transaction_amount": 1000.00,
    "transaction_type": "online",
    "merchant_category": "retail",
    "zip_code": "10001"
}
```

#### Response
```json
{
    "prediction": "legitimate",
    "confidence": 0.95,
    "transaction_id": "tx_123456789",
    "timestamp": "2024-03-23T14:30:00Z"
}
```

### 2. Batch Prediction

#### Request
```http
POST /predict/batch
Content-Type: application/json
Authorization: Bearer YOUR_API_KEY

{
    "transactions": [
        {
            "transaction_amount": 1000.00,
            "transaction_type": "online",
            "merchant_category": "retail",
            "zip_code": "10001"
        },
        {
            "transaction_amount": 500.00,
            "transaction_type": "in-store",
            "merchant_category": "grocery",
            "zip_code": "90210"
        }
    ]
}
```

#### Response
```json
{
    "predictions": [
        {
            "prediction": "legitimate",
            "confidence": 0.95,
            "transaction_id": "tx_123456789"
        },
        {
            "prediction": "fraudulent",
            "confidence": 0.87,
            "transaction_id": "tx_987654321"
        }
    ],
    "batch_id": "batch_123456",
    "timestamp": "2024-03-23T14:30:00Z"
}
```

## Error Handling

### Error Responses
```json
{
    "error": {
        "code": "INVALID_INPUT",
        "message": "Invalid transaction amount",
        "details": "Transaction amount must be greater than 0"
    }
}
```

### HTTP Status Codes
- 200: Success
- 400: Bad Request
- 401: Unauthorized
- 403: Forbidden
- 404: Not Found
- 429: Too Many Requests
- 500: Internal Server Error

## Rate Limiting

- Free Tier: 100 requests/hour
- Pro Tier: 10,000 requests/hour
- Enterprise: Custom limits

## Integration Examples

### Python
```python
import requests

API_KEY = "your_api_key"
BASE_URL = "https://api.fraud-detection.com/v1"

def predict_fraud(transaction_data):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    response = requests.post(
        f"{BASE_URL}/predict",
        json=transaction_data,
        headers=headers
    )
    
    return response.json()
```

### cURL
```bash
curl -X POST "https://api.fraud-detection.com/v1/predict" \
     -H "Authorization: Bearer your_api_key" \
     -H "Content-Type: application/json" \
     -d '{
         "transaction_amount": 1000.00,
         "transaction_type": "online",
         "merchant_category": "retail",
         "zip_code": "10001"
     }'
```

## Webhooks

### Configuration
```http
POST /webhooks/configure
Content-Type: application/json
Authorization: Bearer YOUR_API_KEY

{
    "url": "https://your-server.com/webhook",
    "events": ["fraud_detected", "high_risk_transaction"],
    "secret": "your_webhook_secret"
}
```

### Event Types
- `fraud_detected`
- `high_risk_transaction`
- `model_updated`
- `system_alert`

## API Versioning

- API version included in URL path
- Backward compatibility maintained within major versions
- Deprecation notices provided 6 months in advance

## Security

### Best Practices
1. Store API keys securely
2. Use HTTPS for all requests
3. Implement retry logic with exponential backoff
4. Monitor API usage and set up alerts

### API Key Management
- Rotate keys regularly
- Use separate keys for development and production
- Implement key scoping for different permissions

## Support

- Documentation: [docs.fraud-detection.com](https://docs.fraud-detection.com)
- Email: api-support@fraud-detection.com
- Status Page: [status.fraud-detection.com](https://status.fraud-detection.com)
