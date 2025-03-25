# API Integration Guide

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

### Event Types
- `fraud_detected`
- `high_risk_transaction`
- `model_updated`
- `system_alert`

## Security Best Practices

### API Key Management
1. Store API keys securely
2. Use HTTPS for all requests
3. Implement retry logic with exponential backoff
4. Monitor API usage and set up alerts

### Key Rotation
- Rotate keys regularly
- Use separate keys for development and production
- Implement key scoping for different permissions

## Support Resources

- Documentation: [docs.fraud-detection.com](https://docs.fraud-detection.com)
- Email: api-support@fraud-detection.com
- Status Page: [status.fraud-detection.com](https://status.fraud-detection.com) 