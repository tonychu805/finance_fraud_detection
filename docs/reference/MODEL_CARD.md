# Fraud Detection Model Card

## Current Performance (Last Updated: 2024-03-24)

### LightGBM
- **Version**: 1.0.0
- **Optimal Threshold**: 0.2071
- **Metrics**:
  - F1 Score: 0.7292
  - ROC AUC: 0.9961
  - Average Precision: 0.7422
  - False Positive Rate: 0.92%

### Random Forest
- **Version**: 1.0.0
- **Optimal Threshold**: 0.0994
- **Metrics**:
  - F1 Score: 0.7567
  - ROC AUC: 0.9971
  - Average Precision: 0.7684
  - False Positive Rate: 0.85%

## Performance History

### LightGBM History
| Date       | Version | F1 Score | ROC AUC | FPR  | Notes                    |
|------------|---------|----------|---------|------|--------------------------|
| 2024-03-24 | 1.0.0   | 0.7292   | 0.9961  | 0.92%| Initial release         |

### Random Forest History
| Date       | Version | F1 Score | ROC AUC | FPR  | Notes                    |
|------------|---------|----------|---------|------|--------------------------|
| 2024-03-24 | 1.0.0   | 0.7567   | 0.9971  | 0.85%| Initial release         |

## Model Details

### Architecture
- **LightGBM**:
  - Gradient boosting framework
  - Early stopping enabled
  - Probability calibration using sigmoid method
  - Custom threshold optimization

- **Random Forest**:
  - 200 trees
  - Max depth: 10
  - Class weight: balanced
  - Probability calibration using sigmoid method
  - Custom threshold optimization

### Features
```python
CATEGORICAL_FEATURES = ["type"]
NUMERICAL_FEATURES = ["amount", "oldbalanceOrg", "oldbalanceDest"]
TIME_FEATURES = ["step"]
```

### Data Characteristics
- **Total Samples**: 6,362,620
- **Class Distribution**:
  - Train set: 0.08% fraud rate
  - Validation set: 0.05% fraud rate
  - Test set: 0.33% fraud rate
- **Data Split**: Temporal split (not random) to prevent data leakage

## Preprocessing
1. Outlier handling: Amount capped at 99.9th percentile
2. Feature scaling: StandardScaler for numerical features
3. Categorical encoding: OneHotEncoder
4. Probability calibration using validation set

## Limitations and Biases
1. Class imbalance in training data
2. Temporal distribution shift between train and test sets
3. Limited feature set may miss some fraud patterns
4. Model performance may vary for different transaction types

## Ethical Considerations
1. False positives may inconvenience legitimate customers
2. False negatives may allow fraudulent transactions
3. Model decisions should be explainable to affected users
4. Regular monitoring needed for performance drift

## Future Improvements
1. Feature Engineering:
   - Add more derived features from transaction patterns
   - Incorporate temporal patterns and user behavior

2. Model Enhancements:
   - Implement ensemble of both models
   - Explore deep learning approaches
   - Add anomaly detection components

3. Operational Improvements:
   - Add real-time monitoring
   - Implement A/B testing framework
   - Add model explainability tools

## Maintenance
- Regular retraining schedule: Monthly
- Performance monitoring: Daily
- Data drift detection: Weekly
- Model updates logged in version control

## Version History

### v1.0.0 (Current)
- Initial release with LightGBM and Random Forest models
- Implemented probability calibration
- Added optimal threshold finding
- Documented baseline performance

## Usage Guidelines
1. Load model with optimal threshold:
```python
import joblib

# Load model info
model_info = joblib.load('models/random_forest_calibrated_model.joblib')
model = model_info['model']
threshold = model_info['optimal_threshold']

# Make predictions
y_prob = model.predict_proba(X)[:, 1]
y_pred = (y_prob >= threshold).astype(int)
```

2. Monitor predictions for:
   - Unusual patterns
   - Performance degradation
   - Data drift
   - Business impact

## Contact
- Model Owner: [Your Name]
- Team: Fraud Detection
- Last Updated: [Date] 

## Update Template
Copy this template when adding new performance metrics:

```markdown
### Model Update YYYY-MM-DD
#### [Model Name]
- **Version**: X.Y.Z
- **Optimal Threshold**: 0.XXXX
- **Metrics**:
  - F1 Score: 0.XXXX
  - ROC AUC: 0.XXXX
  - Average Precision: 0.XXXX
  - False Positive Rate: X.XX%

Add row to history table:
| 2024-03-24 | X.Y.Z   | 0.XXXX  | 0.XXXX | X.XX%| Brief description of changes |
``` 