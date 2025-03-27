# Fraud Detection Model Documentation

## Model Overview
- **Version**: 1.0.0
- **Date**: 2024
- **Type**: Binary Classification
- **Framework**: Scikit-learn, LightGBM
- **Models**: 
  - LightGBM Classifier (with probability calibration)
  - Random Forest Classifier (with probability calibration)

## EDA and Model Training Relationship

### EDA Insights and Feature Selection
The model's feature selection and engineering decisions are based on comprehensive EDA results:

1. **Transaction Type Analysis**:
   - TRANSFER transactions show highest fraud risk (5.96x risk ratio, 0.77% fraud rate)
   - CASH_OUT transactions are second highest risk (1.43x risk ratio, 0.18% fraud rate)
   - Other transaction types (PAYMENT, DEBIT, CASH_IN) show no fraud cases

2. **Feature Importance (Information Value)**:
   - oldbalanceOrg: 2.068 (highest predictive power)
   - newbalanceOrig: 1.277
   - amount: 1.269
   - oldbalanceDest: 0.270
   - newbalanceDest: 0.078 (dropped due to low predictive power)

3. **Feature Engineering Decisions**:
   - Time-based features from 'step' column
   - Balance change ratios derived from EDA insights
   - Transaction type risk scores based on EDA risk ratios

### Data Processing Pipeline
The EDA results directly influence our data processing pipeline:
1. Feature selection based on Information Value
2. Outlier handling thresholds determined from EDA
3. Class imbalance handling informed by fraud rate analysis
4. Transaction type encoding based on risk analysis

### Model Training Considerations
EDA insights guide several aspects of model training:
1. Class weights adjusted based on fraud rate analysis
2. Threshold optimization informed by cost analysis
3. Model evaluation metrics chosen based on business impact
4. Feature importance monitoring aligned with EDA findings

## Performance Summary

### LightGBM
- **Optimal Threshold**: 0.2071
- **Metrics**:
  - Precision: 0.7683
  - Recall: 0.6939
  - F1 Score: 0.7292
  - ROC AUC: 0.9961
  - Average Precision: 0.7422

### Random Forest
- **Optimal Threshold**: 0.0994
- **Metrics**:
  - Precision: 0.7875
  - Recall: 0.7283
  - F1 Score: 0.7567
  - ROC AUC: 0.9971
  - Average Precision: 0.7684

## Technical Details

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

### Preprocessing Pipeline
1. Outlier handling: Amount capped at 99.9th percentile
2. Feature scaling: StandardScaler for numerical features
3. Categorical encoding: OneHotEncoder
4. Probability calibration using validation set

## Model Usage

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

## Monitoring and Maintenance

### Key Metrics to Monitor
1. Model Health:
   - Prediction latency
   - Probability distribution drift
   - Feature importance stability

2. Business Impact:
   - False positive rate by customer segment
   - Detection rate by fraud type
   - Financial loss prevention

### Alert Thresholds
- **Critical**: F1 score < 0.70, False positive rate > 1%, Latency > 100ms
- **Warning**: F1 score < 0.75, Data drift detected, High resource usage

## Limitations and Considerations

### Known Limitations
1. Class imbalance in training data
2. Temporal distribution shift between train and test sets
3. Limited feature set may miss some fraud patterns
4. Model performance may vary for different transaction types

### Ethical Considerations
1. False positives may inconvenience legitimate customers
2. False negatives may allow fraudulent transactions
3. Model decisions should be explainable to affected users
4. Regular monitoring needed for performance drift

## Contact
- Model Owner: [Your Name]
- Team: Fraud Detection
- Last Updated: [Date] 