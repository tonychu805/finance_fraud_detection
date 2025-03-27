# Exploratory Data Analysis (EDA) Findings

## 1. Dataset Overview
- Total Transactions: 6,362,620
- Fraud Rate: 0.13% (very imbalanced dataset)
- Time Period: Based on step feature (1 step = 1 hour)

## 2. Data Quality
### 2.1 Outliers
- Found 6,363 outliers in transaction amounts
- Capped at 8,956,797.68 (99.9th percentile)
- This helps prevent extreme values from skewing the model

### 2.2 Missing Values
- No missing values in key features
- All transactions have complete information

## 3. Feature Analysis
### 3.1 Numerical Features
- **amount**: Transaction amount
- **oldbalanceOrg**: Initial balance of sender
- **newbalanceOrig**: Final balance of sender
- **oldbalanceDest**: Initial balance of recipient
- **newbalanceDest**: Final balance of recipient

### 3.2 Categorical Features
- **type**: Transaction type (TRANSFER, CASH_OUT, PAYMENT, DEBIT, CASH_IN)
- **nameOrig**: Sender ID
- **nameDest**: Recipient ID

### 3.3 Time Features
- **step**: Hour of transaction (0-743)
- **hour**: Hour of day (0-23)
- **day**: Day of transaction (0-30)

## 4. Key Findings

### 4.1 Transaction Types and Fraud (risk core = fraud_rate / overall_fraud_rate)
- TRANSFER: Highest risk (5.96 risk score)
- CASH_OUT: Moderate risk (1.43 risk score)
- PAYMENT, DEBIT, CASH_IN: No risk (0.0 risk score)


### 4.2 Time Patterns
#### Time-based Splitting Results:
```
Training (earlier): 0.08% fraud
Validation (middle): 0.05% fraud
Test (later): 0.33% fraud
```
- Fraud rates increase over time
- Latest transactions show 4x higher fraud rate
- Suggests evolving fraud patterns

### 4.3 Feature Importance
- **newbalanceOrig**: High IV (1.277)
- **oldbalanceDest**: Low IV (0.078)
- Transaction amount ratios show strong predictive power

## 5. Data Processing Decisions

### 5.1 Feature Selection
- Kept: newbalanceOrig (high IV)
- Dropped: newbalanceDest (low IV)
- Dropped: nameOrig, nameDest (high cardinality)
- Dropped: isFlaggedFraud (potential data leakage)

### 5.2 Feature Engineering
1. Transaction Amount Ratios:
   - amount_to_balance_ratio
   - amount_to_newbalance_ratio

2. Balance Change Features:
   - balance_change
   - balance_change_ratio
   - balance_depletion

3. Time-based Features:
   - hour
   - day
   - is_high_risk_hour
   - is_high_risk_day

4. Transaction Type Risk Score:
   - Based on observed fraud rates

5. Recipient Balance Features:
   - recipient_balance_ratio

## 6. Data Splitting Strategy Recommendation

For model building, random stratified splitting with SMOTE is recommended over time-based splitting because:

1. Consistent Evaluation:
```
Random Stratified with SMOTE:
Training: 50.00% fraud (after SMOTE)
Validation: 0.13% fraud
Test: 0.13% fraud

vs Time-based:
Training: 0.08% fraud
Validation: 0.05% fraud
Test: 0.33% fraud
```

2. Benefits for Model Building:
- Reliable model evaluation with consistent fraud rates
- Better hyperparameter tuning due to stable validation metrics
- Standard approach for portfolio projects
- Easier to compare model performance across splits

3. Why Not Time-based:
- While time-based splitting shows increasing fraud rates (0.08% → 0.05% → 0.33%), which is realistic
- The varying fraud rates make model evaluation unreliable
- Different fraud rates across splits make it difficult to:
  - Tune hyperparameters consistently
  - Compare model performance
  - Determine if changes in performance are due to model improvements or data distribution changes

4. Implementation:
- Use random stratified splitting with SMOTE for model building
- Apply SMOTE only to training data to maintain realistic validation/test distributions
- Keep validation and test sets with original fraud rates (0.13%) for realistic evaluation

## 7. Next Steps

1. Model Development:
   - Implement random stratified splitting with SMOTE
   - Focus on balanced training data
   - Maintain consistent evaluation metrics

2. Feature Engineering:
   - Monitor engineered features' performance
   - Consider additional time-based features
   - Validate feature importance in model

3. Model Evaluation:
   - Use consistent fraud rates across splits
   - Focus on precision and recall metrics
   - Consider ROC-AUC for imbalanced data 