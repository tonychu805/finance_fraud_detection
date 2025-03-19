# Implementation Guide: Fraud Detection System

This directory contains the core implementation of the fraud detection system. This guide provides detailed technical information about the system's components, their interactions, and usage examples.

## Technical Components

### Core Modules

1. **Data Processing** (`data_processor.py`)
   - Feature engineering pipeline
   - Data cleaning and normalization
   - Train/validation/test split logic

2. **EDA Module** (`eda.py`)
   - Univariate analysis
   - Feature correlation analysis
   - Time series analysis
   - Anomaly detection

3. **Model Development** (`model_development.py`)
   - Model training pipeline
   - Hyperparameter tuning
   - Cross-validation
   - Model ensembling

4. **Evaluation** (`evaluation/`)
   - Custom metrics calculation
   - Model performance analysis
   - Threshold optimization
   - Feature importance analysis

5. **Database Integration** (`supabase_client.py`)
   - Result storage
   - Model metadata tracking
   - Performance logging

## Implementation Details

### Data Processing Pipeline

```python
from implementation.data_processor import DataProcessor

processor = DataProcessor(
    raw_data_path="data/raw/dataset.csv",
    processed_data_dir="data/processed"
)

# Available preprocessing steps:
processor.clean_data()           # Basic cleaning
processor.engineer_features()    # Feature engineering
processor.normalize_features()   # Normalization
processor.split_data()          # Data splitting

# Or run full pipeline:
data = processor.process_data_pipeline(
    sample_size=None,
    test_size=0.2,
    val_size=0.2
)
```

### Model Development

```python
from implementation.model_development import FraudModelDevelopment
from implementation.config import ModelConfig

# Configure model parameters
config = ModelConfig(
    lightgbm_params={
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05
    },
    neural_network_params={
        'layers': [64, 32, 16],
        'dropout': 0.3,
        'learning_rate': 0.001
    }
)

# Initialize development pipeline
model_dev = FraudModelDevelopment(config)

# Train models
model_dev.train_lightgbm()
model_dev.train_neural_network()
model_dev.train_ensemble()
```

### Model Evaluation

```python
from implementation.evaluation import MetricsCalculator

# Calculate comprehensive metrics
metrics = MetricsCalculator.calculate_metrics(
    y_true=y_test,
    y_pred=predictions,
    y_pred_proba=probabilities
)

# Available metrics:
print(f"ROC AUC: {metrics['roc_auc']:.3f}")
print(f"PR AUC: {metrics['pr_auc']:.3f}")
print(f"F1 Score: {metrics['f1']:.3f}")
print(f"Precision: {metrics['precision']:.3f}")
print(f"Recall: {metrics['recall']:.3f}")
```

### Database Operations

```python
from implementation.supabase_client import SupabaseClient

db = SupabaseClient()

# Save model results
db.save_model_results(
    model_name="ensemble_v1",
    metrics={
        'roc_auc': 0.95,
        'precision': 0.92,
        'recall': 0.88
    }
)

# Save feature importance
db.save_feature_importance(
    model_name="ensemble_v1",
    feature_importance={
        'amount': 0.8,
        'transaction_time': 0.6
    }
)
```

## Configuration

### Model Configuration

```python
from implementation.config import ModelConfig

config = ModelConfig(
    data_dir="data/processed",
    output_dir="models",
    model_version="v1.0.0",
    random_state=42,
    # Add model-specific parameters
)
```

### Environment Variables

Required environment variables:
```bash
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
MODEL_OUTPUT_DIR=path/to/output
```

## Testing

Run specific test modules:
```bash
pytest tests/test_data_processor.py
pytest tests/test_model_development.py
pytest tests/test_evaluation.py
```

## Debugging

Common debugging steps:
1. Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

2. Check data processing output:
```python
processor = DataProcessor()
processor.validate_output()  # Validates processed data
```

3. Monitor model training:
```python
model_dev = FraudModelDevelopment(config)
model_dev.enable_training_verbose()  # Enables detailed training logs
```

## Performance Optimization

Tips for optimizing performance:
- Use `sample_size` parameter during development
- Enable multiprocessing for data processing
- Use GPU acceleration for neural networks
- Implement batch processing for large datasets

## Error Handling

The implementation includes comprehensive error handling:
- Data validation errors
- Model training exceptions
- Database connection issues
- Configuration validation

For detailed error messages and solutions, see [error_handling.md](docs/error_handling.md). 