"""
Script to calibrate the ensemble model using Platt Scaling.
"""
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data():
    """Load and preprocess the dataset."""
    logger.info("Loading dataset...")
    df = pd.read_csv("data/raw/dataset.csv")
    
    # Extract features and target
    features = ['step', 'amount', 'oldbalanceOrg', 'oldbalanceDest']
    
    # One-hot encode transaction type
    type_dummies = pd.get_dummies(df['type'], prefix='type')
    
    # Combine features
    X = pd.concat([df[features], type_dummies], axis=1)
    y = df['isFraud']
    
    return X, y

def calibrate_ensemble():
    """Calibrate the ensemble model using Platt Scaling."""
    # Load the ensemble model
    logger.info("Loading ensemble model...")
    model_path = Path("models/fraud_ensemble_model.joblib")
    model_data = joblib.load(model_path)
    
    # Extract the ensemble model
    ensemble_model = model_data['ensemble_model']
    
    # Load and prepare data
    X, y = load_data()
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create calibrated model
    logger.info("Calibrating ensemble model...")
    calibrated_model = CalibratedClassifierCV(
        ensemble_model,
        cv='prefit',
        method='sigmoid'
    )
    
    # Fit calibration
    calibrated_model.fit(X_val, y_val)
    
    # Update model data
    model_data['ensemble_model'] = calibrated_model
    model_data['metadata']['calibration'] = {
        'method': 'sigmoid',
        'calibration_size': len(X_val),
        'date': pd.Timestamp.now().isoformat()
    }
    
    # Save calibrated model
    output_path = Path("models/fraud_ensemble_calibrated_model.joblib")
    logger.info(f"Saving calibrated model to {output_path}")
    joblib.dump(model_data, output_path)
    
    # Evaluate calibration
    from sklearn.metrics import brier_score_loss, roc_auc_score
    from sklearn.calibration import calibration_curve
    
    # Get predictions
    y_pred_proba = calibrated_model.predict_proba(X_val)[:, 1]
    
    # Calculate metrics
    brier = brier_score_loss(y_val, y_pred_proba)
    auc = roc_auc_score(y_val, y_pred_proba)
    
    # Calculate calibration curve
    prob_true, prob_pred = calibration_curve(y_val, y_pred_proba, n_bins=10)
    
    logger.info("\nCalibration Metrics:")
    logger.info(f"Brier Score: {brier:.4f}")
    logger.info(f"ROC AUC: {auc:.4f}")
    
    # Plot calibration curve
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    plt.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
    plt.plot(prob_pred, prob_true, 'o-', label='Model')
    plt.xlabel('Mean predicted probability')
    plt.ylabel('True probability')
    plt.title('Calibration Curve')
    plt.legend()
    plt.grid(True)
    
    # Save plot
    plot_path = Path("reports/figures/calibration_curve.png")
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path)
    plt.close()
    
    logger.info(f"\nCalibration curve saved to {plot_path}")
    
    return calibrated_model

if __name__ == "__main__":
    calibrate_ensemble() 