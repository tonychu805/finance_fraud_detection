"""
Analyze the trained fraud detection model.

This script loads a trained model and analyzes its performance metrics,
feature importance, and generates visualizations.
"""

import argparse
import json
import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
    auc
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Analyze trained fraud detection model")
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/fraud_ensemble_model.joblib",
        help="Path to the trained model"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/model_analysis",
        help="Directory to save analysis results"
    )
    return parser.parse_args()

def load_data():
    """Load the processed test data."""
    test_data = pd.read_csv("data/processed/test_data.csv")
    X_test = test_data.drop("is_fraud", axis=1)
    y_test = test_data["is_fraud"]
    return X_test, y_test

def plot_feature_importance(importance_dict, output_dir):
    """Plot feature importance for each model."""
    for model_name, importance in importance_dict.items():
        # Convert to DataFrame for easier plotting
        df = pd.DataFrame(
            list(importance.items()),
            columns=["Feature", "Importance"]
        ).sort_values("Importance", ascending=True)
        
        # Create horizontal bar plot
        plt.figure(figsize=(10, max(8, len(df) * 0.3)))
        sns.barplot(data=df, y="Feature", x="Importance", palette="viridis")
        plt.title(f"Feature Importance - {model_name}")
        plt.tight_layout()
        plt.savefig(output_dir / f"feature_importance_{model_name}.png")
        plt.close()

def plot_confusion_matrix(y_true, y_pred, output_dir):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Not Fraud", "Fraud"],
        yticklabels=["Not Fraud", "Fraud"]
    )
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png")
    plt.close()

def plot_roc_curve(y_true, y_prob, output_dir):
    """Plot ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_dir / "roc_curve.png")
    plt.close()

def plot_precision_recall_curve(y_true, y_prob, output_dir):
    """Plot precision-recall curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='darkorange', lw=2, label=f'PR curve (AUC = {pr_auc:.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(output_dir / "precision_recall_curve.png")
    plt.close()

def analyze_predictions(y_true, y_pred, y_prob, output_dir):
    """Analyze model predictions and save results."""
    # Calculate prediction statistics
    stats = {
        "total_samples": len(y_true),
        "fraud_samples": int(y_true.sum()),
        "predicted_fraud": int(y_pred.sum()),
        "true_positives": int(((y_true == 1) & (y_pred == 1)).sum()),
        "false_positives": int(((y_true == 0) & (y_pred == 1)).sum()),
        "true_negatives": int(((y_true == 0) & (y_pred == 0)).sum()),
        "false_negatives": int(((y_true == 1) & (y_pred == 0)).sum())
    }
    
    # Add derived metrics
    stats["precision"] = stats["true_positives"] / (stats["true_positives"] + stats["false_positives"])
    stats["recall"] = stats["true_positives"] / stats["fraud_samples"]
    stats["f1_score"] = 2 * (stats["precision"] * stats["recall"]) / (stats["precision"] + stats["recall"])
    
    # Save statistics
    with open(output_dir / "prediction_stats.json", "w") as f:
        json.dump(stats, f, indent=4)
    
    # Generate plots
    plot_confusion_matrix(y_true, y_pred, output_dir)
    plot_roc_curve(y_true, y_prob, output_dir)
    plot_precision_recall_curve(y_true, y_prob, output_dir)

def main():
    """Main analysis function."""
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model and data
    logger.info("Loading model and data...")
    model_data = joblib.load(args.model_path)
    X_test, y_test = load_data()
    
    # Get feature names
    feature_names = X_test.columns.tolist()
    
    # Extract model components
    ensemble_model = model_data["ensemble_model"]
    metadata = model_data["metadata"]
    
    # Update feature names in importance dict
    feature_importance = metadata["feature_importance"]
    for model_name, importance in feature_importance.items():
        named_importance = {}
        for i, (_, score) in enumerate(importance.items()):
            if i < len(feature_names):
                named_importance[feature_names[i]] = score
        feature_importance[model_name] = named_importance
    
    # Plot feature importance
    logger.info("Generating feature importance plots...")
    plot_feature_importance(feature_importance, output_dir)
    
    # Make predictions
    logger.info("Analyzing model predictions...")
    y_pred = ensemble_model.predict(X_test)
    y_prob = ensemble_model.predict_proba(X_test)[:, 1]
    
    # Analyze predictions
    analyze_predictions(y_test, y_pred, y_prob, output_dir)
    
    logger.info(f"Analysis complete. Results saved to {output_dir}")

if __name__ == "__main__":
    main() 