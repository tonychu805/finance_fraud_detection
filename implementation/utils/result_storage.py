"""
Result storage module for saving model results and visualizations.
"""
import json
from pathlib import Path
from typing import Dict, Any, Optional
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

from ..supabase_client import SupabaseClient

class ResultStorage:
    """Handles storage of model results and visualizations."""
    
    def __init__(self, output_dir: str, supabase_client: Optional[SupabaseClient] = None):
        """
        Initialize result storage.
        
        Args:
            output_dir: Directory to store results
            supabase_client: Optional Supabase client for database storage
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.supabase_client = supabase_client
    
    def save_results(self, results: Dict[str, Any], filename: str = "model_results.json", model_version: Optional[str] = None) -> None:
        """
        Save model results to JSON file and optionally to Supabase.
        
        Args:
            results: Dictionary of results to save
            filename: Name of the output file
            model_version: Optional model version for database storage
        """
        # Save to local file
        output_path = self.output_dir / filename
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        
        # Save to Supabase if client is available
        if self.supabase_client and model_version:
            try:
                # Prepare model version data
                model_version_data = {
                    "version": model_version,
                    "algorithm": results.get("algorithm", "Ensemble"),
                    "performance_metrics": results.get("performance_metrics", {}),
                    "feature_importance": results.get("feature_importance", {}),
                    "last_trained": datetime.now().isoformat(),
                }
                
                # Save to database
                self.supabase_client.add_model_version(model_version_data)
            except Exception as e:
                print(f"Error saving results to Supabase: {e}")
    
    def save_feature_importance(self, feature_importance: Dict[str, float], filename: str = "feature_importance.png", model_version: Optional[str] = None) -> None:
        """
        Save feature importance plot and data.
        
        Args:
            feature_importance: Dictionary of feature names and their importance scores
            filename: Name of the output file
            model_version: Optional model version for database storage
        """
        # Save plot locally
        plt.figure(figsize=(12, 6))
        importance_df = pd.DataFrame({
            'feature': list(feature_importance.keys()),
            'importance': list(feature_importance.values())
        }).sort_values('importance', ascending=False)
        
        plt.bar(range(len(importance_df)), importance_df['importance'])
        plt.xticks(range(len(importance_df)), importance_df['feature'], rotation=45)
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.savefig(self.output_dir / filename)
        plt.close()
        
        # Save to Supabase if client is available
        if self.supabase_client and model_version:
            try:
                # Save feature importance data
                feature_data = {
                    "model_version": model_version,
                    "feature_importance": feature_importance,
                    "timestamp": datetime.now().isoformat(),
                }
                
                # Save to database
                self.supabase_client.add_model_version({
                    "version": model_version,
                    "feature_importance": feature_importance,
                    "last_trained": datetime.now().isoformat(),
                })
            except Exception as e:
                print(f"Error saving feature importance to Supabase: {e}")
    
    def save_metrics_plots(self, y_true: Any, y_pred_proba: Any, model_name: str, model_version: Optional[str] = None) -> None:
        """
        Save ROC and precision-recall curves.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            model_name: Name of the model
            model_version: Optional model version for database storage
        """
        from ..evaluation.metrics import MetricsCalculator
        
        # Save ROC curve locally
        roc_path = self.output_dir / f"{model_name}_roc_curve.png"
        MetricsCalculator.plot_roc_curve(y_true, y_pred_proba, save_path=str(roc_path))
        
        # Save precision-recall curve locally
        pr_path = self.output_dir / f"{model_name}_pr_curve.png"
        MetricsCalculator.plot_precision_recall_curve(y_true, y_pred_proba, save_path=str(pr_path))
        
        # Calculate and save metrics to Supabase if client is available
        if self.supabase_client and model_version:
            try:
                metrics = MetricsCalculator.calculate_metrics(y_true, y_pred_proba > 0.5, y_pred_proba)
                
                # Save metrics to database
                self.supabase_client.add_model_version({
                    "version": model_version,
                    "performance_metrics": metrics,
                    "last_trained": datetime.now().isoformat(),
                })
            except Exception as e:
                print(f"Error saving metrics to Supabase: {e}") 