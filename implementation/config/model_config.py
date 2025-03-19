"""
Configuration module for model development.
"""
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class ModelConfig:
    """Configuration for model development pipeline."""
    
    # Data paths
    data_dir: str = "data/processed"
    
    # Random seed for reproducibility
    random_state: int = 42
    
    # Class imbalance handling
    smote_strategy: float = 0.1
    under_sampler_strategy: float = 0.5
    
    # Model parameters
    param_grid: Dict = None
    
    def __post_init__(self):
        """Initialize default parameter grid if not provided."""
        if self.param_grid is None:
            self.param_grid = {
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1],
                'n_estimators': [100, 200],
                'min_child_weight': [1, 3],
                'subsample': [0.8, 1.0]
            }
    
    # Evaluation settings
    cv_folds: int = 5
    scoring_metric: str = 'roc_auc'
    
    # Output settings
    output_dir: str = "data/processed/models"
    feature_importance_plot_name: str = "feature_importance.png"
    results_file_name: str = "model_results.json" 