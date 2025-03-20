"""
ML Tracking module for monitoring model performance.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from fraud_detection.database.supabase_client import SupabaseClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLTracker:
    """
    MLTracker class for managing model versioning and performance tracking.
    
    This class provides methods to:
    - Log model training results
    - Track feature importance
    - Compare model versions
    - Analyze model performance trends
    """
    
    def __init__(self, supabase_client: SupabaseClient):
        """
        Initialize MLTracker with a Supabase client.
        
        Args:
            supabase_client: Initialized SupabaseClient instance
        """
        self.db = supabase_client
        
    async def log_training_result(self,
                                version: str,
                                algorithm: str,
                                metrics: Dict[str, float],
                                params: Dict,
                                importance: Dict[str, float]) -> Dict[str, Any]:
        """
        Log model training results to the database.
        
        Args:
            version: Model version identifier
            algorithm: Algorithm type (e.g., 'lightgbm', 'neural_network')
            metrics: Dictionary of performance metrics
            params: Model parameters
            importance: Feature importance scores
            
        Returns:
            Dict containing the insert result
        """
        try:
            # First, try to authenticate as admin
            await self.db.client.auth.sign_in_with_password({
                "email": "admin@frauddetection.com",
                "password": "admin123"  # This should be in environment variables in production
            })
            
            # Insert model version
            result = await self.db.client.table('model_versions').insert({
                'version': version,
                'algorithm_type': algorithm,
                'training_timestamp': datetime.now().isoformat(),
                'metrics': metrics,
                'parameters': params,
                'feature_importance': importance
            }).execute()
            
            # Also log detailed feature importance
            for feature, score in importance.items():
                await self.db.client.table('feature_importance').insert({
                    'model_version': version,
                    'feature_name': feature,
                    'importance_score': score
                }).execute()
                
            logger.info(f"Successfully logged training results for model version {version}")
            return result.data[0]
            
        except Exception as e:
            logger.error(f"Error logging training results: {str(e)}")
            # If we can't connect to Supabase, save locally as a fallback
            local_path = Path("logs/model_results")
            local_path.mkdir(parents=True, exist_ok=True)
            
            with open(local_path / f"model_{version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", "w") as f:
                json.dump({
                    'version': version,
                    'algorithm_type': algorithm,
                    'training_timestamp': datetime.now().isoformat(),
                    'metrics': metrics,
                    'parameters': params,
                    'feature_importance': importance
                }, f, indent=2)
            
            logger.info(f"Saved model results locally due to database error")
            return {
                'version': version,
                'metrics': metrics,
                'saved_locally': True
            }
            
    async def log_evaluation_result(self,
                                  version: str,
                                  dataset_type: str,
                                  metrics: Dict[str, float],
                                  confusion_matrix: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Log model evaluation results for different datasets.
        
        Args:
            version: Model version identifier
            dataset_type: Type of dataset ('train', 'validation', 'test')
            metrics: Dictionary of evaluation metrics
            confusion_matrix: Optional confusion matrix data
            
        Returns:
            Dict containing the insert result
        """
        try:
            # First, try to authenticate as admin
            await self.db.client.auth.sign_in_with_password({
                "email": "admin@frauddetection.com",
                "password": "admin123"  # This should be in environment variables in production
            })
            
            result = await self.db.client.table('evaluation_results').insert({
                'model_version': version,
                'dataset_type': dataset_type,
                'metrics': metrics,
                'confusion_matrix': confusion_matrix
            }).execute()
            
            logger.info(f"Successfully logged evaluation results for model {version} on {dataset_type} dataset")
            return result.data[0]
            
        except Exception as e:
            logger.error(f"Error logging evaluation results: {str(e)}")
            # If we can't connect to Supabase, save locally as a fallback
            local_path = Path("logs/evaluation_results")
            local_path.mkdir(parents=True, exist_ok=True)
            
            with open(local_path / f"eval_{version}_{dataset_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", "w") as f:
                json.dump({
                    'model_version': version,
                    'dataset_type': dataset_type,
                    'metrics': metrics,
                    'confusion_matrix': confusion_matrix,
                    'timestamp': datetime.now().isoformat()
                }, f, indent=2)
            
            logger.info(f"Saved evaluation results locally due to database error")
            return {
                'version': version,
                'dataset_type': dataset_type,
                'metrics': metrics,
                'saved_locally': True
            }
            
    async def get_best_model(self, metric: str = 'auc') -> Dict[str, Any]:
        """
        Get the best performing model based on a specific metric.
        
        Args:
            metric: Metric to use for comparison (default: 'auc')
            
        Returns:
            Dict containing the best model's information
        """
        try:
            # First, try to authenticate as admin
            await self.db.client.auth.sign_in_with_password({
                "email": "admin@frauddetection.com",
                "password": "admin123"  # This should be in environment variables in production
            })
            
            result = await self.db.client.table('model_versions')\
                .select('version, algorithm_type, metrics, training_timestamp')\
                .order(f"metrics->>'{metric}'", desc=True)\
                .limit(1)\
                .execute()
                
            return result.data[0] if result.data else None
            
        except Exception as e:
            logger.error(f"Error getting best model: {str(e)}")
            # If we can't connect to Supabase, try to read from local files
            local_path = Path("logs/model_results")
            if not local_path.exists():
                return None
                
            best_model = None
            best_score = float('-inf')
            
            for file in local_path.glob("*.json"):
                with open(file, "r") as f:
                    data = json.load(f)
                    score = data.get('metrics', {}).get(metric, float('-inf'))
                    if score > best_score:
                        best_score = score
                        best_model = data
                        
            return best_model
            
    async def compare_models(self, version1: str, version2: str) -> Dict[str, Any]:
        """
        Compare two model versions across all metrics.
        
        Args:
            version1: First model version
            version2: Second model version
            
        Returns:
            Dict containing comparison results
        """
        try:
            # First, try to authenticate as admin
            await self.db.client.auth.sign_in_with_password({
                "email": "admin@frauddetection.com",
                "password": "admin123"  # This should be in environment variables in production
            })
            
            models = await self.db.client.table('model_versions')\
                .select('version, metrics, feature_importance')\
                .in_('version', [version1, version2])\
                .execute()
                
            if len(models.data) != 2:
                raise ValueError("One or both model versions not found")
                
            model1 = next(m for m in models.data if m['version'] == version1)
            model2 = next(m for m in models.data if m['version'] == version2)
            
            return {
                'version_1': version1,
                'version_2': version2,
                'metrics_comparison': {
                    'version_1': model1['metrics'],
                    'version_2': model2['metrics']
                },
                'feature_importance_comparison': {
                    'version_1': model1['feature_importance'],
                    'version_2': model2['feature_importance']
                }
            }
            
        except Exception as e:
            logger.error(f"Error comparing models: {str(e)}")
            # If we can't connect to Supabase, try to read from local files
            local_path = Path("logs/model_results")
            if not local_path.exists():
                return None
                
            model1_data = None
            model2_data = None
            
            for file in local_path.glob("*.json"):
                with open(file, "r") as f:
                    data = json.load(f)
                    if data['version'] == version1:
                        model1_data = data
                    elif data['version'] == version2:
                        model2_data = data
                        
            if not model1_data or not model2_data:
                raise ValueError("One or both model versions not found in local storage")
                
            return {
                'version_1': version1,
                'version_2': version2,
                'metrics_comparison': {
                    'version_1': model1_data['metrics'],
                    'version_2': model2_data['metrics']
                },
                'feature_importance_comparison': {
                    'version_1': model1_data.get('feature_importance', {}),
                    'version_2': model2_data.get('feature_importance', {})
                }
            }
            
    async def get_feature_trends(self, top_n: int = 10) -> List[Dict[str, Any]]:
        """
        Analyze feature importance trends across model versions.
        
        Args:
            top_n: Number of top features to analyze
            
        Returns:
            List of dictionaries containing feature trends
        """
        try:
            # First, try to authenticate as admin
            await self.db.client.auth.sign_in_with_password({
                "email": "admin@frauddetection.com",
                "password": "admin123"  # This should be in environment variables in production
            })
            
            result = await self.db.client.table('feature_importance')\
                .select('feature_name, importance_score, model_version, timestamp')\
                .execute()
                
            # Process results to get feature trends
            feature_data = {}
            for row in result.data:
                feature = row['feature_name']
                if feature not in feature_data:
                    feature_data[feature] = {
                        'scores': [],
                        'versions': set()
                    }
                feature_data[feature]['scores'].append(row['importance_score'])
                feature_data[feature]['versions'].add(row['model_version'])
                
            # Calculate statistics
            trends = []
            for feature, data in feature_data.items():
                if len(data['versions']) > 1:  # Only include features present in multiple versions
                    trends.append({
                        'feature_name': feature,
                        'avg_importance': sum(data['scores']) / len(data['scores']),
                        'version_count': len(data['versions']),
                        'trend': 'stable' if max(data['scores']) - min(data['scores']) < 0.1 else 'variable'
                    })
                    
            # Sort by average importance and return top N
            trends.sort(key=lambda x: x['avg_importance'], reverse=True)
            return trends[:top_n]
            
        except Exception as e:
            logger.error(f"Error analyzing feature trends: {str(e)}")
            # If we can't connect to Supabase, try to read from local files
            local_path = Path("logs/model_results")
            if not local_path.exists():
                return []
                
            feature_data = {}
            
            for file in local_path.glob("*.json"):
                with open(file, "r") as f:
                    data = json.load(f)
                    version = data['version']
                    importance = data.get('feature_importance', {})
                    
                    for feature, score in importance.items():
                        if feature not in feature_data:
                            feature_data[feature] = {
                                'scores': [],
                                'versions': set()
                            }
                        feature_data[feature]['scores'].append(score)
                        feature_data[feature]['versions'].add(version)
                        
            # Calculate statistics
            trends = []
            for feature, data in feature_data.items():
                if len(data['versions']) > 1:  # Only include features present in multiple versions
                    trends.append({
                        'feature_name': feature,
                        'avg_importance': sum(data['scores']) / len(data['scores']),
                        'version_count': len(data['versions']),
                        'trend': 'stable' if max(data['scores']) - min(data['scores']) < 0.1 else 'variable'
                    })
                    
            # Sort by average importance and return top N
            trends.sort(key=lambda x: x['avg_importance'], reverse=True)
            return trends[:top_n] 