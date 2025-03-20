"""
Exploratory Data Analysis (EDA) module for fraud detection.

This module implements comprehensive EDA as outlined in the documentation,
including univariate, bivariate, and time series analysis.
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.feature_selection import mutual_info_classif

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FraudEDA:
    """
    Handles exploratory data analysis for fraud detection.
    """
    
    def __init__(self, data_path: Path):
        """
        Initialize the EDA module.
        
        Args:
            data_path: Path to the raw dataset
        """
        self.data_path = data_path
        self.df = None
        self.numerical_features = [
            "amount", "oldbalanceOrg", "newbalanceOrig",
            "oldbalanceDest", "newbalanceDest"
        ]
        self.categorical_features = ["type"]
        self.time_features = ["step"]
        
    def load_data(self) -> None:
        """Load and perform initial data validation."""
        logger.info(f"Loading data from {self.data_path}")
        self.df = pd.read_csv(self.data_path)
        logger.info(f"Loaded {len(self.df)} transactions")
        
        # Basic data validation
        self._validate_data()
        
    def _validate_data(self) -> None:
        """Perform basic data validation checks."""
        # Check for missing values
        missing = self.df.isnull().sum()
        if missing.any():
            logger.warning(f"Missing values found:\n{missing[missing > 0]}")
        
        # Check data types
        logger.info("Data types:\n" + str(self.df.dtypes))
        
        # Check for duplicates
        duplicates = self.df.duplicated().sum()
        if duplicates > 0:
            logger.warning(f"Found {duplicates} duplicate transactions")
            
    def univariate_analysis(self) -> Dict:
        """
        Perform univariate analysis on all features.
        
        Returns:
            Dictionary containing analysis results
        """
        results = {}
        
        # Numerical features analysis
        for feature in self.numerical_features:
            stats_dict = {
                "mean": self.df[feature].mean(),
                "median": self.df[feature].median(),
                "std": self.df[feature].std(),
                "min": self.df[feature].min(),
                "max": self.df[feature].max(),
                "q1": self.df[feature].quantile(0.25),
                "q3": self.df[feature].quantile(0.75)
            }
            results[feature] = stats_dict
            
            # Create distribution plot
            plt.figure(figsize=(10, 6))
            sns.histplot(data=self.df, x=feature, bins=50)
            plt.title(f"Distribution of {feature}")
            plt.savefig(f"data/processed/eda/{feature}_distribution.png")
            plt.close()
            
        # Categorical features analysis
        for feature in self.categorical_features:
            value_counts = self.df[feature].value_counts()
            results[feature] = {
                "value_counts": value_counts.to_dict(),
                "proportions": (value_counts / len(self.df)).to_dict()
            }
            
            # Create bar plot
            plt.figure(figsize=(10, 6))
            sns.countplot(data=self.df, x=feature)
            plt.title(f"Distribution of {feature}")
            plt.xticks(rotation=45)
            plt.savefig(f"data/processed/eda/{feature}_distribution.png")
            plt.close()
            
        return results
    
    def bivariate_analysis(self) -> Dict:
        """
        Perform bivariate analysis between features and fraud.
        
        Returns:
            Dictionary containing analysis results
        """
        results = {}
        
        # Feature-fraud correlations
        numerical_correlations = self.df[self.numerical_features].corrwith(self.df["isFraud"])
        results["numerical_correlations"] = numerical_correlations.to_dict()
        
        # Create correlation heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(self.df[self.numerical_features + ["isFraud"]].corr(),
                   annot=True, cmap="coolwarm", center=0)
        plt.title("Correlation Heatmap")
        plt.savefig("data/processed/eda/correlation_heatmap.png")
        plt.close()
        
        # Categorical feature analysis
        for feature in self.categorical_features:
            # Chi-square test
            contingency_table = pd.crosstab(self.df[feature], self.df["isFraud"])
            chi2, p_value = stats.chi2_contingency(contingency_table)[:2]
            
            # Calculate fraud rates by category
            fraud_rates = self.df.groupby(feature)["isFraud"].mean()
            
            results[feature] = {
                "chi2": chi2,
                "p_value": p_value,
                "fraud_rates": fraud_rates.to_dict()
            }
            
            # Create fraud rate plot
            plt.figure(figsize=(10, 6))
            fraud_rates.plot(kind="bar")
            plt.title(f"Fraud Rates by {feature}")
            plt.xticks(rotation=45)
            plt.savefig(f"data/processed/eda/{feature}_fraud_rates.png")
            plt.close()
            
        return results
    
    def time_series_analysis(self) -> Dict:
        """
        Perform time series analysis on transaction patterns.
        
        Returns:
            Dictionary containing analysis results
        """
        results = {}
        
        # Convert step to datetime-like features
        self.df["hour"] = self.df["step"] % 24
        self.df["day"] = self.df["step"] // 24
        
        # Transaction frequency over time
        hourly_counts = self.df.groupby("hour").size()
        daily_counts = self.df.groupby("day").size()
        
        results["hourly_counts"] = hourly_counts.to_dict()
        results["daily_counts"] = daily_counts.to_dict()
        
        # Fraud rates over time
        hourly_fraud = self.df.groupby("hour")["isFraud"].mean()
        daily_fraud = self.df.groupby("day")["isFraud"].mean()
        
        results["hourly_fraud_rates"] = hourly_fraud.to_dict()
        results["daily_fraud_rates"] = daily_fraud.to_dict()
        
        # Create time series plots
        plt.figure(figsize=(12, 6))
        hourly_fraud.plot(kind="line")
        plt.title("Fraud Rates by Hour")
        plt.savefig("data/processed/eda/hourly_fraud_rates.png")
        plt.close()
        
        plt.figure(figsize=(12, 6))
        daily_fraud.plot(kind="line")
        plt.title("Fraud Rates by Day")
        plt.savefig("data/processed/eda/daily_fraud_rates.png")
        plt.close()
        
        return results
    
    def run_full_analysis(self) -> Dict:
        """
        Run all EDA analyses and save results.
        
        Returns:
            Dictionary containing all analysis results
        """
        logger.info("Starting comprehensive EDA")
        
        # Create output directory
        output_dir = Path("data/processed/eda")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Run all analyses
        results = {
            "univariate": self.univariate_analysis(),
            "bivariate": self.bivariate_analysis(),
            "time_series": self.time_series_analysis()
        }
        
        # Save results to JSON
        import json
        with open(output_dir / "eda_results.json", "w") as f:
            json.dump(results, f, indent=2)
            
        logger.info("EDA completed successfully")
        return results

if __name__ == "__main__":
    # Example usage
    eda = FraudEDA(Path("data/raw/dataset.csv"))
    eda.load_data()
    results = eda.run_full_analysis()
    
    # Print key findings
    print("\nKey Findings:")
    print("1. Most important features for fraud detection:")
    correlations = results["bivariate"]["numerical_correlations"]
    for feature, corr in sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)[:5]:
        print(f"   - {feature}: {corr:.3f}")
        
    print("\n2. Transaction types with highest fraud rates:")
    fraud_rates = results["bivariate"]["type"]["fraud_rates"]
    for type_, rate in sorted(fraud_rates.items(), key=lambda x: x[1], reverse=True):
        print(f"   - {type_}: {rate:.2%}") 