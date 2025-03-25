#!/usr/bin/env python
"""
Script to run univariate analysis on the fraud detection dataset.
"""
import logging
from pathlib import Path

from fraud_detection.core.eda import FraudEDA

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Run univariate analysis."""
    # Initialize EDA
    eda = FraudEDA(Path("data/raw/dataset.csv"))
    
    # Load data
    eda.load_data()
    
    # Run univariate analysis
    results = eda.univariate_analysis()
    
    # Print key findings
    print("\nUnivariate Analysis Results:")
    
    # Numerical features
    print("\nNumerical Features Statistics:")
    for feature, stats in results.items():
        if feature not in eda.categorical_features:
            print(f"\n{feature}:")
            for stat, value in stats.items():
                print(f"   {stat}: {value:,.2f}")
    
    # Categorical features
    print("\nCategorical Features Distribution:")
    for feature in eda.categorical_features:
        print(f"\n{feature}:")
        value_counts = results[feature]["value_counts"]
        proportions = results[feature]["proportions"]
        for value, count in value_counts.items():
            prop = proportions[value]
            print(f"   {value}: {count:,} ({prop:.2%})")

if __name__ == "__main__":
    main() 