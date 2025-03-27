#!/usr/bin/env python
"""
Script to run all EDA analyses on the fraud detection dataset.
"""
import logging
from pathlib import Path

from fraud_detection.core.eda import FraudEDA

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Run all EDA analyses on the dataset."""
    # Initialize EDA
    eda = FraudEDA(Path("data/raw/dataset.csv"))
    
    # Load data
    eda.load_data()
    
    # Run full analysis
    logger.info("Running full EDA analysis...")
    results = eda.run_full_analysis()
    
    logger.info("EDA completed successfully!")
    
    # Print key findings
    print("\nKey Findings:")
    print("1. Overall Statistics:")
    overall = results["population_analysis"]["overall"]
    print(f"   - Total Transactions: {overall['total_transactions']:,}")
    print(f"   - Overall Fraud Rate: {overall['fraud_rate']:.2%}")
    
    print("\n2. Transaction Types with Highest Risk:")
    type_analysis = results["population_analysis"]["transaction_types"]
    for type_, stats in sorted(type_analysis.items(), key=lambda x: x[1]['risk_ratio'], reverse=True):
        print(f"   - {type_}:")
        print(f"     Risk Ratio: {stats['risk_ratio']:.2f}")
        print(f"     Fraud Rate: {stats['fraud_rate']:.2%}")
        print(f"     Effect Size: {stats['effect_size']:.3f}")
    
    print("\n3. Most Important Numerical Features:")
    for feature in ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]:
        feature_stats = results["population_analysis"][feature]["differences"]
        print(f"   - {feature}:")
        print(f"     Information Value: {feature_stats['information_value']:.3f}")
        print(f"     Mean Difference: {feature_stats['mean_difference']:.2f}")
        print(f"     Variance Ratio: {feature_stats['variance_ratio']:.2f}")

if __name__ == "__main__":
    main() 