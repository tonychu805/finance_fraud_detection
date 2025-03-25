#!/usr/bin/env python
"""
Script to run bivariate analysis on the fraud detection dataset.
"""
import logging
from pathlib import Path

from fraud_detection.core.eda import FraudEDA

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Run bivariate analysis on the dataset."""
    # Initialize EDA
    eda = FraudEDA(Path("data/raw/dataset.csv"))
    
    # Load data
    eda.load_data()
    
    # Run population-based analysis
    results = eda.analyze_population_patterns()
    
    # Print results
    print("\nPopulation-Based Analysis Results:")
    print("=" * 50)
    
    # 1. Overall Statistics
    print("\n1. Overall Statistics:")
    overall = results['overall']
    print(f"Total Transactions: {overall['total_transactions']:,}")
    print(f"Overall Fraud Rate: {overall['fraud_rate']:.2%}")
    print(f"Total Fraud Cases: {overall['fraud_count']:,}")
    
    # 2. Transaction Type Analysis
    print("\n2. Transaction Type Analysis:")
    print("-" * 30)
    type_analysis = results['transaction_types']
    for type_, stats in sorted(type_analysis.items(), key=lambda x: x[1]['risk_ratio'], reverse=True):
        print(f"\n{type_}:")
        print(f"  Transaction Count: {stats['transaction_count']:,}")
        print(f"  Fraud Count: {stats['fraud_count']:,}")
        print(f"  Fraud Rate: {stats['fraud_rate']:.2%}")
        print(f"  Risk Ratio: {stats['risk_ratio']:.2f}")
        print(f"  Effect Size: {stats['effect_size']:.3f}")
    
    # 3. Numerical Features Analysis
    print("\n3. Numerical Features Analysis:")
    print("-" * 30)
    for feature in eda.numerical_features:
        feature_stats = results[feature]
        print(f"\n{feature}:")
        print("  Fraud Transactions:")
        print(f"    Mean: {feature_stats['fraud']['mean']:.2f}")
        print(f"    Median: {feature_stats['fraud']['median']:.2f}")
        print(f"    Std Dev: {feature_stats['fraud']['std']:.2f}")
        print(f"    Q1: {feature_stats['fraud']['q1']:.2f}")
        print(f"    Q3: {feature_stats['fraud']['q3']:.2f}")
        
        print("  Legitimate Transactions:")
        print(f"    Mean: {feature_stats['legitimate']['mean']:.2f}")
        print(f"    Median: {feature_stats['legitimate']['median']:.2f}")
        print(f"    Std Dev: {feature_stats['legitimate']['std']:.2f}")
        print(f"    Q1: {feature_stats['legitimate']['q1']:.2f}")
        print(f"    Q3: {feature_stats['legitimate']['q3']:.2f}")
        
        print("  Differences:")
        print(f"    Mean Difference: {feature_stats['differences']['mean_difference']:.2f}")
        print(f"    Variance Ratio: {feature_stats['differences']['variance_ratio']:.2f}")
        print(f"    Information Value: {feature_stats['differences']['information_value']:.3f}")
    
    # 4. High Cardinality Features Analysis
    print("\n4. High Cardinality Features Analysis:")
    print("-" * 30)
    for feature in eda.high_cardinality_features:
        print(f"\n{feature}:")
        value_analysis = results[feature]
        for value, stats in sorted(value_analysis.items(), key=lambda x: x[1]['risk_ratio'], reverse=True):
            print(f"\n  Value: {value}")
            print(f"    Transaction Count: {stats['transaction_count']:,}")
            print(f"    Fraud Count: {stats['fraud_count']:,}")
            print(f"    Fraud Rate: {stats['fraud_rate']:.2%}")
            print(f"    Risk Ratio: {stats['risk_ratio']:.2f}")
            print(f"    Effect Size: {stats['effect_size']:.3f}")
    
    # 5. Binary Features Analysis
    print("\n5. Binary Features Analysis:")
    print("-" * 30)
    for feature in eda.binary_features:
        print(f"\n{feature}:")
        value_analysis = results[feature]
        for value, stats in sorted(value_analysis.items(), key=lambda x: x[1]['risk_ratio'], reverse=True):
            print(f"\n  Value: {value}")
            print(f"    Transaction Count: {stats['transaction_count']:,}")
            print(f"    Fraud Count: {stats['fraud_count']:,}")
            print(f"    Fraud Rate: {stats['fraud_rate']:.2%}")
            print(f"    Risk Ratio: {stats['risk_ratio']:.2f}")
            print(f"    Effect Size: {stats['effect_size']:.3f}")

if __name__ == "__main__":
    main() 