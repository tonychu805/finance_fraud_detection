#!/usr/bin/env python
"""
Script to run time series analysis on the fraud detection dataset.
"""
import logging
from pathlib import Path

from fraud_detection.core.eda import FraudEDA

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Run time series analysis on the dataset."""
    # Initialize EDA
    eda = FraudEDA(Path("data/raw/dataset.csv"))
    
    # Load data
    eda.load_data()
    
    # Run time series analysis
    results = eda.analyze_time_series()
    
    # Print results
    print("\nTime Series Analysis Results:")
    print("=" * 50)
    
    # 1. Overall Time Patterns
    print("\n1. Overall Time Patterns:")
    print("-" * 30)
    overall = results['overall']
    print(f"Total Time Period: {overall['total_period']}")
    print(f"Average Daily Transactions: {overall['avg_daily_transactions']:.2f}")
    print(f"Average Daily Fraud Rate: {overall['avg_daily_fraud_rate']:.2%}")
    
    # 2. Daily Patterns
    print("\n2. Daily Patterns:")
    print("-" * 30)
    daily = results['daily']
    print("\nDaily Transaction Counts:")
    for day, count in daily['transaction_counts'].items():
        print(f"  {day}: {count:,}")
    
    print("\nDaily Fraud Rates:")
    for day, rate in daily['fraud_rates'].items():
        print(f"  {day}: {rate:.2%}")
    
    # 3. Hourly Patterns
    print("\n3. Hourly Patterns:")
    print("-" * 30)
    hourly = results['hourly']
    print("\nHourly Transaction Counts:")
    for hour, count in hourly['transaction_counts'].items():
        print(f"  {hour:02d}:00 - {hour:02d}:59: {count:,}")
    
    print("\nHourly Fraud Rates:")
    for hour, rate in hourly['fraud_rates'].items():
        print(f"  {hour:02d}:00 - {hour:02d}:59: {rate:.2%}")
    
    # 4. Statistical Significance
    print("\n4. Statistical Significance:")
    print("-" * 30)
    
    # Daily patterns significance
    print("\nDaily Patterns:")
    daily_sig = results['statistical_significance']['daily']
    print(f"  Test Type: {daily_sig['test_type']}")
    print(f"  p-value: {daily_sig['p_value']:.4f}")
    print(f"  Effect Size: {daily_sig['effect_size']:.3f}")
    print(f"  Significant: {'Yes' if daily_sig['significant'] else 'No'}")
    
    # Hourly patterns significance
    print("\nHourly Patterns:")
    hourly_sig = results['statistical_significance']['hourly']
    print(f"  Test Type: {hourly_sig['test_type']}")
    print(f"  p-value: {hourly_sig['p_value']:.4f}")
    print(f"  Effect Size: {hourly_sig['effect_size']:.3f}")
    print(f"  Significant: {'Yes' if hourly_sig['significant'] else 'No'}")

if __name__ == "__main__":
    main() 