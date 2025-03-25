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
    """Run time series analysis."""
    # Initialize EDA
    eda = FraudEDA(Path("data/raw/dataset.csv"))
    
    # Load data
    eda.load_data()
    
    # Run time series analysis
    results = eda.time_series_analysis()
    
    # Print key findings
    print("\nTime Series Analysis Results:")
    print("\n1. Transaction Frequency:")
    print("Hourly transaction counts:")
    for hour, count in results["hourly_counts"].items():
        print(f"   Hour {hour}: {count:,} transactions")
        
    print("\nDaily transaction counts:")
    for day, count in results["daily_counts"].items():
        print(f"   Day {day}: {count:,} transactions")
        
    print("\n2. Fraud Rates:")
    print("Hourly fraud rates:")
    for hour, rate in results["hourly_fraud_rates"].items():
        print(f"   Hour {hour}: {rate:.2%}")
        
    print("\nDaily fraud rates:")
    for day, rate in results["daily_fraud_rates"].items():
        print(f"   Day {day}: {rate:.2%}")

if __name__ == "__main__":
    main() 