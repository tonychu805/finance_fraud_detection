#!/usr/bin/env python
"""
Script to test different data splitting and balancing strategies.
"""
import logging
from pathlib import Path

from fraud_detection.core.data_processor import DataProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Test different data splitting and balancing strategies."""
    # Initialize data processor
    processor = DataProcessor()
    
    # Load and preprocess data
    logger.info("Loading and preprocessing data...")
    df = processor.load_raw_data()
    df_clean = processor.clean_data(df)
    df_features = processor.engineer_features(df_clean)
    df_encoded = processor.encode_categorical_features(df_features)
    df_scaled = processor.scale_numerical_features(df_encoded)
    
    # Test different strategies
    logger.info("\nTesting different data splitting and balancing strategies...")
    strategies = processor.test_data_strategies(
        df_scaled,
        test_size=0.2,
        val_size=0.1,
        random_state=42
    )
    
    # Save results for each strategy
    for name, (train, val, test) in strategies.items():
        # Create strategy-specific directories
        strategy_dir = Path("data/processed") / name
        strategy_dir.mkdir(parents=True, exist_ok=True)
        
        # Save datasets
        train.to_csv(strategy_dir / "train_data.csv", index=False)
        val.to_csv(strategy_dir / "val_data.csv", index=False)
        test.to_csv(strategy_dir / "test_data.csv", index=False)
        
        logger.info(f"\nSaved {name} strategy datasets to {strategy_dir}")

if __name__ == "__main__":
    main() 