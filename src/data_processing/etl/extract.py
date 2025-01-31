import pandas as pd
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class TransactionExtractor:
    def __init__(self):
        """Initialize the extractor with sample data generation capabilities."""
        self.raw_data_path = Path(__file__).resolve().parent.parent.parent.parent / "data" / "raw"
        self.raw_data_path.mkdir(exist_ok=True, parents=True)

    def generate_sample_data(self, size=1000):
        """
        Generate synthetic transaction data for testing.
        
        Args:
            size (int): Number of transactions to generate
            
        Returns:
            pandas.DataFrame: Generated transaction data
        """
        np.random.seed(42)  # For reproducibility
        
        # Generate timestamps
        end_date = pd.Timestamp.now()
        start_date = end_date - pd.Timedelta(days=30)
        timestamps = pd.date_range(start=start_date, end=end_date, periods=size)
        
        # Generate transaction data
        data = {
            'transaction_id': range(1, size + 1),
            'amount': np.random.exponential(100, size),  # Transaction amounts
            'timestamp': timestamps,
            'merchant_category': np.random.choice(
                ['retail', 'restaurant', 'travel', 'entertainment', 'other'],
                size
            ),
            'merchant_country': np.random.choice(
                ['US', 'UK', 'FR', 'DE', 'JP', 'CN', 'BR'],
                size
            ),
            'card_type': np.random.choice(
                ['credit', 'debit', 'virtual'],
                size
            ),
            'transaction_type': np.random.choice(
                ['online', 'in-store', 'mobile'],
                size
            ),
            'is_fraudulent': np.random.choice(
                [0, 1],
                size,
                p=[0.99, 0.01]  # 1% fraud rate
            )
        }
        
        return pd.DataFrame(data)

    def extract_sample_transactions(self, sample_size=1000):
        """
        Extract a sample of transactions (synthetic data for development).
        
        Args:
            sample_size (int): Number of transactions to sample
            
        Returns:
            pandas.DataFrame: Sampled transaction data
        """
        try:
            logger.info(f"Generating {sample_size} sample transactions")
            df = self.generate_sample_data(size=sample_size)
            
            # Save to parquet file
            output_file = self.raw_data_path / "sample_transactions.parquet"
            df.to_parquet(output_file, index=False)
            logger.info(f"Saved {len(df)} transactions to {output_file}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error generating sample transactions: {str(e)}")
            raise 