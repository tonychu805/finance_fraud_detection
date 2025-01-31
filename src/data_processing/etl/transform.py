import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import logging
from pathlib import Path
import sys

# Add src to Python path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))

from src.config import FEATURE_COLUMNS, DATA_DIR

logger = logging.getLogger(__name__)

class TransactionTransformer:
    def __init__(self):
        """Initialize the transformer with necessary encoders."""
        self.label_encoders = {}
        self.categorical_columns = [
            'merchant_category',
            'merchant_country',
            'card_type',
            'transaction_type'
        ]
        self.processed_data_path = DATA_DIR / "processed"
        self.processed_data_path.mkdir(exist_ok=True)
    
    def fit(self, df):
        """
        Fit the transformer on training data.
        
        Args:
            df (pandas.DataFrame): Training data
        """
        logger.info("Fitting transformers on training data")
        for col in self.categorical_columns:
            self.label_encoders[col] = LabelEncoder()
            self.label_encoders[col].fit(df[col].astype(str))
    
    def transform(self, df, save_local=True):
        """
        Transform the transaction data.
        
        Args:
            df (pandas.DataFrame): Transaction data to transform
            save_local (bool): Whether to save transformed data locally
            
        Returns:
            pandas.DataFrame: Transformed data
        """
        logger.info("Transforming transaction data")
        df_transformed = df.copy()
        
        # Encode categorical variables
        for col in self.categorical_columns:
            df_transformed[col] = self.label_encoders[col].transform(
                df_transformed[col].astype(str)
            )
        
        # Add time-based features
        df_transformed['hour_of_day'] = pd.to_datetime(df['timestamp']).dt.hour
        df_transformed['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
        
        # Add time since first and last transaction features
        timestamps = pd.to_datetime(df['timestamp'])
        df_transformed['time_since_first_transaction'] = (timestamps - timestamps.min()).dt.total_seconds()
        df_transformed['time_since_last_transaction'] = (timestamps - timestamps.shift(1)).dt.total_seconds()
        # Fill NaN for first transaction's time difference
        df_transformed['time_since_last_transaction'] = df_transformed['time_since_last_transaction'].fillna(0)
        
        # Calculate transaction amount statistics
        df_transformed['amount_log'] = np.log1p(df_transformed['amount'])
        
        # Select final features
        features = (
            FEATURE_COLUMNS +
            ['hour_of_day', 'day_of_week', 'amount_log']
        )
        
        if save_local:
            output_file = self.processed_data_path / "processed_transactions.parquet"
            df_transformed[features].to_parquet(output_file, index=False)
            logger.info(f"Saved processed data to {output_file}")
        
        return df_transformed[features]
    
    def fit_transform(self, df, save_local=True):
        """
        Fit the transformer and transform the data.
        
        Args:
            df (pandas.DataFrame): Data to fit and transform
            save_local (bool): Whether to save transformed data locally
            
        Returns:
            pandas.DataFrame: Transformed data
        """
        self.fit(df)
        return self.transform(df, save_local) 