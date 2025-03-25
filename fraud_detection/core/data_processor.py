"""
Data processor for fraud detection.

This module handles the preprocessing of transaction data, feature engineering,
and preparation of datasets for model training and evaluation.
"""
import datetime
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
RAW_DATA_PATH = Path("data/raw/dataset.csv")
PROCESSED_DATA_DIR = Path("data/processed")
INTERIM_DATA_DIR = Path("data/interim")

# Features that will be used in the final model
CATEGORICAL_FEATURES = [
    "type",
]

NUMERICAL_FEATURES = [
    "amount",
    "oldbalanceOrg",
    "oldbalanceDest",
]

TIME_FEATURES = [
    "step",  # We'll use this for time-based splitting
]

# These columns will be dropped during preprocessing
COLUMNS_TO_DROP = [
    "nameOrig",
    "nameDest",
    "isFlaggedFraud",  # Remove this to prevent data leakage
    "newbalanceOrig",  # Remove post-transaction features
    "newbalanceDest",  # Remove post-transaction features
]

class DataProcessor:
    """
    Handles the preprocessing of financial transaction data for fraud detection.
    """

    def __init__(
        self, 
        raw_data_path: Optional[Path] = None,
        processed_data_dir: Optional[Path] = None,
        interim_data_dir: Optional[Path] = None,
    ):
        """
        Initialize the data processor.

        Args:
            raw_data_path: Path to the raw data file. If None, uses the default path.
            processed_data_dir: Directory for processed data. If None, uses the default.
            interim_data_dir: Directory for interim data. If None, uses the default.
        """
        self.raw_data_path = raw_data_path or RAW_DATA_PATH
        self.processed_data_dir = processed_data_dir or PROCESSED_DATA_DIR
        self.interim_data_dir = interim_data_dir or INTERIM_DATA_DIR
        
        # Ensure directories exist
        os.makedirs(self.processed_data_dir, exist_ok=True)
        os.makedirs(self.interim_data_dir, exist_ok=True)

        # Preprocessors
        self.categorical_encoders = {}
        self.scaler = None
        
        # Data statistics
        self.feature_stats = {}
        
        # Feature engineering stats (computed only on training data)
        self.amount_quantiles = None
        self.amount_mean = None
        self.amount_std = None

    def load_raw_data(self, sample_size: Optional[int] = None) -> pd.DataFrame:
        """
        Load the raw transaction data from CSV.

        Args:
            sample_size: Number of rows to sample (for quick testing). If None, loads all data.

        Returns:
            pd.DataFrame: Raw transaction data
        """
        logger.info(f"Loading raw data from {self.raw_data_path}")
        
        try:
            if sample_size:
                # Load a random sample for development/testing
                df = pd.read_csv(self.raw_data_path, nrows=sample_size)
                logger.info(f"Loaded random sample of {len(df)} transactions")
            else:
                df = pd.read_csv(self.raw_data_path)
                logger.info(f"Loaded complete dataset with {len(df)} transactions")
                
            return df
            
        except Exception as e:
            logger.error(f"Error loading raw data: {e}")
            # Return a small dummy dataset for demonstration if loading fails
            return self._create_dummy_data()

    def _create_dummy_data(self, n_samples: int = 1000) -> pd.DataFrame:
        """
        Create a dummy dataset for demonstration purposes.

        Args:
            n_samples: Number of dummy samples to generate

        Returns:
            pd.DataFrame: Dummy transaction data
        """
        logger.warning("Creating dummy data since raw data could not be loaded")
        
        # Generate random timestamps within the last 30 days
        now = datetime.datetime.now()
        timestamps = [
            now - datetime.timedelta(
                days=np.random.randint(0, 30),
                hours=np.random.randint(0, 24),
                minutes=np.random.randint(0, 60)
            )
            for _ in range(n_samples)
        ]
        
        # Create dummy categories
        merchant_categories = np.random.choice(
            ["retail", "restaurant", "travel", "entertainment", "electronics"], 
            size=n_samples
        )
        
        transaction_types = np.random.choice(
            ["purchase", "withdrawal", "payment", "refund"], 
            size=n_samples
        )
        
        countries = np.random.choice(
            ["US", "UK", "FR", "DE", "CA", "JP", "AU"], 
            size=n_samples
        )
        
        # Generate transaction data with a fraud rate of about 5%
        data = {
            "transaction_id": [f"dummy_tx_{i}" for i in range(n_samples)],
            "timestamp": [ts.isoformat() for ts in timestamps],
            "amount": np.random.exponential(scale=100, size=n_samples),
            "merchant_category": merchant_categories,
            "merchant_name": [f"Merchant_{i % 20}" for i in range(n_samples)],
            "transaction_type": transaction_types,
            "card_present": np.random.choice([True, False], size=n_samples),
            "country": countries,
            "is_fraud": np.random.choice(
                [True, False], 
                size=n_samples, 
                p=[0.05, 0.95]  # 5% fraud rate
            )
        }
        
        return pd.DataFrame(data)

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the raw data by handling missing values, outliers, etc.

        Args:
            df: Raw transaction data

        Returns:
            pd.DataFrame: Cleaned transaction data
        """
        logger.info("Cleaning transaction data")
        
        # Make a copy to avoid modifying the original
        df_clean = df.copy()
        
        # Handle missing values
        for col in df_clean.columns:
            missing_count = df_clean[col].isna().sum()
            if missing_count > 0:
                logger.info(f"Column {col} has {missing_count} missing values")
                
                if col in NUMERICAL_FEATURES:
                    # Fill numerical missing values with median
                    median_value = df_clean[col].median()
                    df_clean[col].fillna(median_value, inplace=True)
                    logger.info(f"Filled missing values in {col} with median: {median_value}")
                    
                elif col in CATEGORICAL_FEATURES:
                    # Fill categorical missing values with most frequent
                    mode_value = df_clean[col].mode()[0]
                    df_clean[col].fillna(mode_value, inplace=True)
                    logger.info(f"Filled missing values in {col} with mode: {mode_value}")
                    
                elif col in TIME_FEATURES:
                    # Fill time-based missing values with default values
                    df_clean[col].fillna(0, inplace=True)
                    logger.info(f"Filled missing values in {col} with default")
                    
                else:
                    # For other columns, just drop rows with missing values
                    df_clean.dropna(subset=[col], inplace=True)
                    logger.info(f"Dropped {missing_count} rows with missing values in {col}")
        
        # Remove duplicates
        n_duplicates = df_clean.duplicated().sum()
        if n_duplicates > 0:
            df_clean.drop_duplicates(inplace=True)
            logger.info(f"Removed {n_duplicates} duplicate transactions")
        
        # Handle outliers in amount (cap at 99.9th percentile)
        amount_cap = df_clean["amount"].quantile(0.999)
        outliers = df_clean[df_clean["amount"] > amount_cap]
        if len(outliers) > 0:
            df_clean.loc[df_clean["amount"] > amount_cap, "amount"] = amount_cap
            logger.info(f"Capped {len(outliers)} outliers in amount at {amount_cap}")
        
        return df_clean

    def extract_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract time-based features from the timestamp.

        Args:
            df: Transaction data with timestamp column

        Returns:
            pd.DataFrame: Data with additional time features
        """
        logger.info("Extracting time-based features")
        
        # Make a copy to avoid modifying the original
        df_with_features = df.copy()
        
        # Convert timestamp to datetime if it's not already
        if not pd.api.types.is_datetime64_dtype(df_with_features["timestamp"]):
            df_with_features["timestamp"] = pd.to_datetime(df_with_features["timestamp"])
        
        # Extract hour of day (0-23)
        df_with_features["hour_of_day"] = df_with_features["timestamp"].dt.hour
        
        # Extract day of week (0=Monday, 6=Sunday)
        df_with_features["day_of_week"] = df_with_features["timestamp"].dt.dayofweek
        
        # Is weekend (Saturday or Sunday)
        df_with_features["is_weekend"] = df_with_features["day_of_week"].isin([5, 6]).astype(int)
        
        # Is night time (10pm - 6am)
        df_with_features["is_night"] = ((df_with_features["hour_of_day"] >= 22) | 
                                         (df_with_features["hour_of_day"] < 6)).astype(int)
        
        # Month
        df_with_features["month"] = df_with_features["timestamp"].dt.month
        
        # Day of month
        df_with_features["day_of_month"] = df_with_features["timestamp"].dt.day
        
        logger.info("Time features extracted successfully")
        return df_with_features

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer additional features for fraud detection.
        Only uses information available at transaction time.

        Args:
            df: Cleaned transaction data

        Returns:
            pd.DataFrame: Data with engineered features
        """
        logger.info("Engineering additional features")
        
        # Make a copy to avoid modifying the original
        df_features = df.copy()
        
        # Calculate transaction amount ratio to balance
        df_features["amount_to_org_balance_ratio"] = df_features["amount"] / (df_features["oldbalanceOrg"] + 1)  # Add 1 to avoid division by zero
        df_features["amount_to_dest_balance_ratio"] = df_features["amount"] / (df_features["oldbalanceDest"] + 1)
        
        # Flag for zero initial balance
        df_features["zero_org_balance"] = (df_features["oldbalanceOrg"] == 0).astype(int)
        df_features["zero_dest_balance"] = (df_features["oldbalanceDest"] == 0).astype(int)
        
        # Flag for unusual transaction amounts (based on historical data)
        amount_threshold = df_features["amount"].quantile(0.95)
        df_features["unusual_amount"] = (df_features["amount"] > amount_threshold).astype(int)
        
        # Step-based features (assuming step represents time)
        df_features["step_hour"] = df_features["step"] % 24  # 24-hour cycle
        df_features["step_day"] = df_features["step"] // 24  # Day number
        df_features["is_night"] = ((df_features["step_hour"] >= 22) | (df_features["step_hour"] <= 5)).astype(int)
        df_features["is_weekend"] = (df_features["step_day"] % 7 >= 5).astype(int)
        
        logger.info("Feature engineering completed")
        return df_features

    def encode_categorical_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Encode categorical features for model training.

        Args:
            df: Data with categorical features
            fit: Whether to fit encoders on this data (True for training, False for inference)

        Returns:
            pd.DataFrame: Data with encoded categorical features
        """
        logger.info("Encoding categorical features")
        
        # Make a copy to avoid modifying the original
        df_encoded = df.copy()
        
        # Process each categorical feature
        for feature in CATEGORICAL_FEATURES:
            if feature not in df_encoded.columns:
                logger.warning(f"Feature {feature} not found in data")
                continue
                
            if fit:
                # Fit a new encoder for training data
                encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
                
                # Reshape to 2D array for fitting
                feature_values = df_encoded[feature].values.reshape(-1, 1)
                
                # Fit and transform
                encoder.fit(feature_values)
                encoded_features = encoder.transform(feature_values)
                
                # Store the encoder for future use
                self.categorical_encoders[feature] = encoder
                
                # Get feature names from the encoder
                categories = encoder.categories_[0]
                encoded_feature_names = [f"{feature}_{cat}" for cat in categories]
                
            else:
                # Use existing encoder for inference
                if feature not in self.categorical_encoders:
                    logger.error(f"No encoder found for feature {feature}. Using default encoding.")
                    # Create a simple hash-based encoding for inference without a trained encoder
                    df_encoded[f"{feature}_encoded"] = df_encoded[feature].apply(
                        lambda x: hash(str(x)) % 100 / 100
                    )
                    continue
                
                encoder = self.categorical_encoders[feature]
                feature_values = df_encoded[feature].values.reshape(-1, 1)
                encoded_features = encoder.transform(feature_values)
                
                # Use the same feature names as during training
                categories = encoder.categories_[0]
                encoded_feature_names = [f"{feature}_{cat}" for cat in categories]
            
            # Add encoded features to the dataframe
            for i, name in enumerate(encoded_feature_names):
                df_encoded[name] = encoded_features[:, i]
            
            # Drop the original categorical column
            df_encoded.drop(columns=[feature], inplace=True)
        
        logger.info("Categorical encoding completed")
        return df_encoded

    def scale_numerical_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Scale numerical features for model training.

        Args:
            df: Data with numerical features
            fit: Whether to fit scaler on this data (True for training, False for inference)

        Returns:
            pd.DataFrame: Data with scaled numerical features
        """
        logger.info("Scaling numerical features")
        
        # Make a copy to avoid modifying the original
        df_scaled = df.copy()
        
        # Identify numerical columns to scale
        numerical_cols = NUMERICAL_FEATURES + ["amount_log", "country_risk"]
        num_cols_present = [col for col in numerical_cols if col in df_scaled.columns]
        
        if not num_cols_present:
            logger.warning("No numerical features found for scaling")
            return df_scaled
        
        # Extract features to scale
        features_to_scale = df_scaled[num_cols_present].values
        
        if fit:
            # Fit a new scaler for training data
            self.scaler = StandardScaler()
            scaled_features = self.scaler.fit_transform(features_to_scale)
        else:
            # Use existing scaler for inference
            if self.scaler is None:
                logger.error("No scaler found. Using standardization without a fitted scaler.")
                # Simple standardization for inference without a trained scaler
                means = features_to_scale.mean(axis=0)
                stds = features_to_scale.std(axis=0)
                stds[stds == 0] = 1  # Avoid division by zero
                scaled_features = (features_to_scale - means) / stds
            else:
                scaled_features = self.scaler.transform(features_to_scale)
        
        # Update the dataframe with scaled values
        for i, col in enumerate(num_cols_present):
            df_scaled[f"{col}_scaled"] = scaled_features[:, i]
            
            # Keep the original feature for interpretability
            # df_scaled.drop(columns=[col], inplace=True)
        
        logger.info("Numerical scaling completed")
        return df_scaled

    def prepare_features(self, df: pd.DataFrame, training: bool = True) -> pd.DataFrame:
        """
        Prepare final feature set for model training or inference.

        Args:
            df: Processed transaction data
            training: Whether this is for training (True) or inference (False)

        Returns:
            pd.DataFrame: Data with final prepared features
        """
        logger.info("Preparing final features")
        
        # Make a copy
        df_prepared = df.copy()
        
        # Drop unnecessary columns
        df_prepared = df_prepared.drop(columns=COLUMNS_TO_DROP, errors='ignore')
        
        # Rename target column if needed
        if "isFraud" in df_prepared.columns:
            df_prepared = df_prepared.rename(columns={"isFraud": "is_fraud"})
        
        # Encode categorical features
        df_prepared = self.encode_categorical_features(df_prepared, fit=training)
        
        # Scale numerical features
        df_prepared = self.scale_numerical_features(df_prepared, fit=training)
        
        # For training data, we compute feature statistics
        if training and "is_fraud" in df_prepared.columns:
            # Get numeric columns only for correlation
            numeric_cols = df_prepared.select_dtypes(include=[np.number]).columns
            df_numeric = df_prepared[numeric_cols]
            
            if "is_fraud" in df_numeric.columns:
                corr_with_fraud = df_numeric.corrwith(df_numeric["is_fraud"]).abs().sort_values(ascending=False)
                self.feature_stats["correlation_with_fraud"] = corr_with_fraud.to_dict()
                logger.info(f"Top correlated features with fraud: {corr_with_fraud.head(5).to_dict()}")
            
            # Store feature names for later use
            self.feature_stats["feature_names"] = [
                col for col in df_prepared.columns 
                if col != "is_fraud"
            ]
        
        logger.info("Feature preparation completed")
        return df_prepared

    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data by cleaning and engineering features.
        
        Args:
            df: Raw transaction data
            
        Returns:
            pd.DataFrame: Prepared data
        """
        logger.info("Starting data preparation")
        
        # Clean data
        df_clean = self.clean_data(df)
        
        # Basic feature engineering
        df_featured = self._engineer_basic_features(df_clean)
        
        # Save interim data
        df_featured.to_csv(self.interim_data_dir / "featured_data.csv", index=False)
        logger.info("Saved featured data to data/interim/featured_data.csv")
        
        return df_featured

    def _engineer_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer basic features that don't risk data leakage.
        
        Args:
            df: Cleaned transaction data
            
        Returns:
            pd.DataFrame: Data with engineered features
        """
        df_featured = df.copy()
        
        # Time-based features
        df_featured['hour'] = df_featured['step'] % 24
        df_featured['day'] = (df_featured['step'] // 24) % 7
        df_featured['is_weekend'] = df_featured['day'].isin([5, 6]).astype(int)
        df_featured['is_night'] = df_featured['hour'].isin(range(23, 24) + range(0, 6)).astype(int)
        
        # Simple transaction features
        df_featured['zero_org_balance'] = (df_featured['oldbalanceOrg'] == 0).astype(int)
        df_featured['zero_dest_balance'] = (df_featured['oldbalanceDest'] == 0).astype(int)
        
        # One-hot encode transaction type
        df_featured = pd.get_dummies(df_featured, columns=['type'], prefix='type')
        
        return df_featured

    def split_data(
        self,
        df: pd.DataFrame,
        train_size: float = 0.7,
        val_size: float = 0.1,
        target_col: str = "is_fraud"
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, validation, and test sets using time-based splitting.
        
        Args:
            df: Data to split
            train_size: Proportion of data to use for training
            val_size: Proportion of data to use for validation
            target_col: Name of the target column
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Train, validation, and test sets
        """
        logger.info("Preparing train/validation/test splits")
        
        # Sort by step to ensure temporal order
        df_sorted = df.sort_values('step')
        
        # Calculate split indices
        n_samples = len(df_sorted)
        train_idx = int(n_samples * train_size)
        val_idx = int(n_samples * (train_size + val_size))
        
        # Split data
        train_data = df_sorted.iloc[:train_idx]
        val_data = df_sorted.iloc[train_idx:val_idx]
        test_data = df_sorted.iloc[val_idx:]
        
        # Log split information
        logger.info(f"Data split into {len(train_data)} training, {len(val_data)} validation, "
                   f"and {len(test_data)} test samples")
        
        # Log fraud rates
        train_fraud_rate = train_data[target_col].mean() * 100
        val_fraud_rate = val_data[target_col].mean() * 100
        test_fraud_rate = test_data[target_col].mean() * 100
        logger.info(f"Fraud rates - Train: {train_fraud_rate:.2f}%, "
                   f"Validation: {val_fraud_rate:.2f}%, Test: {test_fraud_rate:.2f}%")
        
        # Save splits
        train_data.to_csv(self.processed_data_dir / "train_data.csv", index=False)
        val_data.to_csv(self.processed_data_dir / "val_data.csv", index=False)
        test_data.to_csv(self.processed_data_dir / "test_data.csv", index=False)
        
        return train_data, val_data, test_data

    def scale_features(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        test_data: pd.DataFrame,
        numerical_features: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Scale numerical features using only training data statistics.
        
        Args:
            train_data: Training data
            val_data: Validation data
            test_data: Test data
            numerical_features: List of numerical features to scale
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Scaled datasets
        """
        if numerical_features is None:
            numerical_features = NUMERICAL_FEATURES
            
        # Initialize scaler if not already done
        if self.scaler is None:
            self.scaler = StandardScaler()
            
        # Fit scaler on training data only
        self.scaler.fit(train_data[numerical_features])
        
        # Transform all datasets
        train_scaled = train_data.copy()
        val_scaled = val_data.copy()
        test_scaled = test_data.copy()
        
        train_scaled[numerical_features] = self.scaler.transform(train_data[numerical_features])
        val_scaled[numerical_features] = self.scaler.transform(val_data[numerical_features])
        test_scaled[numerical_features] = self.scaler.transform(test_data[numerical_features])
        
        return train_scaled, val_scaled, test_scaled

    def save_feature_stats(self, train_data: pd.DataFrame) -> None:
        """
        Save feature statistics computed from training data.
        
        Args:
            train_data: Training data
        """
        # Calculate correlation with fraud
        correlations = train_data.corr()['is_fraud'].sort_values(ascending=False)
        
        # Save statistics
        self.feature_stats = {
            "correlation_with_fraud": correlations.to_dict(),
            "feature_names": train_data.columns.tolist()
        }
        
        # Save to file
        with open(self.processed_data_dir / "feature_stats.json", "w") as f:
            json.dump(self.feature_stats, f, indent=2)
            
        logger.info("Saved feature statistics to data/processed/feature_stats.json")

    def process_data_pipeline(
        self, 
        sample_size: Optional[int] = None,
        test_size: float = 0.2,
        val_size: float = 0.1,
        save_processed: bool = True,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        Run the full data processing pipeline.

        Args:
            sample_size: Number of rows to sample (for testing)
            test_size: Proportion of data for testing
            val_size: Proportion of data for validation
            save_processed: Whether to save processed data files
            random_state: Random seed for reproducibility

        Returns:
            Tuple containing X_train, X_val, X_test, y_train, y_val, y_test
        """
        logger.info("Starting full data processing pipeline")
        
        # Load raw data
        raw_df = self.load_raw_data(sample_size=sample_size)
        
        # Clean data
        cleaned_df = self.clean_data(raw_df)
        
        # Save interim data if requested
        if save_processed:
            interim_path = self.interim_data_dir / "cleaned_data.csv"
            cleaned_df.to_csv(interim_path, index=False)
            logger.info(f"Saved cleaned data to {interim_path}")
        
        # Engineer features
        featured_df = self.engineer_features(cleaned_df)
        
        # Save interim data if requested
        if save_processed:
            interim_path = self.interim_data_dir / "featured_data.csv"
            featured_df.to_csv(interim_path, index=False)
            logger.info(f"Saved featured data to {interim_path}")
        
        # Prepare final features
        prepared_df = self.prepare_features(featured_df, training=True)
        
        # Save processed data if requested
        if save_processed:
            processed_path = self.processed_data_dir / "prepared_data.csv"
            prepared_df.to_csv(processed_path, index=False)
            logger.info(f"Saved prepared data to {processed_path}")
            
            # Save feature stats
            self.save_feature_stats(prepared_df)
        
        # Split into train/validation/test sets
        splits = self.split_data(
            prepared_df, test_size=test_size, val_size=val_size, target_col="is_fraud"
        )
        
        # Save splits if requested
        if save_processed:
            X_train, X_val, X_test = splits
            
            train_path = self.processed_data_dir / "train_data.csv"
            pd.concat([X_train, X_train.iloc[:, -1]], axis=1).to_csv(train_path, index=False)
            logger.info(f"Saved training data to {train_path}")
            
            val_path = self.processed_data_dir / "val_data.csv"
            pd.concat([X_val, X_val.iloc[:, -1]], axis=1).to_csv(val_path, index=False)
            logger.info(f"Saved validation data to {val_path}")
            
            test_path = self.processed_data_dir / "test_data.csv"
            pd.concat([X_test, X_test.iloc[:, -1]], axis=1).to_csv(test_path, index=False)
            logger.info(f"Saved test data to {test_path}")
        
        logger.info("Data processing pipeline completed successfully")
        return splits

    def preprocess_transaction(self, transaction: Dict) -> np.ndarray:
        """
        Preprocess a single transaction for inference.

        Args:
            transaction: Transaction data dictionary

        Returns:
            np.ndarray: Preprocessed features ready for model prediction
        """
        # Convert transaction dict to dataframe
        df = pd.DataFrame([transaction])
        
        # Apply the same preprocessing as during training
        df_cleaned = self.clean_data(df)
        df_featured = self.engineer_features(df_cleaned)
        df_prepared = self.prepare_features(df_featured, training=False)
        
        # Select only the columns that were used during training
        if "feature_names" in self.feature_stats:
            feature_names = self.feature_stats["feature_names"]
            # Add any missing columns with default values
            for feature in feature_names:
                if feature not in df_prepared.columns:
                    df_prepared[feature] = 0
            
            # Select only the features used in training, in the correct order
            df_prepared = df_prepared[feature_names]
        
        return df_prepared.values

    def process_training_data(self) -> pd.DataFrame:
        """
        Load and process the training data.

        Returns:
            pd.DataFrame: Processed training data ready for model training
        """
        logger.info("Processing training data")
        
        # Load raw data
        df = self.load_raw_data()
        
        # Clean data
        df_clean = self.clean_data(df)
        
        # Drop unnecessary columns
        df_clean = df_clean.drop(columns=COLUMNS_TO_DROP, errors='ignore')
        
        # Extract time features if timestamp exists
        if 'timestamp' in df_clean.columns:
            df_clean = self.extract_time_features(df_clean)
        
        # Encode categorical features
        for cat_feature in CATEGORICAL_FEATURES:
            if cat_feature in df_clean.columns:
                encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                encoded_features = encoder.fit_transform(df_clean[[cat_feature]])
                feature_names = [f"{cat_feature}_{val}" for val in encoder.categories_[0]]
                encoded_df = pd.DataFrame(encoded_features, columns=feature_names)
                df_clean = pd.concat([df_clean.drop(columns=[cat_feature]), encoded_df], axis=1)
                self.categorical_encoders[cat_feature] = encoder
        
        # Scale numerical features
        if not self.scaler:
            self.scaler = StandardScaler()
            scaled_features = self.scaler.fit_transform(df_clean[NUMERICAL_FEATURES])
        else:
            scaled_features = self.scaler.transform(df_clean[NUMERICAL_FEATURES])
        
        scaled_df = pd.DataFrame(scaled_features, columns=NUMERICAL_FEATURES)
        df_clean[NUMERICAL_FEATURES] = scaled_df
        
        # Ensure target variable is properly named
        if 'isFraud' in df_clean.columns:
            df_clean = df_clean.rename(columns={'isFraud': 'fraud'})
        
        logger.info(f"Processed training data shape: {df_clean.shape}")
        return df_clean


if __name__ == "__main__":
    # Example usage
    processor = DataProcessor()
    
    # Process the full dataset
    X_train, X_val, X_test = processor.process_data_pipeline(
        sample_size=None,  # Use the full dataset
        save_processed=True
    )
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    # Show feature stats
    print("\nTop features correlated with fraud:")
    for feature, corr in list(processor.feature_stats.get("correlation_with_fraud", {}).items())[:10]:
        print(f"{feature}: {corr:.3f}") 