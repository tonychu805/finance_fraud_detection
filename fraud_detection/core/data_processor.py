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
from imblearn.over_sampling import SMOTE

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
    "newbalanceOrig",  # Added based on high IV (1.277)
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
    "newbalanceDest",  # Low IV (0.078) and less predictive
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
        Engineer new features from existing ones.

        Args:
            df: DataFrame with raw features

        Returns:
            pd.DataFrame: DataFrame with additional engineered features
        """
        logger.info("Engineering new features")
        df_engineered = df.copy()

        # 1. Transaction Amount Ratios (with safe division)
        df_engineered["amount_to_balance_ratio"] = np.where(
            df_engineered["oldbalanceOrg"] != 0,
            df_engineered["amount"] / df_engineered["oldbalanceOrg"],
            1.0  # Default value when denominator is 0
        )
        
        # 2. Time-based Features
        df_engineered["hour"] = df_engineered["step"] % 24
        df_engineered["day"] = df_engineered["step"] // 24
        df_engineered["is_high_risk_hour"] = df_engineered["hour"].isin([21, 22, 23]).astype(int)
        df_engineered["is_high_risk_day"] = df_engineered["day"].isin([2, 17, 18, 25, 26, 29]).astype(int)
        
        # 3. Transaction Type Risk Score (with reduced risk scores)
        risk_scores = {
            "TRANSFER": 2.0,  # Reduced from 5.96
            "CASH_OUT": 1.0,  # Reduced from 1.43
            "PAYMENT": 0.0,   # No risk
            "DEBIT": 0.0,     # No risk
            "CASH_IN": 0.0    # No risk
        }
        df_engineered["transaction_type_risk"] = df_engineered["type"].map(risk_scores)
        
        # 4. Recipient Balance Features (with safe division)
        df_engineered["recipient_balance_ratio"] = np.where(
            df_engineered["amount"] != 0,
            df_engineered["oldbalanceDest"] / df_engineered["amount"],
            1.0
        )
        
        # 5. Additional Risk Indicators
        df_engineered["large_transaction"] = (df_engineered["amount"] > df_engineered["amount"].quantile(0.95)).astype(int)
        df_engineered["zero_balance_org"] = (df_engineered["oldbalanceOrg"] == 0).astype(int)
        df_engineered["zero_balance_dest"] = (df_engineered["oldbalanceDest"] == 0).astype(int)
        
        # Handle any remaining infinite values
        for col in df_engineered.columns:
            if df_engineered[col].dtype in ['float64', 'float32']:
                # Replace infinite values with the column's mean
                mean_val = df_engineered[col].replace([np.inf, -np.inf], np.nan).mean()
                df_engineered[col] = df_engineered[col].replace([np.inf, -np.inf], mean_val)
        
        # Add new features to NUMERICAL_FEATURES
        new_numerical_features = [
            "amount_to_balance_ratio",
            "hour",
            "day",
            "is_high_risk_hour",
            "is_high_risk_day",
            "transaction_type_risk",
            "recipient_balance_ratio",
            "large_transaction",
            "zero_balance_org",
            "zero_balance_dest"
        ]
        
        # Update NUMERICAL_FEATURES
        global NUMERICAL_FEATURES
        NUMERICAL_FEATURES.extend(new_numerical_features)
        
        return df_engineered

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

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare final features for modeling."""
        logger.info("Preparing final features")
        
        # Make a copy to avoid modifying the original
        df_prepared = df.copy()
        
        # Drop unnecessary columns
        df_prepared = df_prepared.drop(columns=COLUMNS_TO_DROP, errors='ignore')
        
        # Rename target column if needed
        if "isFraud" in df_prepared.columns:
            df_prepared = df_prepared.rename(columns={"isFraud": "is_fraud"})
        
        # Separate features and target
        target_col = "is_fraud"
        if target_col in df_prepared.columns:
            features = df_prepared.drop(columns=[target_col])
            target = df_prepared[target_col]
        else:
            features = df_prepared
            target = None
        
        # Encode categorical features
        logger.info("Encoding categorical features")
        categorical_features = features.select_dtypes(include=['object']).columns
        for col in categorical_features:
            features[col] = features[col].astype('category').cat.codes
        logger.info("Categorical encoding completed")
        
        # Scale numerical features
        logger.info("Scaling numerical features")
        numerical_features = features.select_dtypes(include=['float64', 'int64']).columns
        scaler = StandardScaler()
        features[numerical_features] = scaler.fit_transform(features[numerical_features])
        logger.info("Numerical scaling completed")
        
        # Calculate feature correlations with target if available
        if target is not None:
            correlations = features.corrwith(target).sort_values(ascending=False)
            top_correlations = correlations.head(10).to_dict()
            logger.info(f"Top correlated features with fraud: {top_correlations}")
            
            # Save feature statistics
            feature_stats = {
                'correlations': correlations.to_dict(),
                'top_correlations': top_correlations
            }
            with open(self.processed_data_dir / "feature_stats.json", 'w') as f:
                json.dump(feature_stats, f, indent=4)
            logger.info("Saved feature statistics to data/processed/feature_stats.json")
        
        # Combine features and target back if target exists
        if target is not None:
            prepared_data = pd.concat([features, target], axis=1)
        else:
            prepared_data = features
        
        # Save prepared data
        prepared_data.to_csv(self.processed_data_dir / "prepared_data.csv", index=False)
        logger.info("Feature preparation completed")
        
        return prepared_data

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
        Split data into train, validation, and test sets using pure time-based splitting.
        
        Args:
            df: Data to split
            train_size: Proportion of data to use for training
            val_size: Proportion of data to use for validation
            target_col: Name of the target column
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Train, validation, and test sets
        """
        logger.info("Preparing train/validation/test splits using time-based approach")
        
        # Sort by step to ensure temporal order
        df = df.sort_values('step')
        
        # Calculate split indices based on number of samples
        n_samples = len(df)
        train_idx = int(n_samples * train_size)
        val_idx = int(n_samples * (train_size + val_size))
        
        # Split data temporally
        train_data = df.iloc[:train_idx].copy()
        val_data = df.iloc[train_idx:val_idx].copy()
        test_data = df.iloc[val_idx:].copy()
        
        # Log split information and fraud rates
        logger.info(f"Data split into {len(train_data)} training, {len(val_data)} validation, "
                   f"and {len(test_data)} test samples")
        
        # Log fraud rates for each split
        train_fraud_rate = train_data[target_col].mean()
        val_fraud_rate = val_data[target_col].mean()
        test_fraud_rate = test_data[target_col].mean()
        
        logger.info(f"Fraud rates:")
        logger.info(f"Training: {train_fraud_rate:.4f} ({train_fraud_rate*100:.2f}%)")
        logger.info(f"Validation: {val_fraud_rate:.4f} ({val_fraud_rate*100:.2f}%)")
        logger.info(f"Test: {test_fraud_rate:.4f} ({test_fraud_rate*100:.2f}%)")
        
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
        random_state: int = 42,
        handle_imbalance: bool = True,
        smote_ratio: float = 0.2
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Run the complete data processing pipeline.

        Args:
            sample_size: Number of rows to sample (for quick testing)
            test_size: Proportion of data to use for testing
            val_size: Proportion of remaining data to use for validation
            random_state: Random state for reproducibility
            handle_imbalance: Whether to handle class imbalance
            smote_ratio: Target ratio of minority class after SMOTE (default: 0.2 or 20%)

        Returns:
            Tuple containing (X_train, X_val, X_test, y_train, y_val, y_test, 
                            amount_train, amount_val, amount_test)
        """
        logger.info("Starting data processing pipeline")
        
        # Load raw data
        df = self.load_raw_data(sample_size)
        
        # Store transaction amounts before preprocessing
        transaction_amounts = pd.DataFrame({
            'amount': df['amount'].copy(),
            'is_synthetic': False
        })
        
        # Clean data
        df_clean = self.clean_data(df)
        
        # Drop high cardinality and potential data leakage columns
        logger.info("Dropping high cardinality and potential data leakage columns")
        columns_to_drop = COLUMNS_TO_DROP + ['newbalanceOrig', 'newbalanceDest']
        df_clean = df_clean.drop(columns=columns_to_drop, errors='ignore')
        
        # Engineer features
        df_features = self.engineer_features(df_clean)
        
        # Encode categorical features
        df_encoded = self.encode_categorical_features(df_features)
        
        # Scale numerical features
        df_scaled = self.scale_numerical_features(df_encoded)
        
        # Handle class imbalance if requested
        if handle_imbalance:
            logger.info("Handling class imbalance using SMOTE")
            
            # Prepare features and target
            X = df_scaled.drop("isFraud", axis=1)
            y = df_scaled["isFraud"].values
            
            # Store column names before converting to numpy array
            feature_columns = X.columns.tolist()
            X = X.values
            
            # Calculate number of samples needed for desired ratio
            n_majority = (y == 0).sum()
            n_minority = (y == 1).sum()
            target_minority = int(n_majority * smote_ratio)
            n_samples_needed = target_minority - n_minority
            
            if n_samples_needed > 0:
                # Apply SMOTE with calculated sampling strategy
                smote = SMOTE(
                    random_state=random_state,
                    sampling_strategy={1: target_minority}
                )
                X_balanced, y_balanced = smote.fit_resample(X, y)
                
                # Reconstruct data using stored column names
                df_scaled = pd.concat([
                    pd.DataFrame(X_balanced, columns=feature_columns),
                    pd.Series(y_balanced, name="isFraud")
                ], axis=1)
                
                # Create synthetic transaction amounts that match the synthetic samples
                fraud_indices = y == 1
                fraud_amounts = transaction_amounts.loc[fraud_indices, 'amount']
                n_repeats = n_samples_needed // len(fraud_amounts) + 1
                synthetic_amounts = pd.concat([fraud_amounts] * n_repeats)
                synthetic_amounts = synthetic_amounts[:n_samples_needed]
                
                # Create synthetic transactions DataFrame
                synthetic_transactions = pd.DataFrame({
                    'amount': synthetic_amounts,
                    'is_synthetic': True
                })
                
                # Combine original and synthetic transactions
                transaction_amounts = pd.concat([
                    transaction_amounts,
                    synthetic_transactions
                ]).reset_index(drop=True)
                
                logger.info(f"Balanced dataset size: {len(df_scaled)}")
                logger.info(f"New fraud rate: {y_balanced.mean():.4f}")
                logger.info(f"Average real transaction amount: ${transaction_amounts[~transaction_amounts['is_synthetic']]['amount'].mean():.2f}")
                logger.info(f"Max real transaction amount: ${transaction_amounts[~transaction_amounts['is_synthetic']]['amount'].max():.2f}")
                logger.info(f"Number of synthetic transactions: {transaction_amounts['is_synthetic'].sum()}")
            else:
                logger.info("No SMOTE needed - target ratio already achieved")
        
        # Convert to numpy arrays for splitting
        X = df_scaled.drop("isFraud", axis=1).values
        y = df_scaled["isFraud"].values
        amounts = transaction_amounts['amount'].values
        is_synthetic = transaction_amounts['is_synthetic'].values
        
        # Split data into train and test first
        X_train, X_test, y_train, y_test, amounts_train, amounts_test, is_synthetic_train, is_synthetic_test = train_test_split(
            X, y, amounts, is_synthetic,
            test_size=test_size,
            random_state=random_state,
            stratify=y if handle_imbalance else None
        )
        
        # Further split training data into train and validation
        X_train, X_val, y_train, y_val, amounts_train, amounts_val, is_synthetic_train, is_synthetic_val = train_test_split(
            X_train, y_train, amounts_train, is_synthetic_train,
            test_size=val_size,
            random_state=random_state,
            stratify=y_train if handle_imbalance else None
        )
        
        # Convert back to pandas DataFrames/Series
        train_data = pd.DataFrame(X_train, columns=df_scaled.drop("isFraud", axis=1).columns)
        train_data["isFraud"] = y_train
        val_data = pd.DataFrame(X_val, columns=df_scaled.drop("isFraud", axis=1).columns)
        val_data["isFraud"] = y_val
        test_data = pd.DataFrame(X_test, columns=df_scaled.drop("isFraud", axis=1).columns)
        test_data["isFraud"] = y_test
        
        train_amounts = pd.DataFrame({
            'amount': amounts_train,
            'is_synthetic': is_synthetic_train
        })
        val_amounts = pd.DataFrame({
            'amount': amounts_val,
            'is_synthetic': is_synthetic_val
        })
        test_amounts = pd.DataFrame({
            'amount': amounts_test,
            'is_synthetic': is_synthetic_test
        })
        
        # Save processed data
        self.save_processed_data(train_data, val_data, test_data)
        
        logger.info("Data processing pipeline completed")
        return (
            train_data.drop("isFraud", axis=1),
            val_data.drop("isFraud", axis=1),
            test_data.drop("isFraud", axis=1),
            train_data["isFraud"],
            val_data["isFraud"],
            test_data["isFraud"],
            train_amounts,
            val_amounts,
            test_amounts
        )

    def save_processed_data(self, train_data: pd.DataFrame, val_data: pd.DataFrame, test_data: pd.DataFrame) -> None:
        """Save processed datasets to files."""
        logger.info("Saving processed datasets")
        
        # Create processed data directory if it doesn't exist
        os.makedirs(self.processed_data_dir, exist_ok=True)
        
        # Save datasets
        train_data.to_csv(self.processed_data_dir / "train_data.csv", index=False)
        val_data.to_csv(self.processed_data_dir / "val_data.csv", index=False)
        test_data.to_csv(self.processed_data_dir / "test_data.csv", index=False)
        
        logger.info(f"Saved processed datasets to {self.processed_data_dir}")

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
        df_prepared = self.prepare_features(df_featured)
        
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

    def test_data_strategies(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42
    ) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
        """
        Test different data splitting and balancing strategies.
        
        Args:
            df: Input data
            test_size: Proportion of data to use for testing
            val_size: Proportion of remaining data to use for validation
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary containing different splitting strategies and their results
        """
        logger.info("Testing different data splitting and balancing strategies")
        
        # Drop string columns that shouldn't be used for modeling
        string_columns = df.select_dtypes(include=['object']).columns
        df_model = df.drop(columns=string_columns)
        
        # 1. Pure time-based splitting (current implementation)
        time_based = self.split_data(
            df_model,
            train_size=1-test_size-val_size,
            val_size=val_size,
            target_col="isFraud"
        )
        
        # 2. Random splitting with stratification
        train_data, test_data = train_test_split(
            df_model,
            test_size=test_size,
            random_state=random_state,
            stratify=df_model["isFraud"]
        )
        train_data, val_data = train_test_split(
            train_data,
            test_size=val_size,
            random_state=random_state,
            stratify=train_data["isFraud"]
        )
        
        # 3. Time-based splitting with SMOTE
        time_based_smote = list(time_based)
        X_train = time_based_smote[0].drop("isFraud", axis=1).values
        y_train = time_based_smote[0]["isFraud"].values
        
        smote = SMOTE(random_state=random_state)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        
        time_based_smote[0] = pd.concat([
            pd.DataFrame(X_train_balanced, columns=time_based_smote[0].drop("isFraud", axis=1).columns),
            pd.Series(y_train_balanced, name="isFraud")
        ], axis=1)
        
        # 4. Random splitting with stratification and SMOTE
        X_train = train_data.drop("isFraud", axis=1).values
        y_train = train_data["isFraud"].values
        
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        
        train_data_balanced = pd.concat([
            pd.DataFrame(X_train_balanced, columns=train_data.drop("isFraud", axis=1).columns),
            pd.Series(y_train_balanced, name="isFraud")
        ], axis=1)
        
        # Log statistics for each strategy
        strategies = {
            "time_based": time_based,
            "random_stratified": (train_data, val_data, test_data),
            "time_based_smote": tuple(time_based_smote),
            "random_stratified_smote": (train_data_balanced, val_data, test_data)
        }
        
        for name, (train, val, test) in strategies.items():
            logger.info(f"\n{name} strategy statistics:")
            logger.info(f"Training set size: {len(train)}")
            logger.info(f"Validation set size: {len(val)}")
            logger.info(f"Test set size: {len(test)}")
            logger.info(f"Training fraud rate: {train['isFraud'].mean():.4f}")
            logger.info(f"Validation fraud rate: {val['isFraud'].mean():.4f}")
            logger.info(f"Test fraud rate: {test['isFraud'].mean():.4f}")
        
        return strategies

if __name__ == "__main__":
    # Example usage
    processor = DataProcessor()
    
    # Process the full dataset
    X_train, X_val, X_test, y_train, y_val, y_test, train_amounts, val_amounts, test_amounts = processor.process_data_pipeline(
        sample_size=None,  # Use the full dataset
        save_processed=True
    )
    
    print(f"Training data shape: {X_train.shape}, {y_train.shape}")
    print(f"Validation data shape: {X_val.shape}, {y_val.shape}")
    print(f"Test data shape: {X_test.shape}, {y_test.shape}")
    
    # Show feature stats
    print("\nTop features correlated with fraud:")
    for feature, corr in list(processor.feature_stats.get("correlation_with_fraud", {}).items())[:10]:
        print(f"{feature}: {corr:.3f}") 