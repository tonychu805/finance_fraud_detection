# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import os
from datetime import datetime

# Add project root to Python path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

# Import our custom modules
from src.data_processing.etl.extract import TransactionExtractor
from src.data_processing.etl.transform import TransactionTransformer

# Set plotting style
plt.style.use('default')  # Use default style instead of seaborn
sns.set_theme()  # This will set the seaborn theme

def load_sample_data():
    """Load sample transaction data"""
    extractor = TransactionExtractor()
    df = extractor.extract_sample_transactions(sample_size=10000)
    print(f"Loaded {len(df)} transactions")
    return df

def analyze_basic_stats(df):
    """Display basic statistics about the dataset"""
    print("\n=== Basic Dataset Information ===")
    print("\nDataset Info:")
    print(df.info())
    
    print("\nBasic Statistics:")
    print(df.describe())
    
    print("\nMissing Values:")
    missing = df.isnull().sum()
    print(missing[missing > 0])

def analyze_fraud_distribution(df):
    """Analyze and plot fraud distribution"""
    print("\n=== Fraud Distribution Analysis ===")
    
    fraud_dist = df['is_fraudulent'].value_counts(normalize=True)
    
    plt.figure(figsize=(10, 6))
    fraud_dist.plot(kind='bar')
    plt.title('Distribution of Fraudulent vs Non-Fraudulent Transactions')
    plt.xlabel('Is Fraudulent')
    plt.ylabel('Percentage')
    plt.xticks(rotation=0)
    plt.show()
    
    print("\nFraud Distribution:")
    print(fraud_dist)
    
    # Calculate fraud rate
    fraud_rate = (df['is_fraudulent'] == 1).mean() * 100
    print(f"\nOverall Fraud Rate: {fraud_rate:.2f}%")

def analyze_amounts(df):
    """Analyze transaction amounts"""
    print("\n=== Transaction Amount Analysis ===")
    
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='is_fraudulent', y='amount', data=df)
    plt.title('Transaction Amount Distribution by Fraud Status')
    plt.yscale('log')
    plt.show()
    
    print("\nAmount Statistics by Fraud Status:")
    print(df.groupby('is_fraudulent')['amount'].describe())
    
    # Additional amount analysis
    plt.figure(figsize=(12, 6))
    sns.histplot(data=df, x='amount', hue='is_fraudulent', bins=50, log_scale=True)
    plt.title('Transaction Amount Distribution')
    plt.xlabel('Amount (log scale)')
    plt.ylabel('Count')
    plt.show()

def analyze_temporal_patterns(df):
    """Analyze temporal patterns in transactions"""
    print("\n=== Temporal Pattern Analysis ===")
    
    # Convert timestamp to datetime if it's not already
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Extract time components
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.day_name()
    df['month'] = df['timestamp'].dt.month
    
    # Fraud rate by hour
    plt.figure(figsize=(15, 5))
    hourly_fraud = df.groupby('hour')['is_fraudulent'].mean()
    hourly_fraud.plot(kind='line', marker='o')
    plt.title('Fraud Rate by Hour of Day')
    plt.xlabel('Hour')
    plt.ylabel('Fraud Rate')
    plt.grid(True)
    plt.show()
    
    # Fraud rate by day of week
    plt.figure(figsize=(12, 5))
    daily_fraud = df.groupby('day_of_week')['is_fraudulent'].mean()
    daily_fraud.plot(kind='bar')
    plt.title('Fraud Rate by Day of Week')
    plt.xlabel('Day of Week')
    plt.ylabel('Fraud Rate')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def analyze_categorical_features(df):
    """Analyze categorical features"""
    print("\n=== Categorical Feature Analysis ===")
    
    categorical_columns = [
        'merchant_category',
        'merchant_country',
        'card_type',
        'transaction_type'
    ]
    
    for column in categorical_columns:
        print(f"\nAnalyzing {column}:")
        
        # Value counts
        print("\nValue Counts:")
        print(df[column].value_counts().head())
        
        # Fraud rate by category
        fraud_rate = df.groupby(column)['is_fraudulent'].mean().sort_values(ascending=False)
        
        plt.figure(figsize=(12, 5))
        fraud_rate.plot(kind='bar')
        plt.title(f'Fraud Rate by {column}')
        plt.xlabel(column)
        plt.ylabel('Fraud Rate')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        print(f"\nTop 5 {column} by Fraud Rate:")
        print(fraud_rate.head())

def analyze_correlations(df):
    """Analyze feature correlations"""
    print("\n=== Feature Correlation Analysis ===")
    
    # Transform categorical variables
    transformer = TransactionTransformer()
    df_transformed = transformer.fit_transform(df, save_local=False)
    
    # Calculate correlations
    corr_matrix = df_transformed.corr()
    
    # Plot correlation matrix
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Feature Correlations')
    plt.tight_layout()
    plt.show()

def save_analysis_results(df):
    """Save analysis results to a report file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = Path("reports") / f"eda_report_{timestamp}.txt"
    report_file.parent.mkdir(exist_ok=True)
    
    with open(report_file, 'w') as f:
        f.write("=== Fraud Detection EDA Report ===\n\n")
        
        # Basic stats
        f.write("Dataset Shape: {}\n".format(df.shape))
        f.write("\nFraud Rate: {:.2f}%\n".format(df['is_fraudulent'].mean() * 100))
        
        # Amount statistics
        f.write("\nAmount Statistics:\n")
        f.write(df['amount'].describe().to_string())
        
        # Categorical feature statistics
        for col in ['merchant_category', 'merchant_country', 'card_type', 'transaction_type']:
            f.write(f"\n\nTop 5 {col}:\n")
            f.write(df[col].value_counts().head().to_string())
    
    print(f"\nAnalysis report saved to: {report_file}")

def main():
    # Load data
    df = load_sample_data()
    
    # Run analyses
    analyze_basic_stats(df)
    analyze_fraud_distribution(df)
    analyze_amounts(df)
    analyze_temporal_patterns(df)
    analyze_categorical_features(df)
    analyze_correlations(df)
    
    # Save results
    save_analysis_results(df)

if __name__ == "__main__":
    main() 