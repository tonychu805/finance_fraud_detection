"""
Exploratory Data Analysis (EDA) module for fraud detection.

This module implements comprehensive EDA as outlined in the documentation,
including univariate, bivariate, and time series analysis.
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.feature_selection import mutual_info_classif

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FraudEDA:
    """
    Handles exploratory data analysis for fraud detection.
    """
    
    def __init__(self, data_path: Path):
        """
        Initialize the EDA module.
        
        Args:
            data_path: Path to the raw dataset
        """
        self.data_path = data_path
        self.df = None
        self.numerical_features = [
            "amount", "oldbalanceOrg", "newbalanceOrig",
            "oldbalanceDest", "newbalanceDest"
        ]
        self.categorical_features = ["type"]
        self.high_cardinality_features = ["nameOrig", "nameDest"]
        self.binary_features = ["isFlaggedFraud"]
        self.time_features = ["step"]
        
    def load_data(self) -> None:
        """Load and perform initial data validation."""
        logger.info(f"Loading data from {self.data_path}")
        self.df = pd.read_csv(self.data_path)
        logger.info(f"Loaded {len(self.df)} transactions")
        
        # Basic data validation
        self._validate_data()
        
    def _validate_data(self) -> None:
        """Perform basic data validation checks."""
        # Check for missing values
        missing = self.df.isnull().sum()
        if missing.any():
            logger.warning(f"Missing values found:\n{missing[missing > 0]}")
        
        # Check data types
        logger.info("Data types:\n" + str(self.df.dtypes))
        
        # Check for duplicates
        duplicates = self.df.duplicated().sum()
        if duplicates > 0:
            logger.warning(f"Found {duplicates} duplicate transactions")
            
    def calculate_effect_size(self, p1: float, p2: float) -> float:
        """
        Calculate Cohen's h effect size for proportions.
        
        Args:
            p1: First proportion
            p2: Second proportion
            
        Returns:
            Cohen's h effect size
        """
        return 2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2))
    
    def calculate_information_value(self, feature: str, target: str) -> float:
        """
        Calculate Information Value (IV) for a feature.
        
        Args:
            feature: Feature name
            target: Target variable name
            
        Returns:
            Information Value
        """
        # For numerical features, create bins
        if feature in self.numerical_features:
            # Create 10 bins, handling duplicate values
            self.df[f'{feature}_bin'] = pd.qcut(self.df[feature], q=10, labels=False, duplicates='drop')
            feature = f'{feature}_bin'
        
        # Calculate WOE and IV
        woe = np.log((self.df[self.df[target] == 1][feature].value_counts() / self.df[self.df[target] == 1][feature].count()) /
                    (self.df[self.df[target] == 0][feature].value_counts() / self.df[self.df[target] == 0][feature].count()))
        iv = ((self.df[self.df[target] == 1][feature].value_counts() / self.df[self.df[target] == 1][feature].count()) -
              (self.df[self.df[target] == 0][feature].value_counts() / self.df[self.df[target] == 0][feature].count())) * woe
        
        return iv.sum()

    def calculate_statistical_significance(self, feature: str, target: str = 'isFraud') -> Dict:
        """
        Calculate statistical significance tests for a feature.
        
        Args:
            feature: Feature name
            target: Target variable name
            
        Returns:
            Dictionary containing statistical test results
        """
        results = {}
        
        if feature in self.numerical_features:
            # For numerical features, use t-test
            fraud_values = self.df[self.df[target] == 1][feature]
            legit_values = self.df[self.df[target] == 0][feature]
            
            # Perform t-test
            t_stat, p_value = stats.ttest_ind(fraud_values, legit_values, equal_var=False)
            
            # Calculate effect size (Cohen's d)
            n1, n2 = len(fraud_values), len(legit_values)
            var1, var2 = fraud_values.var(), legit_values.var()
            pooled_se = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
            cohens_d = (fraud_values.mean() - legit_values.mean()) / pooled_se
            
            results = {
                'test_type': 't-test',
                'statistic': t_stat,
                'p_value': p_value,
                'effect_size': cohens_d,
                'significant': p_value < 0.05
            }
            
        elif feature in self.categorical_features:
            # For categorical features, use chi-square test
            contingency_table = pd.crosstab(self.df[feature], self.df[target])
            chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
            
            # Calculate Cramer's V for effect size
            n = contingency_table.sum().sum()
            min_dim = min(contingency_table.shape) - 1
            cramers_v = np.sqrt(chi2 / (n * min_dim))
            
            results = {
                'test_type': 'chi-square',
                'statistic': chi2,
                'p_value': p_value,
                'effect_size': cramers_v,
                'significant': p_value < 0.05
            }
            
        elif feature in self.binary_features:
            # For binary features, use chi-square test
            contingency_table = pd.crosstab(self.df[feature], self.df[target])
            chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
            
            # Calculate phi coefficient for effect size
            n = contingency_table.sum().sum()
            phi = np.sqrt(chi2 / n)
            
            results = {
                'test_type': 'chi-square',
                'statistic': chi2,
                'p_value': p_value,
                'effect_size': phi,
                'significant': p_value < 0.05
            }
            
        return results
    
    def analyze_population_patterns(self) -> Dict:
        """
        Analyze patterns in the population data.
        
        Returns:
            Dictionary containing analysis results
        """
        results = {}
        
        # 1. Overall Statistics
        results['overall'] = {
            'total_transactions': len(self.df),
            'fraud_rate': self.df['isFraud'].mean(),
            'fraud_count': self.df['isFraud'].sum()
        }
        
        # 2. Transaction Type Analysis
        type_analysis = {}
        overall_fraud_rate = self.df['isFraud'].mean()
        for type_ in self.df['type'].unique():
            type_data = self.df[self.df['type'] == type_]
            fraud_rate = type_data['isFraud'].mean()
            risk_ratio = fraud_rate / overall_fraud_rate
            effect_size = self.calculate_effect_size(fraud_rate, overall_fraud_rate)
            
            # Add statistical significance
            stats_results = self.calculate_statistical_significance('type')
            
            type_analysis[type_] = {
                'fraud_rate': fraud_rate,
                'risk_ratio': risk_ratio,
                'effect_size': effect_size,
                'transaction_count': len(type_data),
                'fraud_count': type_data['isFraud'].sum(),
                'statistical_significance': stats_results
            }
        results['transaction_types'] = type_analysis
        
        # 3. Numerical Features Analysis
        for feature in self.numerical_features:
            fraud_values = self.df[self.df['isFraud'] == 1][feature]
            legit_values = self.df[self.df['isFraud'] == 0][feature]
            
            # Add statistical significance
            stats_results = self.calculate_statistical_significance(feature)
            
            results[feature] = {
                'fraud': {
                    'mean': fraud_values.mean(),
                    'median': fraud_values.median(),
                    'std': fraud_values.std(),
                    'q1': fraud_values.quantile(0.25),
                    'q3': fraud_values.quantile(0.75)
                },
                'legitimate': {
                    'mean': legit_values.mean(),
                    'median': legit_values.median(),
                    'std': legit_values.std(),
                    'q1': legit_values.quantile(0.25),
                    'q3': legit_values.quantile(0.75)
                },
                'differences': {
                    'mean_difference': fraud_values.mean() - legit_values.mean(),
                    'variance_ratio': fraud_values.var() / legit_values.var(),
                    'information_value': self.calculate_information_value(feature, 'isFraud')
                },
                'statistical_significance': stats_results
            }
            
            # Create distribution plot
            plt.figure(figsize=(12, 6))
            # Convert isFraud to string category for better plotting
            plot_df = self.df.copy()
            plot_df['isFraud'] = plot_df['isFraud'].map({0: 'Legitimate', 1: 'Fraud'})
            sns.boxplot(data=plot_df, x='isFraud', y=feature)
            plt.title(f'Distribution of {feature} by Fraud Status')
            plt.savefig(f'data/processed/eda/{feature}_fraud_distribution.png')
            plt.close()
        
        # 4. High Cardinality Features Analysis
        for feature in self.high_cardinality_features:
            # Analyze top 10 most frequent values
            top_10_values = self.df[feature].value_counts().head(10).index
            value_analysis = {}
            
            for value in top_10_values:
                value_data = self.df[self.df[feature] == value]
                fraud_rate = value_data['isFraud'].mean()
                risk_ratio = fraud_rate / overall_fraud_rate
                effect_size = self.calculate_effect_size(fraud_rate, overall_fraud_rate)
                
                # Add statistical significance
                stats_results = self.calculate_statistical_significance(feature)
                
                value_analysis[value] = {
                    'fraud_rate': fraud_rate,
                    'risk_ratio': risk_ratio,
                    'effect_size': effect_size,
                    'transaction_count': len(value_data),
                    'fraud_count': value_data['isFraud'].sum(),
                    'statistical_significance': stats_results
                }
            
            results[feature] = value_analysis
        
        # 5. Binary Features Analysis
        for feature in self.binary_features:
            value_analysis = {}
            
            # Add statistical significance
            stats_results = self.calculate_statistical_significance(feature)
            
            for value in self.df[feature].unique():
                value_data = self.df[self.df[feature] == value]
                fraud_rate = value_data['isFraud'].mean()
                risk_ratio = fraud_rate / overall_fraud_rate
                effect_size = self.calculate_effect_size(fraud_rate, overall_fraud_rate)
                
                value_analysis[value] = {
                    'fraud_rate': fraud_rate,
                    'risk_ratio': risk_ratio,
                    'effect_size': effect_size,
                    'transaction_count': len(value_data),
                    'fraud_count': value_data['isFraud'].sum(),
                    'statistical_significance': stats_results
                }
            
            results[feature] = value_analysis
        
        return results
    
    def time_series_analysis(self) -> Dict:
        """
        Perform time series analysis on transaction patterns.
        
        Returns:
            Dictionary containing analysis results
        """
        results = {}
        
        # Convert step to datetime-like features
        self.df["hour"] = self.df["step"] % 24
        self.df["day"] = self.df["step"] // 24
        
        # Transaction frequency over time
        hourly_counts = self.df.groupby("hour").size()
        daily_counts = self.df.groupby("day").size()
        
        results["hourly_counts"] = hourly_counts.to_dict()
        results["daily_counts"] = daily_counts.to_dict()
        
        # Fraud rates over time
        hourly_fraud = self.df.groupby("hour")["isFraud"].mean()
        daily_fraud = self.df.groupby("day")["isFraud"].mean()
        
        results["hourly_fraud_rates"] = hourly_fraud.to_dict()
        results["daily_fraud_rates"] = daily_fraud.to_dict()
        
        # Create time series plots
        plt.figure(figsize=(12, 6))
        hourly_fraud.plot(kind="line")
        plt.title("Fraud Rates by Hour")
        plt.savefig("data/processed/eda/hourly_fraud_rates.png")
        plt.close()
        
        plt.figure(figsize=(12, 6))
        daily_fraud.plot(kind="line")
        plt.title("Fraud Rates by Day")
        plt.savefig("data/processed/eda/daily_fraud_rates.png")
        plt.close()
        
        return results
    
    def analyze_false_negatives(self) -> Dict:
        """
        Analyze false negatives in fraud detection.
        
        Returns:
            Dictionary containing false negative analysis results
        """
        results = {}
        
        # 1. System Flag Analysis (isFlaggedFraud vs isFraud)
        confusion = pd.crosstab(self.df['isFlaggedFraud'], self.df['isFraud'])
        total_fraud = self.df['isFraud'].sum()
        
        # False negatives are fraud cases (isFraud=1) that weren't flagged (isFlaggedFraud=0)
        false_negatives = confusion.loc[0, 1] if 1 in confusion.columns else 0
        false_negative_rate = false_negatives / total_fraud if total_fraud > 0 else 0
        
        results['system_flags'] = {
            'total_fraud_cases': int(total_fraud),
            'false_negatives': int(false_negatives),
            'false_negative_rate': float(false_negative_rate),
            'missed_fraud_amount': float(self.df[(self.df['isFraud'] == 1) & 
                                               (self.df['isFlaggedFraud'] == 0)]['amount'].sum())
        }
        
        # 2. Analysis by Transaction Type
        type_analysis = {}
        for type_ in self.df['type'].unique():
            type_data = self.df[self.df['type'] == type_]
            type_fraud = type_data['isFraud'].sum()
            type_false_neg = len(type_data[(type_data['isFraud'] == 1) & 
                                         (type_data['isFlaggedFraud'] == 0)])
            
            type_analysis[type_] = {
                'total_fraud': int(type_fraud),
                'false_negatives': int(type_false_neg),
                'false_negative_rate': float(type_false_neg / type_fraud) if type_fraud > 0 else 0,
                'missed_fraud_amount': float(type_data[(type_data['isFraud'] == 1) & 
                                                     (type_data['isFlaggedFraud'] == 0)]['amount'].sum())
            }
        
        results['by_type'] = type_analysis
        
        # 3. Amount Range Analysis for False Negatives
        false_neg_data = self.df[(self.df['isFraud'] == 1) & (self.df['isFlaggedFraud'] == 0)]
        
        results['amount_analysis'] = {
            'min': float(false_neg_data['amount'].min()) if not false_neg_data.empty else 0,
            'max': float(false_neg_data['amount'].max()) if not false_neg_data.empty else 0,
            'mean': float(false_neg_data['amount'].mean()) if not false_neg_data.empty else 0,
            'median': float(false_neg_data['amount'].median()) if not false_neg_data.empty else 0,
            'total': float(false_neg_data['amount'].sum()) if not false_neg_data.empty else 0
        }
        
        return results
    
    def _convert_to_native_types(self, obj):
        """Convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, dict):
            # Convert both keys and values
            return {str(key) if isinstance(key, (np.integer, np.floating)) else key: 
                   self._convert_to_native_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_native_types(item) for item in obj]
        return obj

    def run_full_analysis(self) -> Dict:
        """
        Run all EDA analyses and save results.
        
        Returns:
            Dictionary containing all analysis results
        """
        logger.info("Starting comprehensive EDA")
        
        # Run all analyses
        results = {
            "population_analysis": self.analyze_population_patterns(),
            "time_series_analysis": self.time_series_analysis(),
            "false_negative_analysis": self.analyze_false_negatives()
        }
        
        # Convert numpy types to native Python types
        results = self._convert_to_native_types(results)
        
        # Save results
        output_dir = Path("data/interim/eda")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / "eda_results.json", "w") as f:
            json.dump(results, f, indent=2)
            
        logger.info("EDA results saved to data/interim/eda/eda_results.json")
        return results

if __name__ == "__main__":
    # Example usage
    eda = FraudEDA(Path("data/raw/dataset.csv"))
    eda.load_data()
    results = eda.run_full_analysis()
    
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
    for feature in eda.numerical_features:
        feature_stats = results["population_analysis"][feature]["differences"]
        print(f"   - {feature}:")
        print(f"     Information Value: {feature_stats['information_value']:.3f}")
        print(f"     Mean Difference: {feature_stats['mean_difference']:.2f}")
        print(f"     Variance Ratio: {feature_stats['variance_ratio']:.2f}") 