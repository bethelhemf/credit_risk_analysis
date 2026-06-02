import pandas as pd
import numpy as np
import logging
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extracts Hour, Day, Month, Year from TransactionStartTime."""
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        X['TransactionStartTime'] = pd.to_datetime(X['TransactionStartTime'])
        X['Hour'] = X['TransactionStartTime'].dt.hour
        X['Day'] = X['TransactionStartTime'].dt.day
        X['Month'] = X['TransactionStartTime'].dt.month
        X['Year'] = X['TransactionStartTime'].dt.year
        return X.drop(columns=['TransactionStartTime'])

class CustomerAggregator(BaseEstimator, TransformerMixin):
    """Creates Aggregate Features per Customer."""
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Note: In a production pipeline, aggregation usually happens 
        # before the final X/y split, but here we calculate them per row 
        # based on the provided transactional logic.
        X = X.copy()
        # Defensive: check if necessary columns exist
        if 'Value' not in X.columns:
            logger.error("Value column missing for aggregation")
            return X
            
        customer_stats = X.groupby('CustomerId')['Value'].agg(['sum', 'mean', 'count', 'std']).fillna(0)
        customer_stats.columns = ['Total_Value', 'Avg_Value', 'Transaction_Count', 'Std_Value']
        
        return X.merge(customer_stats, on='CustomerId', how='left')

class WoETransformer(BaseEstimator, TransformerMixin):
    """Simple Weight of Evidence Transformer for Basel II compliance."""
    def __init__(self, columns=None):
        self.columns = columns
        self.woe_maps = {}

    def fit(self, X, y):
        # y is the Proxy Target we defined
        if y is None:
            return self
        X_temp = X.copy()
        X_temp['target'] = y
        for col in self.columns:
            # Binning continuous variables
            X_temp['bin'] = pd.qcut(X_temp[col], 5, duplicates='drop')
            woe = X_temp.groupby('bin')['target'].agg(
                lambda s: np.log((s.count() - s.sum() + 0.5) / (s.sum() + 0.5))
            )
            self.woe_maps[col] = woe
        return self

    def transform(self, X):
        X = X.copy()
        for col, woe_map in self.woe_maps.items():
            bins = pd.qcut(X[col], 5, duplicates='drop')
            X[f'{col}_WoE'] = bins.map(woe_map).fillna(0)
        return X

def build_feature_pipeline(numerical_cols, categorical_cols):
    """
    The Single Fitted Pipeline Object.
    Chains Imputation, Encoding, and Scaling.
    """
    
    # 1. Numerical Pipeline: Impute then Scale
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # 2. Categorical Pipeline: Impute then One-Hot
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # 3. Combine using ColumnTransformer
    preprocessor = ColumnTransformer([
        ('num', num_pipeline, numerical_cols),
        ('cat', cat_pipeline, categorical_cols)
    ])
    
    # 4. Final Full Pipeline
    full_pipeline = Pipeline([
        ('date_extractor', DateFeatureExtractor()),
        ('aggregator', CustomerAggregator()),
        ('preprocessor', preprocessor)
    ])
    
    return full_pipeline

# Example Usage logic
if __name__ == "__main__":
    # Load your data
    df = pd.read_csv('data/raw/data.csv')
    
    # Define features for the pipeline
    num_features = ['Value', 'PricingStrategy']
    cat_features = ['ProductCategory', 'ChannelId']
    
    # Instantiate and fit
    pipeline = build_feature_pipeline(num_features, cat_features)
    # Note: Target 'y' creation happens here using the RFM logic discussed earlier
    logger.info("Pipeline built successfully.")