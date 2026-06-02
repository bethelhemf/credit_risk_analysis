import pandas as pd
import numpy as np
import logging
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extracts Hour, Day, Month, Year from TransactionStartTime."""
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = X.copy()
        X['TransactionStartTime'] = pd.to_datetime(X['TransactionStartTime'])
        X['Hour'] = X['TransactionStartTime'].dt.hour
        X['Day'] = X['TransactionStartTime'].dt.day
        X['Month'] = X['TransactionStartTime'].dt.month
        X['Year'] = X['TransactionStartTime'].dt.year
        return X

class CustomerAggregator(BaseEstimator, TransformerMixin):
    """Creates Aggregate Features per Customer."""
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = X.copy()
        if 'Value' not in X.columns:
            return X
        customer_stats = X.groupby('CustomerId')['Value'].agg(['sum', 'mean', 'count', 'std']).fillna(0)
        customer_stats.columns = ['Total_Value', 'Avg_Value', 'Transaction_Count', 'Std_Value']
        return X.merge(customer_stats, on='CustomerId', how='left')

class KMeansRiskLabeler:
    """Task 4: Identify high-risk customers using K-Means."""
    def __init__(self, n_clusters=3, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        self.scaler = StandardScaler()

    def fit_predict_risk(self, df):
        # Calculate RFM
        snapshot_date = df['TransactionStartTime'].max() + pd.Timedelta(days=1)
        rfm = df.groupby('CustomerId').agg({
            'TransactionStartTime': lambda x: (snapshot_date - x.max()).days,
            'TransactionId': 'count',
            'Value': 'sum'
        }).rename(columns={'TransactionStartTime': 'Recency', 'TransactionId': 'Frequency', 'Value': 'Monetary'})
        
        # Scale and Cluster
        rfm_scaled = self.scaler.fit_transform(rfm)
        rfm['is_high_risk'] = self.kmeans.fit_predict(rfm_scaled)
        
        # Binary target: 1 for highest risk cluster (lowest monetary/freq)
        cluster_map = rfm.groupby('is_high_risk')['Monetary'].mean().idxmin()
        rfm['is_high_risk'] = (rfm['is_high_risk'] == cluster_map).astype(int)
        
        return rfm[['is_high_risk']]

def build_feature_pipeline(numerical_cols, categorical_cols):
    """Task 3: Chains Imputation, Encoding, and Scaling."""
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer([
        ('num', num_pipeline, numerical_cols),
        ('cat', cat_pipeline, categorical_cols)
    ])
    return Pipeline([
        ('date_extractor', DateFeatureExtractor()),
        ('aggregator', CustomerAggregator()),
        ('preprocessor', preprocessor)
    ])
