import pandas as pd
import numpy as np
import logging
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans

# 1. ENHANCED LOGGING (Code Best Practice)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extracts Hour, Day, Month, Year from TransactionStartTime."""
    def fit(self, X, y=None): return self
    def transform(self, X):
        try: # DEFENSIVE ERROR HANDLING
            X = X.copy()
            X['TransactionStartTime'] = pd.to_datetime(X['TransactionStartTime'])
            X['Hour'] = X['TransactionStartTime'].dt.hour
            X['Day'] = X['TransactionStartTime'].dt.day
            X['Month'] = X['TransactionStartTime'].dt.month
            X['Year'] = X['TransactionStartTime'].dt.year
            return X
        except Exception as e:
            logger.error(f"Error in DateFeatureExtractor: {e}")
            raise

class CustomerAggregator(BaseEstimator, TransformerMixin):
    """Creates Aggregate Features per Customer."""
    def fit(self, X, y=None): return self
    def transform(self, X):
        try: # DEFENSIVE ERROR HANDLING
            X = X.copy()
            if 'Value' not in X.columns or 'CustomerId' not in X.columns:
                logger.warning("Required columns missing for aggregation. Skipping.")
                return X
            customer_stats = X.groupby('CustomerId')['Value'].agg(['sum', 'mean', 'count', 'std']).fillna(0)
            customer_stats.columns = ['Total_Value', 'Avg_Value', 'Transaction_Count', 'Std_Value']
            return X.merge(customer_stats, on='CustomerId', how='left')
        except Exception as e:
            logger.error(f"Error in CustomerAggregator: {e}")
            raise

class KMeansRiskLabeler:
    """Task 4: Identify high-risk customers using K-Means."""
    def __init__(self, n_clusters=3, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        self.scaler = StandardScaler()

    def fit_predict_risk(self, df):
        try: # DEFENSIVE ERROR HANDLING
            snapshot_date = df['TransactionStartTime'].max() + pd.Timedelta(days=1)
            rfm = df.groupby('CustomerId').agg({
                'TransactionStartTime': lambda x: (snapshot_date - x.max()).days,
                'TransactionId': 'count',
                'Value': 'sum'
            }).rename(columns={'TransactionStartTime':