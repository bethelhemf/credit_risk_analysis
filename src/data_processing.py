import pandas as pd
import numpy as np
import logging
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans

# 1. Setup Logging (Best Practice: Defensive Programming)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- TASK 3 MODULES ---

class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    """Requirement: Extract Hour, Day, Month, Year."""
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = X.copy()
        X['TransactionStartTime'] = pd.to_datetime(X['TransactionStartTime'])
        X['Hour'] = X['TransactionStartTime'].dt.hour
        X['Day'] = X['TransactionStartTime'].dt.day
        X['Month'] = X['TransactionStartTime'].dt.month
        X['Year'] = X['TransactionStartTime'].dt.year
        return X

class AggregatedFeatureGenerator(BaseEstimator, TransformerMixin):
    """Requirement: Total, Average, Count, and Std per Customer."""
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = X.copy()
        # Group by Customer to get behavioral aggregates
        agg_features = X.groupby('CustomerId')['Value'].agg(['sum', 'mean', 'count', 'std']).fillna(0)
        agg_features.columns = ['Total_Value', 'Avg_Value', 'Transaction_Count', 'Std_Value']
        # Merge back to the original dataframe
        return X.merge(agg_features, on='CustomerId', how='left')

# --- TASK 4 MODULE ---

class KMeansRiskLabeler:
    """Requirement: K-Means clustering (k=3) for High-Risk Proxy."""
    def __init__(self, n_clusters=3, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        self.scaler = StandardScaler()

    def fit_predict_risk(self, df):
        logger.info("Generating RFM for Task 4 Proxy Labels...")
        snapshot_date = df['TransactionStartTime'].max() + pd.Timedelta(days=1)
        
        # Calculate RFM
        rfm = df.groupby('CustomerId').agg({
            'TransactionStartTime': lambda x: (snapshot_date - x.max()).days,
            'TransactionId': 'count',
            'Value': 'sum'
        }).rename(columns={'TransactionStartTime': 'Recency', 'TransactionId': 'Frequency', 'Value': 'Monetary'})
        
        # Scale for Clustering
        rfm_scaled = self.scaler.fit_transform(rfm)
        
        # Cluster
        rfm['Cluster'] = self.kmeans.fit_predict(rfm_scaled)
        
        # Identify High-Risk (Cluster with lowest average Monetary/Frequency)
        risk_map = rfm.groupby('Cluster')[['Monetary', 'Frequency']].mean().sum(axis=1)
        high_risk_cluster = risk_map.idxmin()
        
        rfm['is_high_risk'] = (rfm['Cluster'] == high_risk_cluster).astype(int)
        logger.info(f"High-risk identified as Cluster {high_risk_cluster}")
        return rfm[['is_high_risk']]

# --- WOE LOGIC (Basel II Compliance) ---

def calculate_woe_iv(df, feature, target):
    """Requirement: Implement Weight of Evidence & Information Value."""
    df = df.copy()
    df['bin'] = pd.qcut(df[feature], 5, duplicates='drop')
    
    event_total = df[target].sum()
    non_event_total = df[target].count() - event_total
    
    woe_iv = df.groupby('bin')[target].agg(['count', 'sum'])
    woe_iv.columns = ['Total', 'Bad']
    woe_iv['Good'] = woe_iv['Total'] - woe_iv['Bad']
    
    # WoE formula with epsilon to avoid division by zero
    eps = 0.5 
    woe_iv['WoE'] = np.log(((woe_iv['Good'] + eps) / non_event_total) / 
                           ((woe_iv['Bad'] + eps) / event_total))
    woe_iv['IV'] = (((woe_iv['Good'] / non_event_total) - 
                     (woe_iv['Bad'] / event_total)) * woe_iv['WoE'])
    
    return woe_iv, woe_iv['IV'].sum()

# --- THE MAIN PIPELINE FUNCTION ---

def get_fitted_pipeline(df):
    """Requirement: Chain all transformations into a single Fitted Pipeline."""
    
    numerical_features = ['Value', 'Total_Value', 'Avg_Value', 'Std_Value', 'Hour']
    categorical_features = ['ProductCategory', 'ChannelId']

    # 1. Define Column Transformer (Handling Missing Values & Encoding)
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), numerical_features),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('ohe', OneHotEncoder(handle_unknown='ignore'))
            ]), categorical_features)
        ]
    )

    # 2. Build the Full Pipeline (Task 3)
    full_pipeline = Pipeline([
        ('date_extract', DateFeatureExtractor()),
        ('aggregator', AggregatedFeatureGenerator()),
        ('preprocessor', preprocessor)
    ])
    
    return full_pipeline

if __name__ == "__main__":
    try:
        # Load Raw Data
        df_raw = pd.read_csv('data/raw/data.csv')
        df_raw['TransactionStartTime'] = pd.to_datetime(df_raw['TransactionStartTime'])

        # Task 4: Generate Proxy Label first
        labeler = KMeansRiskLabeler()
        risk_labels = labeler.fit_predict_risk(df_raw)
        
        # Merge Labels
        df_ready = df_raw.merge(risk_labels, on='CustomerId', how='left')
        
        # Task 3: Run the Pipeline
        pipeline = get_fitted_pipeline(df_ready)
        processed_features = pipeline.fit_transform(df_ready)
        
        # Export for Task 5
        logger.info("Feature engineering and Labeling complete.")
        # Note: In Task 5 you will feed 'processed_features' and 'df_ready[is_high_risk]' into models
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")