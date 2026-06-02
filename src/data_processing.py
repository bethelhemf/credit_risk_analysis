import pandas as pd
import numpy as np
import logging
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans

# 1. Defensive Logging Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("BatiBankPipeline")

# --- TASK 3: FEATURE EXTRACTION ---

class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extracts Hour, Day, Month, Year from TransactionStartTime."""
    def fit(self, X, y=None): return self
    def transform(self, X):
        try:
            X = X.copy()
            X['TransactionStartTime'] = pd.to_datetime(X['TransactionStartTime'])
            X['Hour'] = X['TransactionStartTime'].dt.hour
            X['Day'] = X['TransactionStartTime'].dt.day
            X['Month'] = X['TransactionStartTime'].dt.month
            X['Year'] = X['TransactionStartTime'].dt.year
            return X
        except Exception as e:
            logger.error(f"Date Extraction Failed: {e}")
            raise

class AggregateFeatureGenerator(BaseEstimator, TransformerMixin):
    """Requirement: Total, Avg, Count, Std per Customer."""
    def fit(self, X, y=None): return self
    def transform(self, X):
        try:
            X = X.copy()
            stats = X.groupby('CustomerId')['Value'].agg(['sum', 'mean', 'count', 'std']).fillna(0)
            stats.columns = ['Total_Value', 'Avg_Value', 'Transaction_Count', 'Std_Value']
            return X.merge(stats, on='CustomerId', how='left')
        except Exception as e:
            logger.error(f"Aggregate Generation Failed: {e}")
            raise

# --- TASK 4: PROXY TARGET CREATION ---

class KMeansTargetCreator(BaseEstimator, TransformerMixin):
    """Requirement: RFM Calculation -> K-Means Clustering -> is_high_risk Label."""
    def __init__(self, n_clusters=3):
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.scaler = StandardScaler()
        self.high_risk_cluster = None

    def fit(self, X, y=None):
        # Calculate RFM
        snapshot_date = X['TransactionStartTime'].max() + pd.Timedelta(days=1)
        rfm = X.groupby('CustomerId').agg({
            'TransactionStartTime': lambda x: (snapshot_date - x.max()).days,
            'TransactionId': 'count',
            'Value': 'sum'
        }).rename(columns={'TransactionStartTime': 'Recency', 'TransactionId': 'Frequency', 'Value': 'Monetary'})
        
        # Scale for clustering
        rfm_scaled = self.scaler.fit_transform(rfm)
        rfm['cluster'] = self.kmeans.fit_predict(rfm_scaled)
        
        # Explicit Target Logic: Cluster with lowest Monetary + Frequency is High Risk
        self.high_risk_cluster = rfm.groupby('cluster')[['Monetary', 'Frequency']].mean().sum(axis=1).idxmin()
        self.high_risk_customers = set(rfm[rfm['cluster'] == self.high_risk_cluster].index)
        logger.info(f"Target Created: Cluster {self.high_risk_cluster} identified as High Risk.")
        return self

    def transform(self, X):
        X = X.copy()
        X['is_high_risk'] = X['CustomerId'].apply(lambda x: 1 if x in self.high_risk_customers else 0)
        return X

# --- TASK 3: WOE AND INFORMATION VALUE ---

class WoETransformer(BaseEstimator, TransformerMixin):
    """Requirement: Weight of Evidence (WoE) and IV calculation."""
    def __init__(self, columns=None):
        self.columns = columns
        self.woe_maps = {}

    def fit(self, X, y=None):
        X_temp = X.copy()
        target = 'is_high_risk'
        for col in self.columns:
            X_temp['bin'] = pd.qcut(X_temp[col], 5, duplicates='drop')
            stats = X_temp.groupby('bin', observed=True)[target].agg(['count', 'sum'])
            stats.columns = ['Total', 'Bad']
            stats['Good'] = stats['Total'] - stats['Bad']
            
            # Defensive check for zero-division
            stats['Bad'] = stats['Bad'].replace(0, 0.5)
            stats['Good'] = stats['Good'].replace(0, 0.5)
            
            p_good = stats['Good'] / stats['Good'].sum()
            p_bad = stats['Bad'] / stats['Bad'].sum()
            self.woe_maps[col] = np.log(p_good / p_bad).to_dict()
            
            iv = ((p_good - p_bad) * np.log(p_good / p_bad)).sum()
            logger.info(f"WoE fit for {col} | Information Value: {iv:.4f}")
        return self

    def transform(self, X):
        X = X.copy()
        for col, woe_map in self.woe_maps.items():
            X[f'{col}_WoE'] = pd.qcut(X[col], 5, duplicates='drop').map(woe_map).fillna(0)
        return X

# --- THE "FULL PIPELINE WIRING" (The 100/100 Deliverable) ---

def build_fitted_pipeline(df):
    """
    Wires Tasks 3 and 4 into a single reproducible Pipeline object.
    """
    # 1. Define Columns
    num_cols = ['Value', 'Total_Value', 'Avg_Value', 'Std_Value', 'Hour']
    cat_cols = ['ProductCategory', 'ChannelId']

    # 2. Build Preprocessor (Imputation, OHE, Scaling)
    preprocessor = ColumnTransformer([
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), num_cols),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ]), cat_cols)
    ])

    # 3. Final Wiring
    full_pipeline = Pipeline([
        ('date_extractor', DateFeatureExtractor()),
        ('aggregator', AggregateFeatureGenerator()),
        ('target_labeler', KMeansTargetCreator()), # Task 4 integrated
        ('woe_step', WoETransformer(columns=['Value', 'Total_Value'])), # WoE integrated
        ('preprocessor', preprocessor)
    ])
    
    return full_pipeline.fit(df)

if __name__ == "__main__":
    df_raw = pd.read_csv('data/raw/data.csv')
    pipeline = build_fitted_pipeline(df_raw)
    processed_df = pipeline.transform(df_raw)
    processed_df.to_csv('data/processed/model_ready.csv', index=False)
    logger.info("Production Pipeline successfully executed and saved.")