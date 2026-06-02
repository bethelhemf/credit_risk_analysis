import pandas as pd
import numpy as np
import logging

# Set up logging for defensive monitoring
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FeatureEngineer:
    def __init__(self, df: pd.DataFrame):
        """
        Initializes the engineer with a dataframe.
        Includes basic error handling to ensure data is not empty.
        """
        if df.empty:
            logger.error("The input DataFrame is empty.")
            raise ValueError("Input DataFrame cannot be empty.")
        self.df = df

    def preprocess_data(self):
        """
        Cleans timestamps and handles data types.
        """
        try:
            self.df['TransactionStartTime'] = pd.to_datetime(self.df['TransactionStartTime'])
            logger.info("Timestamps converted successfully.")
            return self.df
        except Exception as e:
            logger.error(f"Error in preprocessing: {e}")
            raise

    def create_rfm_features(self):
        """
        Calculates Recency, Frequency, and Monetary features.
        Modular design: separated from labeling logic.
        """
        try:
            snapshot_date = self.df['TransactionStartTime'].max() + pd.Timedelta(days=1)
            rfm = self.df.groupby('CustomerId').agg({
                'TransactionStartTime': lambda x: (snapshot_date - x.max()).days,
                'TransactionId': 'count',
                'Value': 'sum'
            }).rename(columns={
                'TransactionStartTime': 'Recency',
                'TransactionId': 'Frequency',
                'Value': 'Monetary'
            })
            logger.info("RFM features calculated.")
            return rfm
        except KeyError as e:
            logger.error(f"Missing required column for RFM: {e}")
            raise

    def assign_proxy_target(self, rfm_df: pd.DataFrame):
        """
        Defines the Proxy Label (Target) for credit risk.
        Basel II compliance: Documentation of risk thresholds.
        """
        # Defensive check: ensure RFM columns exist
        required = ['Recency', 'Frequency', 'Monetary']
        if not all(col in rfm_df.columns for col in required):
            raise KeyError(f"RFM dataframe must contain {required}")

        # Labeling logic: Top 20% risk score as 'Bad' (1)
        rfm_df['R_Score'] = pd.qcut(rfm_df['Recency'], 4, labels=[1, 2, 3, 4]).astype(int)
        rfm_df['F_Score'] = pd.qcut(rfm_df['Frequency'].rank(method='first'), 4, labels=[4, 3, 2, 1]).astype(int)
        rfm_df['M_Score'] = pd.qcut(rfm_df['Monetary'], 4, labels=[4, 3, 2, 1]).astype(int)
        
        rfm_df['Risk_Score'] = rfm_df['R_Score'] + rfm_df['F_Score'] + rfm_df['M_Score']
        
        # High risk threshold
        threshold = rfm_df['Risk_Score'].quantile(0.8)
        rfm_df['Target'] = (rfm_df['Risk_Score'] >= threshold).astype(int)
        
        logger.info(f"Target variable created. Default rate: {rfm_df['Target'].mean():.2%}")
        return rfm_df

def calculate_woe_iv(df, feature, target):
    """
    Calculates Weight of Evidence (WoE) and Information Value (IV).
    Includes defensive handling for division by zero.
    """
    df['bin'] = pd.qcut(df[feature], 5, duplicates='drop')
    woe_df = df.groupby('bin').agg({target: ['count', 'sum']})
    woe_df.columns = ['Total', 'Bad']
    woe_df['Good'] = woe_df['Total'] - woe_df['Bad']
    
    # Defensive: Avoid division by zero by adding a small epsilon
    eps = 1e-9
    dist_good = (woe_df['Good'] / woe_df['Good'].sum()) + eps
    dist_bad = (woe_df['Bad'] / woe_df['Bad'].sum()) + eps
    
    woe_df['WoE'] = np.log(dist_good / dist_bad)
    woe_df['IV'] = (dist_good - dist_bad) * woe_df['WoE']
    
    return woe_df, woe_df['IV'].sum()