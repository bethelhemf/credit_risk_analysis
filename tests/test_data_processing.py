import pytest
import pandas as pd
from src.data_processing import DateFeatureExtractor, CustomerAggregator

def test_date_extractor():
    """Test if hour extraction works."""
    df = pd.DataFrame({'TransactionStartTime': ['2018-11-15 02:18:49']})
    extractor = DateFeatureExtractor()
    result = extractor.transform(df)
    assert 'Hour' in result.columns
    assert result['Hour'].iloc[0] == 2

def test_customer_aggregator():
    """Test if total value per customer is correct."""
    df = pd.DataFrame({
        'CustomerId': ['C1', 'C1'],
        'Value': [1000, 2000]
    })
    aggregator = CustomerAggregator()
    result = aggregator.transform(df)
    assert 'Total_Value' in result.columns
    assert result['Total_Value'].iloc[0] == 3000