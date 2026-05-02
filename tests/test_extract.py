import pandas as pd 
from unittest.mock import patch, MagicMock
from ingestion.extract import fetch_daily_stock


MOCK_RESPONSE = {
    "Time Series (Daily)": {
        "2024-01-01": {
            "1. open": "150.0", "2. high": "155.0",
            "3. low": "149.0", "4. close": "153.0", "5. volume": "1000000"
        }
    }
}

@patch("ingestion.extract.requests.get")
def test_fetch_returns_dataframe(mock_get):
    mock_get.return_value = MagicMock(
        status_code = 200,
        json = lambda: MOCK_RESPONSE
    )

    df = fetch_daily_stock("AAPL")
    assert isinstance(df , pd.DataFrame)
    assert "close" in df.columns
    assert df['symbol'].iloc[0] == "AAPL"