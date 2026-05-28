import os 
import pandas as pd 
from google.cloud import bigquery
from dotenv import load_dotenv

load_dotenv()

PROJECT_ID =os.getenv("GCP_PROJECT_ID")
DATASET    = os.getenv("BQ_DATASET")
TABLE      = os.getenv("BQ_TABLE")

def fetch_stock_data()-> pd.DataFrame:
    client = bigquery.Client(project =PROJECT_ID)

    query = f"""
        SELECT date, open , high, low, close, volume, symbol
        FROM `{PROJECT_ID}.{DATASET}.{TABLE}`
        ORDER BY symbol, date  ASC
    """

    df = client.query(query).to_dataframe()

    df["date"]  = pd.to_datetime(df["date"])
    df["close"] = df["close"].astype(float)
    df["open"]  = df["open"].astype(float)
    df["high"]  = df["high"].astype(float)
    df["low"]   = df["low"].astype(float)
    df["volume"]= df["volume"].astype(float)

    df = df.drop_duplicates(subset=["symbol" , "date"] , keep="last")
    df = df.sort_values(["symbol" , "date"]).reset_index(drop=True)

    print(f"Fetched {len(df)} rows - symbol: {df['symbol'].unique()}")
    return df 

if __name__ == "__main__":
    df = fetch_stock_data()
    print(df.head(10))
    print(df.shape)
    