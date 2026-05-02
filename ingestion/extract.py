import requests
import pandas as pd 
from dotenv import load_dotenv
import os 

load_dotenv()

API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
BASE_URL = "https://www.alphavantage.co/query"


def fetch_daily_stock(symbol : str) -> pd.DataFrame:
    params = {
        "function" : "TIME_SERIES_DAILY",
        "symbol": symbol,
        "apikey": API_KEY,
        "outputsize":"compact"   
    }

    response = requests.get(BASE_URL , params= params)
    response.raise_for_status()

    data = response.json()

    time_series = data.get("Time Series (Daily)", {})

    if not time_series:
        print(f"Raw API response for {symbol}: {data}")
        raise ValueError(f"No data returned for {symbol}. Check your API key or symbol.")
        
    df = pd.DataFrame.from_dict(time_series , orient='index')
    df.index.name = 'date'
    df.reset_index(inplace = True)

    df.columns = ["date" , "open" , "high" , "low" , "close" , "volume"]
    df['symbol'] = symbol
    df['date'] = pd.to_datetime(df['date'])

    return df 

if __name__ == "__main__":
    df = fetch_daily_stock("AAPL")
    print(df.head())
    print(f"\nShape: {df.head}")



