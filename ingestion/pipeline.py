from extract import fetch_daily_stock
from load import load_to_bigquery


SYMBOLS = ["AAPL" , "MSFT" , "GOOGL"]

def run():
    for symbol in SYMBOLS:
        print(f"Extracting {symbol}...")
        df = fetch_daily_stock(symbol)
        load_to_bigquery(df)
        print(f"Done: {symbol}")

if __name__ == "__main__":
    run()
