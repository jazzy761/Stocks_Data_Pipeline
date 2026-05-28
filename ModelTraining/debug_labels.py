from fetch import fetch_stock_data
import pandas as pd

df = fetch_stock_data()

for symbol in df["symbol"].unique():
    sdf = df[df["symbol"] == symbol].copy()
    sdf = sdf.sort_values("date").reset_index(drop=True)
    
    print(f"\n--- {symbol} ---")
    print(f"Rows: {len(sdf)}")
    print(f"Date range: {sdf['date'].min()} → {sdf['date'].max()}")
    print(f"Close sample (first 5): {sdf['close'].values[:5]}")
    
    sdf["label"] = (sdf["close"].shift(-1) > sdf["close"]).astype(int)
    sdf = sdf.dropna(subset=["label"])
    
    print(f"Label counts: {sdf['label'].value_counts().to_dict()}")
    print(f"Label balance: {sdf['label'].mean():.2f}")
    print(f"Sample closes:\n{sdf[['date','close','label']].head(10).to_string()}")