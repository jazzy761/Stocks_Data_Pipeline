import numpy as np
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler
import os 

#features.py

WINDOW = 20

def add_technical_indicators(df:pd.DataFrame) -> pd.DataFrame:
    df = df.copy() 

    # trend
    df["sma_5"]      = df["close"].rolling(5).mean()
    df["sma_20"]     = df["close"].rolling(20).mean()
    df["ema_12"]     = df["close"].ewm(span=12).mean()
    df["ema_26"]     = df["close"].ewm(span=26).mean()

    # momentum
    df["macd"]      = df["ema_12"] - df["ema_26"]
    df["roc_5"]     = df["close"].pct_change(5) # rate of change
    df["momentum"]  = df["close"] - df["close"].shift(5)

    # Volatility
    df["daily_return"]   = df["close"].pct_change()
    df["volatility_10"]  = df["daily_return"].rolling(10).std()
    df["high_low_range"] = df["high"] - df["low"]

    # RSI
    delta       = df["close"].diff()
    gain        = delta.clip(lower=0).rolling(14).mean()
    loss        = (-delta.clip(upper=0)).rolling(14).mean()
    rs          = gain / (loss + 1e-9)
    df["rsi"]   = 100-(100 / (1 + rs)) 

    # Volume Signal
    df["volume_sma_10"] = df["volume"].rolling(10).mean()
    df["volume_ratio"]  = df["volume"] / (df["volume_sma_10"] + 1e-9)

    # Price position within recent range
    rolling_min = df["close"].rolling(20).min()
    rolling_max = df["close"].rolling(20).max()
    df["price_position"] = (df["close"] - rolling_min) / (rolling_max - rolling_min + 1e-9)

    df = df.dropna()
    return df 

def build_sequences(df:pd.DataFrame):
    all_X , all_y = [], []

    features = ["close", "open", "high", "low", "volume",
        "sma_5", "sma_20", "ema_12", "ema_26",
        "macd", "roc_5", "momentum",
        "daily_return", "volatility_10", "high_low_range",
        "rsi", "volume_ratio", "price_position"
        ]

    for symbol in df["symbol"].unique():
        sdf = df[df["symbol"] == symbol].copy()
        sdf = sdf.sort_values("date").reset_index(drop=True)
        sdf = add_technical_indicators(sdf)

        sdf["label"] = (sdf["close"].shift(-1) > sdf["close"]).astype(int)
        sdf = sdf.dropna(subset=["label"])

        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(sdf[features])
        labels = sdf["label"].values

        for i in range(WINDOW , len(scaled)):
            all_X.append(scaled[i - WINDOW:i])
            all_y.append(labels[i])

    X = np.array(all_X)
    y = np.array(all_y)

    return X , y

def train_test_split_temporal(X , y , split = 0.8):
    n = int(len(X) * split)
    return X[:n] , X[n:] , y[:n] , y[n:]

def save_sequences(X_train , X_test , y_train , y_test , path = "ModelTraining/data"):
    os.makedirs(path , exist_ok = True)
    np.save(f"{path}/X_train.npy" , X_train)
    np.save(f"{path}/X_test.npy" , X_test)
    np.save(f"{path}/y_train.npy" , y_train)
    np.save(f"{path}/y_test.npy" , y_test)
    print(f"saved sequences to {path}/")


if __name__ == "__main__":
    from fetch import fetch_stock_data
    df = fetch_stock_data()
    X , y = build_sequences(df)
    X_train , X_test , y_train , y_test = train_test_split_temporal(X , y)
    save_sequences( X_train , X_test , y_train , y_test)

    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Train samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Label balance: {y.mean():.2f}")