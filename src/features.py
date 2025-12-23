"""
feature_engineer.py

Feature Engineering & Labeling

This module creates lagged, rolling and technical indictator-based features, 
such as moving averages, RSI, and volatility, and generates binary classification labels 
representing the next day's price direction.

Example:
    python features.py
"""
import os
import pandas as pd

# path setup
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DATA_DIR = os.path.join(ROOT_DIR, "data", "processed")
FEATURES_DIR = os.path.dirname(os.path.join(ROOT_DIR, "data", "features"))
os.makedirs(FEATURES_DIR, exist_ok=True)

def load_data(price_path: str) -> pd.DataFrame:
    """
    Load processed price CSV files.

    Parameters
    ----------
    price_path : str
        Path to the processed prices CSV.
    Returns:
        pd.DataFrame: Loaded price DataFrame.
    """
    prices = pd.read_csv(price_path)
    return prices

def compute_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute common moving averages: 5, 10, and 20 day.
    """
    df["ma_5"] = df["Close"].rolling(5).mean()
    df["ma_10"] = df["Close"].rolling(10).mean()
    df["ma_20"] = df["Close"].rolling(20).mean()
    return df

def compute_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Compute 14-day RSI using standard Wilder smoothing.
    """
    delta = df["Close"].diff()
    
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    # Wilder smoothing
    for i in range(period, len(df)):
        avg_gain.iat[i] = (avg_gain.iat[i - 1] * (period - 1) + gain.iat[i]) / period
        avg_loss.iat[i] = (avg_loss.iat[i - 1] * (period - 1) + loss.iat[i]) / period
        
    # edge case: zero loss period - avoid division by zero and inf values
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    rsi = 100 - (100 / (1 + rs))
    
    # handle cases where avg_loss is zero
    rsi.loc[(avg_loss == 0) & (avg_gain > 0)] = 100 # heavy gains
    rsi.loc[(avg_loss == 0) & (avg_gain == 0)] = 50 # neutral momentum

    df["rsi_14"] = rsi

    return df

def compute_volatility(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    Rolling volatility (std of 1-day returns).
    """
    df["volatility_14"] = df["return_1d"].rolling(window).std()
    return df

def compute_price_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute intraday price range features.
    """
    df["range_high_low"] = df["High"] - df["Low"]
    df["range_open_close"] = df["Close"] - df["Open"]
    df["range_hl_pct"] = (df["High"] - df["Low"]) / df["Close"]
    return df

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create all engineered features:
        - returns (1d, 5d)
        - volume change
        - moving averages
        - RSI
        - volatility
        - price ranges

    Parameters
    ----------
    df : pd.DataFrame
        Price DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame with engineered features added.
    """
    # Change in daily return in past day and past 5 days
    df["return_1d"] = df["Close"].pct_change()
    df["return_5d"] = df["Close"].pct_change(5)

    # Volume change (past day)
    df["volume_change"] = df["Volume"].pct_change()
    
    # Technical indicators
    df = compute_moving_averages(df)
    df = compute_rsi(df)
    df = compute_volatility(df)
    df = compute_price_ranges(df)

    return df

def create_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create binary classification labels for next-day price direction.

    Args:
        df (pd.DataFrame): DataFrame with daily returns.

    Returns:
        pd.DataFrame: DataFrame including a 'target' column, where
        target=1 indicates tomorrow's price goes up and
        target=0 indicates tomorrow's price remains the same or goes down
    """
    # The sign of the next day's return determines what today's target should have been
    df["target"] = (df["return_1d"].shift(-1) > 0).astype(int)
    return df

def save_features(df: pd.DataFrame, output_prefix: str) -> None:
    """
    Finalize dataset by dropping NaNs and saving to /data/features/.

    Args:
        df (pd.DataFrame): Fully engineered feature DataFrame.
        output_prefix (str): Filename prefix.
    """
    df.dropna().reset_index(drop=True)
    os.makedirs("data/features", exist_ok=True) # create folder if necessary

    output_path = f"data/features/{output_prefix}_features.csv"
    df.to_csv(output_path, index=False)
    
    # debug
    print(f"Feature dataset saved to: {output_path}")

def main():
    """
    Main entry point for the feature engineering pipeline.
    """
    for file in os.listdir(PROCESSED_DATA_DIR):
        path = os.path.join(PROCESSED_DATA_DIR, file)
        df = load_data(path)
        df = create_features(df)
        df = create_labels(df)
        save_features(df, os.path.basename(path).split("_")[0])

if __name__ == "__main__":
    main()

