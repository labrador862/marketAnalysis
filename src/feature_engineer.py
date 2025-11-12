"""
feature_engineer.py

Feature Engineering & Labeling

This module merges processed price and sentiment data into a unified dataset 
suitable for machine learning. It creates lagged and rolling sentiment features, 
computes daily price-based features, and generates binary classification labels 
representing the next day's price direction.

Example:
    python feature_engineer.py --price data/processed/NVDA_prices_processed_2025-10-29_00-05.csv \
                               --sentiment data/processed/NVDA_sentiment_2025-10-30_22-11.csv
"""
import os
import argparse
import pandas as pd

def load_data(price_path: str, sentiment_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load processed price and sentiment CSV files.

    Parameters
    ----------
    price_path : str
        Path to the processed prices CSV.
    sentiment_path : str
        Path to the processed sentiment CSV.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Loaded price and sentiment DataFrames.
    """
    prices = pd.read_csv(price_path)
    sentiment = pd.read_csv(sentiment_path)
    return prices, sentiment

def merge_data(prices: pd.DataFrame, sentiment: pd.DataFrame) -> pd.DataFrame:
    """
    Merge price and sentiment data on the 'date' column.

    Parameters
    ----------
        prices : pd.DataFrame
            Processed price data with daily OHLCV values.
        sentiment : pd.DataFrame
            Daily sentiment data with 'avg_sentiment' and 'article_count' columns.

    Returns:
        pd.DataFrame: Merged DataFrame aligned by date.
    """
    # debug
    print(prices.columns)
    print(sentiment.columns)
    print("=== PRICE DATES SAMPLE ===")
    print(prices["date"].head(), prices["date"].dtype)
    print("=== SENTIMENT DATES SAMPLE ===")
    print(sentiment["date"].head(), sentiment["date"].dtype)
    print("Unique dates in prices:", prices["date"].nunique())
    print("Unique dates in sentiment:", sentiment["date"].nunique())
    common_dates = set(prices["date"]).intersection(set(sentiment["date"]))
    print("Common date count:", len(common_dates))


    merged = pd.merge(prices, sentiment, on="date", how="inner")
    # debug
    print("Merged shape:", merged.shape)
    print(merged)

    merged.sort_values("date", inplace=True)
    merged.reset_index(drop=True, inplace=True)
    
    return merged

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create lagged, rolling, and price-based features for model input.

    Parameters
    ----------
    df : pd.DataFrame
        Merged price + sentiment DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame with engineered features added.
    """
    # Change in daily return in past day and past 5 days
    df["return_1d"] = df["Close"].pct_change()
    df["return_5d"] = df["Close"].pct_change(5)

    # Change in volume traded in past day
    df["volume_change"] = df["Volume"].pct_change()

    # Shift average sentiment and num articles for a given day to
    # the next day such that yesterday's news impacts today's analysis
    df["sentiment_lag1"] = df["avg_sentiment"].shift(1)
    df["article_count_lag1"] = df["article_count"].shift(1)

    # Three day rolling sentiment averages 
    df["sentiment_rolling3"] = df["avg_sentiment"].rolling(3).mean().shift(1)

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
    df.fillna(0, inplace=True)
    os.makedirs("data/features", exist_ok=True) # create folder if necessary

    output_path = f"data/features/{output_prefix}_features.csv"
    df.to_csv(output_path, index=False)
    
    # debug
    print(f"Feature dataset saved to: {output_path}")

def main():
    """
    Main entry point for the feature engineering pipeline.
    """
    parser = argparse.ArgumentParser(description="Merge sentiment and price data for ML feature generation.")
    parser.add_argument("--prices", required=True, help="Path to processed prices CSV.")
    parser.add_argument("--sentiment", required=True, help="Path to processed sentiment CSV.")
    args = parser.parse_args()

    prices, sentiment = load_data(args.prices, args.sentiment)
    df = merge_data(prices, sentiment)
    df = create_features(df)
    df = create_labels(df)
    
    # debug
    print("Final shape before save:", df.shape)

    save_features(df, os.path.basename(args.prices).split("_")[0])


if __name__ == "__main__":
    main()

