import os
import pandas as pd
from datetime import datetime

# path setup
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(ROOT_DIR, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(ROOT_DIR, "data", "processed")
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

def clean_stock_data(file_path):
    """
    Load and clean a raw stock price CSV. This includes datetime parsing, 
    sorting, numeric conversion, and duplicate removal.
    
    Parameters
    ----------
    file_path : str
        Absolute path to the raw stock price CSV file.

    Returns:
    pandas.DataFrame
        Cleaned and standardized price data.
    """
    df = pd.read_csv(file_path)
    
    # convert Datetime from str to pandas datetime64 value and rename to 'date',
    # important naming style for consistency
    df.rename(columns={"Datetime": "Date"}, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.date

    # sort in chronological order, important for time-series operations later
    df.sort_values("Date", inplace=True)
    df.drop_duplicates(subset=["Date"], inplace=True)
    
    # convert all values in each column to a numeric dtype (e.g., float64, int64)
    # errors="coerce" turns unparsable values into NaN
    numeric_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # give the DataFrame a clean index, ensures clean and continuous
    # row labeling after sorting/dropping
    df.reset_index(drop=True, inplace=True)
    
    return df

def save_processed_data(df, raw_filename):
    """
    Save a cleaned DataFrame to data/processed/ with a timestamped filename.
    
    Parameters
    ----------
    df : DataFrame
        The cleaned stock DataFrame.
    raw_filename : str
        Original raw CSV filename, e.g. 'NVDA_prices_2025-10-28_23-32.csv'.
    """
    # extract base prefix and create timestamp for versioning
    prefix = raw_filename.split("_")[0] + "_" + raw_filename.split("_")[1]
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    filename = f"{prefix}_processed_{timestamp}.csv"
    
    df.to_csv(os.path.join(PROCESSED_DATA_DIR, filename), index=False)
    print(f"Saved processed data: {filename}")

if __name__ == "__main__":
    for file in os.listdir(RAW_DATA_DIR):
        path = os.path.join(RAW_DATA_DIR, file)
        df = clean_stock_data(path)
        save_processed_data(df, file.replace(".csv", ""))

