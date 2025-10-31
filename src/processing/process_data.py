import os
import pandas as pd
from datetime import datetime
from langdetect import detect, LangDetectException

# path setup
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DATA_DIR = os.path.join(ROOT_DIR, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(ROOT_DIR, "data", "processed")
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

def filter_english_articles(df):
    """
    Remove non-English articles from the data.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The input news DataFrame containing a 'description' column.
    
    Returns
    -------
    df : pandas.DataFrame
        DataFrame consisting of only English articles.
    """
    mask = [] # boolean mask
    for text in df["description"].fillna(""):
        # append only English articles to the mask
        try:
            mask.append(detect(text) == "en")
        except LangDetectException:
            mask.append(False)
    
    # summary for logging
    filtered_df = df[mask]
    total = len(df)
    kept = len(filtered_df)
    percent = (kept / total * 100)
    print(f"Filtered English articles: {kept} / {total} ({percent:.1f}%)")
    
    return df[mask]

def clean_stock_data(file_path):
    """
    Load and clean a raw stock price CSV. This includes datetime parsing, 
    sorting, numeric conversion, and duplicate removal.
    
    Parameters
    ----------
    file_path : str
        Absolute path to the raw stock price CSV file.

    Returns
    -------
    pandas.DataFrame
        Cleaned and standardized price data.
    """
    df = pd.read_csv(file_path)
    
    # convert Datetime from str to pandas datetime64 values
    df["Datetime"] = pd.to_datetime(df["Datetime"], utc=True)
    
    # sort in chronological order, important for time-series operations later
    df.sort_values("Datetime", inplace=True)
    df.drop_duplicates(subset=["Datetime"], inplace=True)
    
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

def clean_news_data(file_path):
    """
    Load and clean a raw news CSV. Performs column extraction, 
    text cleanup, duplicate removal, and language filtering.
    
    Parameters
    ----------
    file_path : str
        Absolute path to the raw stock price CSV file.
        
    Returns
    -------
    pandas.DataFrame
        Cleaned and standardized news data.
    """
    df = pd.read_csv(file_path)
    df = filter_english_articles(df)
    
    # extract publisher name
    if "source" in df.columns:
        df["source_name"] = df["source"].astype(str).str.extract(r"'name': '([^']+)'", expand=False)
        df.drop(columns=["source"], inplace=True, errors="ignore")

    # parse publication timestamps and convert to timezone-aware datetime object
    if "publishedAt" in df.columns:
        df["publishedAt"] = pd.to_datetime(df["publishedAt"], errors="coerce", utc=True)

    # drop incomplete or duplicate rows; title and description
    # are critical for sentiment analysis
    df.dropna(subset=["title", "description", "url"], inplace=True)
    df.drop_duplicates(subset=["url"], inplace=True)
    
    # clean text formatting for NLP
    for col in ["author", "title", "description"]:
        df[col] = (
            df[col]
            .fillna("")     # replace NaN with empty string
            .astype(str)    # ensure string dtype
            .str.replace(r"[\r\n]+", " ", regex=True)  # replace newlines with space
            .str.replace(r"\s+", " ", regex=True) # collapse whitespace
            .str.strip()    # remove leading/trailing whitespace
    )

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
        The cleaned DataFrame (stock or news).
    raw_filename : str
        Original raw CSV filename, e.g. 'NVDA_news_2025-10-28_23-32.csv'.
    """
    # extract base prefirm and create timestamp for versioning
    prefix = raw_filename.split("_")[0] + "_" + raw_filename.split("_")[1]
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    filename = f"{prefix}_processed_{timestamp}.csv"
    
    df.to_csv(os.path.join(PROCESSED_DATA_DIR, filename), index=False)
    print(f"Saved processed data: {filename}")

if __name__ == "__main__":
    for file in os.listdir(RAW_DATA_DIR):
        # ensure only CSVs are processed
        if file.endswith(".csv"):
            path = os.path.join(RAW_DATA_DIR, file)
            # identify stock or news data, clean respectively
            if "prices" in file:
                df = clean_stock_data(path)
                save_processed_data(df, file.replace(".csv", ""))
            elif "news" in file:
                df = clean_news_data(path)
                save_processed_data(df, file.replace(".csv", ""))

