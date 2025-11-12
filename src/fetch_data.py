import os
import requests
import yfinance as yf
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime, timezone, timedelta

# path setup
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(ROOT_DIR, "data", "raw")
os.makedirs(RAW_DATA_DIR, exist_ok=True)

# load .env variables
load_dotenv()

def fetch_stock_data(ticker, period, interval, auto_adjust=False):
    """
    Fetch stock price data for a given ticker using yfinance and save it
    as a timestamped CSV file under data/raw/.
    
    Parameters
    ----------
    ticker : str
        Stock ticker symbol (e.g., "AAPL", "NVDA").
    period : str
        Time period of historical data to fetch (e.g., "7d", "1mo", "1y").
    interval : str
        Frequency of data intervals (e.g., "1h", "1d").
    auto_adjust : bool, optional
        Whether to automatically adjust prices for corporate actions such as 
        stock splits and dividends. If True (default), all OHLC values are adjusted and
        the 'Adj Close' column is removed. If False, raw market prices are returned
        and the 'Adj Close' column is included. 
    """
    try:
        # check for invalid ticker or no data, download it otherwise
        data = yf.download(ticker, period=period, interval=interval, auto_adjust=auto_adjust)
        if data.empty:
            print(f"No data returned for {ticker}. Skipping.")
            return
        
        # move datetime into a column
        data.reset_index(inplace=True)
        
        # yfinance often returns a MultiIndex DataFrame as it
        # is designed to handle multiple tickers at once, for single ticker
        # requests we flatten the MultiIndex structure such that a messy and
        # unnecessary row of ['ticker', 'ticker', 'ticker', ...] is removed
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        # create timestamp for versioning
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        filename = f"{ticker}_prices_{timestamp}.csv"
        
        # save file to data/raw/
        data.to_csv(os.path.join(RAW_DATA_DIR, filename), index=False)
        print(f"Saved price data for {ticker}")
        
    except Exception as e:
        print(f"Failed to fetch data for {ticker}: {e}")

def fetch_news(ticker):
    """
    Fetch recent news articles related to a ticker using NewsAPI and save them
    as a timestamped CSV under data/raw/.
    
    Parameters
    ----------
    ticker : str
        Stock ticker symbol (e.g., "AAPL", "NVDA").
    """
    try: 
        key = os.getenv("NEWS_API_KEY")
        if not key:
            raise ValueError("Missing NEWS_API_KEY in .env file")
        now = datetime.now(timezone.utc)
        
        all_articles = []
        # manually check for relevant articles in the past month one week at a time
        for offset in range(0, 28, 7):
            to_date = now - timedelta(days=offset)  # number of days before today
            from_date = to_date - timedelta(days=7) # 7 days prior to to_date value
            url = (
                f"https://newsapi.org/v2/everything?"
                f"q={ticker}&from={from_date.date()}&to={to_date.date()}"
                f"&language=en&sortBy=publishedAt&pageSize=100&page=1&apiKey={key}"
            )
            response = requests.get(url, timeout=10) # wait up to 10s for response
            response.raise_for_status() # check for bad response
            articles = response.json().get("articles", [])
            if articles:
                all_articles.extend(articles) # add articles from current page to total
                print(f"Fetched {len(articles)} articles from {from_date.date()} to {to_date.date()}.")
        
        if not all_articles:
            print(f"No news found for {ticker}.")
            return
        
        df = pd.DataFrame(all_articles)

        # create timestamp for versioning
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        filename = f"{ticker}_news_{timestamp}.csv"
        df.to_csv(os.path.join(RAW_DATA_DIR, filename), index=False)
        print(f"Saved news data ({len(df)} articles) for {ticker}.")
        
    except Exception as e:
        print(f"Failed to fetch news for {ticker}: {e}")

if __name__ == "__main__":
    # Settings
    tickers =  ["NVDA"]
    period = "1mo"
    interval = "1h" 
    
    # Execution
    for ticker in tickers:
        fetch_stock_data(ticker, period, interval)
        fetch_news(ticker)
