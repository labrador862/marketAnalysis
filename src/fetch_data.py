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
    Fetch recent news articles related to a ticker using NewsData.io and save them
    as a timestamped CSV under data/raw/.
    
    Parameters
    ----------
    ticker : str
        Stock ticker symbol (e.g., "AAPL", "NVDA").
    """
    try: 
        key = os.getenv("NEWS_DATA_IO_KEY")
        if not key:
            raise ValueError("Missing NEWS_DATA_IO_KEY in .env file")

        years = 5
        max_pages = 1000
        base_url = "https://newsdata.io/api/1/news"
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=years*365)
        
        all_articles = []
        next_page = None
        page_count = 0
        
        seen_urls = set()
        all_articles = []

        current_end = datetime.now(timezone.utc)
        cutoff_date = current_end - timedelta(days=365 * years)

        while current_end > cutoff_date:
            current_start = current_end - timedelta(days=30)

            next_page = None
            page_count = 0

            while page_count < max_pages:
                params = {
                    "apikey": key,
                    "q": ticker,
                    "language": "en",
                    "from_date": current_start.date().isoformat(),
                    "to_date": current_end.date().isoformat(),
                }

                if next_page:
                    params["page"] = next_page

                response = requests.get(base_url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()

                articles = data.get("results", [])
                if not articles:
                    break

                for article in articles:
                    link = article.get("link")
                    if not link or link in seen_urls:
                        continue

                    seen_urls.add(link)

                    published = pd.to_datetime(
                        article.get("pubDate"), utc=True, errors="coerce"
                    )
                    if pd.isna(published):
                        continue

                    creator = article.get("creator")
                    author = creator[0] if isinstance(creator, list) and creator else None

                    all_articles.append({
                        "author": author,
                        "title": article.get("title"),
                        "description": article.get("description"),
                        "url": link,
                        "publishedAt": published.isoformat(),
                        "source": article.get("source_id"),
                    })

                next_page = data.get("nextPage")
                if not next_page:
                    break

                page_count += 1

            # move window backward
            current_end = current_start

    except Exception as e:
        print(f"Failed to fetch news for {ticker}: {e}")

if __name__ == "__main__":
    # Settings
    tickers =  ["NVDA"]
    period = "2y"
    interval = "1d" 
    
    # Execution
    for ticker in tickers:
        fetch_stock_data(ticker, period, interval)
        fetch_news(ticker)
