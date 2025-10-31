import os
import argparse
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from transformers import pipeline
import torch

# path setup
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROCESSED_DATA_DIR = os.path.join(ROOT_DIR, "data", "processed")
SENTIMENT_DATA_DIR = os.path.join(ROOT_DIR, "data", "sentiment")
os.makedirs(SENTIMENT_DATA_DIR, exist_ok=True)

def load_finbert():
    # determine if a CUDA supported GPU is available; if so, use it
    device = 0 if torch.cuda.is_available() else -1 # 0 refers to gpu, -1 refers to cpu
    print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU detected")
    print("Torch version:", torch.__version__)
    print("Built with CUDA:", torch.version.cuda)
    return pipeline("sentiment-analysis", model="ProsusAI/finbert", device=device)

def analyze_news_sentiment(df, sentiment_model):
    texts = (df["title"] + ". " + df["description"]).tolist()
    results = []

    for text in tqdm(texts, desc="Analyzing sentiment..."):
        try:
            output = sentiment_model(text[:512])[0]  # truncate to FinBERT limit
            results.append(output)
        except Exception as e:
            print(f"Skipping one entry due to error: {e}")
            results.append({"label": "neutral", "score": 0.0})

    df["sentiment_label"] = [r["label"] for r in results]
    df["sentiment_score"] = [r["score"] for r in results]

    # convert labels to numeric polarity for averaging
    label_map = {"positive": 1, "neutral": 0, "negative": -1}
    df["sentiment_value"] = df["sentiment_label"].map(label_map) * df["sentiment_score"]
    
    # debug
    readable_cols = ["publishedAt", "author", "title", "sentiment_label", "sentiment_score"]
    readable_df = df[readable_cols].copy()
    readable_df.to_csv(os.path.join(SENTIMENT_DATA_DIR, "readable.csv"), index=False)

    return df

def aggregate_daily_sentiment(df):
    df["date"] = pd.to_datetime(df["publishedAt"], utc=True).dt.date
    daily = (
        df.groupby("date")["sentiment_value"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "avg_sentiment", "count": "article_count"})
        .reset_index()
    )
    daily.to_csv(os.path.join(SENTIMENT_DATA_DIR, "daily.csv"), index=False)
    return daily

def save_results(daily_df, raw_filename):
    prefix = raw_filename.split("_")[0]  # grab ticker
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    filename = f"{prefix}_sentiment_{timestamp}.csv"
    daily_df.to_csv(os.path.join(SENTIMENT_DATA_DIR, filename), index=False)
    print(f"Saved sentiment data: {filename}")
    
def save_results_by_article(df, raw_filename):
    prefix = raw_filename.split("_")[0]  # grab ticker
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    filename = f"{prefix}_sentiment_by_article_{timestamp}.csv"
    readable_cols = ["publishedAt", "author", "title", "sentiment_label", "sentiment_score"]
    readable_df = df[readable_cols].copy()
    readable_df.to_csv(os.path.join(SENTIMENT_DATA_DIR, filename), index=False)
    print(f"Saved sentiment data by article: {filename}")

def main(input_path):
    df = pd.read_csv(input_path)
    model = load_finbert()
    df = analyze_news_sentiment(df, model)
    save_results_by_article(df, os.path.basename(input_path))
    aggregated_df = aggregate_daily_sentiment(df)
    save_results(aggregated_df, os.path.basename(input_path))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to processed news CSV file")
    args = parser.parse_args()
    main(args.input)
