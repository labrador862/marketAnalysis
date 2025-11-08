import os
import argparse
import pandas as pd
from datetime import datetime
from transformers import pipeline
import torch

# path setup
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DATA_DIR = os.path.join(ROOT_DIR, "data", "processed")
SENTIMENT_DATA_DIR = os.path.join(ROOT_DIR, "data", "sentiment")
os.makedirs(SENTIMENT_DATA_DIR, exist_ok=True)

def load_finbert():
    """
    Load the FinBERT sentiment analysis model and automatically set the device
    (GPU if available, otherwise CPU).
    
    Returns:
    transformers.pipelines.text_classification.TextClassificationPipeline
        A FinBERT model pipeline ready to process text and return sentiment outputs.
        
    Notes:
    - Prints GPU information and library versioning for transparency.
    """
    # determine if a CUDA supported GPU is available; if so, use it
    device = 0 if torch.cuda.is_available() else -1 # 0 refers to gpu, -1 refers to cpu
    print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU detected")
    print("Torch version:", torch.__version__)
    print("Built with CUDA:", torch.version.cuda)
    return pipeline("sentiment-analysis", model="ProsusAI/finbert", device=device)

def analyze_news_sentiment(df, sentiment_model):
    """
    Run FinBERT sentinment analysis on each news article, producing sentiment labels, scores, 
    and numeric values for aggregation. 
    
    Parameters
    ----------
    df : pandas.DataFrame
        Processed news DataFrame containing at least 'title' and 'description' columns.
    sentiment_model : transformers.pipelines.text_classification.TextClassificationPipeline
        Preloaded FinBERT model pipeline.
    
    Returns
    -------
    pandas.DataFrame
        Original DataFrame with additional columns:
            - 'sentiment_label' : str
                Sentiment classification ('positive', 'neutral', 'negative').
            - 'sentiment_score' : float
                Confidence score from FinBERT (0-1).
            - 'sentiment_value' : float
                Signed sentiment score used for averaging:
                (1 x score) for positive, (0 x score) for neutral, (-1 x score) for negative.
    
    Notes
    -----
    - Each article's text is constructed as "<title>. <description>" when fed into FinBERT
    - Texts are truncated to 512 tokens as this is the maximum allowed with FinBERT
    - Article inference errors default to a neutral sentiment with score 0.0.
    """
    texts = (df["title"] + ". " + df["description"]).tolist()
    results = []

    for text in texts:
        try:
            # truncates input beyond 512 tokens (FinBERT max)
            output = sentiment_model(text, truncation=True, max_length=512)[0]
            print(f"debug: {output}")
            results.append(output)
        except Exception as e:
            print(f"Skipping one entry due to error: {e}")
            results.append({"label": "neutral", "score": 0.0})

    df["sentiment_label"] = [r["label"] for r in results]
    df["sentiment_score"] = [r["score"] for r in results]

    # convert labels to numeric polarity for averaging
    label_map = {"positive": 1, "neutral": 0, "negative": -1}
    df["sentiment_value"] = df["sentiment_label"].map(label_map) * df["sentiment_score"]

    return df

def aggregate_daily_sentiment(df):
    df["date"] = pd.to_datetime(df["publishedAt"], utc=True).dt.date
    aggregated = (
        df.groupby("date")["sentiment_value"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "avg_sentiment", "count": "article_count"})
        .reset_index()
    )
    return aggregated

def save_results(aggregated_df, raw_filename):
    prefix = raw_filename.split("_")[0]  # grab ticker
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    filename = f"{prefix}_sentiment_{timestamp}.csv"
    
    aggregated_df.to_csv(os.path.join(SENTIMENT_DATA_DIR, filename), index=False)
    print(f"Saved sentiment data: {filename}")
    
def save_results_by_article(df, raw_filename):
    prefix = raw_filename.split("_")[0]  # grab ticker
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    filename = f"{prefix}_sentiment_by_article_{timestamp}.csv"
    
    readable_cols = ["publishedAt", "author", "title", "sentiment_label", "sentiment_score"]
    readable_df = df[readable_cols].copy()
    readable_df.rename(columns={"publishedAt": "date"}, inplace=True)
    readable_df["date"] = pd.to_datetime(readable_df["date"], utc=True).dt.date
    readable_df["sentiment_score"] = readable_df["sentiment_score"].round(2)
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
