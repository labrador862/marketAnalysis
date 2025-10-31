from transformers import pipeline
import torch

# Detect if CUDA is available
device = 0 if torch.cuda.is_available() else -1
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU detected")
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Built with CUDA:", torch.version.cuda)



# Load the FinBERT model for financial sentiment analysis
sentiment_model = pipeline("sentiment-analysis", model="ProsusAI/finbert", device=device)

# Some example financial news headlines
headlines = [
    "NVIDIA stock soars after strong quarterly earnings report",
    "Tesla faces regulatory challenges in Europe amid safety concerns",
    "Apple launches new MacBook models, market reacts with optimism",
    "Investors worry about rising inflation and slower economic growth",
    "Amazon's revenue growth slows, but cloud services continue to expand"
]

# Run sentiment analysis
results = sentiment_model(headlines)

# Display results neatly
for headline, result in zip(headlines, results):
    label = result['label']
    score = result['score']
    print(f"{headline}\n → Sentiment: {label} (confidence: {score:.2f})\n")
