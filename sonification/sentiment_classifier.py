from transformers import pipeline
import pandas as pd
from .sentiment import Sentiment


class SentimentClassifier:
    def __init__(self, model_name : str = "cardiffnlp/twitter-roberta-base-sentiment-latest",
                 batch_size : int = 64, max_length : int = 512):
        self.pipe = pipeline("text-classification",
                              model = model_name,
                              truncation = True,
                              batch_size = batch_size,
                              max_length = max_length)

    def classify(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        results = self.pipe(df["Reviews"].tolist())
        df["Sentiment Label"] = [Sentiment(r["label"]) for r in results]
        df["Sentiment Score"] = [r["score"] for r in results]
        return df
