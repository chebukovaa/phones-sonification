from __future__ import annotations
from dataclasses import dataclass
from .sentiment import Sentiment


@dataclass
class BrandStats:
    name: str
    sentiment_lengths: dict[Sentiment, list[int]]

    @property
    def total_reviews(self) -> int:
        return sum(len(lengths_list) for lengths_list in self.sentiment_lengths.values())

    @property
    def sentiment_probabilities(self) -> dict[Sentiment, float]:
        total = self.total_reviews
        if total == 0:
            return {}
        return {sentiment: len(lengths_list) / total for sentiment, lengths_list in self.sentiment_lengths.items()}
