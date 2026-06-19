from __future__ import annotations
import pandas as pd
from .brandstats import BrandStats


class BrandAnalyzer:
    """Extracts BrandStats from a DataFrame with a 'Sentiment Label' column."""

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def analyze_all(self) -> dict[str, BrandStats]:
        results = {}
        for name, group in self.df.groupby("Brand Name"):
            results[name] = self._analyze_group(name, group)
        return results

    def analyze_brand(self, brand_name: str) -> BrandStats:
        group = self.df[self.df["Brand Name"] == brand_name]
        return self._analyze_group(brand_name, group)

    @staticmethod
    def _analyze_group(name: str, group: pd.DataFrame) -> BrandStats:
        sentiments = group.groupby("Sentiment Label")  # группируем по сантименту
        lengths = sentiments["Review Length"].apply(lambda x: sorted(x.tolist())).to_dict()  # сохраняем длины отзывов
                                                                                            # в отсортированном порядке
        return BrandStats(
            name=name,
            sentiment_lengths=lengths,
        )
