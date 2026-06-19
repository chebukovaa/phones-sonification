from __future__ import annotations
import re
import pandas as pd


class DataLoader:
    """Loads the CSV and performs cleaning, deduplication, trimming."""

    BRAND_REPLACEMENTS = {
        "apple computer": "apple", "asus computers": "asus", "cat phones": "cat",
        "black berry": "blackberry", "blackberrry": "blackberry", "blackberry (rim)": "blackberry",
        "att": "at&t", "yezz wireless ltd.": "yezz", "ut starcom": "utstarcom",
        "zte corporation": "zte", "zte(usa) wireless": "zte",
        "sonim technologies": "sonim",
        "sony ericsson mobile": "sony", "sony/ericsson": "sony",
        "sony ericsson": "sony", "sonyericsson": "sony",
        "samssung": "samsung", "samsung galaxy": "samsung",
        "samsung galaxy international inc": "samsung", "samsung international": "samsung",
        "samsung korea": "samsung", "samsung korea ltd": "samsung",
        "samsung/straight talk": "samsung", "samsybg galaxy": "samsung",
        "the nokia": "nokia", "homtom ht7": "homtom",
        "hp handheld": "hp", "htc america": "htc",
        "lg electronics mobilecomm usa": "lg", "lg electronics": "lg",
        "lg electronic": "lg", "lgg": "lg",
        "lenovo manufacturer": "lenovo",
        "motorola x 2nd gen xt1093": "motorola", "moto x": "motorola",
        "google pixel": "google", "indigi®": "indigi",
        "ipro group": "ipro", "worryfree gadgets": "worryfree",
    }
    UNKNOWN_BRANDS = {"unassigned", "unknown", "unlocked cell phone"}

    def __init__(self, path: str):
        self.path = path

    def load(self) -> pd.DataFrame:
        df = pd.read_csv(self.path)  # считываем данные в формат dataframe
        df = df.drop(columns=["Product Name", "Price", "Review Votes"],
                     errors="ignore")  #  удаляем нерелевантные столбцы
        df = df.dropna(subset=["Brand Name", "Reviews"])  # удаляем пропуски в значимых столбцах
        df = self._clean_brands(df)  # приводим названия брендов к единому виду
        df["Review Length"] = df["Reviews"].str.split().str.len()  # сохраняем изначальную длину отзыва
        df = self._deduplicate(df)  # убираем дубликаты
        df = self._strip_long_reviews(df)  # обрезаем длинные отзывы для NLP
        return df

    def _clean_brands(self, df: pd.DataFrame) -> pd.DataFrame:
        df["Brand Name"] = (df["Brand Name"]
                            .str.lower()  # приводим к нижнему регистру
                            .apply(lambda x: re.sub(r"[.,]", " ", str(x)).strip()))  # точки, запятые заменяем пробелами
        for old, new in self.BRAND_REPLACEMENTS.items():  # реализуем все исправления в названиях бренда
            df["Brand Name"] = df["Brand Name"].str.replace(old, new, regex=False)  
        df = df[~df["Brand Name"].isin(self.UNKNOWN_BRANDS)]  # удаляем названия по типу "unknown"
        return df

    @staticmethod
    def _deduplicate(df: pd.DataFrame) -> pd.DataFrame:
        long_duplicates = df[(df.duplicated(subset="Reviews", keep="first")) & (df["Review Length"] >= 6)]
        df = df.drop(long_duplicates.index)  # удаляем длинные дубликаты
        df = (df.groupby("Brand Name", group_keys=False)  # убираем короткие дубликаты в группах
              .apply(lambda x: x.drop(x[x.duplicated(subset="Reviews", keep="first")].index))
              .reset_index(drop=True))
        return df

    @staticmethod
    def _strip_long_reviews(df: pd.DataFrame, max_words: int = 200) -> pd.DataFrame:
        def _trim(text: str) -> str:
            sentences = re.split(r'(?<=[.!?])\s*(?=[A-Za-z0-9]|$)', text)  # разделяем текст на предложения
            if len(sentences) > 6:  # если больше 6 предложений
                text = " ".join(sentences[:3] + sentences[-3:])  # берём первые 3 и последние 3
            if len(text.split()) > max_words:  # если всё ещё больше чем нужно
                phrases = re.split(r'(?<=[.!?,:\-*])\s*(?=[A-Za-z0-9]|$)', text)  # разделяем на фразы
                text = " ".join(phrases[:3] + phrases[-4:])  # берём первые 3 и последние 4
            if len(text.split()) > max_words:  # если всё ещё больше чем нужно
                text = " ".join(text.split()[:max_words])  # разделяем на слова и обрезаем по верхней границе
            return text

        mask = df["Review Length"] > max_words  # ищем отзывы, превышающие лимит слов
        df = df.copy()  # не меняем исходный DataFrame
        df.loc[mask, "Reviews"] = df.loc[mask, "Reviews"].apply(_trim)  # применяем функцию обрезки к длинным отзывам
        return df

if __name__ == "__main__":
    print(len(DataLoader.BRAND_REPLACEMENTS))