"""Usage: streamlit run server.py"""
import os
import streamlit as st
from pathlib import Path
import pandas as pd
from sonification.brand_analyzer import BrandAnalyzer


output_dir = Path(os.environ.get("SONIFICATION_OUTPUT_DIR", "output/sonified_brands"))  # задаем директорию с результатами сонификации

# Загрузка данных (кэшированная)
@st.cache_data
def load_stats(parquet_path):
    df = pd.read_parquet(parquet_path)
    return BrandAnalyzer(df).analyze_all()

# Заголовок
st.title("Сонификация данных о телефонах")

# Статистики и имена брендов
all_stats = load_stats("data/phones_data.parquet")
all_stats = {name: stats for name, stats in all_stats.items() if stats.total_reviews > 500}
brands = sorted(all_stats.keys())

# Боковая панель
brand = st.sidebar.selectbox("Brand", brands)
brand_stats = all_stats[brand]

# Основной контент — сверху вниз
st.header("Статистика")
st.metric("Total reviews", brand_stats.total_reviews)

probs = brand_stats.sentiment_probabilities
cols = st.columns(len(probs))
for col, (sentiment, prob) in zip(cols, probs.items()):
    col.metric(sentiment, f"{prob:.0%}")

st.header("Сравнение алгоритмов")
tab_length, tab_velocity = st.tabs(["LengthSonifier", "VelocitySonifier"])

with tab_length:
    path = output_dir / f"{brand}_LengthSonifier.wav"
    if path.exists():
        st.audio(str(path))
    else:
        st.info("No file for this algorithm")

with tab_velocity:
    path = output_dir / f"{brand}_VelocitySonifier.wav"
    if path.exists():
        st.audio(str(path))
    else:
        st.info("No file for this algorithm")
