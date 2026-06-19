"""Usage: python -m sonification prepare | sonify"""

import pandas as pd
import argparse
from pathlib import Path
from midi2audio import FluidSynth

from .data_loader import DataLoader
from .sentiment_classifier import SentimentClassifier
from .brand_analyzer import BrandAnalyzer
from .sonifier import LengthSonifier, VelocitySonifier


RAW_PATH = "data/Amazon_Unlocked_Mobile.csv"
PARQUET_PATH = "data/phones_data.parquet"
OUTPUT_DIR = "output/sonified_brands"


def prepare(raw_csv_path: str = RAW_PATH, parquet_path: str = PARQUET_PATH):
    Path(raw_csv_path).parent.mkdir(parents=True, exist_ok=True)
    Path(parquet_path).parent.mkdir(parents=True, exist_ok=True)
    df = DataLoader(raw_csv_path).load()
    df = SentimentClassifier().classify(df)
    df.to_parquet(parquet_path, index=False)
    print(f"Saved {len(df)} rows to {parquet_path}")


def sonify(parquet_path: str = PARQUET_PATH, output_dir: str = OUTPUT_DIR) -> None:
    df = pd.read_parquet(parquet_path)

    all_stats = BrandAnalyzer(df).analyze_all()
    sonifiers = [LengthSonifier(), VelocitySonifier()]

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fs = FluidSynth("/usr/share/sounds/sf2/FluidR3_GM.sf2")
    for name, stats in all_stats.items():
        for sonifier in sonifiers:
            title = f"{name}_{sonifier.__class__.__name__}"
            midi_path = f"{output_dir}/{title}.mid"
            wav_path = f"{output_dir}/{title}.wav"
            melody = sonifier.sonify(stats)
            score = sonifier.to_score(melody, title=title)
            score.write("midi", f"{output_dir}/{title}.mid")
            fs.midi_to_audio(midi_path, wav_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("prepare")
    p.add_argument("--raw-csv-path", default=RAW_PATH, type=str, help="raw csv path")
    p.add_argument("--parquet-path", default=PARQUET_PATH, type=str, help="parquet path")

    s = sub.add_parser("sonify")
    s.add_argument("--parquet-path", default=PARQUET_PATH, type=str, help="parquet path")
    s.add_argument("--output-dir", default=OUTPUT_DIR, type=str, help="output dir")

    args = parser.parse_args()

    if args.command == "prepare":
        prepare(args.raw_csv_path, args.parquet_path)
    elif args.command == "sonify":
        sonify(args.parquet_path, args.output_dir)
    else:
        parser.print_help()
