"""Download Wisesight sentiment data and save as CSV.

Migrated from sentence-classification/get_wisesight_sentiment.py
(originally fetched txt files; sentence-classification-train kept raw txt).
"""
import argparse
from pathlib import Path

import pandas as pd
import requests

TRAIN_TEXT_URL = "https://raw.githubusercontent.com/PyThaiNLP/wisesight-sentiment/master/kaggle-competition/train.txt"
TRAIN_LABEL_URL = "https://raw.githubusercontent.com/PyThaiNLP/wisesight-sentiment/master/kaggle-competition/train_label.txt"
TEST_TEXT_URL = "https://raw.githubusercontent.com/PyThaiNLP/wisesight-sentiment/master/kaggle-competition/test.txt"
TEST_LABEL_URL = "https://raw.githubusercontent.com/PyThaiNLP/wisesight-sentiment/master/kaggle-competition/test_label.txt"


def download_text(url: str) -> str:
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.text


def main(out_dir: str = "data") -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_text = download_text(TRAIN_TEXT_URL)
    train_label = download_text(TRAIN_LABEL_URL)
    test_text = download_text(TEST_TEXT_URL)
    test_label = download_text(TEST_LABEL_URL)

    train_df = pd.DataFrame(
        {"text": train_text.split("\n")[:-1], "label": train_label.split("\n")[:-1]}
    )
    test_df = pd.DataFrame(
        {"text": test_text.split("\n")[:-1], "label": test_label.split("\n")[:-1]}
    )

    train_df.to_csv(out / "train.csv", index=False)
    test_df.to_csv(out / "test.csv", index=False)
    print(f"wrote {out / 'train.csv'} ({len(train_df)} rows)")
    print(f"wrote {out / 'test.csv'} ({len(test_df)} rows)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data")
    args = parser.parse_args()
    main(args.out_dir)
