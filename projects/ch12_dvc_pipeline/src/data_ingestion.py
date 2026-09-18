"""Stage 1 · Data ingestion
Downloads the tweet-emotion CSV, keeps two classes, encodes the label and
writes a train/test split.            OUTPUT: data/raw/train.csv, data/raw/test.csv
"""
import os
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(__file__))
from utils import get_logger  # noqa: E402

logger = get_logger("data_ingestion")

DATA_URL = "https://raw.githubusercontent.com/entbappy/Branching-tutorial/refs/heads/master/tweet_emotions.csv"
POSITIVE_CLASS = "happiness"   # the demo later swaps this to "neutral" to show DVC re-running the stage
NEGATIVE_CLASS = "sadness"


def load_data(url: str) -> pd.DataFrame:
    df = pd.read_csv(url)
    logger.info("loaded %d rows from %s", len(df), url)
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=["tweet_id"])
    df = df[df["sentiment"].isin([POSITIVE_CLASS, NEGATIVE_CLASS])].copy()
    df["sentiment"] = df["sentiment"].map({POSITIVE_CLASS: 1, NEGATIVE_CLASS: 0})
    logger.info("kept %d rows (%s=1, %s=0)", len(df), POSITIVE_CLASS, NEGATIVE_CLASS)
    return df


def save_data(train: pd.DataFrame, test: pd.DataFrame, out_dir: str = "data/raw") -> None:
    os.makedirs(out_dir, exist_ok=True)
    train.to_csv(os.path.join(out_dir, "train.csv"), index=False)
    test.to_csv(os.path.join(out_dir, "test.csv"), index=False)
    logger.info("saved %d train / %d test rows to %s", len(train), len(test), out_dir)


def main() -> None:
    try:
        df = preprocess(load_data(DATA_URL))
        train, test = train_test_split(df, test_size=0.2, random_state=42)
        save_data(train, test)
    except Exception:
        logger.exception("data ingestion failed")
        raise


if __name__ == "__main__":
    main()
