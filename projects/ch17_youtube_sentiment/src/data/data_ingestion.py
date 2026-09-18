"""Stage 1 · data ingestion
Download the Reddit sentiment CSV, apply the basic cleaning found during EDA,
split it, and write data/raw/{train,test}.csv."""
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.common import ROOT, get_logger, load_params  # noqa: E402

logger = get_logger("data_ingestion")
DATA_URL = "https://raw.githubusercontent.com/Himanshu-1703/reddit-sentiment-analysis/refs/heads/main/data/reddit.csv"


def load_data(url: str) -> pd.DataFrame:
    df = pd.read_csv(url)
    logger.info("downloaded %d rows, columns=%s", len(df), list(df.columns))
    return df


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.dropna(subset=["clean_comment"])
    df = df.drop_duplicates()
    df = df[df["clean_comment"].str.strip() != ""]          # comments that are only whitespace
    logger.info("basic clean: %d → %d rows", before, len(df))
    return df.reset_index(drop=True)


def main() -> None:
    try:
        test_size = load_params("data_ingestion")["test_size"]
        df = basic_clean(load_data(DATA_URL))
        train, test = train_test_split(df, test_size=test_size, random_state=42, stratify=df["category"])
        out = ROOT / "data" / "raw"
        out.mkdir(parents=True, exist_ok=True)
        train.to_csv(out / "train.csv", index=False)
        test.to_csv(out / "test.csv", index=False)
        logger.info("saved %d train / %d test rows to %s", len(train), len(test), out)
    except Exception:
        logger.exception("data ingestion failed")
        raise


if __name__ == "__main__":
    main()
