"""Stage 2 · text preprocessing
data/raw/*.csv → data/interim/{train,test}_processed.csv"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.common import ROOT, get_logger, preprocess_comment  # noqa: E402

logger = get_logger("data_preprocessing")


def process(name: str) -> None:
    df = pd.read_csv(ROOT / "data" / "raw" / f"{name}.csv")
    df["clean_comment"] = df["clean_comment"].astype(str).map(preprocess_comment)
    df = df[df["clean_comment"].str.len() > 0]
    out = ROOT / "data" / "interim"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"{name}_processed.csv", index=False)
    logger.info("%s: %d rows written", name, len(df))


def main() -> None:
    try:
        for split in ("train", "test"):
            process(split)
    except Exception:
        logger.exception("preprocessing failed")
        raise


if __name__ == "__main__":
    main()
