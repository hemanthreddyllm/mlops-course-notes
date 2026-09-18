"""Stage 2 · Text preprocessing
Cleans the tweet text: lowercase, strip URLs / numbers / punctuation,
drop stop words, lemmatise.     INPUT: data/raw   OUTPUT: data/processed
"""
import os
import re
import string
import sys

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

sys.path.append(os.path.dirname(__file__))
from utils import get_logger  # noqa: E402

logger = get_logger("data_preprocessing")

for pkg in ("stopwords", "wordnet", "omw-1.4"):
    nltk.download(pkg, quiet=True)

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()
URL_RE = re.compile(r"https?://\S+|www\.\S+")
PUNCT_TABLE = str.maketrans({c: " " for c in string.punctuation})


def normalize_text(text: str) -> str:
    text = str(text).lower()
    text = URL_RE.sub(" ", text)
    text = re.sub(r"\d+", " ", text)
    text = text.translate(PUNCT_TABLE)
    words = [LEMMATIZER.lemmatize(w) for w in text.split() if w not in STOP_WORDS]
    return " ".join(words)


def process_split(name: str, in_dir: str = "data/raw", out_dir: str = "data/processed") -> None:
    df = pd.read_csv(os.path.join(in_dir, f"{name}.csv"))
    df["content"] = df["content"].map(normalize_text)
    df = df[df["content"].str.len() > 0]          # tweets that became empty carry no signal
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, f"{name}_processed.csv"), index=False)
    logger.info("%s: %d rows written", name, len(df))


def main() -> None:
    try:
        for split in ("train", "test"):
            process_split(split)
    except Exception:
        logger.exception("preprocessing failed")
        raise


if __name__ == "__main__":
    main()
