"""Helpers shared by the pipeline stages and the Flask API:
logging, params.yaml loading, and the ONE text-cleaning function
(training and serving must clean text identically)."""
import logging
import os
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]          # project root

try:  # python.org macOS builds lack a CA bundle → HTTPS downloads fail without this
    import certifi
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
except ImportError:
    pass


def get_logger(name: str) -> logging.Logger:
    (ROOT / "logs").mkdir(exist_ok=True)
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")
        for h in (logging.StreamHandler(), logging.FileHandler(ROOT / "logs" / f"{name}.log")):
            h.setFormatter(fmt)
            logger.addHandler(h)
    return logger


def load_params(section: str | None = None) -> dict:
    with open(ROOT / "params.yaml") as f:
        params = yaml.safe_load(f)
    return params[section] if section else params


# ── text cleaning ────────────────────────────────────────────────────────────
_NLTK_READY = False
_STOP_WORDS: set[str] = set()
_LEMMATIZER = None
# negations and contrast words carry sentiment ("not good"), so they are kept
KEEP_WORDS = {"not", "no", "but", "however", "yet", "nor", "never"}


def _ensure_nltk() -> None:
    global _NLTK_READY, _STOP_WORDS, _LEMMATIZER
    if _NLTK_READY:
        return
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    for pkg in ("stopwords", "wordnet", "omw-1.4"):
        try:
            nltk.data.find(f"corpora/{pkg}")
        except LookupError:
            nltk.download(pkg, quiet=True)
    _STOP_WORDS = set(stopwords.words("english")) - KEEP_WORDS
    _LEMMATIZER = WordNetLemmatizer()
    _NLTK_READY = True


def preprocess_comment(comment: str) -> str:
    """lowercase → strip → drop newlines → keep letters/digits/basic punctuation
    → remove stop words (except negations) → lemmatise"""
    _ensure_nltk()
    text = str(comment).lower().strip()
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"[^a-z0-9\s!?.,]", "", text)          # removes emojis, non-English script, symbols
    words = [w for w in text.split() if w not in _STOP_WORDS]
    return " ".join(_LEMMATIZER.lemmatize(w) for w in words)
