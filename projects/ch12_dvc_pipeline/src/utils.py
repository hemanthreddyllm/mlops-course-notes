"""Shared helpers for the pipeline components: logging setup + HTTPS certificates."""
import logging
import os

try:
    # python.org builds of Python on macOS ship without a CA bundle, so HTTPS downloads
    # (pandas.read_csv(url), nltk.download) fail with CERTIFICATE_VERIFY_FAILED. Point them at certifi's.
    import certifi

    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
except ImportError:
    pass


def get_logger(name: str) -> logging.Logger:
    """Log to the console and to logs/<name>.log (the instructor's 'log to a file instead of print')."""
    os.makedirs("logs", exist_ok=True)
    logger = logging.getLogger(name)
    if logger.handlers:  # already configured
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")
    for handler in (logging.StreamHandler(), logging.FileHandler(f"logs/{name}.log")):
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger
