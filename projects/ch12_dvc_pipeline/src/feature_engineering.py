"""Stage 3 · Feature engineering
Bag-of-words with CountVectorizer (fit on train only).
INPUT: data/processed   OUTPUT: data/features (train_bow.csv, test_bow.csv, vectorizer.pkl)
"""
import os
import pickle
import sys

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

sys.path.append(os.path.dirname(__file__))
from utils import get_logger  # noqa: E402

logger = get_logger("feature_engineering")

MAX_FEATURES = 1000


def main() -> None:
    try:
        train = pd.read_csv("data/processed/train_processed.csv").fillna("")
        test = pd.read_csv("data/processed/test_processed.csv").fillna("")

        vectorizer = CountVectorizer(max_features=MAX_FEATURES)
        X_train = vectorizer.fit_transform(train["content"])   # learn the vocabulary from TRAIN only
        X_test = vectorizer.transform(test["content"])

        os.makedirs("data/features", exist_ok=True)
        for name, X, y in (("train", X_train, train["sentiment"]), ("test", X_test, test["sentiment"])):
            out = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
            out["label"] = y.values
            out.to_csv(f"data/features/{name}_bow.csv", index=False)
            logger.info("%s features: %s", name, X.shape)

        # the prediction pipeline must reuse the SAME fitted vectorizer
        with open("data/features/vectorizer.pkl", "wb") as f:
            pickle.dump(vectorizer, f)
    except Exception:
        logger.exception("feature engineering failed")
        raise


if __name__ == "__main__":
    main()
