"""Stage 4 · Model building
Trains an XGBoost classifier on the bag-of-words features.
INPUT: data/features   OUTPUT: model.pkl
"""
import os
import pickle
import sys

import pandas as pd
from xgboost import XGBClassifier

sys.path.append(os.path.dirname(__file__))
from utils import get_logger  # noqa: E402

logger = get_logger("model_building")

PARAMS = {"n_estimators": 100, "learning_rate": 0.1, "eval_metric": "logloss", "random_state": 42}


def main() -> None:
    try:
        df = pd.read_csv("data/features/train_bow.csv")
        X, y = df.drop(columns=["label"]).values, df["label"].values
        model = XGBClassifier(**PARAMS)
        model.fit(X, y)
        with open("model.pkl", "wb") as f:
            pickle.dump(model, f)
        logger.info("trained on %d rows with %s", len(y), PARAMS)
    except Exception:
        logger.exception("model building failed")
        raise


if __name__ == "__main__":
    main()
