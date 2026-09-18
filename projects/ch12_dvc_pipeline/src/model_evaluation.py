"""Stage 5 · Model evaluation
Scores model.pkl on the test features.
INPUT: model.pkl + data/features/test_bow.csv   OUTPUT: metrics.json
"""
import json
import os
import pickle
import sys

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

sys.path.append(os.path.dirname(__file__))
from utils import get_logger  # noqa: E402

logger = get_logger("model_evaluation")


def main() -> None:
    try:
        with open("model.pkl", "rb") as f:
            model = pickle.load(f)
        df = pd.read_csv("data/features/test_bow.csv")
        X, y = df.drop(columns=["label"]).values, df["label"].values

        pred = model.predict(X)
        proba = model.predict_proba(X)[:, 1]
        metrics = {
            "accuracy": round(float(accuracy_score(y, pred)), 4),
            "precision": round(float(precision_score(y, pred)), 4),
            "recall": round(float(recall_score(y, pred)), 4),
            "auc": round(float(roc_auc_score(y, proba)), 4),
        }
        with open("metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)
        logger.info("metrics: %s", metrics)
    except Exception:
        logger.exception("model evaluation failed")
        raise


if __name__ == "__main__":
    main()
