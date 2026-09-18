"""Stage 3 · model building
TF-IDF (settings chosen in experiments 2–3) → SMOTE oversampling (experiment 4)
→ LightGBM with tuned parameters (experiment 5). Writes lgbm_model.pkl + tfidf_vectorizer.pkl."""
import pickle
import sys
from pathlib import Path

import lightgbm as lgb
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.common import ROOT, get_logger, load_params  # noqa: E402

logger = get_logger("model_building")


def main() -> None:
    try:
        p = load_params("model_building")
        train = pd.read_csv(ROOT / "data" / "interim" / "train_processed.csv").dropna(subset=["clean_comment"])

        vectorizer = TfidfVectorizer(ngram_range=tuple(p["ngram_range"]), max_features=p["max_features"])
        X = vectorizer.fit_transform(train["clean_comment"])
        y = train["category"].values
        logger.info("TF-IDF matrix: %s", X.shape)

        if p.get("oversample", True):
            X, y = SMOTE(random_state=42).fit_resample(X, y)
            logger.info("after SMOTE: %s rows, class counts=%s", X.shape[0], pd.Series(y).value_counts().to_dict())

        model = lgb.LGBMClassifier(
            objective="multiclass", num_class=3, metric="multi_logloss",
            learning_rate=p["learning_rate"], max_depth=p["max_depth"], n_estimators=p["n_estimators"],
            num_leaves=p.get("num_leaves", 31), reg_alpha=p.get("reg_alpha", 0.0), reg_lambda=p.get("reg_lambda", 0.0),
            random_state=42, n_jobs=-1, verbose=-1,
        )
        model.fit(X, y)

        with open(ROOT / "lgbm_model.pkl", "wb") as f:
            pickle.dump(model, f)
        with open(ROOT / "tfidf_vectorizer.pkl", "wb") as f:
            pickle.dump(vectorizer, f)
        logger.info("saved lgbm_model.pkl and tfidf_vectorizer.pkl")
    except Exception:
        logger.exception("model building failed")
        raise


if __name__ == "__main__":
    main()
