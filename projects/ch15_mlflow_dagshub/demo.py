"""ElasticNet on the red-wine-quality data, tracked with MLflow.

Usage:
    python demo.py [alpha] [l1_ratio]          # defaults: 0.5 0.5

Where the runs go (checked in this order):
    1. DagsHub   - set DAGSHUB_REPO_OWNER and DAGSHUB_REPO_NAME (log in once with `dagshub login`)
    2. Any MLflow tracking server - set MLFLOW_TRACKING_URI, e.g. http://127.0.0.1:5000
    3. Otherwise - a local ./mlruns folder (model registration is skipped there)
"""
import logging
import os
import sys
import warnings
from urllib.parse import urlparse

import certifi
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

os.environ.setdefault("SSL_CERT_FILE", certifi.where())   # macOS python.org builds need this for HTTPS
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402
from mlflow.models import infer_signature  # noqa: E402

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("demo")

DATA_URL = "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv"
EXPERIMENT = "wine-quality-elasticnet"
REGISTERED_MODEL = "ElasticnetWineModel"


def configure_tracking() -> str:
    """Point MLflow at DagsHub, an explicit server, or the local folder."""
    owner, repo = os.getenv("DAGSHUB_REPO_OWNER"), os.getenv("DAGSHUB_REPO_NAME")
    if owner and repo:
        import dagshub
        dagshub.init(repo_owner=owner, repo_name=repo, mlflow=True)   # sets the tracking URI for us
    # otherwise MLflow reads MLFLOW_TRACKING_URI itself, or falls back to ./mlruns
    mlflow.set_experiment(EXPERIMENT)
    return mlflow.get_tracking_uri()


def eval_metrics(actual, pred):
    rmse = float(np.sqrt(mean_squared_error(actual, pred)))
    mae = float(mean_absolute_error(actual, pred))
    r2 = float(r2_score(actual, pred))
    return rmse, mae, r2


def main() -> None:
    warnings.filterwarnings("ignore")
    np.random.seed(40)            # same seed as the course demo, so the split is reproducible

    try:
        data = pd.read_csv(DATA_URL, sep=";")    # the file is semicolon-separated
    except Exception:
        logger.exception("could not download the training CSV; check your internet connection")
        raise

    train, test = train_test_split(data)          # 75% / 25%
    train_x, test_x = train.drop(columns=["quality"]), test.drop(columns=["quality"])
    train_y, test_y = train[["quality"]], test[["quality"]]

    # hyperparameters from the command line (sys.argv), with defaults
    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    tracking_uri = configure_tracking()

    with mlflow.start_run(run_name=f"alpha={alpha}_l1={l1_ratio}") as run:
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        model.fit(train_x, train_y)
        predictions = model.predict(test_x)
        rmse, mae, r2 = eval_metrics(test_y, predictions)

        print(f"ElasticNet (alpha={alpha:g}, l1_ratio={l1_ratio:g})")
        print(f"  RMSE: {rmse:.4f}\n  MAE:  {mae:.4f}\n  R2:   {r2:.4f}")

        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        signature = infer_signature(train_x, model.predict(train_x))
        if urlparse(tracking_uri).scheme == "file":
            # a plain local folder has no model registry: just store the model with the run
            mlflow.sklearn.log_model(model, name="model", signature=signature)
        else:
            # a tracking server (local server or DagsHub) also registers a new model version
            mlflow.sklearn.log_model(model, name="model", signature=signature,
                                     registered_model_name=REGISTERED_MODEL)

        print(f"  run id: {run.info.run_id}\n  tracked at: {tracking_uri}")


if __name__ == "__main__":
    main()
