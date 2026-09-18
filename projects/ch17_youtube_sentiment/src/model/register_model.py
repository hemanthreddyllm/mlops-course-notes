"""Stage 5 · model registration
Registers the model logged by model_evaluation.py and points the alias `staging` at it.
(The course uses the old stage API, transition_model_version_stage(..., "Staging");
MLflow 3 replaces stages with aliases.)"""
import json
import os
import sys
from pathlib import Path

import mlflow
from mlflow import MlflowClient

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.common import ROOT, get_logger  # noqa: E402

logger = get_logger("register_model")
os.environ.setdefault("MLFLOW_TRACKING_URI", "http://127.0.0.1:5050")
MODEL_NAME = "yt_chrome_plugin_model"


def main() -> None:
    try:
        info = json.loads((ROOT / "experiment_info.json").read_text())
        version = mlflow.register_model(info["model_uri"], MODEL_NAME)
        client = MlflowClient()
        client.set_registered_model_alias(MODEL_NAME, "staging", version.version)
        client.set_model_version_tag(MODEL_NAME, version.version, "accuracy", f"{info['accuracy']:.4f}")
        logger.info("registered %s v%s (alias: staging) from run %s",
                    MODEL_NAME, version.version, info["run_id"])
    except Exception:
        logger.exception("model registration failed")
        raise


if __name__ == "__main__":
    main()
