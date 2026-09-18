"""Stage 4 · model evaluation, tracked in MLflow
Scores the model on the test set, logs params / metrics / confusion matrix / model / vectorizer
to the MLflow server (MLFLOW_TRACKING_URI), and writes experiment_info.json for registration."""
import json
import os
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import mlflow.lightgbm  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from mlflow.models import infer_signature  # noqa: E402
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.common import ROOT, get_logger, load_params  # noqa: E402

logger = get_logger("model_evaluation")
os.environ.setdefault("MLFLOW_TRACKING_URI", "http://127.0.0.1:5050")
EXPERIMENT = "dvc-pipeline-runs"
LABELS = {-1: "negative", 0: "neutral", 1: "positive"}


def main() -> None:
    try:
        mlflow.set_experiment(EXPERIMENT)
        params = load_params()
        model = pickle.load(open(ROOT / "lgbm_model.pkl", "rb"))
        vectorizer = pickle.load(open(ROOT / "tfidf_vectorizer.pkl", "rb"))
        test = pd.read_csv(ROOT / "data" / "interim" / "test_processed.csv").dropna(subset=["clean_comment"])
        X_test = vectorizer.transform(test["clean_comment"])
        y_test = test["category"].values

        with mlflow.start_run(run_name="lightgbm-tfidf-pipeline") as run:
            mlflow.set_tags({"model_type": "LightGBM", "task": "sentiment", "source": "dvc pipeline"})
            for section, values in params.items():
                mlflow.log_params({f"{section}.{k}": str(v) for k, v in values.items()})

            pred = model.predict(X_test)
            report = classification_report(y_test, pred, output_dict=True)
            mlflow.log_metric("accuracy", accuracy_score(y_test, pred))
            for label, name in LABELS.items():
                stats = report[str(label)]
                mlflow.log_metrics({f"{name}_precision": stats["precision"],
                                    f"{name}_recall": stats["recall"],
                                    f"{name}_f1": stats["f1-score"]})
            mlflow.log_metric("macro_f1", report["macro avg"]["f1-score"])

            # confusion matrix as an artifact
            cm = confusion_matrix(y_test, pred, labels=list(LABELS))
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=LABELS.values(), yticklabels=LABELS.values(), ax=ax)
            ax.set_xlabel("predicted"); ax.set_ylabel("actual"); ax.set_title("Confusion matrix (test set)")
            fig.tight_layout()
            cm_path = ROOT / "reports" / "confusion_matrix.png"
            cm_path.parent.mkdir(exist_ok=True)
            fig.savefig(cm_path, dpi=120)
            plt.close(fig)
            mlflow.log_artifact(str(cm_path))

            # the model (with signature) and the vectorizer it needs
            sample = X_test[:5].toarray()
            signature = infer_signature(pd.DataFrame(sample, columns=vectorizer.get_feature_names_out()),
                                        model.predict(sample))
            # mlflow.lightgbm, not mlflow.sklearn: MLflow 3 saves sklearn models with skops, which rejects LightGBM types
            info = mlflow.lightgbm.log_model(model, name="lgbm_model", signature=signature)
            mlflow.log_artifact(str(ROOT / "tfidf_vectorizer.pkl"))

            with open(ROOT / "experiment_info.json", "w") as f:
                json.dump({"run_id": run.info.run_id, "model_uri": info.model_uri,
                           "accuracy": report["accuracy"], "macro_f1": report["macro avg"]["f1-score"]}, f, indent=4)
            logger.info("accuracy=%.4f macro_f1=%.4f run=%s", report["accuracy"], report["macro avg"]["f1-score"],
                        run.info.run_id)
    except Exception:
        logger.exception("model evaluation failed")
        raise


if __name__ == "__main__":
    main()
