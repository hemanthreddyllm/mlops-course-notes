"""Copy the Chapter 15 champion model out of MLflow into model/model.joblib,
so the Docker image doesn't need MLflow or the tracking database.

Chapter 15 logged through a tracking *server*, so artifact locations look like
`mlflow-artifacts:/…`; the server stores the files under ch15's `mlartifacts/<exp>/models/<model id>/artifacts`. We read that folder directly, so no server needs to be running."""
import os
from pathlib import Path

import joblib

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
import mlflow  # noqa: E402
from mlflow import MlflowClient  # noqa: E402

HERE = Path(__file__).parent
CH15 = HERE.parent.parent / "ch15_mlflow_dagshub"

mlflow.set_tracking_uri(f"sqlite:///{CH15 / 'mlflow.db'}")
client = MlflowClient()
version = client.get_model_version_by_alias("ElasticnetWineModel", "champion")
model_id = version.source.removeprefix("models:/")          # e.g. m-2c3a…
matches = list((CH15 / "mlartifacts").glob(f"*/models/{model_id}/artifacts"))
assert matches, f"artifacts for {model_id} not found under {CH15 / 'mlartifacts'}"
model = mlflow.sklearn.load_model(str(matches[0]))

out = HERE / "model" / "model.joblib"
out.parent.mkdir(exist_ok=True)
joblib.dump(model, out)
print(f"exported ElasticnetWineModel v{version.version} (@champion) → {out.relative_to(HERE)} "
      f"({out.stat().st_size} bytes) | alpha={model.alpha}, l1_ratio={model.l1_ratio}")
