# MLOps course: hands-on projects

Runnable companions to the HTML notes in `../notes/`. Each coding chapter has a notebook or a small project here. Every notebook has already been **executed**, so you can read the outputs without running anything.

| Chapter | Folder | What's inside |
|---|---|---|
| 06 · Linux commands | `ch06_linux_commands/` | `linux_commands_practice.ipynb`: every command from the video run in a throw-away `sandbox/` |
| 08–10 · Git & GitHub | `ch08_10_git/` | `git_workflow_practice.ipynb`: clone → status/add/commit/push/pull → rollback → branches → merge → conflict, all offline (a local bare repo plays GitHub) |
| 12 · DVC pipelines | `ch12_dvc_pipeline/` | the full project: `notebooks/01_experiment_twitter_emotion.ipynb` (the notebook version), `src/` (5 pipeline components), `dvc.yaml`, and `02_dvc_pipeline_walkthrough.ipynb` (git init → dvc init → dvc repro → tracking demos) |
| 13 · Cloud fundamentals | `ch13_cloud_fundamentals/` | `aws_services_offline.ipynb`: S3, EC2, ECR and IAM (least privilege) with real `boto3` code against **moto**, a local fake of AWS (no account, no cost) |
| 14 · MLflow intro | `ch14_mlflow_experiment_tracking/` | `experiment_tracking_intro.ipynb`: parameter counts (ML vs DL), manual "spreadsheet" tracking vs a 9-run MLflow grid, `search_runs`, parallel coordinates, autolog. Browse it with `mlflow ui --backend-store-uri sqlite:///mlflow.db` |
| 15 · MLflow + DagsHub | `ch15_mlflow_dagshub/` | `demo.py` (the course script; sends runs to DagsHub, any MLflow server, or ./mlruns) + `mlflow_remote_tracking_walkthrough.ipynb`: local `mlflow server` in place of DagsHub, the 3 video runs, compare, model registry `@champion`, **`mlflow models serve` REST call** |
| 16 · Docker | `ch16_docker/` | `calculator_app/` (Flask + Dockerfile), `wine_api/` (FastAPI serving the ch15 champion model + Dockerfile) and `docker_walkthrough.ipynb`. **Docker cells are skipped until Docker Desktop is installed** (see `notes/todo_after_course.html`) |
| 17 · End-to-end project | `ch17_youtube_sentiment/` | YouTube Sentiment Insights: `notebooks/1_preprocessing_eda.ipynb`, `notebooks/2_experiments.ipynb` (6 MLflow experiment rounds), `pipeline_walkthrough.ipynb` (DVC 5-stage pipeline → registry alias → Flask API → pytest → Docker), `src/`, `flask_api/app.py`, `yt_chrome_plugin_frontend/` (Chrome extension + offline `dev_preview.html`), `Dockerfile`, `.github/workflows/cicd.yaml` (used in ch18). Run the notebooks in order 1 → 2 → walkthrough; they share an MLflow server on port 5050 |
| 18 · CI/CD with GitHub Actions | `ch18_github_actions/` | `github_actions_walkthrough.ipynb` (anatomy, actionlint, triggers, fresh-checkout test, full workflow run with `toy_runner.py`, secret masking, ECR URIs with moto, redeploy conflict), `ec2_setup.sh`. Uses the improved `ch17_youtube_sentiment/.github/workflows/cicd.yaml`. **No GitHub/AWS account used: cloud and Docker steps are printed, not run** |
| 21 · Kubernetes | `ch21_kubernetes/` | `manifests/` (Namespace, Deployment with probes + resources + securityContext, LoadBalancer Service, HPA, PDB — validated with `kubeconform`), `toy_k8s.py` (control-loop simulator), `kubernetes_walkthrough.ipynb` (self-healing, load balancing, rolling update, failed release + undo, HPA, node failure). **No cluster needed** |
| 22 · AWS SageMaker | `ch22_sagemaker/` | `sagemaker_walkthrough.ipynb`: the video's flow (S3 → `CreateTrainingJob` → `CreateModel`/`EndpointConfig`/`Endpoint` → `InvokeEndpoint` → delete) run against **moto**, plus a local scikit-learn reference model and a serving-cost comparison. **No AWS account needed** |
| 23 · Grafana monitoring | `ch23_grafana/` | A real Prometheus + Grafana stack watching the Chapter 17 API: `instrumented_api.py` (adds `/metrics`), `prometheus.yml`, `alerts.yml`, provisioned Grafana data source + `mlops.json` dashboard, `traffic.py` (baseline → spike → errors → drift), `stack.py`, and `grafana_walkthrough.ipynb`. Downloads Prometheus + Grafana (~1.5 GB) into `~/.cache/mlops-course-tools` — `rm -rf` it when done |

`../course_repo/` is a clone of the instructor's repository (github.com/entbappy/Ultimate-MLOps-Full-Course), kept for reference.
The code here is written separately and follows the same structure.

## Setup (already done on this machine)

```bash
cd projects
python3 -m venv .venv
source .venv/bin/activate
pip install jupyter nbconvert ipykernel pandas scikit-learn xgboost nltk dvc pyyaml matplotlib certifi mlflow dagshub flask gunicorn fastapi "uvicorn[standard]" httpx joblib boto3 "moto[s3,iam,ec2,ecr,sts]" lightgbm imbalanced-learn optuna wordcloud seaborn flask-cors pytest
```

In VS Code, open a notebook and pick the **`.venv`** interpreter as the kernel.

### Two macOS fixes that were needed
1. **XGBoost can't load (`libomp.dylib` not found).** Normally you'd run `brew install libomp`. Without Homebrew, reuse the copy that scikit-learn ships:
   ```bash
   SP=.venv/lib/python3.13/site-packages
   cp $SP/sklearn/.dylibs/libomp.dylib $SP/xgboost/lib/
   install_name_tool -add_rpath @loader_path $SP/xgboost/lib/libxgboost.dylib
   codesign -f -s - $SP/xgboost/lib/libxgboost.dylib
   ```
   (Reinstalling xgboost undoes this, so repeat it afterwards.) **LightGBM** (Chapter 17) needs the same fix: replace `xgboost/lib/libxgboost.dylib` with `lightgbm/lib/lib_lightgbm.dylib` and copy `libomp.dylib` into `lightgbm/lib/`.
2. **HTTPS downloads fail (`CERTIFICATE_VERIFY_FAILED`).** python.org builds of Python on macOS ship without CA certificates. The project code sets `SSL_CERT_FILE` to `certifi`'s bundle. The alternative is to run `/Applications/Python 3.13/Install Certificates.command` once.

## Re-running

* Open any notebook and choose **Run All**. Each one resets its own sandbox or generated files in its first cell.
* The DVC project can also be run from a terminal:
  ```bash
  cd ch12_dvc_pipeline
  dvc repro && dvc dag && dvc metrics show
  ```
* `_tools/build_notebooks.py` (ch06/08–10/12), `_tools/build_ch13.py`, `_tools/build_ch14.py`, `_tools/build_ch15.py`, `_tools/build_ch16.py`, `_tools/build_ch17.py`, `_tools/build_ch18.py`, `_tools/build_ch21.py`, `_tools/build_ch22.py` and `_tools/build_ch23.py` regenerate the notebooks (without outputs). After regenerating, execute them with
  `jupyter nbconvert --to notebook --execute --inplace <notebook>`.

### actionlint · kubeconform
`_tools/bin/actionlint` is the prebuilt [actionlint](https://github.com/rhysd/actionlint) binary (darwin/arm64, downloaded from its releases page; the pip package needs Go to build). Lint a workflow with `_tools/bin/actionlint path/to/workflow.yaml`.
`_tools/bin/kubeconform` (same idea, from its GitHub releases) validates Kubernetes manifests against the official schemas: `_tools/bin/kubeconform -strict -summary ch21_kubernetes/manifests/*.yaml`.
