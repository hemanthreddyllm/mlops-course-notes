# Handoff — what exists, what runs, what's next

Written at the end of the build on macOS (Apple silicon), for continuing on another machine (Windows).
If you are Claude Code: read this together with `/CLAUDE.md`, then say what you plan to do before changing anything.

---

## 1 · Status

**Every chapter of the course is written and its hands-on work executed.** Nothing is half-finished.

| Notes page | Video time | Hands-on project | Ran for real? |
|---|---|---|---|
| `01_introduction_overview` … `03_intro_to_mlops_importance` | 00:00 – 00:29 | – | concepts only |
| `04_why_linux_for_mlops`, `05_linux_on_aws_ec2` | 00:29 – 01:00 | – | concepts + EC2 walkthrough (no account used) |
| `06_linux_commands` | 01:00 – 01:20 | `ch06_linux_commands` | ✅ notebook |
| `07_git_and_github_intro` … `10_git_branch_management` | 01:20 – 02:05 | `ch08_10_git` | ✅ local bare repo stands in for GitHub |
| `11_introduction_to_dvc`, `12_ml_pipelines_with_dvc` | 02:05 – 02:47 | `ch12_dvc_pipeline` | ✅ 5-stage DVC pipeline, accuracy 0.7401 |
| `13_cloud_fundamentals_for_mlops` | 02:47 – 03:07 | `ch13_cloud_fundamentals` | ✅ S3/IAM/EC2/ECR against **moto** |
| `14_intro_to_mlflow_experiment_tracking` | 03:07 – 03:42 | `ch14_mlflow_experiment_tracking` | ✅ 9 runs, local MLflow + UI screenshots |
| `15_mlflow_tracking_with_dagshub` | 03:42 – 04:05 | `ch15_mlflow_dagshub` | ✅ local MLflow server stands in for DagsHub; model served |
| `16_docker` | 04:05 – 04:52 | `ch16_docker` | ⚠️ apps run; **Docker steps skipped** (no Docker on the build machine) |
| `17_end_to_end_youtube_sentiment` | 04:52 – 07:20 | `ch17_youtube_sentiment` | ✅ EDA + 31 MLflow runs + DVC pipeline + registry + Flask API + extension preview |
| `18_cicd_github_actions` | 07:20 – 07:43 | `ch18_github_actions` | ✅ workflow linted + executed by a local toy runner |
| `19_cicd_jenkins`, `20_cicd_circleci` | 07:43 – 10:18 | – | **placeholders by choice** (one CI tool is enough) |
| `21_kubernetes` | 10:18 – 10:46 | `ch21_kubernetes` | ✅ manifests validated (kubeconform) + control-loop simulator |
| `22_sagemaker` | 10:46 – 11:09 | `ch22_sagemaker` | ✅ full train→deploy→invoke→delete against **moto** |
| `23_grafana` | 11:09 – end | `ch23_grafana` | ✅ **real Prometheus + Grafana** watching the ch17 API |

Transcript ↔ chapter mapping: transcript files `16–19` are all Docker (notes ch16); `20` is the end-to-end project
(ch17); `21+22` are CI/CD intro + GitHub Actions (ch18); `23`→ch19, `24`→ch20, `25`→ch21, `26`→ch22, `27`→ch23.

## 2 · What is deliberately *not* done

* **Anything needing your own accounts**: AWS, GCP, DagsHub, Docker Hub, a real GitHub push, a YouTube Data API key.
  All of it is listed as 39 checkbox tasks in `notes/todo_after_course.html`, grouped by chapter, with steps.
* **Docker** was never installed, so ch16's image builds, ch17's image, ch18's real pipeline and ch23's container
  version print their commands and skip. These are the highest-value things to run first on a machine with Docker.
* **Jenkins and CircleCI** notes (ch19, ch20) — placeholders; ask if you want them.

## 3 · Setting up on Windows

```powershell
git clone <this repo>
cd MLOps\projects
py -m venv .venv
.venv\Scripts\pip install -r ..\docs\requirements.txt
.venv\Scripts\jupyter lab
```

Then open any `projects\chNN_*\*.ipynb` and Run All. Notebooks are committed **with their outputs**, so you can read
them without running anything.

### Platform differences to expect

| Area | Why it breaks on Windows | Fix |
|---|---|---|
| `pkill` / `xattr` / `bash -c` in notebooks | POSIX-only | run in **WSL** or Git Bash, or swap for `taskkill` / `psutil` |
| `projects/_tools/bin/` (actionlint, kubeconform) | macOS-arm64 binaries, not committed | download the windows build from each project's GitHub releases into the same folder |
| ch23 `stack.py` tool download | picks `darwin`/`linux` only | add a `windows-amd64` branch, or run the chapter under WSL |
| ch18 `toy_runner.py` | executes steps with `bash` | Git Bash on PATH, or WSL |
| Port 5001 for the API | chosen because macOS AirPlay owns 5000 | fine as-is; 5000 also works on Windows |
| LightGBM/XGBoost `libomp` fix in `projects/README.md` | macOS-only problem | ignore; pip wheels work on Windows |
| Headless Chrome screenshots | app path differs | `C:\Program Files\Google\Chrome\Application\chrome.exe` |

### Re-running the heavy chapters

| Chapter | Command | Time | Notes |
|---|---|---|---|
| 17 | run `notebooks/1_…`, `notebooks/2_…`, then `pipeline_walkthrough.ipynb` | ~10 min | starts a local MLflow server on :5050; downloads the Reddit dataset |
| 18 | `github_actions_walkthrough.ipynb` | ~2 min | clones ch17 as a fixture; needs ch17's git repo, which its walkthrough creates |
| 21 | `kubernetes_walkthrough.ipynb` | ~1 min | no cluster needed |
| 22 | `sagemaker_walkthrough.ipynb` | ~1 min | moto only |
| 23 | `grafana_walkthrough.ipynb` | ~8 min | downloads Prometheus + Grafana (~1.5 GB) to `~/.cache/mlops-course-tools`; needs ch17's `lgbm_model.pkl`, so run ch17 first |

Generated artefacts (data, `*.pkl`, `mlflow.db`, DVC caches) are gitignored — the notebooks recreate them.

## 4 · How the notes are maintained

`CLAUDE.md` has the full working agreement. In short: notebooks come from `projects/_tools/build_chNN.py`
(edit the generator, re-execute, then re-run the matching `inject_chNN_results.py` to refresh the numbers and
images inside the HTML page). Every chapter also touches `notes/index.html`, the previous page's pager,
`projects/README.md` and the to-do page.

## 5 · Suggested next steps

1. **Install Docker Desktop** and re-run ch16 → ch17 → ch18 so the container steps execute for real.
2. Work down `notes/todo_after_course.html` — it is ordered so each chapter's account-based steps build on the last.
3. The capstone at the bottom of that list (deploy the Chapter 17 API to EC2 through GitHub Actions) is the single
   best interview story: it uses DVC, MLflow, Docker, ECR, EC2, CI/CD and monitoring together.
4. Optional: ask for the Jenkins/CircleCI chapters, or for a deeper drift-monitoring chapter beyond ch23.

## 6 · Claude's memory

`docs/memory/` holds the memory entries from the machine where this was built (working agreement, the to-do-list
habit, and the chapter-scope decisions). Claude Code's memory is per-machine, so on a new machine either point
Claude at that folder or let it re-create the entries from `CLAUDE.md`.
