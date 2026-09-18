# MLOps course — notes and hands-on projects

Study archive for the 12-hour **Ultimate MLOps Full Course** (YouTube `w71RHxAWxaM`): written notes for every
chapter, plus working code for every chapter that involves code — all of it executed, with real outputs saved.

📖 **Notes site:** https://hemanthreddyllm.github.io/mlops-course-notes/ — *public, even though this repo is private* (see `docs/HANDOFF.md` § Publishing).
🧭 **New machine? Start with [`docs/HANDOFF.md`](docs/HANDOFF.md).**

---

## What's here

```
notes/          one HTML page per chapter + index + after-course to-do list
projects/       the hands-on work, one folder per chapter (notebooks with outputs)
transcript/     the course transcript, split per chapter (source material)
docs/           HANDOFF.md (status + how to run things) and the Claude memory export
CLAUDE.md       working agreement for Claude Code
```

Open `notes/index.html` locally, or use the hosted site. Every chapter page has: concepts and diagrams, the
instructor's steps with video timestamps, **precision notes** (mistakes and outdated advice in the video),
a **hands-on** section linking to the executed code, interview questions and a recap.

## The chapters

| # | Chapter | Hands-on |
|---|---|---|
| 01–03 | Introduction, real-world analogy, why MLOps | – |
| 04–06 | Linux for MLOps, EC2, commands | `ch06_linux_commands` |
| 07–10 | Git and GitHub: repos, code management, branches | `ch08_10_git` |
| 11–12 | DVC: pipelines and tracking | `ch12_dvc_pipeline` |
| 13 | Cloud fundamentals | `ch13_cloud_fundamentals` (moto) |
| 14–15 | MLflow tracking, DagsHub | `ch14_…`, `ch15_…` |
| 16 | Docker | `ch16_docker` |
| 17 | **End-to-end project:** YouTube sentiment (EDA → experiments → DVC pipeline → registry → Flask API → Chrome extension) | `ch17_youtube_sentiment` |
| 18 | CI/CD with GitHub Actions | `ch18_github_actions` |
| 19–20 | Jenkins, CircleCI | *placeholders — deliberately skipped* |
| 21 | Kubernetes | `ch21_kubernetes` |
| 22 | AWS SageMaker | `ch22_sagemaker` |
| 23 | Grafana + Prometheus monitoring | `ch23_grafana` |

## Quick start

```bash
cd projects
python -m venv .venv                 # Windows: py -m venv .venv
.venv/bin/pip install -r ../docs/requirements.txt    # Windows: .venv\Scripts\pip install -r ..\docs\requirements.txt
.venv/bin/jupyter lab                # open any projects/chNN_*/ notebook
```

Nothing in the repo needs a cloud account to run: AWS is mocked with `moto`, MLflow and Prometheus/Grafana run
locally, and Kubernetes and GitHub Actions are simulated. The steps that *do* need your own accounts are collected
in `notes/todo_after_course.html` — 39 tasks with checkboxes, grouped by chapter.

## Notes

* Generated data, models, the virtualenv and downloaded tool binaries are **not** committed (see `.gitignore`);
  the notebooks recreate them.
* `transcript/` is source material for the notes and is not published with the site.
