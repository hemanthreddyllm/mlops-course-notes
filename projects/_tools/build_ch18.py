"""Generate the Chapter 18 notebook: GitHub Actions CI/CD, explored and simulated locally."""
import os
import nbformat as nbf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "ch18_github_actions", "github_actions_walkthrough.ipynb")


def md(s): return nbf.v4.new_markdown_cell(s.strip("\n"))
def py(s): return nbf.v4.new_code_cell(s.strip("\n"))


cells = [
md(r"""
# Chapter 18 · CI/CD with GitHub Actions: walkthrough

Companion to `notes/18_cicd_github_actions.html` (video 07:20:16 – 07:42:46).

In the video, a push to GitHub triggers a workflow with three jobs:
1. **CI**: lint and tests (only `echo` placeholders in the course).
2. **Continuous delivery**: build the Docker image and push it to **Amazon ECR**.
3. **Continuous deployment**: a **self-hosted runner on EC2** pulls the image and runs it on port 8080.

A real run needs your GitHub repo and an AWS account, so this notebook does everything that works offline:

| Part | What happens |
|---|---|
| 1 | read both workflows (the course's and our Chapter 17 one): triggers, jobs, `needs:` order |
| 2 | lint them with **actionlint** (the course file has outdated actions and a deprecated command) |
| 3 | which pushes start a run? (`branches`, `paths-ignore`) |
| 4 | a clean checkout has **no model files**, so `pytest` fails. That's why our CI rebuilds them with DVC |
| 5 | run the whole Chapter 17 workflow with a **toy runner** (`toy_runner.py`); cloud and Docker steps are shown, not run |
| 6 | secrets: masking, and why a job output containing a secret gets dropped |
| 7 | ECR names and URIs, simulated with **moto** |
| 8 | deploying twice: the course's commented-out lines vs a working replacement |
| 9 | what to run on the EC2 instance, and clean-up |
"""),
py(r"""
import os, sys, re, json, shutil, subprocess, textwrap
from pathlib import Path
import certifi
os.environ["SSL_CERT_FILE"] = certifi.where()
import yaml
import pandas as pd

HERE = Path.cwd()
assert (HERE / "toy_runner.py").exists(), "run this notebook from projects/ch18_github_actions"
sys.path.insert(0, str(HERE))
import importlib, toy_runner as tr
importlib.reload(tr)

PROJECTS = HERE.parent
ACTIONLINT = PROJECTS / "_tools" / "bin" / "actionlint"
OURS = PROJECTS / "ch17_youtube_sentiment" / ".github" / "workflows" / "cicd.yaml"
COURSE = PROJECTS.parent / "course_repo" / "CICD" / "github actions" / ".github" / "workflows" / "cicd.yaml"
SANDBOX = HERE / "sandbox"
shutil.rmtree(SANDBOX, ignore_errors=True); SANDBOX.mkdir()
for k in ("CLICOLOR", "CLICOLOR_FORCE"):
    os.environ.pop(k, None)
os.environ["PY_COLORS"] = "0"                      # plain pytest output
pd.set_option("display.max_colwidth", 120)
pd.set_option("display.width", 200)
print("our workflow   :", OURS.relative_to(PROJECTS))
print("course workflow:", COURSE.exists() and COURSE.relative_to(PROJECTS.parent))
print("actionlint     :", ACTIONLINT.exists() and subprocess.run([ACTIONLINT, "--version"], capture_output=True, text=True).stdout.split()[0])
"""),
md(r"""
## 1 · Anatomy of a workflow

A **workflow** is a YAML file in `.github/workflows/`, and the folder name must be exact.
* `on:` lists the **events** that start it.
* `jobs:` run on **runners**. Jobs run in parallel unless `needs:` chains them.
* Each job has **steps**. A step either `run:`s shell commands or `uses:` a published **action** (reusable code such as `actions/checkout`).
"""),
py(r"""
course_wf = tr.load(COURSE)
print(tr.describe(course_wf))
"""),
py(r"""
ours_wf = tr.load(OURS)
print(tr.describe(ours_wf))
"""),
py(r"""
# side by side: what each workflow does in each phase
def summary(wf):
    rows = []
    for j in tr.job_order(wf)[0]:
        spec = wf["jobs"][j]
        runs = [s["run"] for s in spec["steps"] if "run" in s]
        rows.append({"job": j, "runs-on": spec["runs-on"], "steps": len(spec["steps"]),
                     "does real testing?": any(re.search(r"pytest|unittest", r) for r in runs),
                     "tags image with commit?": any("github.sha" in r for r in runs) or any("github.sha" in json.dumps(s.get("env", {})) for s in spec["steps"]),
                     "replaces old container?": any("rm -f" in r or "docker stop" in r for r in runs)})
    return pd.DataFrame(rows)
display(summary(course_wf).assign(workflow="course"))
display(summary(ours_wf).assign(workflow="ours (ch17)"))
"""),
md(r"""
Things to notice in the course file:
* **CI only echoes** "Linting repository" and "Running unit tests". Nothing is actually checked.
* It always pushes `:latest`, so there's no record of which commit is running and no easy rollback.
* The "stop and remove container" step is **commented out**. See part 8 for why that breaks the second deployment.
"""),
md("## 2 · Lint the workflows with actionlint"),
py(r"""
def lint(path):
    res = subprocess.run([ACTIONLINT, "-no-color", "-oneline", str(path)], capture_output=True, text=True)
    rows = []
    for line in res.stdout.splitlines():
        m = re.match(r".*?:(\d+):(\d+): (.*) \[(.+)\]$", line)
        if m:
            rows.append({"line": int(m[1]), "rule": m[4], "message": m[3][:110]})
    return res.returncode, pd.DataFrame(rows)

rc, course_issues = lint(COURSE)
print(f"course workflow: {len(course_issues)} problems (exit code {rc})")
display(course_issues)
rc, our_issues = lint(OURS)
print(f"our workflow: {len(our_issues)} problems (exit code {rc})")
"""),
md(r"""
* `actions/checkout@v3` and `aws-actions/configure-aws-credentials@v1` run on Node versions GitHub no longer supports. Use `@v4`.
* `echo "::set-output name=…"` was deprecated in 2022. Write to the `$GITHUB_OUTPUT` file instead.

**Tip:** run `actionlint` locally or in a pre-commit hook. A broken workflow otherwise only shows up after you push.
"""),
md("## 3 · Which pushes start a run?"),
py(r"""
cases = [
    ("push", "main", ["src/model/model_building.py"]),
    ("push", "main", ["README.md"]),
    ("push", "main", ["README.md", "params.yaml"]),
    ("push", "feature/new-chart", ["flask_api/app.py"]),
    ("pull_request", "main", ["flask_api/app.py"]),
    ("workflow_dispatch", "main", []),
]
rows = []
for event, branch, files in cases:
    runs_course, why_course = tr.would_trigger(course_wf, event, branch, files)
    runs_ours, why_ours = tr.would_trigger(ours_wf, event, branch, files)
    rows.append({"event": event, "branch": branch, "changed": ", ".join(files) or "–",
                 "course": "▶ runs" if runs_course else "· no", "ours": "▶ runs" if runs_ours else "· no",
                 "reason (ours)": why_ours})
pd.DataFrame(rows)
"""),
md(r"""
`paths-ignore: README.md` skips a run only when **every** changed file is ignored.
Our workflow also has `workflow_dispatch`, which adds a **Run workflow** button in the Actions tab.
Neither workflow runs on pull requests. Adding `pull_request:` to the CI job (not the deploy jobs) is a common next step.
"""),
md(r"""
## 4 · Why CI must build the model: a fresh checkout
GitHub's runner starts from an **empty machine** and runs `actions/checkout`, which gets only what git tracks.
Our model files are DVC outputs, so git ignores them.
"""),
py(r"""
fresh = SANDBOX / "fresh_checkout"
subprocess.run(["git", "clone", "-q", str(PROJECTS / "ch17_youtube_sentiment"), str(fresh)], check=True)
print("model files in the checkout:", sorted(p.name for p in fresh.glob("*.pkl")) or "none")
res = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests"], cwd=fresh, capture_output=True, text=True)
print("pytest exit code:", res.returncode)
print("\n".join(l for l in (res.stdout + res.stderr).splitlines() if "Error" in l or "error" in l)[:600])
"""),
md(r"""
The tests fail on a clean machine even though they pass on the laptop: **"works on my machine"** again.
The first version of our Chapter 17 workflow had exactly this bug, and the toy runner caught it.
The fix, now in the workflow, is `dvc repro model_building` before `pytest`.
(With a DVC remote on S3 you would use `dvc pull` instead of retraining.)
"""),
md(r"""
## 5 · Run the whole Chapter 17 workflow with the toy runner
`toy_runner.py` does what a runner does, on a much smaller scale:
* checks out the repo into an empty folder
* runs each `run:` step with bash
* fills in `${{ … }}` expressions
* passes files between jobs as **artifacts**
* skips the jobs that depend on a failed job

It can't reach AWS and there's no Docker engine here, so those steps are **printed, not run**.
The secrets below are fake.
"""),
py(r"""
fake_secrets = {"AWS_ACCESS_KEY_ID": "AKIAFAKEFAKEFAKE1234", "AWS_SECRET_ACCESS_KEY": "fake/SecretKey+for+the+demo"}
repo_vars = {"AWS_REGION": "ap-south-1", "ECR_REPOSITORY_NAME": "yt-sentiment"}
runner = tr.Runner(SANDBOX / "work", secrets=fake_secrets, variables=repo_vars,
                   source_repo=PROJECTS / "ch17_youtube_sentiment", log_tail=4)
result = runner.run_workflow(ours_wf)
result
"""),
md(r"""
What a real run would add:
* the Docker build (about 2–4 min on GitHub's runner) and push
* the deploy job, which waits until the **self-hosted runner on EC2** picks it up
* the smoke test, which curls the new container

On GitHub, each job's log is in the **Actions** tab. A yellow dot means running, green means passed, red means failed.
"""),
md("## 6 · Secrets: masking, and outputs that contain secrets"),
py(r"""
print("secrets referenced by the course workflow:", tr.secrets_used(COURSE))
print("secrets referenced by our workflow      :", tr.secrets_used(OURS))
print("variables referenced by our workflow    :", sorted(set(re.findall(r"vars\.(\w+)", OURS.read_text()))))
"""),
py(r"""
# A tiny workflow that passes a value containing a secret from one job to the next
demo = yaml.safe_load('''
name: output-demo
on: push
jobs:
  build:
    runs-on: ubuntu-latest
    outputs:
      image: ${{ steps.tag.outputs.image }}
    steps:
      - id: tag
        run: |
          echo "building for repo ${{ secrets.ECR_REPOSITORY_NAME }}"
          echo "image=123456789012.dkr.ecr.ap-south-1.amazonaws.com/${{ secrets.ECR_REPOSITORY_NAME }}:abc123" >> "$GITHUB_OUTPUT"
  deploy:
    needs: build
    runs-on: self-hosted
    steps:
      - run: echo "deploying '${{ needs.build.outputs.image }}'"
''')
(SANDBOX / "work2").mkdir()
r = tr.Runner(SANDBOX / "work2", secrets={"ECR_REPOSITORY_NAME": "yt-sentiment"})
r.run_workflow(demo);
"""),
md(r"""
Two rules:
1. Secret values are replaced with `***` in logs. The same happens for anything else that equals a secret, which is why storing the repository name or region as a secret makes logs hard to read.
2. GitHub **drops a job output that contains a secret** ("Skip output … since it may contain secret"). The next job receives an empty string, and `docker pull ''` fails.

That's why our workflow keeps only the two credentials as **secrets**, and stores `AWS_REGION` and `ECR_REPOSITORY_NAME` as **variables**
(Settings → Secrets and variables → Actions → *Variables* tab). The course stores all five as secrets.
"""),
md("## 7 · ECR names and URIs (moto stands in for AWS)"),
py(r"""
import boto3
from moto import mock_aws

with mock_aws():
    ecr = boto3.client("ecr", region_name="ap-south-1",
                       aws_access_key_id="testing", aws_secret_access_key="testing")
    repo = ecr.create_repository(repositoryName="yt-sentiment",
                                 imageScanningConfiguration={"scanOnPush": True})["repository"]
    token = ecr.get_authorization_token()["authorizationData"][0]
    rows = [
        ("repository URI (the video copies this)", repo["repositoryUri"]),
        ("registry = AWS_ECR_LOGIN_URI", repo["repositoryUri"].split("/")[0]),
        ("ECR_REPOSITORY_NAME", repo["repositoryName"]),
        ("docker login endpoint", token["proxyEndpoint"]),
        ("image pushed by our workflow", repo["repositoryUri"] + ":<commit-sha>  and  :latest"),
    ]
pd.DataFrame(rows, columns=["what", "value"])
"""),
md(r"""
The registry host is `<account-id>.dkr.ecr.<region>.amazonaws.com`. The region must be the **same** in the ECR URI and in the region setting.
The video works in `ap-south-1` (Mumbai), while the course README example says `us-east-1`. A mismatch makes the login or push fail.
The ECR login token lasts **12 hours**, so the workflow logs in again on every run.
"""),
md(r"""
## 8 · Deploying twice: why the course's commented-out lines matter
This is a small imitation of Docker's container list. Container **names must be unique**.
"""),
py(r"""
class FakeDocker:
    def __init__(self): self.containers = {}
    def run(self, name, image, port):
        if name in self.containers:
            raise RuntimeError(f'Conflict. The container name "/{name}" is already in use')
        if any(c["port"] == port for c in self.containers.values()):
            raise RuntimeError(f"Bind for 0.0.0.0:{port} failed: port is already allocated")
        self.containers[name] = {"image": image, "port": port}
    def rm_f(self, name): self.containers.pop(name, None)

def course_deploy(d, image, stop_lines_enabled):
    if stop_lines_enabled and "cnncls" in d.containers:     # the commented-out step
        d.rm_f("cnncls")
    d.run("cnncls", image, 8080)

def our_deploy(d, image):
    d.rm_f("yt-sentiment")                                   # docker rm -f … || true
    d.run("yt-sentiment", image, 8080)

rows = []
for label, fn in [("course, lines commented", lambda d, i: course_deploy(d, i, False)),
                  ("course, lines enabled",   lambda d, i: course_deploy(d, i, True)),
                  ("ours",                    our_deploy)]:
    d = FakeDocker()
    for push, image in enumerate(["app:v1", "app:v2", "app:v3"], 1):
        try:
            fn(d, image); outcome = f"✔ running {image}"
        except RuntimeError as e:
            outcome = f"✘ {e}"
        rows.append({"deploy script": label, "push": push, "result": outcome,
                     "serving": ", ".join(c["image"] for c in d.containers.values())})
pd.DataFrame(rows)
"""),
md(r"""
With the lines commented out, the **second push fails**, and users keep getting the old version.
The instructor mentions this at the end of the video: uncomment the three lines before your second push.
Our workflow always removes the old container, so every push works.

Both versions have a few seconds of downtime between `rm` and `run`. Zero-downtime deploys need two containers behind a load balancer
(blue/green or rolling updates), which is what ECS, Kubernetes (Chapter 21) or App Runner do for you.
"""),
md("## 9 · On the EC2 instance, and clean-up"),
py(r"""
print((HERE / "ec2_setup.sh").read_text())
"""),
py(r"""
cleanup = pd.DataFrame([
    ("EC2", "Instances → select → Instance state → Terminate", "t2.large is not free tier (about $0.09/hour in ap-south-1)"),
    ("GitHub runner", "Settings → Actions → Runners → … → Remove", "an offline runner leaves deploy jobs queued forever"),
    ("ECR", "Repositories → select → Delete", "stored images cost a little per GB-month"),
    ("IAM", "Users → select → Delete (or deactivate its access keys)", "leaked keys are the biggest risk"),
    ("GitHub secrets", "Settings → Secrets and variables → Actions → delete", "remove keys you deleted in IAM"),
    ("Security group", "remove the 0.0.0.0/0 rule on port 8080", "if you keep the instance"),
], columns=["resource", "how", "why"])
cleanup
"""),
py(r"""
shutil.rmtree(SANDBOX, ignore_errors=True)
print("sandbox removed")
"""),
]

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, OUT)
print("wrote", OUT)
