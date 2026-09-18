"""Generate the Chapter 16 notebook: Docker basics, the calculator image, and a containerised model API.
Docker cells run only when a Docker engine is available; otherwise they print the command they would run."""
import os
import nbformat as nbf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "ch16_docker", "docker_walkthrough.ipynb")


def md(s): return nbf.v4.new_markdown_cell(s.strip("\n"))
def py(s): return nbf.v4.new_code_cell(s.strip("\n"))


cells = [
md("""
# Chapter 16 · Docker: walkthrough

Companion to `notes/16_docker.html` (video 04:05:17 – 04:52:33: introduction, installation, practical demo, custom images).

| Part | What happens | Needs Docker? |
|---|---|---|
| 0 | check whether a Docker engine is running | – |
| 1 | `hello-world`: pull → images → run → ps → rm → rmi | yes |
| 2 | the calculator app **without** Docker (plain Python) | no |
| 3 | its Dockerfile → `docker build` → `docker run -p` / `-d` → `docker push` | yes |
| 4 | beyond the video: containerise the **Chapter 15 wine-quality model** as a FastAPI service | build/run: yes |

Cells that need Docker check first. If Docker isn't installed they print **⏭ skipped** plus the exact command, so this notebook runs either way.
After installing Docker Desktop, choose **Run All** again and every step will execute for real.
"""),
py("""
import os, sys, json, time, shutil, subprocess, urllib.request, urllib.parse, re
HERE = os.getcwd()
assert os.path.isdir("calculator_app"), "run this notebook from projects/ch16_docker"
BIN = os.path.dirname(sys.executable)
ENV = {**os.environ, "PATH": BIN + os.pathsep + os.environ["PATH"]}
for k in ("CLICOLOR", "CLICOLOR_FORCE"): ENV.pop(k, None)

DOCKER = shutil.which("docker")
ENGINE_UP = bool(DOCKER) and subprocess.run([DOCKER, "info"], capture_output=True).returncode == 0
HUB_USER = os.getenv("DOCKERHUB_USERNAME", "<your-dockerhub-username>")

def docker(cmd, check=True, cwd=None):
    \"\"\"Run a docker command if the engine is up; otherwise show what would run.\"\"\"
    print("$ docker " + cmd)
    if not ENGINE_UP:
        print("   ⏭ skipped: no running Docker engine on this machine")
        return None
    out = subprocess.run(f"{DOCKER} {cmd}", shell=True, capture_output=True, text=True, cwd=cwd)
    print((out.stdout + out.stderr).strip()[-3000:])
    if check and out.returncode != 0:
        raise RuntimeError(f"docker {cmd} failed")
    return out

def wait_for(url, timeout=60):
    for _ in range(timeout * 2):
        try:
            with urllib.request.urlopen(url, timeout=1) as r:
                return r.status
        except Exception:
            time.sleep(0.5)
    raise TimeoutError(url)
"""),
md("## 0 · Is Docker installed?  (video 04:29 · `docker --version`, `docker ps`)"),
py("""
print("docker CLI :", DOCKER or "not found")
print("engine up  :", ENGINE_UP)
if ENGINE_UP:
    docker("--version"); docker("ps")
else:
    print(\"\"\"
Install Docker Desktop first (see the notes, section 'Installation'):
  • macOS / Windows: https://docs.docker.com/desktop/  (Windows also needs WSL 2)
  • Ubuntu server  : https://docs.docker.com/engine/install/ubuntu/
Then check with:  docker --version  and  docker ps\"\"\")
"""),
md("""
## 1 · The `hello-world` image  (video 04:32 – 04:36)
"""),
py("""
docker("pull hello-world")
docker("images")
docker("run hello-world")
docker("ps -a")
"""),
py("""
# clean up: remove the stopped hello-world containers, then the image
if ENGINE_UP:
    ids = subprocess.run(f"{DOCKER} ps -aq --filter ancestor=hello-world", shell=True, capture_output=True, text=True).stdout.split()
    for cid in ids:
        docker(f"rm {cid}")
else:
    docker("rm <container-id>")
docker("rmi hello-world")
"""),
md("""
## 2 · The calculator app without Docker  (video 04:38)

First make sure the app works as plain Python: `python app.py`, then open <http://localhost:8080>.
"""),
py("""
print("calculator_app/ contains:", sorted(os.listdir("calculator_app")))
app = subprocess.Popen([sys.executable, "app.py"], cwd="calculator_app", env={**ENV, "PORT": "8080"},
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
wait_for("http://127.0.0.1:8080/health")

def calc(a, b, op, port=8080):
    data = urllib.parse.urlencode({"a": a, "b": b, "operation": op}).encode()
    html = urllib.request.urlopen(f"http://127.0.0.1:{port}/", data=data).read().decode()
    m = re.search(r'id="(result|error)">([^<]+)', html)
    return m.group(2) if m else "?"

for a, b, op in [(23, 4, "add"), (133, 7, "subtract"), (6, 7, "multiply"), (1, 0, "divide")]:
    print(f"{op:9s} {a}, {b}  →  {calc(a, b, op)}")
app.terminate(); app.wait()
print("app stopped")
"""),
md("""
## 3 · Containerise it  (video 04:40 – 04:51)

### 3.1 · The Dockerfile
"""),
py("""
print(open("calculator_app/Dockerfile").read())
"""),
md("""
### 3.2 · Build, run, detach, stop
`docker build -t <user>/mycalapp:latest .`, then `docker run -p 8080:8080 …` (host port : container port).
"""),
py("""
IMAGE = f"{HUB_USER.lower().strip('<>') if ENGINE_UP else HUB_USER}/mycalapp:latest"
docker(f"build -t {IMAGE} .", cwd="calculator_app")
docker("images")
"""),
py("""
# -d = detached (keeps running after the terminal closes); --name makes it easy to stop later
docker(f"run -d --name calc -p 8080:8080 {IMAGE}")
if ENGINE_UP:
    wait_for("http://127.0.0.1:8080/health")
    print("from the container:", calc(23, 4, "add"))
docker("ps")
docker("logs calc")
"""),
py("""
docker("stop calc")
docker("rm calc")
"""),
md("""
### 3.3 · Push to Docker Hub, then pull and run anywhere  (video 04:48)
Needs a Docker Hub account: set `DOCKERHUB_USERNAME` before starting Jupyter and run `docker login` once in a terminal.
"""),
py("""
if ENGINE_UP and "DOCKERHUB_USERNAME" in os.environ:
    docker(f"push {IMAGE}")
else:
    print("$ docker login")
    print(f"$ docker push {IMAGE}")
    print(f"$ docker pull {IMAGE}        # on any other machine")
    print(f"$ docker run -d -p 8080:8080 {IMAGE}")
    print("   ⏭ skipped: needs Docker and a Docker Hub login")
"""),
md("""
## 4 · Beyond the video: containerise a real model API

The Chapter 15 champion model (`ElasticnetWineModel@champion`) is exported to a small file and served with **FastAPI**.
This is the pattern the CI/CD chapters deploy: model file + API code + Dockerfile → image → container on a server.
"""),
py("""
out = subprocess.run([sys.executable, "export_model.py"], cwd="wine_api", env=ENV, capture_output=True, text=True)
print(out.stdout.strip() or out.stderr[-1500:])
print(open("wine_api/Dockerfile").read())
"""),
py("""
api = subprocess.Popen([f"{BIN}/uvicorn", "app:app", "--port", "8000"], cwd="wine_api", env=ENV,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
wait_for("http://127.0.0.1:8000/health")

wines = [
    {"fixed acidity": 7.4, "volatile acidity": 0.70, "citric acid": 0.00, "residual sugar": 1.9, "chlorides": 0.076,
     "free sulfur dioxide": 11, "total sulfur dioxide": 34, "density": 0.9978, "pH": 3.51, "sulphates": 0.56, "alcohol": 9.4},
    {"fixed acidity": 8.5, "volatile acidity": 0.28, "citric acid": 0.56, "residual sugar": 1.8, "chlorides": 0.092,
     "free sulfur dioxide": 35, "total sulfur dioxide": 103, "density": 0.9969, "pH": 3.30, "sulphates": 0.75, "alcohol": 10.5},
]
def post(url, payload):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r: return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())

print("GET  /health  →", json.loads(urllib.request.urlopen("http://127.0.0.1:8000/health").read()))
print("POST /predict →", post("http://127.0.0.1:8000/predict", wines))
status, body = post("http://127.0.0.1:8000/predict", [{"alcohol": 10}])
print(f"POST /predict with missing fields → HTTP {status}: {body['detail'][0]['msg']} ({len(body['detail'])} errors)")
spec = json.loads(urllib.request.urlopen("http://127.0.0.1:8000/openapi.json").read())
print("auto-generated docs at http://127.0.0.1:8000/docs · endpoints:", list(spec["paths"]))
api.terminate(); api.wait()
"""),
py("""
# the same service as a container
WINE_IMAGE = "wine-api:1.0"
docker(f"build -t {WINE_IMAGE} .", cwd="wine_api")
docker(f"run -d --name wine -p 8000:8000 {WINE_IMAGE}")
if ENGINE_UP:
    wait_for("http://127.0.0.1:8000/health")
    print("from the container:", post("http://127.0.0.1:8000/predict", wines))
    docker("stop wine"); docker("rm wine")
print("\\nFor an x86 EC2 server, build from an Apple-silicon Mac with:")
print(f"$ docker build --platform linux/amd64 -t {WINE_IMAGE} wine_api")
"""),
md("""
## Cheat sheet

| Task | Command |
|---|---|
| check install | `docker --version`, `docker ps` |
| get an image | `docker pull hello-world` |
| list images / containers | `docker images`, `docker ps` (running), `docker ps -a` (all) |
| run | `docker run IMAGE`, `-p HOST:CONTAINER`, `-d` detached, `--name NAME` |
| logs / stop / remove | `docker logs NAME`, `docker stop NAME`, `docker rm ID`, `docker rmi IMAGE` |
| remove all containers | `docker rm $(docker ps -aq)` (add `-f` to force-stop running ones) |
| build | `docker build -t user/app:tag .` |
| share | `docker login`, `docker push user/app:tag`, `docker pull user/app:tag` |
"""),
]

book = nbf.v4.new_notebook()
book.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3 (MLOps .venv)", "language": "python"},
                 "language_info": {"name": "python"}}
book.cells = cells
nbf.write(book, OUT)
print("wrote", OUT)
