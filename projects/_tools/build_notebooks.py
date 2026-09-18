"""Generate the hands-on notebooks for chapters 06, 08-10 and 12."""
import os
import nbformat as nbf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # the projects/ folder


def nb(cells):
    book = nbf.v4.new_notebook()
    book.metadata = {
        "kernelspec": {"name": "python3", "display_name": "Python 3 (MLOps .venv)", "language": "python"},
        "language_info": {"name": "python"},
    }
    out = []
    for kind, src in cells:
        src = src.strip("\n")
        out.append(nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src))
    book.cells = out
    return book


def save(path, cells):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    nbf.write(nb(cells), path)
    print("wrote", path)


def md(s): return ("md", s)
def py(s): return ("code", s)
def sh(s): return ("code", "%%bash\n" + s.strip("\n"))


# ─────────────────────────────────────────────────────────────── Chapter 06
linux = [
md("""
# Chapter 06 · Linux commands for MLOps: hands-on

Companion to `notes/06_linux_commands.html` (video 01:31:55 – 01:54:09).

Every command the instructor runs on the EC2 Ubuntu server, run here for real inside a throw-away **`sandbox/`** folder.

* Cells starting with `%%bash` run in a bash shell. Each cell is a new shell, but they all start in the sandbox folder.
* This notebook works on **macOS and Linux**. The few commands that only exist on Ubuntu servers (`apt`, `useradd`) are explained, and the notebook checks whether they're available.
* To follow along on the real server instead, launch EC2 (Chapter 05) and type the same commands there.
"""),
py("""
import os, shutil, subprocess

# All work happens inside ./sandbox, which is wiped each time this cell runs
SANDBOX = os.path.abspath("sandbox")
shutil.rmtree(SANDBOX, ignore_errors=True)
os.makedirs(SANDBOX)
os.chdir(SANDBOX)            # %%bash cells inherit this working directory
for k in ("CLICOLOR", "CLICOLOR_FORCE", "LS_COLORS"):
    os.environ.pop(k, None)  # plain (uncoloured) ls output
print("working in:", os.getcwd())
"""),
md("""
## 1 · `sudo` and package updates  (video 01:32)

On a fresh Ubuntu server the first thing to run is:

```bash
sudo apt update          # refresh the package list (sudo = run as admin/root)
sudo apt-get update      # same job, older tool; preferred in scripts/Dockerfiles
sudo apt upgrade -y      # actually install newer versions (beyond the video)
```

Let's check what OS this notebook is running on and whether `apt` exists here:
"""),
sh("""
uname -s -m
command -v apt >/dev/null && echo "apt is available: this is Debian/Ubuntu" \\
  || echo "apt not found: not Ubuntu (on macOS the equivalent is Homebrew). Run 'sudo apt update' on EC2."
"""),
md("""
## 2 · Looking around: `ls`, `ls -al`, `pwd`  (video 01:34)

A fresh home folder looks empty with `ls`, but `ls -al` also shows **hidden** files (names starting with `.`).
Let's create one visible file and one hidden file to see the difference.
"""),
sh("""
touch visible.txt .hidden_config
echo "--- ls ---";      ls
echo "--- ls -al ---";  ls -al
echo "--- pwd (print working directory) ---"; pwd
"""),
md("""
## 3 · Folders: `mkdir`, `cd`, `cd ..`, `rmdir`  (video 01:35)
"""),
sh("""
mkdir test
ls
cd test && echo "inside: $(pwd)" && ls -l   # empty folder
cd ..   && echo "back in: $(pwd)"
rmdir test                                  # rmdir only works on EMPTY folders
ls
mkdir bappy_test                            # the folder used for the rest of the demo
ls -d */
"""),
sh("""
# rmdir refuses a non-empty folder; rm -r deletes a folder and its contents
mkdir -p not_empty && touch not_empty/file.txt
rmdir not_empty || echo "↑ rmdir failed as expected"
rm -r not_empty && echo "rm -r removed it"
"""),
md("""
## 4 · Files: `touch`, editing, `cat`  (video 01:38)

The instructor edits files with **vim**: `vim bappy.txt` → press `i` (insert mode) → type → `Esc` → `:wq` → Enter.
Vim is interactive and can't run inside a notebook, so here we write the file with `printf` and use `cat` to read it back.
"""),
sh("""
touch bappy.txt
printf 'hello\\n' > bappy.txt     # what you'd type in vim
cat bappy.txt
"""),
md("""
## 5 · Running a Python file  (video 01:39)
"""),
sh("""
cat > test.py <<'EOF'
print("hello")
EOF
cat test.py
python3 test.py
"""),
md("""
## 6 · Copy, move, delete: `cp`, `mv`, `rm`  (video 01:40)

`cp` leaves the original in place. `mv` removes it from the source (and also renames files). `rm` deletes permanently; there's no recycle bin.
"""),
sh("""
cp test.py bappy_test/
echo "after cp → here: $(ls | tr '\\n' ' ') | bappy_test: $(ls bappy_test)"
rm bappy_test/test.py
mv test.py bappy_test/
echo "after mv → here: $(ls | tr '\\n' ' ') | bappy_test: $(ls bappy_test)"
rm bappy_test/test.py
echo "after rm → bappy_test: '$(ls bappy_test)'"
mv bappy.txt notes.txt && mv notes.txt bappy.txt && echo "mv also renames"
"""),
md("""
## 7 · Help and processes: `man`, `top`  (video 01:42)

`man ls` opens the manual (press `q` to quit). `top` is Linux's Task Manager and is interactive (`Ctrl+C` or `q` to quit), so here we print one snapshot of it.
"""),
sh("""
man ls 2>/dev/null | col -b | sed -n '1,12p' || ls --help | head -12
"""),
sh("""
if [ "$(uname)" = "Darwin" ]; then
  top -l 1 -n 5 -stats pid,command,cpu,mem | head -20   # macOS flags
else
  top -b -n 1 | head -15                                # Linux flags (what you'd use on EC2)
fi
echo "--- non-interactive alternative: ps ---"
ps aux | head -5
"""),
md("""
## 8 · Permissions: `ls -l` and `chmod`  (video 01:43)

The permission string has 3 groups (owner / group / others), each made of `r`=4, `w`=2, `x`=1.
"""),
sh("""
ls -l bappy.txt
chmod 700 bappy.txt && ls -l bappy.txt     # owner rwx, nobody else anything
chmod 664 bappy.txt && ls -l bappy.txt     # the Ubuntu default in the video: rw-rw-r--
chmod 644 bappy.txt && ls -l bappy.txt     # typical for normal files
"""),
py("""
def to_octal(perm: str) -> str:
    \"\"\"'rw-rw-r--' -> '664'\"\"\"
    perm = perm[-9:]
    val = {"r": 4, "w": 2, "x": 1, "-": 0}
    return "".join(str(sum(val[c] for c in perm[i:i + 3])) for i in (0, 3, 6))

def to_symbolic(octal: str) -> str:
    \"\"\"'755' -> 'rwxr-xr-x'\"\"\"
    return "".join(("r" if d & 4 else "-") + ("w" if d & 2 else "-") + ("x" if d & 1 else "-")
                   for d in map(int, octal))

for mode, use in [("700", "video example: owner only"), ("664", "video default"), ("755", "scripts & folders"),
                  ("644", "normal files"), ("600", "private files"), ("400", "SSH .pem key")]:
    print(f"chmod {mode}  ->  {to_symbolic(mode)}   ({use})")
assert to_octal("-rw-rw-r--") == "664"
"""),
sh("""
# execute permission in action: a script can't run until it has +x
printf '#!/bin/bash\\necho "script ran"\\n' > run.sh
./run.sh 2>&1 || echo "↑ no execute permission yet"
chmod +x run.sh && ./run.sh
"""),
md("""
## 9 · Archives: `tar` and `zip`  (video 01:47)

The instructor's first attempt failed because `-f` has to be followed by the **archive name**. The correct forms:
"""),
sh("""
printf 'print("hello")\\n' > test.py
tar -cvf archive.tar test.py bappy.txt        # c=create v=verbose f=file name
echo "--- list contents (t) ---"
tar -tvf archive.tar
mkdir -p extracted && tar -xvf archive.tar -C extracted   # x=extract (-C = into this folder)
ls extracted
echo "--- gzip-compressed (beyond the video) ---"
tar -czvf archive.tar.gz test.py bappy.txt
ls -l archive.tar archive.tar.gz
"""),
sh("""
if command -v zip >/dev/null; then
  zip files.zip test.py bappy.txt && unzip -l files.zip
else
  echo "zip not installed (on Ubuntu: sudo apt install -y zip unzip)"
fi
"""),
md("""
## 10 · Users and utilities: `whoami`, `date`, `echo`, `head`, `tail`  (video 01:50)

User management (`sudo useradd bappy`, `sudo userdel bappy`) needs root on a Linux server, so it isn't run here.
The instructor also says it isn't needed for this course.
"""),
sh("""
whoami
date
echo "hello from echo"
"""),
sh("""
# head / tail are most useful on long files such as logs
for i in $(seq 1 30); do echo "2026-09-16 10:$(printf %02d $i) INFO request $i served"; done > app.log
echo "request 17 failed" | sed 's/^/2026-09-16 10:17 ERROR /' >> app.log
echo "--- head -n 3 ---"; head -n 3 app.log
echo "--- tail -n 3 ---"; tail -n 3 app.log
echo "--- grep ERROR (beyond the video) ---"; grep ERROR app.log
"""),
md("""
## 11 · Extra commands for deployment (beyond the video)

These come up constantly when you serve a model from a server.
"""),
sh("""
echo "--- disk space ---"; df -h . | head -3
echo "--- folder sizes ---"; du -sh * | sort -h | tail -5
echo "--- environment variables ---"
export MODEL_NAME=emotion-xgb && echo "MODEL_NAME=$MODEL_NAME"
echo "--- is a process running? ---"
ps aux | grep -i "[j]upyter" | head -2 | cut -c1-120
echo "--- HTTP check (what you'd do against your API) ---"
curl -s -o /dev/null -w "github.com answered HTTP %{http_code}\\n" https://github.com || echo "no network"
"""),
md("""
## 12 · Summary

| Task | Command |
|---|---|
| update packages (Ubuntu) | `sudo apt update` |
| where am I / what's here | `pwd`, `ls`, `ls -al` |
| folders | `mkdir`, `cd`, `cd ..`, `rmdir`, `rm -r` |
| files | `touch`, `vim` (i → Esc → `:wq`), `cat`, `head`, `tail` |
| run code | `python3 file.py` |
| copy / move / delete | `cp`, `mv`, `rm` |
| help / processes | `man`, `top`, `ps aux` |
| permissions | `ls -l`, `chmod 700`, `chmod +x` |
| archives | `tar -cvf` / `-xvf` / `-czvf`, `zip` |
| misc | `whoami`, `date`, `echo` |

The `sandbox/` folder can be deleted at any time; re-running the first cell recreates it.
"""),
]
save(f"{ROOT}/ch06_linux_commands/linux_commands_practice.ipynb", linux)


# ─────────────────────────────────────────────────────────── Chapters 08-10
git = [
md("""
# Chapters 08–10 · Git & GitHub: hands-on

Companion to `notes/08_…`, `09_…` and `10_…` (video 02:08:22 – 02:25:53).

Everything the instructor does with Git and GitHub, run for real, **offline**:

* A **bare repository** (`github-learning-test.git`) plays the role of GitHub. Pushing to it and pulling from it works exactly like with github.com.
* Two clones, **`dev1`** and **`dev2`**, play two developers on different machines.
* Git settings come from a sandbox config file, so **your real `~/.gitconfig` is never read or changed**.

To do it on the real GitHub, create the repo on github.com (Chapter 08) and use its URL instead of the local path. The commands are the same.
"""),
py("""
import os, shutil

SANDBOX = os.path.abspath("git_sandbox")
shutil.rmtree(SANDBOX, ignore_errors=True)
os.makedirs(SANDBOX)
os.chdir(SANDBOX)

# isolated identity/config for this notebook only
cfg = os.path.join(SANDBOX, "sandbox.gitconfig")
with open(cfg, "w") as f:
    f.write(\"\"\"[user]
    name = Learner
    email = learner@example.com
[init]
    defaultBranch = main
[pull]
    rebase = false
[advice]
    detachedHead = false
[color]
    ui = never
\"\"\")
os.environ["GIT_CONFIG_GLOBAL"] = cfg     # use this instead of ~/.gitconfig
os.environ["GIT_CONFIG_NOSYSTEM"] = "1"
os.environ["GIT_PAGER"] = "cat"
for k in ("CLICOLOR", "CLICOLOR_FORCE", "LS_COLORS"):
    os.environ.pop(k, None)  # plain (uncoloured) ls output
print("sandbox:", SANDBOX)
"""),
md("""
---
# Chapter 08 · Create the remote repo and clone it

## 8.1 · "Create repository" on GitHub
On github.com you click **New → name → Public → ☑ README → .gitignore: Python → License: MIT → Create**.
GitHub then makes an **initial commit** containing those three files. We do the same by hand:
"""),
sh("""
git init --bare -q github-learning-test.git          # the empty "GitHub" repo

# what GitHub's form does for you: an initial commit with README, .gitignore, LICENSE
git clone -q github-learning-test.git _github_web 2>/dev/null
cd _github_web
printf '# github-learning-test\\nLearning Git & GitHub for MLOps.\\n' > README.md
printf '__pycache__/\\n*.py[cod]\\n.venv/\\nvenv/\\n.env\\n.ipynb_checkpoints\\n*.log\\n' > .gitignore
printf 'MIT License\\n\\nCopyright (c) 2026 Learner\\n\\n(Full MIT text: https://opensource.org/license/mit)\\n' > LICENSE
git add . && git commit -q -m "Initial commit"
git push -q origin main
cd .. && rm -rf _github_web
echo "remote repo ready"
"""),
md("""
## 8.2 · `git clone`  (video 02:11)
On GitHub: **Code → HTTPS → copy**, then `git clone <url>`. Here the "URL" is the local path.
"""),
sh("""
git clone github-learning-test.git dev1
cd dev1
echo "--- ls -a (note the hidden .git folder) ---"; ls -a
echo "--- inside .git ---"; ls .git
echo "--- the remote called origin ---"; git remote -v
"""),
md("""
## 8.3 · What `.gitignore` does
`git check-ignore -v` tells you which rule (if any) makes Git skip a file.
"""),
sh("""
cd dev1
mkdir -p src/__pycache__ && touch app.py .env server.log src/__pycache__/utils.cpython-313.pyc
for f in app.py .env server.log src/__pycache__/utils.cpython-313.pyc; do
  git check-ignore -q "$f" && echo "IGNORED  $f  ← $(git check-ignore -v "$f" | cut -f1)" || echo "tracked  $f"
done
git status --short         # only app.py shows up as a new file
rm -rf app.py .env server.log src
"""),
md("""
---
# Chapter 09 · status → add → commit → push → pull

## 9.1 · A new file shows up as *untracked*  (video 02:14)
"""),
sh("""
cd dev1
printf 'print("feature 1")\\n' > test.py
git status
"""),
md("""
## 9.2 · Stage just one file  (video 02:15)
With two new files, `git add test.py` stages only `test.py`, and `test2.py` stays untracked.
"""),
sh("""
cd dev1
printf 'print("bappy")\\n' > test2.py
git add test.py
git status
echo "--- short form: A = staged new file, ?? = untracked ---"
git status --short
"""),
md("""
## 9.3 · Commit, then push to `origin main`  (video 02:16 – 02:17)
"""),
sh("""
cd dev1
git commit -m "test.py file added"
echo "--- status: test.py is gone from the list (it's committed and tracked now) ---"
git status --short
git push origin main 2>&1
git log --oneline
"""),
md("""
## 9.4 · A second developer pulls, adds feature 2 and pushes  (video 02:18)
"""),
sh("""
git clone -q github-learning-test.git dev2
cd dev2
echo "--- dev2 received ---"; cat test.py
printf 'print("feature 2")\\n' >> test.py
git status --short                   # M = modified
git add test.py
git commit -q -m "test.py file updated"
git push -q origin main && echo "dev2 pushed feature 2"
"""),
sh("""
cd dev1
git pull 2>&1 | tail -3
echo "--- dev1 now has ---"; cat test.py
"""),
md("""
## 9.5 · Why you pull before you push (beyond the video)
If someone pushed while you were working, **your push is rejected** until you pull their commits.
"""),
sh("""
cd dev2 && printf 'x = 1\\n' > utils.py && git add . && git commit -q -m "add utils" && git push -q origin main && cd ..
cd dev1 && printf '# notes\\n' > NOTES.md && git add . && git commit -q -m "add notes"
git push origin main 2>&1 | head -3 || true
echo "↑ rejected: the remote has a commit dev1 doesn't have"
git pull --no-edit 2>&1 | tail -2        # different files, so Git merges automatically
git push -q origin main && echo "push OK after pulling"
"""),
md("""
## 9.6 · Going back to a previous version  (video 02:19)

The video suggests using `git pull` to get old code back. **That doesn't work:** `pull` only brings the *latest* commits.
The real tools are `git log`, `git show`, `git restore --source` and `git revert`:
"""),
sh("""
cd dev1
git log --oneline
OLD=$(git log --format=%h --grep="test.py file added" -n1)
echo "--- test.py as it was in $OLD ---"
git show "$OLD":test.py
"""),
sh("""
cd dev1
OLD=$(git log --format=%h --grep="test.py file added" -n1)
git restore --source "$OLD" test.py        # bring the old file back into the working folder
echo "--- working copy now ---"; cat test.py
git diff --stat
git restore test.py                        # undo: back to the latest version
echo "--- restored latest ---"; cat test.py
"""),
sh("""
cd dev1
printf 'print(1/0)  # bug!\\n' >> test.py
git commit -q -am "buggy change" && git push -q origin main
BAD=$(git log --format=%h -n1)
git revert --no-edit "$BAD" 2>&1 | head -2   # a NEW commit that undoes the bad one (safe after pushing)
git push -q origin main
cat test.py
git log --oneline -4
"""),
md("""
## 9.7 · Commit everything with `git add .`  (video 02:20)
"""),
sh("""
cd dev1
touch test3.py
git status --short
git add .
git commit -q -m "files added"
git push -q origin main
git log --oneline -3
"""),
md("""
---
# Chapter 10 · Branches

## 10.1 · List branches, then create and switch  (video 02:22)
"""),
sh("""
cd dev1
git branch
git checkout -b bappy          # newer equivalent: git switch -c bappy
git branch
"""),
md("""
## 10.2 · Commit on the branch and push the branch  (video 02:23)
"""),
sh("""
cd dev1
printf 'print("hello world")\\nprint("this is new")\\n' > test3.py
git add . && git commit -q -m "new code added"
git push origin bappy 2>&1 | tail -2
echo "--- branches that exist on the remote ---"
git ls-remote --heads origin
"""),
md("""
## 10.3 · Switch back: `main` is unchanged  (video 02:24)
"""),
sh("""
cd dev1
git checkout -q main
echo "on $(git branch --show-current): test3.py has $(wc -l < test3.py | tr -d ' ') lines"
git checkout -q bappy
echo "on $(git branch --show-current): test3.py has $(wc -l < test3.py | tr -d ' ') lines"
cat test3.py
git checkout -q main
"""),
md("""
## 10.4 · Merge the branch (beyond the video: the step the video leaves out)
On GitHub you'd open a **pull request** (bappy → main) and click *Merge*. Locally, it looks like this:
"""),
sh("""
cd dev1
git merge bappy 2>&1 | tail -3
git push -q origin main
git branch -d bappy
git push -q origin --delete bappy && echo "branch deleted locally and on the remote"
cat test3.py
"""),
md("""
## 10.5 · A merge conflict, and how to resolve it (beyond the video)
Two branches change **the same line** of a file differently.
"""),
sh("""
cd dev1
git checkout -q -b feature/greeting
printf 'print("hello from the feature branch")\\nprint("this is new")\\n' > test3.py
git commit -q -am "feature: new greeting"
git checkout -q main
printf 'print("hello from main")\\nprint("this is new")\\n' > test3.py
git commit -q -am "main: different greeting"
git merge feature/greeting 2>&1 || true
echo "===== test3.py with conflict markers ====="
cat test3.py
"""),
sh("""
cd dev1
# resolve: write the version we want (keep both greetings) and remove the markers
printf 'print("hello from main")\\nprint("hello from the feature branch")\\nprint("this is new")\\n' > test3.py
git add test3.py
git commit -q --no-edit
git push -q origin main
echo "resolved:"; cat test3.py
"""),
md("""
## 10.6 · The whole history as a graph
"""),
sh("""
cd dev1
git log --oneline --graph --all
"""),
md("""
## Summary

| Chapter | Commands practised |
|---|---|
| 08 | `git init --bare` (stands in for GitHub), `git clone`, `.git/`, `git remote -v`, `.gitignore` + `git check-ignore` |
| 09 | `git status`, `git add <file>` / `git add .`, `git commit -m`, `git push origin main`, `git pull`, rejected push, `git log`, `git show`, `git restore --source`, `git revert` |
| 10 | `git branch`, `git checkout -b`, `git push origin <branch>`, `git checkout main`, `git merge`, resolving a conflict, `git branch -d`, `git log --graph` |

Delete `git_sandbox/` whenever you like; re-running the notebook rebuilds it.
"""),
]
save(f"{ROOT}/ch08_10_git/git_workflow_practice.ipynb", git)


# ─────────────────────────────────────────────── Chapter 12 · experiment notebook
experiment = [
md("""
# Twitter emotion detection: the notebook version

Companion to `notes/12_ml_pipelines_with_dvc.html` (video 02:41 – 02:46).

This is the **"traditional approach"**: the whole project in one notebook. The instructor walks through a notebook like this, then splits it into `src/` components run by DVC (see `../02_dvc_pipeline_walkthrough.ipynb`).

Steps: **ingest → text preprocessing → bag-of-words → XGBoost → evaluate**.
"""),
py("""
import os, re, string
import certifi
os.environ.setdefault("SSL_CERT_FILE", certifi.where())   # macOS python.org builds need this for HTTPS

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from xgboost import XGBClassifier

for pkg in ("stopwords", "wordnet", "omw-1.4"):
    nltk.download(pkg, quiet=True)
"""),
md("## 1 · Data ingestion"),
py("""
DATA_URL = "https://raw.githubusercontent.com/entbappy/Branching-tutorial/refs/heads/master/tweet_emotions.csv"
df = pd.read_csv(DATA_URL)
print(df.shape)
df.head()
"""),
py("""
counts = df["sentiment"].value_counts()
kept = {"happiness", "sadness"}

fig, ax = plt.subplots(figsize=(8, 4.2))
colors = ["#2a78d6" if s in kept else "#c3c2b7" for s in counts.index]
bars = ax.barh(counts.index[::-1], counts.values[::-1], color=colors[::-1], height=0.6)
ax.bar_label(bars, fmt="{:,.0f}", padding=3, fontsize=8, color="#52514e")
ax.set_title("Tweets per emotion (blue = the two classes the project keeps)", loc="left", fontsize=11)
ax.set_xlabel("tweets")
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.grid(axis="x", color="#e7e6e1", linewidth=0.8)
ax.set_axisbelow(True)
plt.tight_layout()
plt.show()
"""),
md("The instructor keeps only **happiness (→ 1)** and **sadness (→ 0)** to make it a simple binary problem, and drops `tweet_id`."),
py("""
data = df.drop(columns=["tweet_id"])
data = data[data["sentiment"].isin(["happiness", "sadness"])].copy()
data["sentiment"] = data["sentiment"].map({"happiness": 1, "sadness": 0})
# note: the course repo uses df['sentiment'].replace(..., inplace=True) on a filtered slice,
# which silently does nothing under pandas >= 3 (copy-on-write). .map() on a .copy() is safe.
print(data.shape)
print(data["sentiment"].value_counts().rename({1: "happiness (1)", 0: "sadness (0)"}))
train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)
len(train_df), len(test_df)
"""),
md("## 2 · Text preprocessing\nLowercase → remove URLs → remove numbers → remove punctuation → drop stop words → lemmatise."),
py("""
STOP = set(stopwords.words("english"))
LEM = WordNetLemmatizer()

def normalize_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"https?://\\S+|www\\.\\S+", " ", text)
    text = re.sub(r"\\d+", " ", text)
    text = text.translate(str.maketrans({c: " " for c in string.punctuation}))
    return " ".join(LEM.lemmatize(w) for w in text.split() if w not in STOP)

sample = "Watching 3 movies tonight!!! Check http://example.com - the kids were SO happy :)"
print("before:", sample)
print("after: ", normalize_text(sample))
"""),
py("""
train_df["content"] = train_df["content"].map(normalize_text)
test_df["content"] = test_df["content"].map(normalize_text)
train_df = train_df[train_df["content"].str.len() > 0]
test_df = test_df[test_df["content"].str.len() > 0]
train_df.head()
"""),
md("## 3 · Feature engineering: bag of words\nFit the vocabulary on **train only**, then transform both splits."),
py("""
vectorizer = CountVectorizer(max_features=1000)
X_train = vectorizer.fit_transform(train_df["content"])
X_test = vectorizer.transform(test_df["content"])
y_train, y_test = train_df["sentiment"].values, test_df["sentiment"].values
print(X_train.shape, X_test.shape)
print("first vocabulary words:", list(vectorizer.get_feature_names_out()[:12]))
"""),
md("## 4 · Model building"),
py("""
model = XGBClassifier(n_estimators=100, learning_rate=0.1, eval_metric="logloss", random_state=42)
model.fit(X_train, y_train)
"""),
md("## 5 · Evaluation"),
py("""
pred = model.predict(X_test)
proba = model.predict_proba(X_test)[:, 1]
metrics = {
    "accuracy": accuracy_score(y_test, pred),
    "precision": precision_score(y_test, pred),
    "recall": recall_score(y_test, pred),
    "auc": roc_auc_score(y_test, proba),
}
print(pd.Series(metrics).round(4).to_string())
cm = pd.DataFrame(confusion_matrix(y_test, pred),
                  index=["actual sadness", "actual happiness"],
                  columns=["pred sadness", "pred happiness"])
cm
"""),
py("""
# which words push a tweet towards "happiness"? (XGBoost gain importance)
imp = pd.Series(model.get_booster().get_score(importance_type="gain"))
imp.index = [vectorizer.get_feature_names_out()[int(k[1:])] for k in imp.index]
imp.sort_values(ascending=False).head(10).round(1)
"""),
py("""
def predict(texts):
    X = vectorizer.transform([normalize_text(t) for t in texts])
    return ["happiness" if p else "sadness" for p in model.predict(X)]

predict(["I love this sunny day with my friends", "I miss you so much, feeling lonely"])
"""),
md("""
## Why this notebook isn't enough for production

It works, but everything is in one place: no modularity, hard to collaborate on, hard to re-run only what changed.
The same code is split into five components in **`src/`** and wired together with **`dvc.yaml`**.
See **`02_dvc_pipeline_walkthrough.ipynb`** in the project root.

| Notebook section | Component | Output |
|---|---|---|
| 1 · ingestion | `src/data_ingestion.py` | `data/raw/` |
| 2 · preprocessing | `src/data_preprocessing.py` | `data/processed/` |
| 3 · features | `src/feature_engineering.py` | `data/features/` |
| 4 · model | `src/model_building.py` | `model.pkl` |
| 5 · evaluation | `src/model_evaluation.py` | `metrics.json` |
"""),
]
save(f"{ROOT}/ch12_dvc_pipeline/notebooks/01_experiment_twitter_emotion.ipynb", experiment)


# ─────────────────────────────────────────────── Chapter 12 · DVC walkthrough
walk = [
md("""
# Build & track the ML pipeline with DVC: walkthrough

Companion to `notes/12_ml_pipelines_with_dvc.html` (video 02:46 – 03:09). Run it from the **project root** (`ch12_dvc_pipeline/`).

What happens, in the same order as the video:
1. run a component by hand → 2. `git init` + `dvc init` → 3. `dvc.yaml` with **one** stage → `dvc repro`
4. re-run (skipped) → 5. edit the code (re-runs) → 6. add the other stages one by one
7. `dvc dag`, `dvc metrics show` → 8. extras: `dvc status`, `dvc metrics diff`, a deleted output, what to commit.

> ⚠️ The first code cell **resets** this folder's generated state (`.git`, `.dvc`, `dvc.lock`, `data/`, `model.pkl`, `metrics.json`, `logs/`) so the notebook can be re-run from scratch. Your `src/`, `notebooks/` and `dvc.yaml` are kept.
"""),
py("""
import os, sys, shutil, json, pathlib, yaml, certifi

PROJECT = pathlib.Path.cwd()
assert (PROJECT / "src" / "data_ingestion.py").exists(), "open this notebook from the ch12_dvc_pipeline folder"

# make `dvc` and `python` resolve to this environment inside %%bash cells
os.environ["PATH"] = os.path.dirname(sys.executable) + os.pathsep + os.environ["PATH"]
os.environ["DVC_NO_ANALYTICS"] = "1"
os.environ["SSL_CERT_FILE"] = certifi.where()
# isolated git identity (doesn't touch ~/.gitconfig)
os.environ.update(GIT_AUTHOR_NAME="Learner", GIT_AUTHOR_EMAIL="learner@example.com",
                  GIT_COMMITTER_NAME="Learner", GIT_COMMITTER_EMAIL="learner@example.com",
                  GIT_CONFIG_GLOBAL=os.devnull, GIT_PAGER="cat")
for k in ("CLICOLOR", "CLICOLOR_FORCE", "LS_COLORS"):
    os.environ.pop(k, None)

# reset generated state only
for name in [".git", ".dvc", "data", "logs"]:
    shutil.rmtree(PROJECT / name, ignore_errors=True)
for name in ["dvc.lock", "model.pkl", "metrics.json", ".dvcignore", ".gitignore"]:
    (PROJECT / name).unlink(missing_ok=True)

# a normal Python .gitignore (DVC will append its own entries to it later)
(PROJECT / ".gitignore").write_text("\\n".join(["__pycache__/", "*.py[cod]", "logs/", ".ipynb_checkpoints/", ""]))

FULL_DVC_YAML = (PROJECT / "dvc.yaml").read_text()        # the finished pipeline, restored at the end
STAGES = yaml.safe_load(FULL_DVC_YAML)["stages"]

def write_pipeline(n_stages: int) -> None:
    \"\"\"Write dvc.yaml containing only the first n stages (to add them one by one, as in the video).\"\"\"
    if n_stages == len(STAGES):
        (PROJECT / "dvc.yaml").write_text(FULL_DVC_YAML)
    else:
        subset = dict(list(STAGES.items())[:n_stages])
        (PROJECT / "dvc.yaml").write_text(yaml.safe_dump({"stages": subset}, sort_keys=False))
    print((PROJECT / "dvc.yaml").read_text())

print("project:", PROJECT)
print("stages available:", list(STAGES))
"""),
md("""
## 1 · Each component runs on its own  (video 02:48)
Before DVC, the instructor tests each file with `python src/<file>.py`. Here's ingestion; then the output is deleted again so DVC can recreate it.
"""),
sh("""
python src/data_ingestion.py
ls -l data/raw
rm -rf data logs
"""),
md("""
## 2 · `git init` → `dvc init`  (video 02:57)
DVC runs on top of Git, so the folder must be a Git repository.
"""),
sh("""
git init -q -b main && echo "git initialised"
dvc init
echo "--- new files ---"
ls -a
echo "--- .dvc/ ---"
ls -a .dvc
"""),
md("""
## 3 · `dvc.yaml` with just the first stage → `dvc repro`  (video 02:55 – 02:59)
Every stage has `cmd` (what to run), `deps` (script + inputs) and `outs` (what it produces).
"""),
py("write_pipeline(1)"),
sh("dvc repro 2>&1"),
md("""
`dvc repro` wrote **`dvc.lock`**, a fingerprint (md5 hash + size) of every dependency and output.
This is how DVC "remembers" what it ran:
"""),
sh("cat dvc.lock"),
md("## 4 · Run it again: nothing to do  (video 03:00)"),
sh("dvc repro 2>&1"),
md("""
## 5 · Change the code: the stage re-runs  (video 03:01)
As in the video, switch the positive class from **happiness** to **neutral**.
"""),
py("""
src_file = PROJECT / "src" / "data_ingestion.py"
original_code = src_file.read_text()
src_file.write_text(original_code.replace('POSITIVE_CLASS = "happiness"', 'POSITIVE_CLASS = "neutral"'))
print([l for l in src_file.read_text().splitlines() if l.startswith("POSITIVE_CLASS")])
"""),
sh("""
dvc status
dvc repro 2>&1
"""),
py("""
import pandas as pd
print("rows in data/raw/train.csv:", len(pd.read_csv("data/raw/train.csv")),
      "(neutral + sadness, so more rows than happiness + sadness)")
src_file.write_text(original_code)   # change it back, like the instructor
print([l for l in src_file.read_text().splitlines() if l.startswith("POSITIVE_CLASS")])
"""),
md("""
Changing the code **back** makes the stage "changed" again. Recent DVC versions have a **run-cache**, so if this exact combination was run before, DVC restores the old outputs instead of recomputing (look for *"cached"* / *"checking out"* in the output):
"""),
sh("dvc repro 2>&1"),
md("""
## 6 · Add the remaining stages one at a time  (video 03:03 – 03:06)
Each time, the earlier stages are **skipped** and only the new stage runs.
"""),
py("write_pipeline(2)   # + data_preprocessing"),
sh("dvc repro 2>&1"),
py("write_pipeline(3)   # + feature_engineering"),
sh("dvc repro 2>&1"),
py("write_pipeline(4)   # + model_building"),
sh("dvc repro 2>&1"),
py("write_pipeline(5)   # + model_evaluation (the full file, with metrics:)"),
sh("dvc repro 2>&1"),
sh("""
echo "--- one more time: everything up to date ---"
dvc repro 2>&1
"""),
md("## 7 · `dvc dag` and `dvc metrics show`  (video 03:07)"),
sh("dvc dag"),
sh("""
dvc metrics show
echo "--- metrics.json ---"
cat metrics.json
"""),
md("""
---
## 8 · Beyond the video

### 8.1 · What to commit
DVC has already added its outputs to `.gitignore` files, so large data never goes into Git. You commit the **pipeline definition + lock file**:
"""),
sh("""
echo "--- data/.gitignore ---"; cat data/.gitignore
echo "--- /.gitignore ---"; cat .gitignore
git add .
git status --short
git commit -q -m "pipeline: 5 DVC stages" && git log --oneline
"""),
md("""
### 8.2 · Change only the model → only 2 stages re-run, then compare metrics
"""),
py("""
mb = PROJECT / "src" / "model_building.py"
mb_original = mb.read_text()
mb.write_text(mb_original.replace('"n_estimators": 100', '"n_estimators": 300'))
print([l for l in mb.read_text().splitlines() if l.startswith("PARAMS")])
"""),
sh("""
dvc status
dvc repro 2>&1
echo "--- metrics vs the last commit ---"
dvc metrics diff
"""),
py("""
_ = mb.write_text(mb_original)   # restore the committed version
"""),
sh("""
dvc repro 2>&1 | tail -4
dvc metrics diff && echo "(no difference: back to the committed model)"
"""),
md("""
### 8.3 · Accidentally deleted output → `dvc status` notices, `dvc repro` fixes it
"""),
sh("""
rm model.pkl
dvc status
dvc repro 2>&1 | tail -6
ls -l model.pkl
"""),
md("""
### 8.4 · The URL trap
`data_ingestion` only depends on its **script**. If new rows were added to the CSV **at the URL**, `dvc status` would still report *up to date* and the stage would be skipped.
Fixes: `dvc import-url <url> data/source.csv` + `dvc update`, or `always_changed: true` on that stage, or a data-version value in `params.yaml`.
"""),
sh("dvc status"),
md("""
## Summary

| Step | Command | What we saw |
|---|---|---|
| set up | `git init`, `dvc init` | `.dvc/`, `.dvcignore` created |
| run pipeline | `dvc repro` | only stages whose deps/outs changed run; `dvc.lock` updated |
| nothing changed | `dvc repro` | *didn't change, skipping* |
| code changed | `dvc repro` | that stage + everything downstream re-runs |
| inspect | `dvc dag`, `dvc metrics show`, `dvc status` | graph, metrics table, what's out of date |
| compare | `dvc metrics diff` | metric change vs last commit |
| commit | `git add dvc.yaml dvc.lock .gitignore …` | data stays out of Git |

Re-run this notebook from the top at any time: the first cell resets the generated state.
"""),
]
save(f"{ROOT}/ch12_dvc_pipeline/02_dvc_pipeline_walkthrough.ipynb", walk)
