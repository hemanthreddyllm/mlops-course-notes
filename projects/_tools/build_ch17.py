"""Generate the Chapter 17 notebooks (YouTube Sentiment Insights):
  notebooks/1_preprocessing_eda.ipynb   data understanding + cleaning
  notebooks/2_experiments.ipynb         the six MLflow experiment rounds
  pipeline_walkthrough.ipynb            DVC pipeline → registry → Flask API → tests → Docker
Run order: 1 → 2 → walkthrough (all three share the MLflow server on port 5050)."""
import os
import nbformat as nbf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJ = os.path.join(ROOT, "ch17_youtube_sentiment")


def md(s): return nbf.v4.new_markdown_cell(s.strip("\n"))
def py(s): return nbf.v4.new_code_cell(s.strip("\n"))


def save(cells, path):
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nbf.write(nb, path)
    print("wrote", path)


# ---------------------------------------------------------------------------------------------
# shared snippet: start (or reuse) the local MLflow server that stands in for the EC2 server
SERVER = r'''
import os, sys, time, shutil, subprocess, urllib.request
PROJECT = os.path.abspath(os.path.join(os.getcwd(), "..")) if os.path.basename(os.getcwd()) == "notebooks" else os.getcwd()
BIN = os.path.dirname(sys.executable)
MLFLOW_PORT = 5050
TRACKING_URI = f"http://127.0.0.1:{MLFLOW_PORT}"
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
os.environ["MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR"] = "false"

def server_up():
    try:
        with urllib.request.urlopen(TRACKING_URI + "/health", timeout=1) as r:
            return r.status == 200
    except Exception:
        return False

def start_mlflow_server(fresh=False):
    """Local stand-in for `mlflow server` on EC2: SQLite for runs, a folder for artifacts (S3 in the video)."""
    if fresh:
        stop_mlflow_server()
        for p in ("mlflow.db",):
            if os.path.exists(os.path.join(PROJECT, p)): os.remove(os.path.join(PROJECT, p))
        shutil.rmtree(os.path.join(PROJECT, "mlartifacts"), ignore_errors=True)
    if server_up():
        print("MLflow server already running at", TRACKING_URI); return
    log = open(os.path.join(PROJECT, "logs_mlflow_server.txt"), "w")
    subprocess.Popen([os.path.join(BIN, "mlflow"), "server",
                      "--backend-store-uri", "sqlite:///mlflow.db",
                      "--artifacts-destination", "./mlartifacts",
                      "--host", "127.0.0.1", "--port", str(MLFLOW_PORT)],
                     cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    for _ in range(120):
        if server_up():
            print("MLflow server started at", TRACKING_URI); return
        time.sleep(0.5)
    raise RuntimeError("MLflow server did not start; see logs_mlflow_server.txt")

def stop_mlflow_server():
    subprocess.run(["pkill", "-f", f"mlflow server.*port {MLFLOW_PORT}"])
    subprocess.run(["pkill", "-f", "mlflow.server.jobs._huey_consumer"])
    time.sleep(1)
'''

# =============================================================================================
# 1 · preprocessing + EDA
# =============================================================================================
eda = [
md(r"""
# Chapter 17 · Notebook 1: data understanding, cleaning and EDA

Companion to `notes/17_end_to_end_youtube_sentiment.html` (video 04:52:33 onwards, "data collection / preprocessing / EDA").

**Goal of the project:** a Chrome extension that shows the sentiment of a YouTube video's comments.
We need a model that labels a comment as **positive (1)**, **neutral (0)** or **negative (−1)**.
YouTube comments with labels aren't freely available, so the course trains on a labelled **Reddit comments** dataset.

| Step | What we check |
|---|---|
| 1 | load, shape, missing values, duplicates, blank comments |
| 2 | class balance |
| 3 | comment length: words and characters per class |
| 4 | stop words, punctuation, non-English characters |
| 5 | most common bigrams and trigrams |
| 6 | the cleaning function, then before/after examples |
| 7 | word clouds and top words per class |
| 8 | save the cleaned data for notebook 2 |
"""),
py(r"""
import os, re, json, warnings
from collections import Counter
import certifi
os.environ["SSL_CERT_FILE"] = certifi.where()
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud

for pkg in ("stopwords", "wordnet", "omw-1.4"):
    nltk.download(pkg, quiet=True)
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

pd.set_option("display.max_colwidth", 90)
sns.set_theme(style="whitegrid")
LABELS = {-1: "negative", 0: "neutral", 1: "positive"}
PALETTE = {"negative": "#e34948", "neutral": "#9a9890", "positive": "#1baf7a"}
os.makedirs("../reports", exist_ok=True)
DATA_URL = "https://raw.githubusercontent.com/Himanshu-1703/reddit-sentiment-analysis/refs/heads/main/data/reddit.csv"
"""),
md("## 1 · Load and basic checks"),
py(r"""
df = pd.read_csv(DATA_URL)
print("shape:", df.shape)
print(df.dtypes, "\n")
df.head()
"""),
py(r"""
print("missing values:\n", df.isna().sum(), "\n")
print("duplicate rows:", df.duplicated().sum())
print("blank comments (only spaces):", (df["clean_comment"].fillna("").str.strip() == "").sum())

df = df.dropna(subset=["clean_comment"]).drop_duplicates()
df = df[df["clean_comment"].str.strip() != ""].reset_index(drop=True)
print("after cleaning:", df.shape)
"""),
md("## 2 · Class balance: the dataset is imbalanced"),
py(r"""
df["label"] = df["category"].map(LABELS)
dist = df["label"].value_counts()
share = (dist / len(df) * 100).round(1)
print(pd.DataFrame({"rows": dist, "percent": share}))

fig, ax = plt.subplots(figsize=(6, 3.2))
sns.countplot(data=df, x="label", order=["positive", "neutral", "negative"], palette=PALETTE, ax=ax)
ax.set_title("Comments per class"); ax.set_xlabel("")
plt.tight_layout(); plt.savefig("../reports/eda_class_balance.png", dpi=110); plt.show()
EDA = {"rows": int(len(df)), "class_percent": share.to_dict()}
"""),
md(r"""
Positive comments are the largest class and negative the smallest (about 22%).
A model can score well on accuracy while missing many negative comments, so later we watch **negative-class recall**
and try imbalance fixes (experiment 4).
"""),
md("## 3 · How long are comments?"),
py(r"""
df["word_count"] = df["clean_comment"].str.split().str.len()
df["char_count"] = df["clean_comment"].str.len()
print(df.groupby("label")[["word_count", "char_count"]].describe().round(1).T)

fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
for lab, color in PALETTE.items():
    sns.kdeplot(df.loc[df.label == lab, "word_count"].clip(upper=150), ax=axes[0], color=color, label=lab, fill=True, alpha=.15)
axes[0].set_title("Words per comment (clipped at 150)"); axes[0].legend()
sns.boxplot(data=df, x="label", y="word_count", order=["positive", "neutral", "negative"], palette=PALETTE, ax=axes[1], showfliers=False)
axes[1].set_title("Words per comment (outliers hidden)"); axes[1].set_xlabel("")
plt.tight_layout(); plt.savefig("../reports/eda_word_count.png", dpi=110); plt.show()
EDA["median_words"] = df.groupby("label")["word_count"].median().to_dict()
EDA["comments_over_100_words_pct"] = round(float((df.word_count > 100).mean() * 100), 1)
"""),
md(r"""
Most comments are short, with long right tails. **Neutral** comments are the shortest; opinions (positive or negative) take more words.
This is why the course keeps comment length in mind but doesn't use it as a feature.
"""),
md("## 4 · Stop words, punctuation and non-English characters"),
py(r"""
STOP = set(stopwords.words("english"))
tokens = df["clean_comment"].str.lower().str.split()
df["stopword_count"] = tokens.apply(lambda ws: sum(w in STOP for w in ws))
print("stop-word share of all words: %.1f%%" % (df.stopword_count.sum() / df.word_count.sum() * 100))

top_stop = Counter(w for ws in tokens for w in ws if w in STOP).most_common(15)
print("most common stop words:", top_stop)

NEGATIONS = ["not", "no", "but", "however", "yet", "nor", "never"]
neg_counts = {w: int(tokens.apply(lambda ws: w in ws).sum()) for w in NEGATIONS}
print("comments containing negation words:", neg_counts)

odd = df["clean_comment"].str.contains(r"[^A-Za-z0-9\s!?.,']")
print(f"comments with non-English / special characters: {odd.sum()} ({odd.mean():.1%})")
df.loc[odd, "clean_comment"].head(5)
"""),
md(r"""
The dataset was **already partly cleaned** by its author: apostrophes were stripped ("don", "won") and two-letter words seem to be gone (`no` appears in 0 comments).
About 6% of comments still contain non-English script (Hindi, Arabic, emoji). Our cleaning step removes those characters.

**Why keep the negation words?** The standard NLTK list removes `not`, `no` and `but`. Then *"this is **not** good"* becomes *"good"* and the label flips.
The course removes stop words **except** a small keep-list (`not, but, however, no, yet`; we also keep `nor, never`).
"""),
md("## 5 · Most frequent bigrams and trigrams"),
py(r"""
def top_ngrams(texts, n, k=15):
    vec = CountVectorizer(ngram_range=(n, n), stop_words="english", max_features=20000)
    counts = vec.fit_transform(texts).sum(axis=0).A1
    return sorted(zip(vec.get_feature_names_out(), counts), key=lambda x: -x[1])[:k]

fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
for ax, n in zip(axes, (2, 3)):
    words, counts = zip(*top_ngrams(df["clean_comment"], n))
    ax.barh(words[::-1], counts[::-1], color="#2a78d6")
    ax.set_title(f"Top {'bi' if n == 2 else 'tri'}grams")
plt.tight_layout(); plt.savefig("../reports/eda_ngrams.png", dpi=110); plt.show()
EDA["top_bigrams"] = [w for w, _ in top_ngrams(df["clean_comment"], 2, 10)]
"""),
md(r"""
Three things stand out:
* **Template or bot text**: "free encyclopedia team reached …" and "best overall submitter …" repeat hundreds of times. They are boilerplate, not opinions.
* **Repetition spam**: "good good good", "movie movie movie", "lot lot lot".
* **Politics**: "prime minister", "narendra modi", "rahul gandhi", "political party", "right wing".

The training data comes from **(mostly Indian political) Reddit threads**, while the extension will read **YouTube comments about all kinds of videos**.
That mismatch between training data and live data (a **domain shift**) is the project's biggest weakness. It's a good point to raise in interviews.
"""),
md("## 6 · The cleaning function (identical to `src/common.py`)"),
py(r"""
KEEP_WORDS = {"not", "no", "but", "however", "yet", "nor", "never"}
STOP_WORDS = set(stopwords.words("english")) - KEEP_WORDS
lemmatizer = WordNetLemmatizer()

def preprocess_comment(comment: str) -> str:
    text = str(comment).lower().strip()
    text = re.sub(r"\n", " ", text)                  # newlines → spaces
    text = re.sub(r"[^a-z0-9\s!?.,]", "", text)      # keep letters, digits and basic punctuation
    words = [w for w in text.split() if w not in STOP_WORDS]
    return " ".join(lemmatizer.lemmatize(w) for w in words)

examples = ["This video is NOT good at all!!\nWaste of time 😡",
            "The ministers were running the campaigns better than expected",
            "however, I still think yet another tutorial won't help"]
pd.DataFrame({"before": examples, "after": [preprocess_comment(e) for e in examples]})
"""),
py(r"""
import sys; sys.path.insert(0, "..")
from src.common import preprocess_comment as project_version
assert all(preprocess_comment(e) == project_version(e) for e in examples), "notebook and src/common.py disagree!"
print("notebook function == src/common.py ✔  (training and serving clean text identically)")

df["processed"] = df["clean_comment"].apply(preprocess_comment)
empty = (df["processed"].str.strip() == "")
print("comments that became empty after cleaning:", empty.sum())
df = df[~empty].reset_index(drop=True)
df[["clean_comment", "processed", "label"]].sample(6, random_state=1)
"""),
md("## 7 · Word clouds and top words per class"),
py(r"""
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
cmaps = {"positive": "Greens", "neutral": "Greys", "negative": "Reds"}
for ax, lab in zip(axes, ["positive", "neutral", "negative"]):
    text = " ".join(df.loc[df.label == lab, "processed"])
    wc = WordCloud(width=600, height=360, background_color="white", colormap=cmaps[lab],
                   collocations=False, max_words=100, random_state=0).generate(text)
    ax.imshow(wc, interpolation="bilinear"); ax.axis("off"); ax.set_title(lab)
plt.tight_layout(); plt.savefig("../reports/eda_wordclouds.png", dpi=110); plt.show()
"""),
py(r"""
top = {}
for lab in ["positive", "neutral", "negative"]:
    c = Counter(w for text in df.loc[df.label == lab, "processed"] for w in text.split())
    top[lab] = [w for w, _ in c.most_common(12)]
pd.DataFrame(top)
"""),
md(r"""
The top words overlap heavily between classes (`modi`, `india`, `bjp`, …), so single words alone separate the classes poorly.
`not` is the most frequent word in **every** class once we keep it, so on its own it doesn't tell the classes apart. In pairs such as "not good" it does.
That's why **n-grams** get their own experiment in notebook 2.

Look at the random sample in step 6: some labels look wrong (an angry comment labelled *positive*). The labels were produced automatically, so the dataset has **label noise**.
That caps achievable accuracy, whatever model we pick.
"""),
md("## 8 · Save for notebook 2"),
py(r"""
out = df[["processed", "category"]].rename(columns={"processed": "clean_comment"})
out.to_csv("reddit_preprocessing.csv", index=False)
EDA["rows_after_cleaning"] = int(len(out))
json.dump(EDA, open("../reports/eda_summary.json", "w"), indent=1)
print(out.shape); print(json.dumps(EDA, indent=1))
"""),
]

# =============================================================================================
# 2 · experiments
# =============================================================================================
exp = [
md(r"""
# Chapter 17 · Notebook 2: experiments tracked in MLflow

In the video every experiment is logged to an **MLflow server on an EC2 instance** with an **S3 bucket** for artifacts.
Here a local `mlflow server` stands in for it. The code is identical except for the tracking URI.

| Round | Question | Model | Winner in the video |
|---|---|---|---|
| 1 | baseline: how good is a simple model? | Random Forest + bag of words | – |
| 2 | bag of words or TF-IDF? unigrams, bigrams or trigrams? | RF | TF-IDF (1,3) |
| 3 | how many features? (1000 → 10000) | RF | 1000 |
| 4 | how should we handle class imbalance? | RF | SMOTE oversampling |
| 5 | which algorithm, with Optuna tuning? | LR, SVM, NB, KNN, XGB, LightGBM, RF | LightGBM |
| 6 | does stacking beat LightGBM? | LR + KNN + LightGBM | no, keep LightGBM |

**What we compare:** accuracy, plus per-class precision and recall. The main criterion is **recall on the negative class**,
because negative comments are the minority class and the one a creator most wants to catch.

Two deliberate differences from the video (both explained in the notes):
* the vectorizer and resampling are fitted on the **training split only**. The course's baseline notebook fits the vectorizer on all rows, and its LightGBM-tuning notebook applies SMOTE before splitting; both leak test information.
* Optuna tunes on a **validation split** carved out of the training data, not on the test set (the course scores trials on the test set).
"""),
py(SERVER + r'''
start_mlflow_server(fresh=True)     # notebook 2 is the first to write to the server: start clean
'''),
py(r"""
import json, warnings, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import optuna
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
mlflow.set_tracking_uri(TRACKING_URI)
os.makedirs("cm", exist_ok=True)

df = pd.read_csv("reddit_preprocessing.csv").dropna()
X_train_txt, X_test_txt, y_train, y_test = train_test_split(
    df["clean_comment"], df["category"], test_size=0.2, stratify=df["category"], random_state=42)
print("train", X_train_txt.shape, " test", X_test_txt.shape)
RESULTS = []          # one row per run → summary charts + reports/experiment_summary.json
"""),
py(r"""
LABELS = {-1: "negative", 0: "neutral", 1: "positive"}

def log_and_score(experiment, run_name, model, X_tr, y_tr, X_te, params, tags=None, fit=True):
    # fit → predict → log params, metrics and a confusion-matrix image to MLflow
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name):
        t0 = time.time()
        if fit:
            model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        if pred.ndim > 1: pred = pred.ravel()
        rep = classification_report(y_test, pred, output_dict=True, zero_division=0)
        metrics = {"accuracy": accuracy_score(y_test, pred), "macro_f1": rep["macro avg"]["f1-score"],
                   "fit_seconds": time.time() - t0}
        for k, name in LABELS.items():
            metrics[f"{name}_precision"] = rep[str(k)]["precision"]
            metrics[f"{name}_recall"] = rep[str(k)]["recall"]
            metrics[f"{name}_f1"] = rep[str(k)]["f1-score"]
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.set_tags({"experiment_round": experiment, **(tags or {})})

        fig, ax = plt.subplots(figsize=(4, 3.4))
        sns.heatmap(confusion_matrix(y_test, pred, labels=[-1, 0, 1]), annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=list(LABELS.values()), yticklabels=list(LABELS.values()), ax=ax)
        ax.set_xlabel("predicted"); ax.set_ylabel("actual"); ax.set_title(run_name, fontsize=9)
        path = f"cm/{experiment}__{run_name}.png"
        plt.tight_layout(); fig.savefig(path, dpi=90); plt.close(fig)
        mlflow.log_artifact(path)
    row = {"experiment": experiment, "run": run_name, **{k: round(v, 4) for k, v in metrics.items()}}
    RESULTS.append(row)
    print(f"{run_name:<32} acc={metrics['accuracy']:.3f}  neg_recall={metrics['negative_recall']:.3f}  "
          f"macro_f1={metrics['macro_f1']:.3f}  ({metrics['fit_seconds']:.0f}s)")
    return row

def rf():
    return RandomForestClassifier(n_estimators=200, max_depth=15, n_jobs=-1, random_state=42)

def table(experiment):
    cols = ["run", "accuracy", "negative_recall", "negative_precision", "neutral_recall", "positive_recall", "macro_f1"]
    return pd.DataFrame([r for r in RESULTS if r["experiment"] == experiment])[cols].set_index("run")
"""),
md(r"""
## Round 1 · Baseline  (video: "Exp 1 – baseline model")
Bag of words (`CountVectorizer`, 10,000 features) + Random Forest (200 trees, depth 15).
"""),
py(r"""
vec = CountVectorizer(max_features=10000)
Xtr, Xte = vec.fit_transform(X_train_txt), vec.transform(X_test_txt)
log_and_score("exp1-baseline", "rf-bow-10k", rf(), Xtr, y_train, Xte,
              {"vectorizer": "bow", "max_features": 10000, "ngram_range": "(1,1)", "n_estimators": 200, "max_depth": 15})
table("exp1-baseline")
"""),
md(r"""
**≈64% accuracy, the same as in the video**. But look at negative recall: almost **zero**.
A depth-15 forest on 10,000 sparse count features almost never predicts the minority class. Its negative *precision* looks perfect only because it makes so few negative predictions.
That's exactly why the instructor says "don't rely on precision, look at recall".
"""),
md(r"""
## Round 2 · Bag of words vs TF-IDF × n-gram range  (6 runs, 5000 features)
* **Bag of words**: raw counts of each word or n-gram.
* **TF-IDF**: counts down-weighted for terms that appear in many comments, so rare but informative terms stand out.
"""),
py(r"""
for vec_name, Vec in [("bow", CountVectorizer), ("tfidf", TfidfVectorizer)]:
    for ngram in [(1, 1), (1, 2), (1, 3)]:
        vec = Vec(ngram_range=ngram, max_features=5000)
        Xtr, Xte = vec.fit_transform(X_train_txt), vec.transform(X_test_txt)
        log_and_score("exp2-bow-vs-tfidf", f"{vec_name}-{ngram}", rf(), Xtr, y_train, Xte,
                      {"vectorizer": vec_name, "ngram_range": str(ngram), "max_features": 5000})
table("exp2-bow-vs-tfidf")
"""),
md(r"""
## Round 3 · TF-IDF (1,3): how many features?  (10 runs: 1000, 2000, …, 10000)
"""),
py(r"""
for mf in range(1000, 10001, 1000):
    vec = TfidfVectorizer(ngram_range=(1, 3), max_features=mf)
    Xtr, Xte = vec.fit_transform(X_train_txt), vec.transform(X_test_txt)
    log_and_score("exp3-tfidf-max-features", f"tfidf-trigram-{mf}", rf(), Xtr, y_train, Xte,
                  {"vectorizer": "tfidf", "ngram_range": "(1,3)", "max_features": mf})
t3 = table("exp3-tfidf-max-features"); t3
"""),
py(r"""
fig, ax = plt.subplots(figsize=(7, 3.2))
xs = range(1000, 10001, 1000)
ax.plot(xs, t3["accuracy"], marker="o", label="accuracy", color="#2a78d6")
ax.plot(xs, t3["negative_recall"], marker="o", label="negative recall", color="#e34948")
ax.set_xlabel("max_features"); ax.set_ylim(0, 1); ax.legend(); ax.set_title("Round 3: more features doesn't help a depth-15 forest")
plt.tight_layout(); plt.show()
"""),
md(r"""
With the forest capped at depth 15, extra features mostly add noise. The smallest setting (**1000**) is as good as or better than the rest,
and it gives a smaller, faster model. The course picks **TF-IDF, trigrams, 1000 features** from here on.
"""),
md(r"""
## Round 4 · Handling the class imbalance  (5 runs)
| Technique | What it does |
|---|---|
| `class_weight="balanced"` | errors on rare classes cost more; the data is unchanged |
| SMOTE (oversampling) | creates synthetic minority samples by interpolating between neighbours |
| ADASYN | like SMOTE, but creates more samples where the minority class is hard to learn |
| random undersampling | drops majority-class rows until the classes are equal |
| SMOTEENN | SMOTE, then removes noisy rows with Edited Nearest Neighbours |

Resampling is applied to the **training split only**. The test set must keep the real class distribution.
"""),
py(r"""
vec = TfidfVectorizer(ngram_range=(1, 3), max_features=1000)
Xtr, Xte = vec.fit_transform(X_train_txt), vec.transform(X_test_txt)
BASE = {"vectorizer": "tfidf", "ngram_range": "(1,3)", "max_features": 1000}

m = RandomForestClassifier(n_estimators=200, max_depth=15, n_jobs=-1, random_state=42, class_weight="balanced")
log_and_score("exp4-imbalance", "class_weight", m, Xtr, y_train, Xte, {**BASE, "imbalance": "class_weight"})

for name, sampler in [("oversampling_smote", SMOTE(random_state=42)),
                      ("adasyn", ADASYN(random_state=42)),
                      ("undersampling", RandomUnderSampler(random_state=42)),
                      ("smote_enn", SMOTEENN(random_state=42))]:
    Xr, yr = sampler.fit_resample(Xtr, y_train)
    counts = pd.Series(yr).value_counts().to_dict()
    log_and_score("exp4-imbalance", name, rf(), Xr, yr, Xte, {**BASE, "imbalance": name, "train_rows": len(yr)})
    print("   resampled class counts:", counts)
table("exp4-imbalance")
"""),
md(r"""
Every technique multiplies negative recall several times over round 3 (0.13 → about 0.45).
* **SMOTEENN** has the highest negative recall, as in the video, but here it **collapses**: its cleaning step removes so many rows that the model never predicts *positive* (positive recall 0, accuracy about 0.45).
* The other four are **within about one point** of each other in our run, so there is no clear winner.

The course picks **SMOTE oversampling**, which keeps every original row (undersampling throws data away).
`class_weight="balanced"` is just as good here and simpler: no synthetic rows at all.

## Round 5 · Which algorithm? (Optuna tuning, TF-IDF(1,3) 1000 + SMOTE)
In the video each algorithm gets its own notebook and 30 Optuna trials. Here each gets a small, fixed number of trials so the notebook finishes in minutes.
Optuna scores each trial on a **validation split** (20% of train); only the best configuration is scored once on the test set and logged.
"""),
py(r"""
Xs, ys = SMOTE(random_state=42).fit_resample(Xtr, y_train)
Xfit, Xval, yfit, yval = train_test_split(Xs, ys, test_size=0.2, stratify=ys, random_state=42)
# note: validation rows here include synthetic SMOTE rows. Stricter setups resample inside each CV fold.
ENC = {-1: 0, 0: 1, 1: 2}; DEC = {v: k for k, v in ENC.items()}    # XGBoost needs labels 0..n-1

class XGBWrapped(XGBClassifier):                                   # map −1/0/1 ⇄ 0/1/2 transparently
    def fit(self, X, y, **kw): return super().fit(X, pd.Series(y).map(ENC), **kw)
    def predict(self, X): return pd.Series(super().predict(X)).map(DEC).to_numpy()

SPACES = {
    "LogisticRegression": (8, lambda t: LogisticRegression(C=t.suggest_float("C", 1e-3, 10, log=True), max_iter=2000)),
    "LinearSVC":          (8, lambda t: LinearSVC(C=t.suggest_float("C", 1e-3, 10, log=True))),
    "MultinomialNB":      (8, lambda t: MultinomialNB(alpha=t.suggest_float("alpha", 1e-3, 5, log=True))),
    "KNN":                (6, lambda t: KNeighborsClassifier(n_neighbors=t.suggest_int("n_neighbors", 3, 30),
                                                             p=t.suggest_categorical("p", [1, 2]), n_jobs=-1)),
    "RandomForest":       (6, lambda t: RandomForestClassifier(n_estimators=t.suggest_int("n_estimators", 100, 300),
                                                               max_depth=t.suggest_int("max_depth", 10, 40),
                                                               min_samples_split=t.suggest_int("min_samples_split", 2, 10),
                                                               n_jobs=-1, random_state=42)),
    "XGBoost":            (6, lambda t: XGBWrapped(n_estimators=t.suggest_int("n_estimators", 100, 300),
                                                   learning_rate=t.suggest_float("learning_rate", 0.03, 0.3, log=True),
                                                   max_depth=t.suggest_int("max_depth", 3, 10),
                                                   n_jobs=-1, random_state=42, verbosity=0)),
    "LightGBM":           (8, lambda t: LGBMClassifier(n_estimators=t.suggest_int("n_estimators", 100, 400),
                                                       learning_rate=t.suggest_float("learning_rate", 0.03, 0.3, log=True),
                                                       max_depth=t.suggest_int("max_depth", 5, 25),
                                                       num_leaves=t.suggest_int("num_leaves", 20, 80),
                                                       n_jobs=-1, random_state=42, verbose=-1)),
}

BEST = {}
for name, (n_trials, build) in SPACES.items():
    def objective(trial):
        model = build(trial)
        model.fit(Xfit, yfit)
        pred = model.predict(Xval)
        return classification_report(yval, pred, output_dict=True, zero_division=0)["-1"]["recall"] * 0.5 + \
               accuracy_score(yval, pred) * 0.5                     # balance negative recall and accuracy
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials)
    best_model = build(optuna.trial.FixedTrial(study.best_params))
    BEST[name] = study.best_params
    log_and_score("exp5-algorithms-optuna", name, best_model, Xs, ys, Xte,
                  {**BASE, "imbalance": "smote", "algorithm": name, "optuna_trials": n_trials,
                   **{f"best_{k}": v for k, v in study.best_params.items()}})
t5 = table("exp5-algorithms-optuna").sort_values("accuracy", ascending=False); t5
"""),
py(r"""
ax = t5[["accuracy", "negative_recall", "macro_f1"]].plot.barh(figsize=(8, 4), color=["#2a78d6", "#e34948", "#1baf7a"], width=.8)
ax.set_xlim(0, 1); ax.invert_yaxis(); ax.set_title("Round 5: tuned algorithms (test set)"); plt.tight_layout(); plt.show()
print(json.dumps(BEST, indent=1))
"""),
md(r"""
In our run, **XGBoost and a tuned linear SVM tie for first place** (≈0.787). LightGBM (only 8 quick trials) follows closely and trains about 4× faster than XGBoost.
Tuned linear models are strong, cheap baselines for TF-IDF text. KNN does badly on 1000-dimensional sparse vectors, and naive Bayes lags behind.
The course picks **LightGBM** after a much longer search.
The course then tuned LightGBM on its own (100 Optuna trials). That produced the values in `params.yaml`:
`learning_rate=0.09, max_depth=20, n_estimators=367`.

## Round 6 · LightGBM with the course's parameters, then stacking
"""),
py(r"""
final_lgbm = LGBMClassifier(objective="multiclass", num_class=3, learning_rate=0.09, max_depth=20, n_estimators=367,
                            n_jobs=-1, random_state=42, verbose=-1)
log_and_score("exp6-final-vs-stacking", "lightgbm-params-yaml", final_lgbm, Xs, ys, Xte,
              {**BASE, "imbalance": "smote", "algorithm": "LightGBM", "learning_rate": 0.09, "max_depth": 20, "n_estimators": 367})

stack = StackingClassifier(
    estimators=[("lr", LogisticRegression(C=1.0, max_iter=2000)),
                ("knn", KNeighborsClassifier(n_neighbors=10, n_jobs=-1)),
                ("lgbm", LGBMClassifier(learning_rate=0.09, max_depth=20, n_estimators=367, n_jobs=-1, random_state=42, verbose=-1))],
    final_estimator=LogisticRegression(max_iter=2000), cv=3, n_jobs=1)
log_and_score("exp6-final-vs-stacking", "stacking-lr-knn-lgbm", stack, Xs, ys, Xte,
              {**BASE, "imbalance": "smote", "algorithm": "Stacking(LR,KNN,LGBM)+LR", "cv": 3})
table("exp6-final-vs-stacking")
"""),
md(r"""
Stacking lands at about the same score as LightGBM alone, but it trains several models, runs slower at prediction time and is harder to maintain.
**Decision: ship LightGBM.** That's the model the DVC pipeline trains (`pipeline_walkthrough.ipynb`).
"""),
md("## Summary of all rounds"),
py(r"""
summary = pd.DataFrame(RESULTS)
best_per_round = summary.loc[summary.groupby("experiment")["accuracy"].idxmax(), ["experiment", "run", "accuracy", "negative_recall", "macro_f1"]]
print(best_per_round.to_string(index=False))
json.dump({"runs": RESULTS, "best_params_round5": BEST}, open("../reports/experiment_summary.json", "w"), indent=1, default=str)
print("\nruns logged:", len(RESULTS), "→ open", TRACKING_URI)
"""),
md(r"""
Open the MLflow UI (`http://127.0.0.1:5050`), select all runs in one experiment, and click **Compare**
to see the parallel-coordinates and scatter plots shown in the video.
The server keeps running for the pipeline notebook. To stop it: `stop_mlflow_server()`.
"""),
]

# =============================================================================================
# 3 · pipeline walkthrough (project root)
# =============================================================================================
walk = [
md(r"""
# Chapter 17 · Pipeline walkthrough: DVC → MLflow registry → Flask API → tests → Docker

The second half of the chapter turns the notebook experiments into a **reproducible project**:

```
data_ingestion → data_preprocessing → model_building → model_evaluation → model_registration
   (dvc.yaml)                                         (logs to MLflow)     (registry alias "staging")
                                        ↓
                       flask_api/app.py  ←  Chrome extension (popup.js)
                                        ↓
                       Dockerfile  →  .github/workflows/cicd.yaml  (Chapter 18)
```

| Part | What happens |
|---|---|
| 0 | start the local MLflow server (stands in for the EC2 + S3 server) |
| 1 | reset generated files, `git init`, `dvc init` |
| 2 | read `params.yaml` and `dvc.yaml`; `dvc dag` |
| 3 | `dvc repro`: all five stages |
| 4 | `dvc repro` again: nothing to do; then change a parameter and re-run only what's affected |
| 5 | model registry: versions and aliases (the video uses stages) |
| 6 | start the Flask API and call every endpoint (the video uses Postman) |
| 7 | `pytest` |
| 8 | Docker image (skips if Docker isn't installed) |
| 9 | shut everything down |
"""),
py(SERVER + r'''
import json, re, pathlib, textwrap
assert os.path.exists("dvc.yaml"), "run this notebook from projects/ch17_youtube_sentiment"
ENV = {**os.environ, "PATH": BIN + os.pathsep + os.environ["PATH"], "MLFLOW_TRACKING_URI": TRACKING_URI}
for k in ("CLICOLOR", "CLICOLOR_FORCE", "LS_COLORS"):
    ENV.pop(k, None)

def sh(cmd, check=True, tail=60):
    print("$", cmd)
    out = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=ENV)
    text = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", out.stdout + out.stderr).rstrip()
    lines = [l for l in text.splitlines() if "mlflow.agent.hint" not in l and "Downloading artifacts" not in l]
    print("\n".join(lines[-tail:]))
    if check and out.returncode != 0:
        raise RuntimeError(f"command failed ({out.returncode})")

start_mlflow_server()   # reuse the server from notebook 2 (keeps its experiments), or start a new one
'''),
md("## 1 · Fresh start: `git init` + `dvc init`"),
py(r"""
# remove only what the pipeline generates (not mlflow.db: it holds notebook 2's experiments)
for p in ["data", "logs", ".dvc", ".git", ".pytest_cache", "dvc.lock", "lgbm_model.pkl", "tfidf_vectorizer.pkl",
          "experiment_info.json", "reports/confusion_matrix.png", ".gitignore"]:
    if os.path.isdir(p): shutil.rmtree(p)
    elif os.path.exists(p): os.remove(p)

# a private git config so your ~/.gitconfig is never read or changed
cfg = os.path.abspath(".git_sandbox_config")
open(cfg, "w").write("[user]\n    name = Learner\n    email = learner@example.com\n[init]\n    defaultBranch = main\n[color]\n    ui = never\n")
ENV.update(GIT_CONFIG_GLOBAL=cfg, GIT_CONFIG_NOSYSTEM="1", GIT_PAGER="cat")

open(".gitignore", "w").write(textwrap.dedent('''
    .venv/
    __pycache__/
    logs/
    mlflow.db
    mlartifacts/
    logs_mlflow_server.txt
    .git_sandbox_config
    notebooks/reddit_preprocessing.csv
    notebooks/cm/
    .pytest_cache/
''').lstrip())
# this demo registers versions 1, 2, 3: start from an empty registry entry
from mlflow import MlflowClient
mlflow_client = MlflowClient(TRACKING_URI)
try:
    mlflow_client.delete_registered_model("yt_chrome_plugin_model"); print("removed old registry entry")
except Exception:
    pass
sh("git init -q && dvc init -q && git status --short | head -20")
"""),
md("## 2 · The two files that define the pipeline"),
py(r"""
print(open("params.yaml").read())
print(open("dvc.yaml").read())
"""),
py(r"""
sh("dvc dag")
"""),
md(r"""
## 3 · `dvc repro`: run every stage
`MLFLOW_TRACKING_URI` points `model_evaluation` and `model_registration` at the tracking server.
In the video it's hard-coded as `http://ec2-…amazonaws.com:5000/`; here it comes from an environment variable.
"""),
py(r"""
t0 = time.time()
sh("dvc repro", tail=70)
print(f"\n⏱ {time.time() - t0:.0f}s")
"""),
py(r"""
info = json.load(open("experiment_info.json"))
print(json.dumps(info, indent=1))
from IPython.display import Image, display
display(Image("reports/confusion_matrix.png", width=380))
sh("git add -A && git commit -qm 'pipeline: first full run' && git log --oneline")
"""),
md("## 4 · Caching: re-run only what changed"),
py(r"""
sh("dvc repro", tail=10)          # nothing changed → every stage is skipped
"""),
py(r"""
# change one hyper-parameter: only model_building and later stages are out of date
params = open("params.yaml").read()
open("params.yaml", "w").write(params.replace("n_estimators: 367", "n_estimators: 150"))
sh("dvc status")
sh("dvc params diff")
t0 = time.time()
sh("dvc repro", tail=25)
print(f"\n⏱ {time.time() - t0:.0f}s (ingestion and preprocessing were skipped)")
info_150 = json.load(open("experiment_info.json"))
print(f"accuracy with n_estimators=150: {info_150['accuracy']:.4f}  vs 367: {info['accuracy']:.4f}")
"""),
py(r"""
# undo the change: DVC's run-cache restores the old outputs without retraining.
# But model_registration has a side effect (it writes to the MLflow registry), and DVC can't undo that.
# It re-runs, and the registry gets a *third* version that points to the first run's model.
sh("git checkout -- params.yaml")
sh("dvc repro", tail=25)
sh("git status --short")
"""),
md(r"""
## 5 · Model registry: versions and aliases
The video (MLflow 2.x) moves a version to the **Staging** stage. MLflow 3 replaces stages with **aliases**:
free-form, movable pointers such as `@staging` and `@production`.
"""),
py(r"""
NAME = "yt_chrome_plugin_model"
rows = []
for v in sorted(mlflow_client.search_model_versions(f"name='{NAME}'"), key=lambda v: int(v.version)):
    run = mlflow_client.get_run(v.run_id)
    rows.append({"version": v.version, "run_id": v.run_id[:8],
                 "n_estimators": run.data.params.get("model_building.n_estimators"),
                 "accuracy": round(run.data.metrics["accuracy"], 4),
                 "aliases": mlflow_client.get_model_version(NAME, v.version).aliases})
import pandas as pd
pd.DataFrame(rows)
"""),
py(r"""
# promote: the version with the best accuracy gets @production (in the video: "transition to Production")
best = max(rows, key=lambda r: (r["accuracy"], -int(r["version"])))
mlflow_client.set_registered_model_alias(NAME, "production", best["version"])
print(f"@production → v{best['version']}   @staging → v{mlflow_client.get_model_version_by_alias(NAME, 'staging').version}")
# note: n_estimators=150 scored slightly higher than the course's 367 on this split. Small differences like this
# are why a promotion rule (e.g. "beats production by > X on a fixed test set") is better than eyeballing.
"""),
md(r"""
## 6 · Run the Flask API and call it (Postman in the video)
The API loads `models:/yt_chrome_plugin_model@staging` from the registry and the vectorizer from `tfidf_vectorizer.pkl`.
It runs on **port 5001** because macOS AirPlay already uses 5000.
"""),
py(r"""
API_PORT = 5001
API = f"http://127.0.0.1:{API_PORT}"
subprocess.run(["pkill", "-f", "flask_api/app.py"])
api_log = open("logs/api.txt", "w")
api_proc = subprocess.Popen([sys.executable, "flask_api/app.py"], env={**ENV, "PORT": str(API_PORT)},
                            stdout=api_log, stderr=subprocess.STDOUT, start_new_session=True)
for _ in range(120):
    try:
        print(urllib.request.urlopen(API + "/", timeout=1).read().decode()); break
    except Exception:
        time.sleep(0.5)

def post(path, payload):
    req = urllib.request.Request(API + path, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.headers.get_content_type(), r.read()
"""),
py(r"""
ctype, body = post("/predict", {"comments": [
    "This video is awesome! I loved it a lot",
    "Very bad explanation. poor video",
    "the tutorial starts at minute five",
    "I don't think this is not helpful",
]})
pd.DataFrame(json.loads(body))
"""),
md(r"""
The last example is a double negative (*"don't … not helpful"*), which is hard for a bag-of-n-grams model.
Short, sarcastic or double-negative comments are the typical failure cases.
"""),
py(r"""
demo = json.load(open("yt_chrome_plugin_frontend/dev/sample_comments.json"))   # fake YouTube-style comments
_, body = post("/predict_with_timestamps", {"comments": [{"text": c["text"], "timestamp": c["timestamp"]} for c in demo]})
preds = json.loads(body)
counts = pd.Series([p["sentiment"] for p in preds]).value_counts().reindex([1, 0, -1], fill_value=0)
print("demo comments:", len(preds)); print(counts.rename({1: "positive", 0: "neutral", -1: "negative"}))

images = {
    "api_pie.png": post("/generate_chart", {"sentiment_counts": {str(k): int(v) for k, v in counts.items()}}),
    "api_trend.png": post("/generate_trend_graph", {"sentiment_data": preds}),
    "api_wordcloud.png": post("/generate_wordcloud", {"comments": [c["text"] for c in demo]}),
}
for fname, (ctype, data) in images.items():
    open(f"reports/{fname}", "wb").write(data)
    print(fname, ctype, len(data), "bytes")
    display(Image(data=data, width=420))
json.dump({"counts": {int(k): int(v) for k, v in counts.items()}, "n": len(preds)}, open("reports/api_demo_counts.json", "w"))
"""),
md(r"""
The extension's popup shows the same results. To see the real popup without a YouTube API key, open
`yt_chrome_plugin_frontend/dev_preview.html` in a browser while the API is running. It uses the fake comments above.
"""),
md("## 7 · Automated tests"),
py(r"""
sh(f"{sys.executable} -m pytest -q tests", tail=15)
"""),
md("## 8 · Docker image  (the full deployment is in Chapter 18)"),
py(r"""
DOCKER = shutil.which("docker")
if DOCKER and subprocess.run([DOCKER, "info"], capture_output=True).returncode == 0:
    sh("docker build -t yt-sentiment-api .", tail=10)
    sh("docker run -d --rm --name yt-sentiment -p 8080:8080 yt-sentiment-api")
    time.sleep(8)
    print(urllib.request.urlopen("http://127.0.0.1:8080/").read().decode())
    sh("docker stop yt-sentiment")
else:
    print("⏭ Docker isn't running here. Commands to run once it is:")
    print("   docker build -t yt-sentiment-api .")
    print("   docker run -d -p 8080:8080 yt-sentiment-api     # then open http://localhost:8080/")
print(open("Dockerfile").read())
"""),
md("## 9 · Shut down"),
py(r"""
api_proc.terminate()
stop_mlflow_server()
sh("dvc dag --md", tail=30)
print("API and MLflow server stopped. To restart the UI:  mlflow server --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./mlartifacts --port 5050")
"""),
]

save(eda, os.path.join(PROJ, "notebooks", "1_preprocessing_eda.ipynb"))
save(exp, os.path.join(PROJ, "notebooks", "2_experiments.ipynb"))
save(walk, os.path.join(PROJ, "pipeline_walkthrough.ipynb"))
