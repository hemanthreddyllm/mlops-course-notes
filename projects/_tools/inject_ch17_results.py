"""Copy Chapter 17 results (experiment summary + walkthrough outputs) and images into notes/17_*.html."""
import json
import re
import shutil
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[2]
PROJ = ROOT / "projects" / "ch17_youtube_sentiment"
PAGE = ROOT / "notes" / "17_end_to_end_youtube_sentiment.html"
IMG = ROOT / "notes" / "assets" / "img"

summary = json.load(open(PROJ / "reports" / "experiment_summary.json"))
runs = summary["runs"]
exp = {"runs": runs, "agree": (
    "Our run agrees with the video on four conclusions: the baseline scores about 64%, TF-IDF with trigrams gives the best negative recall, "
    "1000 features is best, and stacking only matches LightGBM. It differs on two: in round 4 the sensible imbalance fixes land within about "
    "a point of each other (SMOTEENN collapses the positive class), and in round 5 a tuned linear SVM and XGBoost tie with LightGBM.")}

# --- text for the hands-on boxes -------------------------------------------------------------
lines = [f"{'round':<26}{'best run (by accuracy)':<24}{'accuracy':>9}{'neg recall':>12}{'macro F1':>10}"]
for e in dict.fromkeys(r["experiment"] for r in runs):
    rows = [r for r in runs if r["experiment"] == e]
    b = max(rows, key=lambda r: r["accuracy"])
    lines.append(f"{e:<26}{b['run']:<24}{b['accuracy']:>9.3f}{b['negative_recall']:>12.3f}{b['macro_f1']:>10.3f}")
lines.append(f"\n{len(runs)} runs logged to MLflow (local server, port 5050).\nRound 5 top 3 by accuracy: "
             + ", ".join(f"{r['run']} {r['accuracy']:.3f}" for r in sorted(
                 [r for r in runs if r["experiment"] == "exp5-algorithms-optuna"], key=lambda r: -r["accuracy"])[:3]))
ho_exp = "\n".join(lines)

nb = nbformat.read(PROJ / "pipeline_walkthrough.ipynb", 4)
outs = []
for c in nb.cells:
    if c.cell_type != "code":
        continue
    outs.append("".join(o.get("text", "") or o.get("data", {}).get("text/plain", "")
                        for o in c.get("outputs", [])))
alltext = "\n".join(outs)

def grab(pattern, text=alltext, flags=re.M):
    return [m.group(0).strip() for m in re.finditer(pattern, text, flags)]

pipe = []
pipe += grab(r"^Running stage '.*':$")[:5]
pipe += grab(r"model_evaluation \| INFO \| accuracy=.*$")[:1]
pipe.append("dvc repro (again)        → " + ("all 5 stages skipped: 'Data and pipelines are up to date.'" if "up to date" in alltext else "?"))
pipe += grab(r"^params.yaml\s+model_building.n_estimators.*$")
pipe += grab(r"^accuracy with n_estimators=150.*$")
pipe += [l.replace("Stage ", "revert → Stage ") for l in grab(r"^Stage '(model_building|model_evaluation)' is cached.*$")]
pipe.append("")
table = re.search(r"^ +version +run_id.*(?:\n\d+ .*)+", alltext, re.M)
if table:
    pipe.append(table.group(0).rstrip())
pipe += grab(r"^@production → .*$")[:1]
ho_pipe = "\n".join(pipe)

pred = re.search(r"^ +comment +label +sentiment(?:\n\d+ .*)+", alltext, re.M)
api = grab(r'^\{"message".*$')[:1]
api.append("")
api.append("POST /predict →")
api.append(pred.group(0).rstrip() if pred else "?")
api.append("")
api += grab(r"^demo comments: \d+$")
counts = re.search(r"^positive\s+\d+\nneutral\s+\d+\nnegative\s+\d+", alltext, re.M)
if counts:
    api.append(counts.group(0))
api += grab(r"api_\w+\.png image/png \d+ bytes", flags=0)
api += ["", "pytest → " + (grab(r"^\d+ passed.*$") or ["?"])[0]]
ho_api = "\n".join(api)

api_result = ("POST /predict  (our API, model loaded from models:/yt_chrome_plugin_model@staging)\n"
              + (pred.group(0).rstrip() if pred else ""))

page = PAGE.read_text()
page = re.sub(r"const EXP = .*?;\n", lambda m: "const EXP = " + json.dumps(exp, ensure_ascii=False) + ";\n", page, count=1)
page = re.sub(r"const HANDS_ON = .*?;\n", lambda m: "const HANDS_ON = " + json.dumps(
    {"ho-exp": ho_exp, "ho-pipe": ho_pipe, "ho-api": ho_api, "api-result": api_result}, ensure_ascii=False) + ";\n", page, count=1)
PAGE.write_text(page)

for src, dst in [("reports/eda_wordclouds.png", "ch17_eda_wordclouds.png"),
                 ("reports/api_pie.png", "ch17_api_pie.png"),
                 ("reports/api_trend.png", "ch17_api_trend.png"),
                 ("reports/api_wordcloud.png", "ch17_api_wordcloud.png"),
                 ("reports/confusion_matrix.png", "ch17_confusion_matrix.png")]:
    shutil.copy(PROJ / src, IMG / dst)

print(ho_exp, "\n\n", ho_pipe, "\n\n", ho_api, sep="")
