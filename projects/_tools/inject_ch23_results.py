"""Copy Chapter 23 results (notebook outputs + Grafana screenshot) into notes/23_grafana.html."""
import json
import re
import shutil
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[2]
PROJ = ROOT / "projects" / "ch23_grafana"
PAGE = ROOT / "notes" / "23_grafana.html"
IMG = ROOT / "notes" / "assets" / "img"

nb = nbformat.read(PROJ / "grafana_walkthrough.ipynb", 4)
cells = [c for c in nb.cells if c.cell_type == "code"]
text = "\n".join("".join(o.get("text", "") or o.get("data", {}).get("text/plain", "")
                        for o in c.get("outputs", [])) for c in cells)

def grab(pattern, flags=re.M, n=None):
    found = [m.group(0).rstrip() for m in re.finditer(pattern, text, flags)]
    return found[:n] if n else found

lines = []
lines += grab(r"^(API|Prometheus|Grafana) +http://\S+ +\w+$")
lines.append("")
lines += grab(r"^\s+\d+ +(sentiment-api|prometheus) .*$")[:2] or ["(scrape targets: sentiment-api, prometheus — both up)"]
lines.append("")
lines += grab(r"^ +\d\d:\d\d:\d\d +\w+ +\d+ rps for \d+s.*$")
lines += grab(r"^total \d+s of traffic$")
lines.append("")
lines += grab(r"^requests served *: .*$") + grab(r"^predictions made: .*$")
lines += grab(r"^ +(positive|neutral|negative) +\d+$")
lines.append("")
alerts = grab(r"^\d+ +\w+ +(firing|inactive|pending) +\d+ +\d+$")
if alerts:
    lines.append("alert rules (state at the end of the run, after recovery):")
    lines += ["   " + re.sub(r"^\d+ +", "", a) for a in alerts]
fired = grab(r"^alerts that fired during the run:$")
if fired:
    lines += fired + ["   " + f for f in grab(r"^ {3,}\w+, firing.*$")]
lines.append("")
lines += grab(r"^grafana +: \d.*$") + grab(r"^data source +: .*$") + grab(r"^dashboard +: .*$")
lines += grab(r"^saved ch23_grafana_dashboard\.png.*$")
ho = "\n".join(l for l in lines if l is not None)

page = PAGE.read_text()
page = re.sub(r"const HANDS_ON = .*?;\n",
              lambda m: "const HANDS_ON = " + json.dumps({"ho-main": ho}, ensure_ascii=False) + ";\n",
              page, count=1)
PAGE.write_text(page)

for name in ("ch23_grafana_dashboard.png", "ch23_promql.png"):
    src = PROJ / "reports" / name
    if src.exists():
        shutil.copy(src, IMG / name)
        print("copied", name, f"({src.stat().st_size // 1024} KB)")
    else:
        print("MISSING", name)
print("\n" + ho)
