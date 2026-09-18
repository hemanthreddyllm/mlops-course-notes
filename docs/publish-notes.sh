#!/usr/bin/env bash
# Republish notes/ to GitHub Pages (the gh-pages branch).
# Run from the repo root after changing anything in notes/:   bash docs/publish-notes.sh
set -euo pipefail

REPO_URL="https://github.com/hemanthreddyllm/mlops-course-notes.git"
BLOB="https://github.com/hemanthreddyllm/mlops-course-notes/blob/main/"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TMP="$(mktemp -d)"

cp -R "$ROOT/notes/." "$TMP"/
touch "$TMP/.nojekyll"

# only notes/ is published, so point the hands-on links back at the repo
python3 - "$TMP" "$BLOB" <<'PY'
import pathlib, re, sys
tmp, blob = sys.argv[1], sys.argv[2]
for p in pathlib.Path(tmp).glob("*.html"):
    s = p.read_text()
    p.write_text(re.sub(r'href="\.\./(projects/[^"]*)"', lambda m: f'href="{blob}{m.group(1)}"', s))
PY

cd "$TMP"
git init -q && git checkout -qb gh-pages
git config user.name "hemanthreddyllm"
git config user.email "hemanthreddyllm@users.noreply.github.com"
git add -A
git commit -qm "Publish notes site ($(date -u +%Y-%m-%dT%H:%MZ))"
git push -q --force "$REPO_URL" gh-pages
echo "published → https://hemanthreddyllm.github.io/mlops-course-notes/"
rm -rf "$TMP"
