---
name: mlops-notes-format
description: "How the user wants MLOps course notes built — HTML per TOC chapter, plus runnable notebooks/project files for any coding chapter"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 3d81a703-4b11-4d08-8037-0e69f466b79c
  modified: 2026-09-17T05:21:49.185Z
---

The user is studying a 12-hour MLOps YouTube course (w71RHxAWxaM) for interviews and to deploy/serve models. Notes live in `notes/` (one HTML per table-of-contents chapter, shared `notes/assets/`), transcript in `transcript/`. User asks for chapters one or two at a time.

For any chapter that involves coding or working with a project, also build the work as runnable files — Jupyter notebooks and/or project folders under `projects/` — execute them so outputs are saved, and have the HTML page summarize "what we did" and link to those files. The instructor's course repo (entbappy on GitHub) may be used as a reference.

**Why:** the user said explanations that fit a notebook should live in the notebook itself, with the HTML summarizing.

**How to apply:** each chapter's HTML = concepts + diagrams + interview Qs + a "Hands-on files" section; the code lives in `projects/chNN_*`. Mark my additions as "Beyond the video" and flag instructor mistakes. Render-check pages in headless Chrome before reporting.

Environment gotchas on this Mac:
* **Headless Chrome:** `--virtual-time-budget` can hang indefinitely on the chapter pages. Omit it and run Chrome in the background, then check for the PNG. Tall pages render in 16000-px slices, shifting the second slice with a `margin-top` CSS override in a temporary copy.
* **Ports:** port 5000 is taken by AirPlay. Local APIs use 5001; the ch17 MLflow server uses 5050.
* **Generators:** `_tools/build_chNN.py` rewrites every notebook it owns, which wipes their outputs, so re-execute all of them after regenerating.
