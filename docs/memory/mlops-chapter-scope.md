---
name: mlops-chapter-scope
description: "Which course segments the user chose to skip in the MLOps notes (Jenkins, CircleCI) and how chapters are numbered now"
metadata: 
  node_type: memory
  type: project
  originSessionId: 3d81a703-4b11-4d08-8037-0e69f466b79c
  modified: 2026-09-18T18:29:18.314Z
---

On 2026-09-17 the user asked for CI/CD notes on **GitHub Actions only**. Jenkins and CircleCI are "just tools like GitHub Actions", so one is enough.

The two skipped segments exist as placeholder pages with no notes: `notes/19_cicd_jenkins.html` and `notes/20_cicd_circleci.html`. The user may ask for them later.

Chapter numbering in `notes/index.html` since then:
- 18: GitHub Actions (transcript files 21 and 22)
- 19: Jenkins (transcript 23)
- 20: CircleCI (transcript 24)
- 21: Kubernetes (transcript 25)
- 22: SageMaker (transcript 26)
- 23: Grafana (transcript 27)

**Why:** the user wants to avoid duplicate tool coverage.

**Status (2026-09-18): the course is finished.** Chapters 01–18 and 21–23 all have notes plus executed hands-on projects; 19 and 20 are the deliberate placeholders. What remains is the account-based practice in `notes/todo_after_course.html`.

Chapter 23 downloads Prometheus and Grafana (~1.5 GB) into `~/.cache/mlops-course-tools`; remove it when the user is done.

**How to apply:**
- Don't build notes for Jenkins or CircleCI unless the user asks.
- When a new chapter references later ones, use the numbers above.
- If the user does ask for 19 or 20, replace the placeholder page and flip its index row from "Not written" to "Ready".

Related: [[mlops-notes-format]], [[mlops-after-course-todo]]
