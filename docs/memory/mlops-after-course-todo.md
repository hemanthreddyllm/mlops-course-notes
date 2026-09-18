---
name: mlops-after-course-todo
description: The user keeps a to-do list of hands-on tasks to do after finishing the 12-hour MLOps course; add to it every chapter
metadata: 
  node_type: memory
  type: project
  originSessionId: 3d81a703-4b11-4d08-8037-0e69f466b79c
  modified: 2026-09-17T01:10:18.274Z
---

The user asked (2026-09-16, while on Chapter 16 · Docker) for a saved note of everything they still need to do themselves, to work through **after finishing the whole 12-hour course**.

It lives at `notes/todo_after_course.html` (linked from `notes/index.html`), grouped by chapter, with checkboxes. The items are the steps I couldn't do for them: Docker Desktop install, AWS/GCP/DagsHub/Docker Hub accounts, real GitHub pushes/PRs, EC2 deployment, DVC S3 remote, etc.

**Why:** some chapters need the user's own accounts or installs (no Docker, no cloud creds on this machine), so those steps get deferred.

**How to apply:** each time a new chapter's notes are built, append any deferred steps from that chapter to the to-do page (keep the item ids stable so saved checkbox state survives), and mention the additions in the reply. Related: [[mlops-notes-format]].
