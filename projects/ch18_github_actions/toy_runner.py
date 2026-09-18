"""A tiny, local imitation of a GitHub Actions runner, for learning only.

What it does
  * reads a workflow YAML file and lists its triggers, jobs and the order `needs:` implies
  * decides whether a push event would start the workflow (branches / paths-ignore)
  * runs a job's `run:` steps with bash, filling in ${{ secrets.X }}, ${{ vars.X }}, ${{ env.X }}, ${{ github.sha }},
    ${{ steps.<id>.outputs.<k> }} and ${{ needs.<job>.outputs.<k> }}, and masks secret values in the log
  * imitates a few common actions (checkout, setup-python, upload/download-artifact); cloud and Docker
    steps are shown but skipped, because there is no AWS account or Docker engine here

What it does NOT do: containers, matrix builds, `if:` conditions, caching, real GitHub or AWS calls.
The real runner is https://github.com/actions/runner.
"""
import fnmatch
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

EXPR = re.compile(r"\$\{\{\s*(.+?)\s*\}\}")


def load(path):
    with open(path) as f:
        wf = yaml.safe_load(f)
    # YAML 1.1 reads the bare key `on` as the boolean True
    if True in wf and "on" not in wf:
        wf["on"] = wf.pop(True)
    return wf


def job_order(wf):
    """Jobs in an order that respects `needs:` (a topological sort)."""
    jobs = wf["jobs"]
    needs = {j: ([spec["needs"]] if isinstance(spec.get("needs"), str) else spec.get("needs", []))
             for j, spec in jobs.items()}
    order, done = [], set()
    while len(order) < len(jobs):
        ready = [j for j in jobs if j not in done and all(n in done for n in needs[j])]
        if not ready:
            raise ValueError("cycle in needs:")
        for j in ready:
            order.append(j)
            done.add(j)
    return order, needs


def describe(wf):
    order, needs = job_order(wf)
    lines = [f"workflow: {wf.get('name', '(unnamed)')}",
             f"triggers: {', '.join(wf['on']) if isinstance(wf['on'], dict) else wf['on']}"]
    for j in order:
        spec = wf["jobs"][j]
        after = f"  (after {', '.join(needs[j])})" if needs[j] else ""
        lines.append(f"\njob '{j}' · {spec.get('name', j)} · runs-on: {spec['runs-on']}{after}")
        for i, st in enumerate(spec["steps"], 1):
            label = st.get("name") or st.get("uses", "")
            kind = "uses an action" if "uses" in st else "runs shell commands"
            lines.append(f"   {i:>2}. {label[:46]:<46} {kind}")
    return "\n".join(lines)


def secrets_used(path):
    return sorted(set(re.findall(r"secrets\.([A-Za-z_][A-Za-z0-9_]*)", Path(path).read_text())))


def would_trigger(wf, event, branch, changed_files):
    """Would a `push` (or other event) start this workflow? Returns (bool, reason)."""
    on = wf["on"]
    if isinstance(on, str):
        on = {on: None}
    elif isinstance(on, list):
        on = {e: None for e in on}
    if event not in on:
        return False, f"workflow doesn't listen to '{event}'"
    rules = on[event] or {}
    if "branches" in rules and not any(fnmatch.fnmatch(branch, b) for b in rules["branches"]):
        return False, f"branch '{branch}' not in {rules['branches']}"
    ignore = rules.get("paths-ignore")
    if ignore and all(any(fnmatch.fnmatch(f, p) for p in ignore) for f in changed_files):
        return False, f"every changed file matches paths-ignore {ignore}"
    return True, "runs"


def _ca_bundle():
    """python.org builds on macOS ship without CA certificates (GitHub's Ubuntu runners have them)."""
    try:
        import certifi
        return {"SSL_CERT_FILE": certifi.where()}
    except ImportError:
        return {}


class Runner:
    def __init__(self, workdir, secrets=None, variables=None, env=None, artifacts_dir=None, source_repo=None,
                 docker_available=None, log_tail=25):
        self.workdir = Path(workdir)
        self.secrets = secrets or {}
        self.vars = variables or {}
        self.env = env or {}
        self.artifacts = Path(artifacts_dir or self.workdir.parent / "_artifacts")
        self.source_repo = source_repo
        self.docker = shutil.which("docker") is not None if docker_available is None else docker_available
        self.log_tail = log_tail
        self.job_outputs = {}
        self.sha = "0" * 40

    # -- expressions -------------------------------------------------------------------------
    def _value(self, expr, ctx):
        parts = expr.split(".")
        if parts[0] == "secrets":
            return self.secrets.get(parts[1], "")
        if parts[0] == "vars":
            return self.vars.get(parts[1], "")
        if parts[0] == "env":
            return ctx["env"].get(parts[1], "")
        if parts[0] == "github":
            return {"sha": ctx["sha"], "ref": "refs/heads/main", "repository": "you/yt-sentiment"}.get(parts[1], "")
        if parts[0] == "steps":
            return ctx["steps"].get(parts[1], {}).get(parts[3], "")
        if parts[0] == "needs":
            return self.job_outputs.get(parts[1], {}).get(parts[3], "")
        return f"<{expr}?>"

    def render(self, text, ctx):
        return EXPR.sub(lambda m: str(self._value(m.group(1), ctx)), str(text))

    def mask(self, text):
        for v in self.secrets.values():
            if v:
                text = text.replace(v, "***")
        return text

    # -- actions we can imitate --------------------------------------------------------------
    def _uses(self, st, ctx):
        action = st["uses"].split("@")[0]
        w = {k: self.render(v, ctx) for k, v in (st.get("with") or {}).items()}
        if action == "actions/checkout":
            if self.workdir.exists():
                shutil.rmtree(self.workdir)
            subprocess.run(["git", "clone", "-q", str(self.source_repo), str(self.workdir)], check=True)
            ctx["sha"] = self.sha = subprocess.run(["git", "-C", str(self.workdir), "rev-parse", "HEAD"],
                                                   capture_output=True, text=True).stdout.strip()
            return "ok", f"cloned {Path(self.source_repo).name} @ {ctx['sha'][:7]} (a fresh checkout: only files tracked by git)"
        if action == "actions/setup-python":
            return "ok", f"(imitated) using this machine's Python instead of installing {w.get('python-version')}"
        if action == "actions/upload-artifact":
            dest = self.artifacts / w["name"]
            dest.mkdir(parents=True, exist_ok=True)
            files = [p for p in w["path"].split() if p]
            for p in files:
                shutil.copy(self.workdir / p, dest / Path(p).name)
            return "ok", f"stored {len(files)} file(s) as artifact '{w['name']}'"
        if action == "actions/download-artifact":
            src = self.artifacts / w["name"]
            for f in src.iterdir():
                shutil.copy(f, self.workdir / f.name)
            return "ok", f"restored artifact '{w['name']}' ({', '.join(f.name for f in src.iterdir())})"
        if action.startswith("aws-actions/"):
            if st.get("id"):
                ctx["steps"][st["id"]] = {"registry": "123456789012.dkr.ecr.ap-south-1.amazonaws.com"}
            return "skipped", "needs a real AWS account (outputs faked: registry=123456789012.dkr.ecr.ap-south-1.amazonaws.com)"
        return "skipped", "action not imitated by the toy runner"

    def _run_script(self, script, st, ctx, tag):
        """Run one step's script with bash, like the real runner, and collect GITHUB_OUTPUT / GITHUB_ENV."""
        out_file = self.workdir.parent / f"_output_{tag}"
        env_file = self.workdir.parent / f"_env_{tag}"
        out_file.write_text("")
        env_file.write_text("")
        step_env = {k: self.render(v, ctx) for k, v in (st.get("env") or {}).items()}
        env = {**os.environ, **ctx["env"], **step_env, "CI": "true",
               "GITHUB_OUTPUT": str(out_file), "GITHUB_ENV": str(env_file),
               "PATH": os.path.dirname(sys.executable) + os.pathsep + os.environ["PATH"],
               **_ca_bundle()}
        cwd = self.workdir if self.workdir.exists() else self.workdir.parent
        res = subprocess.run(["bash", "-eo", "pipefail", "-c", script], cwd=cwd,
                             capture_output=True, text=True, env=env)
        parse = lambda f: dict(l.split("=", 1) for l in f.read_text().splitlines() if "=" in l)
        if st.get("id"):
            ctx["steps"][st["id"]] = parse(out_file)
        ctx["env"].update(parse(env_file))      # GITHUB_ENV: visible to the following steps
        return res

    # -- run a job -----------------------------------------------------------------------------
    def run_job(self, wf, job_id):
        spec = wf["jobs"][job_id]
        ctx = {"env": {**wf.get("env", {}), **spec.get("env", {})}, "steps": {}, "sha": self.sha}
        print(f"▶ job '{job_id}' ({spec.get('name', job_id)}) on {spec['runs-on']}")
        t_job = time.time()
        for i, st in enumerate(spec["steps"], 1):
            name = st.get("name") or st.get("uses") or st["run"].splitlines()[0]
            t0 = time.time()
            if "uses" in st:
                status, msg = self._uses(st, ctx)
            else:
                script = self.render(st["run"], ctx)
                if not self.docker and re.search(r"(^|\s)docker\s", script):
                    status, msg = "skipped", "no Docker engine here; would run:\n" + self.mask(script).rstrip()
                    # still record the step's outputs so later jobs can use them
                    keep = [l for l in script.splitlines()
                            if "GITHUB_OUTPUT" in l or "GITHUB_ENV" in l or re.match(r"\s*[A-Za-z_]\w*=", l)]
                    if keep:
                        self._run_script("\n".join(keep), st, ctx, tag=f"{job_id}_{i}")
                else:
                    res = self._run_script(script, st, ctx, tag=f"{job_id}_{i}")
                    log = self.mask((res.stdout + res.stderr).strip())
                    status = "ok" if res.returncode == 0 else f"FAILED (exit {res.returncode})"
                    msg = "\n".join(log.splitlines()[-self.log_tail:])
            mark = {"ok": "✔", "skipped": "⏭"}.get(status, "✘")
            print(f"  {mark} {i}. {name}  [{time.time() - t0:.1f}s]")
            if msg:
                print("     " + msg.replace("\n", "\n     "))
            if status.startswith("FAILED"):
                print(f"✘ job '{job_id}' failed; later steps and dependent jobs don't run")
                return False
        outs = spec.get("outputs") or {}
        self.job_outputs[job_id] = {}
        for k, v in outs.items():
            value = self.render(v, ctx)
            if any(sv and sv in value for sv in self.secrets.values()):
                # the real runner does the same: "Skip output 'k' since it may contain secret."
                print(f"  ⚠ Skip output '{k}' since it may contain secret.")
                value = ""
            self.job_outputs[job_id][k] = value
        print(f"✔ job '{job_id}' finished in {time.time() - t_job:.0f}s  outputs={self.job_outputs[job_id]}")
        return True

    def run_workflow(self, wf):
        """Run every job in `needs:` order; a failed job skips the jobs that depend on it."""
        order, needs = job_order(wf)
        result = {}
        for j in order:
            if any(result.get(n) != "success" for n in needs[j]):
                result[j] = "skipped"
                print(f"⏭ job '{j}' skipped because {', '.join(needs[j])} did not succeed\n")
                continue
            result[j] = "success" if self.run_job(wf, j) else "failure"
            print()
        return result
