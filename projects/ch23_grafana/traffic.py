"""Generate traffic against the instrumented API so the dashboards have something to show.

Five phases, so the graphs tell a story:
  1. baseline   – quiet, mixed comments
  2. spike      – a burst of traffic (watch latency and in-flight)
  3. errors     – malformed requests → 400s
  4. drift      – the incoming comments turn negative; the *prediction mix* moves, with no code change
  5. recovery   – back to baseline
"""
import json
import random
import sys
import threading
import time
import urllib.error
import urllib.request

API = "http://127.0.0.1:5002"

POSITIVE = ["great video, thanks a lot", "this finally made mlops click for me", "excellent explanation",
            "loved the docker part", "really helpful, subscribed", "best course i have watched"]
NEUTRAL = ["where can i find the dataset", "which python version is this", "is there a github link",
           "timestamp for the dvc section", "does this work on windows", "watching from india"]
NEGATIVE = ["terrible audio, i cannot hear anything", "this does not work with the latest version",
            "waste of time, too slow", "worst explanation, very confusing", "broken code and no support"]


def post(path, payload, timeout=10):
    req = urllib.request.Request(API + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code
    except Exception:
        return 0


def sample(mix, n):
    """mix = (positive, neutral, negative) weights."""
    pools, weights = (POSITIVE, NEUTRAL, NEGATIVE), mix
    return [random.choice(random.choices(pools, weights=weights)[0]) for _ in range(n)]


def burst(mix, requests_per_second, seconds, batch=(1, 8), bad=0.0, workers=4):
    stop = time.time() + seconds
    delay = workers / max(requests_per_second, 0.01)

    def worker():
        while time.time() < stop:
            if random.random() < bad:
                post("/predict", {"wrong_key": "oops"})          # 400: the client sent nonsense
            else:
                post("/predict", {"comments": sample(mix, random.randint(*batch))})
            time.sleep(max(delay * random.uniform(0.6, 1.4), 0.01))

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(workers)]
    [t.start() for t in threads]
    [t.join() for t in threads]


PHASES = [
    ("baseline",  dict(mix=(5, 4, 1), requests_per_second=2,  seconds=60, batch=(1, 6))),
    ("spike",     dict(mix=(5, 4, 1), requests_per_second=12, seconds=45, batch=(5, 25), workers=8)),
    ("errors",    dict(mix=(5, 4, 1), requests_per_second=4,  seconds=30, batch=(1, 6), bad=0.35)),
    ("drift",     dict(mix=(1, 2, 8), requests_per_second=3,  seconds=75, batch=(1, 8))),
    ("recovery",  dict(mix=(5, 4, 1), requests_per_second=2,  seconds=30, batch=(1, 6))),
]


def run(scale=1.0, log=print):
    marks = []
    for name, kwargs in PHASES:
        kwargs = {**kwargs, "seconds": max(int(kwargs["seconds"] * scale), 5)}
        log(f"  {time.strftime('%H:%M:%S')}  {name:<9} {kwargs['requests_per_second']} rps for {kwargs['seconds']}s"
            + ("  (35% malformed requests)" if kwargs.get("bad") else "")
            + ("  (comments turn negative)" if name == "drift" else ""))
        marks.append((name, time.time()))
        burst(**kwargs)
    marks.append(("end", time.time()))
    return marks


if __name__ == "__main__":
    run(scale=float(sys.argv[1]) if len(sys.argv) > 1 else 1.0)
