"""A tiny imitation of the Kubernetes control plane, for learning only.

It models the loops that make Kubernetes feel "automatic":
  * Deployment controller  → owns ReplicaSets, drives rolling updates (maxSurge / maxUnavailable)
  * ReplicaSet controller  → keeps exactly `replicas` pods alive  (self-healing)
  * Scheduler              → places Pending pods on a node with room (by CPU requests)
  * Kubelet + probes       → starts containers, marks them ready, restarts unhealthy ones
  * Service                → load-balances across *ready* pods only
  * HorizontalPodAutoscaler→ changes `replicas` from CPU utilisation

Not modelled: the API server, etcd, real networking, storage, RBAC, StatefulSets, DaemonSets, taints,
affinity, PodDisruptionBudgets. One tick = 5 simulated seconds. Real Kubernetes: https://kubernetes.io
"""
import math
from dataclasses import dataclass, field

TICK_SECONDS = 5


@dataclass
class Node:
    name: str
    cpu: float = 2.0            # allocatable cores
    ready: bool = True


@dataclass
class Pod:
    name: str
    rs: str
    image: str
    cpu_request: float = 0.25
    node: str | None = None
    phase: str = "Pending"      # Pending → Running (→ Terminating)
    ready: bool = False
    restarts: int = 0
    created: int = 0
    started: int | None = None


@dataclass
class ReplicaSet:
    name: str
    deployment: str
    image: str
    revision: int
    desired: int = 0


@dataclass
class Deployment:
    name: str
    image: str
    replicas: int
    max_surge: int = 1
    max_unavailable: int = 0
    revision: int = 1
    ready_after: int = 2        # ticks from start to "ready" (startup + readiness probe)
    history: dict = field(default_factory=dict)


class Cluster:
    """`bad` anywhere in an image tag means its readiness probe never succeeds (a broken release)."""

    def __init__(self, nodes=("node-1", "node-2"), node_cpu=2.0, seed=0):
        self.t = 0
        self.nodes = {n: Node(n, node_cpu) for n in nodes}
        self.deployments: dict[str, Deployment] = {}
        self.replicasets: dict[str, ReplicaSet] = {}
        self.pods: dict[str, Pod] = {}
        self.services: dict[str, dict] = {}
        self.hpas: dict[str, dict] = {}
        self.events: list[str] = []
        self.history: list[dict] = []       # one row per tick, for charts
        self.load_rps = 0.0                 # total requests/second hitting the Service
        self.cpu_per_request = 0.02         # cores per request/second
        self._n = 0
        self._rr = 0
        self._cooldown = 0

    # ---------------------------------------------------------------- helpers
    def _log(self, kind, msg):
        self.events.append(f"{self.t * TICK_SECONDS:>4}s  {kind:<10} {msg}")

    def _name(self, prefix):
        self._n += 1
        return f"{prefix}-{self._n:04x}"

    def _pods_of(self, rs_name, alive_only=True):
        return [p for p in self.pods.values() if p.rs == rs_name and (not alive_only or p.phase != "Terminating")]

    def _dep_pods(self, dep):
        rs_names = [r.name for r in self.replicasets.values() if r.deployment == dep]
        return [p for p in self.pods.values() if p.rs in rs_names]

    def ready_pods(self, dep):
        return [p for p in self._dep_pods(dep) if p.ready and p.phase == "Running"]

    # ---------------------------------------------------------------- "kubectl apply"
    def apply(self, manifest: dict):
        kind, meta, spec = manifest["kind"], manifest.get("metadata", {}), manifest.get("spec", {})
        name = meta.get("name")
        if kind == "Deployment":
            ru = (spec.get("strategy", {}) or {}).get("rollingUpdate", {}) or {}
            c = spec["template"]["spec"]["containers"][0]
            cpu = c.get("resources", {}).get("requests", {}).get("cpu", "250m")
            dep = Deployment(name=name, image=c["image"], replicas=spec.get("replicas", 1),
                             max_surge=int(ru.get("maxSurge", 1)), max_unavailable=int(ru.get("maxUnavailable", 1)))
            dep.history[1] = dep.image
            self.deployments[name] = dep
            self._cpu_request = float(cpu[:-1]) / 1000 if str(cpu).endswith("m") else float(cpu)
            self._log("deployment", f"created {name} (replicas={dep.replicas}, image={dep.image.split('/')[-1]})")
        elif kind == "Service":
            self.services[name] = {"selector": spec["selector"], "port": spec["ports"][0]["port"],
                                   "type": spec.get("type", "ClusterIP")}
            ip = "34.118.0.42" if spec.get("type") == "LoadBalancer" else "10.96.0.42"
            self._log("service", f"created {name} ({spec.get('type', 'ClusterIP')}, external IP {ip})")
        elif kind == "HorizontalPodAutoscaler":
            m = spec["metrics"][0]["resource"]["target"]["averageUtilization"]
            self.hpas[spec["scaleTargetRef"]["name"]] = {"min": spec["minReplicas"], "max": spec["maxReplicas"],
                                                         "target": m / 100}
            self._log("hpa", f"created {name} (min={spec['minReplicas']}, max={spec['maxReplicas']}, cpu target={m}%)")
        return self

    # ---------------------------------------------------------------- user actions
    def set_image(self, dep_name, image):
        """kubectl set image … — starts a rolling update."""
        dep = self.deployments[dep_name]
        dep.revision += 1
        dep.image = image
        dep.history[dep.revision] = image
        self._log("rollout", f"{dep_name} → {image.split('/')[-1]} (revision {dep.revision})")
        return self

    def rollout_undo(self, dep_name):
        dep = self.deployments[dep_name]
        prev = max(r for r in dep.history if r < dep.revision)
        self._log("rollout", f"undo {dep_name}: back to revision {prev} ({dep.history[prev].split('/')[-1]})")
        image = dep.history[prev]
        dep.revision += 1
        dep.image = image
        dep.history[dep.revision] = image
        return self

    def scale(self, dep_name, replicas):
        self.deployments[dep_name].replicas = replicas
        self._log("scale", f"{dep_name} → {replicas} replicas")
        return self

    def delete_pod(self, name):
        p = self.pods.pop(name, None)
        if p:
            self._log("kubectl", f"deleted pod {name} (the ReplicaSet will notice)")
        return self

    def fail_node(self, node):
        self.nodes[node].ready = False
        gone = [p for p in self.pods.values() if p.node == node]
        for p in gone:
            self.pods.pop(p.name)
        self._log("node", f"{node} went NotReady — {len(gone)} pod(s) lost")
        return self

    def break_container(self, name):
        """Make one container start failing its liveness probe."""
        self.pods[name].ready = False
        self.pods[name].phase = "Unhealthy"
        self._log("probe", f"liveness probe failing on {name}")
        return self

    # ---------------------------------------------------------------- control loops
    def _deployment_controller(self, dep):
        rs_list = sorted([r for r in self.replicasets.values() if r.deployment == dep.name],
                         key=lambda r: r.revision)
        current = next((r for r in rs_list if r.image == dep.image), None)
        if current is None:
            current = ReplicaSet(self._name(dep.name), dep.name, dep.image, dep.revision)
            self.replicasets[current.name] = current
            self._log("replicaset", f"created {current.name} for revision {dep.revision}")
        others = [r for r in rs_list if r.name != current.name and r.desired > 0]

        if not others:                              # steady state
            current.desired = dep.replicas
            return
        # rolling update
        alive = [p for p in self._dep_pods(dep.name) if p.phase != "Terminating"]
        available = len([p for p in alive if p.ready])
        ready_current = len([p for p in self._pods_of(current.name) if p.ready])

        if ready_current >= dep.replicas and current.desired >= dep.replicas:
            # the new version is fully available → retire every older ReplicaSet (this is also what
            # `kubectl rollout undo` relies on: the good ReplicaSet wins and the bad one goes to 0)
            for r in others:
                self._log("replicaset", f"scaling {r.name} down to 0 (revision {r.revision} retired)")
                r.desired = 0
            return
        if current.desired < dep.replicas and len(alive) < dep.replicas + dep.max_surge:
            current.desired += 1
        if available > dep.replicas - dep.max_unavailable:
            for r in others:
                if r.desired > 0:
                    r.desired -= 1
                    break

    def _replicaset_controller(self, rs):
        pods = self._pods_of(rs.name)
        while len(pods) < rs.desired:
            name = self._name(rs.name)
            self.pods[name] = Pod(name=name, rs=rs.name, image=rs.image,
                                  cpu_request=getattr(self, "_cpu_request", 0.25), created=self.t)
            self._log("replicaset", f"{rs.name} created pod {name} (have {len(pods) + 1}/{rs.desired})")
            pods = self._pods_of(rs.name)
        while len(pods) > rs.desired:
            victim = sorted(pods, key=lambda p: (p.ready, p.created))[0]
            self.pods.pop(victim.name)
            self._log("replicaset", f"{rs.name} removed pod {victim.name}")
            pods = self._pods_of(rs.name)

    def _scheduler(self):
        for p in self.pods.values():
            if p.node:
                continue
            used = {n: sum(q.cpu_request for q in self.pods.values() if q.node == n) for n in self.nodes}
            fits = [n for n, node in self.nodes.items() if node.ready and used[n] + p.cpu_request <= node.cpu]
            if not fits:
                self._log("scheduler", f"{p.name} stays Pending: no node has {p.cpu_request} free CPU")
                continue
            p.node = min(fits, key=lambda n: used[n])     # least-loaded node
            p.started = self.t
            self._log("scheduler", f"assigned {p.name} → {p.node}")

    def _kubelet(self):
        for p in self.pods.values():
            if p.phase == "Unhealthy":                    # liveness probe failed → restart the container
                p.restarts += 1
                p.phase, p.ready, p.started = "Running", False, self.t
                self._log("kubelet", f"restarted {p.name} (restart count {p.restarts})")
            elif p.node and p.phase == "Pending":
                p.phase = "Running"
            if p.phase == "Running" and not p.ready and p.started is not None:
                if self.t - p.started >= self.deployments[self._dep_of(p)].ready_after:
                    if "bad" in p.image:
                        if (self.t - p.started) % 4 == 0:
                            self._log("probe", f"readiness probe failed on {p.name} (image {p.image.split(':')[-1]})")
                    else:
                        p.ready = True
                        self._log("kubelet", f"{p.name} is Ready")

    def _dep_of(self, pod):
        return self.replicasets[pod.rs].deployment

    def _hpa(self, dep):
        cfg = self.hpas.get(dep.name)
        if not cfg or self.t % 3:
            return
        ready = self.ready_pods(dep.name)
        if not ready:
            return
        cpu_each = (self.load_rps * self.cpu_per_request) / len(ready)
        util = cpu_each / ready[0].cpu_request
        desired = max(cfg["min"], min(cfg["max"], math.ceil(len(ready) * util / cfg["target"])))
        if desired > dep.replicas:
            self._log("hpa", f"cpu {util:.0%} of request > target {cfg['target']:.0%} → scale up to {desired}")
            dep.replicas = desired
            self._cooldown = 4                       # stand-in for the 5-minute scale-down stabilisation window
        elif desired < dep.replicas:
            if self._cooldown > 0:
                self._cooldown -= 1
                self._log("hpa", f"cpu {util:.0%} < target, but still inside the stabilisation window "
                                 f"({self._cooldown} check(s) to go)")
            else:
                self._log("hpa", f"cpu {util:.0%} < target for a while → scale down to {desired}")
                dep.replicas = desired

    def tick(self, n=1, quiet=False):
        for _ in range(n):
            self.t += 1
            for dep in self.deployments.values():
                self._hpa(dep)
                self._deployment_controller(dep)
            for rs in list(self.replicasets.values()):
                self._replicaset_controller(rs)
            self._scheduler()
            self._kubelet()
            for dep in self.deployments.values():
                self.history.append({"t": self.t * TICK_SECONDS, "deployment": dep.name,
                                     "desired": dep.replicas,
                                     "pods": len([p for p in self._dep_pods(dep.name)]),
                                     "ready": len(self.ready_pods(dep.name)),
                                     "rps": self.load_rps,
                                     "image": dep.image.split(":")[-1]})
        if not quiet:
            print(self.get_pods())
        return self

    def settle(self, max_ticks=60, stable_for=3):
        """Run until nothing changes for a few ticks, like waiting for `kubectl rollout status`.
        Returns False if it never settles (e.g. a rollout stuck on a failing readiness probe)."""
        stable = 0
        for _ in range(max_ticks):
            before = self._snapshot()
            self.tick(quiet=True)
            stable = stable + 1 if self._snapshot() == before else 0
            if stable >= stable_for:
                return True
        return False

    def rollout_complete(self, dep_name):
        """Like `kubectl rollout status`: are all replicas up to date and available?"""
        dep = self.deployments[dep_name]
        current = [r for r in self.replicasets.values() if r.deployment == dep_name and r.image == dep.image]
        if not current:
            return False
        ready_current = len([p for p in self._pods_of(current[0].name) if p.ready])
        others = sum(r.desired for r in self.replicasets.values()
                     if r.deployment == dep_name and r.name != current[0].name)
        return ready_current >= dep.replicas and others == 0

    def _snapshot(self):
        return sorted((p.name, p.phase, p.ready, p.node, p.restarts) for p in self.pods.values())

    # ---------------------------------------------------------------- "kubectl get"
    def get_pods(self):
        rows = [f"{'NAME':<26}{'READY':<7}{'STATUS':<12}{'RESTARTS':<10}{'NODE':<9}{'IMAGE':<10}{'AGE':>5}"]
        for p in sorted(self.pods.values(), key=lambda p: p.name):
            rows.append(f"{p.name:<26}{'1/1' if p.ready else '0/1':<7}"
                        f"{('Running' if p.ready else p.phase):<12}{p.restarts:<10}"
                        f"{(p.node or '<none>'):<9}{p.image.split(':')[-1]:<10}{(self.t - p.created) * TICK_SECONDS:>4}s")
        return "\n".join(rows)

    def get_deployments(self):
        rows = [f"{'NAME':<16}{'READY':<8}{'UP-TO-DATE':<12}{'AVAILABLE':<11}{'REVISION':<9}IMAGE"]
        for d in self.deployments.values():
            cur = [r for r in self.replicasets.values() if r.deployment == d.name and r.image == d.image]
            up = len(self._pods_of(cur[0].name)) if cur else 0
            rows.append(f"{d.name:<16}{f'{len(self.ready_pods(d.name))}/{d.replicas}':<8}{up:<12}"
                        f"{len(self.ready_pods(d.name)):<11}{d.revision:<9}{d.image.split('/')[-1]}")
        return "\n".join(rows)

    def get_replicasets(self):
        rows = [f"{'NAME':<26}{'DESIRED':<9}{'CURRENT':<9}{'READY':<7}{'REVISION':<9}IMAGE"]
        for r in sorted(self.replicasets.values(), key=lambda r: r.revision):
            pods = self._pods_of(r.name)
            rows.append(f"{r.name:<26}{r.desired:<9}{len(pods):<9}"
                        f"{len([p for p in pods if p.ready]):<7}{r.revision:<9}{r.image.split(':')[-1]}")
        return "\n".join(rows)

    # ---------------------------------------------------------------- Service
    def route(self, dep_name, n_requests):
        """Round-robin across ready pods, the way kube-proxy spreads connections."""
        ready = self.ready_pods(dep_name)
        if not ready:
            return {}, n_requests           # every request fails: 503 from the load balancer
        hits = {p.name: 0 for p in ready}
        for _ in range(n_requests):
            hits[ready[self._rr % len(ready)].name] += 1
            self._rr += 1
        return hits, 0
