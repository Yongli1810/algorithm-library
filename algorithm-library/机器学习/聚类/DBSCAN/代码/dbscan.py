"""Handwritten DBSCAN with interchangeable exact 2-D range queries.

NumPy is used for array arithmetic, not for clustering. All three methods
share the same distance kernel, input validation, and cluster expansion.
"""
from collections import deque
from dataclasses import asdict, dataclass
from math import isfinite, nextafter, inf, hypot
from time import perf_counter

import numpy as np

METHODS = ("brute", "grid", "pruned")


def validate(points, eps, min_samples=5):
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (n, 2)")
    if not np.isfinite(points).all():
        raise ValueError("coordinates must be finite")
    if not isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    if isinstance(min_samples, (bool, np.bool_)) or not isinstance(min_samples, (int, np.integer)) or min_samples < 1:
        raise ValueError("min_samples must be a positive integer")
    return points, float(eps), int(min_samples)


@dataclass
class QueryStats:
    queries: int = 0
    candidate_cells: int = 0
    box_checks: int = 0
    skipped_cells: int = 0
    accepted_cells: int = 0
    distance_evaluations: int = 0
    bulk_accepted_points: int = 0
    neighbor_entries: int = 0
    fallback_queries: int = 0


@dataclass
class Cell:
    ids: np.ndarray
    bounds: tuple | None


class RangeIndex:
    def __init__(self, points, eps, method="pruned"):
        self.points, self.eps, _ = validate(points, eps)
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        self.method = method
        self.stats = QueryStats()
        self.all_ids = np.arange(len(self.points), dtype=np.intp)
        self.cells = {}
        self.eps_num, self.eps_den = self.eps.as_integer_ratio()
        # Conservatively widen cell lookup and tighten whole-box acceptance.
        # The final point test remains hypot(dx, dy) <= eps, with no tolerance.
        self.outer = nextafter(nextafter(self.eps, inf), inf)
        self.inner = nextafter(nextafter(self.eps, 0.0), 0.0)
        if method != "brute":
            groups = {}
            for i, (x, y) in enumerate(self.points):
                groups.setdefault((self.cell_index(float(x)), self.cell_index(float(y))), []).append(i)
            for key, ids in groups.items():
                ids = np.asarray(ids, dtype=np.intp)
                bounds = None
                if method == "pruned":
                    block = self.points[ids]
                    lo, hi = block.min(axis=0), block.max(axis=0)
                    bounds = (float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1]))
                self.cells[key] = Cell(ids, bounds)

    def cell_index(self, coordinate):
        """Exact floor of the ratio of the two stored binary floats.

        This avoids int truncation for negatives and division rounding at a
        grid boundary. Python integer arithmetic also avoids int64 overflow.
        """
        a, b = coordinate.as_integer_ratio()
        return (a * self.eps_den) // (b * self.eps_num)

    def candidate_keys(self, point):
        ranges = []
        for value in point:
            v = float(value)
            lo = nextafter(v - self.outer, -inf)
            hi = nextafter(v + self.outer, inf)
            if not isfinite(lo) or not isfinite(hi):
                self.stats.fallback_queries += 1
                return list(self.cells)
            first, last = self.cell_index(lo), self.cell_index(hi)
            # An extreme coordinate/eps ratio may put many representational
            # cells between adjacent floats. Scan occupied cells in that case.
            if last - first > 8:
                self.stats.fallback_queries += 1
                return list(self.cells)
            ranges.append(range(first, last + 1))
        return [(x, y) for x in ranges[0] for y in ranges[1] if (x, y) in self.cells]

    def box_action(self, point, bounds):
        x, y = map(float, point)
        x0, y0, x1, y1 = bounds
        dx, dy = max(x0 - x, 0.0, x - x1), max(y0 - y, 0.0, y - y1)
        fx, fy = max(abs(x - x0), abs(x - x1)), max(abs(y - y0), abs(y - y1))
        lower, upper = hypot(dx, dy), hypot(fx, fy)
        if lower > self.outer:
            return "skip"
        if upper < self.inner:
            return "accept"
        return "test"

    def _filter(self, point, ids):
        self.stats.distance_evaluations += len(ids)
        with np.errstate(over="ignore", invalid="ignore"):
            delta = self.points[ids] - point
            distance = np.hypot(delta[:, 0], delta[:, 1])
        return ids[distance <= self.eps]

    def query(self, point_id, explain=False):
        if not 0 <= point_id < len(self.points):
            raise IndexError("point_id out of range")
        before = asdict(self.stats) if explain else None
        self.stats.queries += 1
        point = self.points[point_id]
        details = []
        if self.method == "brute":
            neighbors = self._filter(point, self.all_ids)
        else:
            candidates, accepted = [], []
            for key in self.candidate_keys(point):
                cell = self.cells[key]
                self.stats.candidate_cells += 1
                action = "test"
                if self.method == "pruned":
                    self.stats.box_checks += 1
                    action = self.box_action(point, cell.bounds)
                if action == "skip":
                    self.stats.skipped_cells += 1
                elif action == "accept":
                    self.stats.accepted_cells += 1
                    self.stats.bulk_accepted_points += len(cell.ids)
                    accepted.append(cell.ids)
                else:
                    candidates.append(cell.ids)
                if explain:
                    details.append({"key": list(key), "action": action,
                                    "count": len(cell.ids), "bounds": cell.bounds})
            if candidates:
                accepted.append(self._filter(point, np.concatenate(candidates)))
            neighbors = np.sort(np.concatenate(accepted)) if accepted else np.empty(0, dtype=np.intp)
        self.stats.neighbor_entries += len(neighbors)
        if explain:
            delta = {key: value - before[key] for key, value in asdict(self.stats).items()}
            return neighbors, {"cells": details, "counts": delta}
        return neighbors


@dataclass
class Result:
    labels: np.ndarray
    core: np.ndarray
    stats: dict
    timings: dict
    frames: list

    @property
    def clusters(self):
        return len(set(self.labels.tolist()) - {-1, -2})


def fit(points, eps, min_samples=5, method="pruned", trace=False):
    start = perf_counter()
    points, eps, min_samples = validate(points, eps, min_samples)
    n = len(points)
    index_start = perf_counter()
    index = RangeIndex(points, eps, method)
    index_seconds = perf_counter() - index_start
    labels = np.full(n, -2, dtype=np.intp)  # -2 unassigned; -1 noise
    visited, enqueued, core = (np.zeros(n, dtype=bool) for _ in range(3))
    frames, query_seconds = [], 0.0
    cluster = 0

    def get_neighbors(i):
        nonlocal query_seconds
        tick = perf_counter()
        neighbors = index.query(i)
        query_seconds += perf_counter() - tick
        core[i] = len(neighbors) >= min_samples  # self is included
        visited[i] = True
        return neighbors

    def frame(i, neighbors, queried):
        if trace:
            frames.append({"point": int(i), "queried": queried,
                           "neighbors": neighbors.tolist(), "core": bool(core[i]),
                           "labels": labels.tolist(), "visited": int(visited.sum()),
                           "distance_evaluations": index.stats.distance_evaluations})

    for seed in range(n):
        if visited[seed]:
            continue
        neighbors = get_neighbors(seed)
        if not core[seed]:
            labels[seed] = -1
            frame(seed, neighbors, True)
            continue
        labels[seed] = cluster
        enqueued[seed] = True
        fresh = neighbors[~enqueued[neighbors]]
        enqueued[fresh] = True
        queue = deque(fresh.tolist())
        frame(seed, neighbors, True)
        while queue:
            i = queue.popleft()
            queried = not visited[i]
            neighbors = np.empty(0, dtype=np.intp)
            if queried:
                neighbors = get_neighbors(i)
                if core[i]:
                    fresh = neighbors[~enqueued[neighbors]]
                    enqueued[fresh] = True
                    queue.extend(fresh.tolist())
            if labels[i] < 0:
                labels[i] = cluster
            frame(i, neighbors, bool(queried))
        cluster += 1
    total = perf_counter() - start
    return Result(labels, core, asdict(index.stats),
                  {"total_seconds": total, "index_seconds": index_seconds,
                   "query_seconds": query_seconds}, frames)


def canonical_labels(labels):
    """Compare partitions without relying on arbitrary cluster numbers."""
    mapping, answer = {}, []
    for label in map(int, labels):
        if label < 0:
            answer.append(label)
        else:
            if label not in mapping:
                mapping[label] = len(mapping)
            answer.append(mapping[label])
    return np.asarray(answer, dtype=np.intp)


def reference_fit(points, eps, min_samples=5):
    """External practical reference, explicitly KDTree and one worker."""
    from sklearn.cluster import DBSCAN
    start = perf_counter()
    points, eps, min_samples = validate(points, eps, min_samples)
    model = DBSCAN(eps=eps, min_samples=min_samples, algorithm="kd_tree",
                   leaf_size=40, metric="euclidean", n_jobs=1).fit(points)
    total = perf_counter() - start
    core = np.zeros(len(points), dtype=bool)
    core[model.core_sample_indices_] = True
    return Result(model.labels_.copy(), core, {}, {"total_seconds": total}, [])
