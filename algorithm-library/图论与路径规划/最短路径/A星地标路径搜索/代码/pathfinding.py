"""Exact shortest paths on static, undirected four-neighbour unit grids.

The A* kernel and tie-breaking are shared by all five methods.  Landmark
selection never sees the benchmark queries.  Only Python's standard library
is needed for the algorithm and tests.
"""
from array import array
from collections import deque
from dataclasses import dataclass
from heapq import heappop, heappush

METHODS = ("dijkstra", "manhattan", "alt_corners", "alt_farthest", "alt_boundary")
STRATEGIES = ("corners", "farthest", "boundary")


class Grid:
    """Immutable map rows, deterministic adjacency, and connected components."""

    def __init__(self, rows):
        rows = tuple(rows)
        if not rows or not isinstance(rows[0], str) or not rows[0]:
            raise ValueError("Map must have at least one nonempty row")
        if any(not isinstance(row, str) or len(row) != len(rows[0]) for row in rows):
            raise ValueError("Map rows must be rectangular strings")
        if any(set(row) - {".", "#"} for row in rows):
            raise ValueError("Use '.' for free cells and '#' for walls")
        self.rows, self.height, self.width = rows, len(rows), len(rows[0])
        self.n = self.width * self.height
        if self.n >= 2**31:
            raise ValueError("Map is too large for signed 32-bit distances")
        self.x = tuple(v % self.width for v in range(self.n))
        self.y = tuple(v // self.width for v in range(self.n))
        free = tuple(v for v in range(self.n) if rows[self.y[v]][self.x[v]] == ".")
        self.free = free
        allowed = set(free)
        adj = [()] * self.n
        for v in free:
            x, y = self.x[v], self.y[v]
            # Up, left, right, down, independent of method.
            adj[v] = tuple(u for u, valid in ((v-self.width, y > 0), (v-1, x > 0),
                                            (v+1, x+1 < self.width), (v+self.width, y+1 < self.height))
                           if valid and u in allowed)
        self.adj = tuple(adj)
        labels, components = [-1] * self.n, []
        for start in free:
            if labels[start] >= 0:
                continue
            cid = len(components)
            labels[start] = cid
            queue, component = deque([start]), []
            while queue:
                v = queue.popleft()
                component.append(v)
                for u in adj[v]:
                    if labels[u] < 0:
                        labels[u] = cid
                        queue.append(u)
            components.append(tuple(sorted(component)))
        self.component = tuple(labels)
        self.components = tuple(components)

    def vertex(self, x, y):
        if type(x) is not int or type(y) is not int or not (0 <= x < self.width and 0 <= y < self.height):
            raise ValueError("Coordinates must be in-range integers (x, y)")
        v = y * self.width + x
        self.check_vertex(v)
        return v

    def check_vertex(self, v):
        if type(v) is not int or not 0 <= v < self.n or self.component[v] < 0:
            raise ValueError("Endpoint must be a free cell with a valid integer id")


def _fill_distances(grid, start, distance):
    """Fill one component of a shared distance slot; other components survive."""
    distance[start] = 0
    queue = deque([start])
    while queue:
        v = queue.popleft()
        next_distance = distance[v] + 1
        for u in grid.adj[v]:
            if distance[u] < 0:
                distance[u] = next_distance
                queue.append(u)


def bfs_distances(grid, start):
    grid.check_vertex(start)
    distance = array("i", [-1]) * grid.n
    _fill_distances(grid, start, distance)
    return distance


def touches_obstacle(grid, v):
    """Only actual wall cells count; outside the rectangle is not an obstacle."""
    x, y, w, h = grid.x[v], grid.y[v], grid.width, grid.height
    return ((x > 0 and grid.rows[y][x-1] == "#") or
            (x+1 < w and grid.rows[y][x+1] == "#") or
            (y > 0 and grid.rows[y-1][x] == "#") or
            (y+1 < h and grid.rows[y+1][x] == "#"))


@dataclass(frozen=True)
class LandmarkIndex:
    grid: Grid
    strategy: str
    k: int
    landmarks: tuple
    tables: tuple
    boundary_candidates: tuple

    @property
    def table_payload_bytes(self):
        """Exact array payload, not total Python heap usage or peak RSS."""
        return sum(table.itemsize * len(table) for table in self.tables)


def build_landmarks(grid, strategy="farthest", k=4):
    if strategy not in STRATEGIES:
        raise ValueError("Unknown landmark strategy")
    if type(k) is not int or not 1 <= k <= 64:
        raise ValueError("k must be an integer from 1 through 64")
    # Slot j stores the j-th landmark's distances in each disconnected component.
    # Thus storage is O(k*N), not O(k*number_of_components*N).
    slots = min(k, max(map(len, grid.components), default=0))
    tables = tuple(array("i", [-1]) * grid.n for _ in range(slots))
    all_landmarks, candidate_counts = [], []
    corners = ((0, 0), (grid.width-1, 0), (0, grid.height-1), (grid.width-1, grid.height-1))
    for component in grid.components:
        chosen, chosen_set = [], set()
        nearest = {v: grid.n + 1 for v in component}
        boundary = tuple(v for v in component if touches_obstacle(grid, v)) if strategy == "boundary" else ()
        candidate_counts.append(len(boundary))
        for j in range(min(k, len(component))):
            available = [v for v in component if v not in chosen_set]
            if strategy == "corners" and j < 4:
                x, y = corners[j]
                landmark = min(available, key=lambda v: (abs(grid.x[v]-x)+abs(grid.y[v]-y), v))
            elif j == 0:
                landmark = component[0]  # same anchor for both farthest strategies
            else:
                candidates = [v for v in boundary if v not in chosen_set] if strategy == "boundary" else []
                if not candidates:
                    candidates = available  # explicit fallback for open/small components
                landmark = max(candidates, key=lambda v: (nearest[v], -v))
            chosen.append(landmark)
            chosen_set.add(landmark)
            _fill_distances(grid, landmark, tables[j])
            for v in component:
                nearest[v] = min(nearest[v], tables[j][v])
        all_landmarks.append(tuple(chosen))
    return LandmarkIndex(grid, strategy, k, tuple(all_landmarks), tables, tuple(candidate_counts))


def make_heuristic(grid, target, method, index=None):
    grid.check_vertex(target)
    if method not in METHODS:
        raise ValueError("Unknown search method")
    tx, ty, xs, ys = grid.x[target], grid.y[target], grid.x, grid.y
    if method == "dijkstra":
        return lambda v: 0
    if method == "manhattan":
        return lambda v: abs(xs[v]-tx) + abs(ys[v]-ty)
    if index is None or index.grid is not grid or method != "alt_" + index.strategy:
        raise ValueError("ALT needs the matching index for this exact Grid instance")
    count = len(index.landmarks[grid.component[target]])
    tables = index.tables[:count]
    targets = tuple(table[target] for table in tables)

    def lower_bound(v):
        result = abs(xs[v]-tx) + abs(ys[v]-ty)
        if grid.component[v] != grid.component[target]:
            return result
        for table, to_target in zip(tables, targets):
            # Public helper is also defined outside target's component. Search
            # itself rejects unreachable endpoint pairs before using this bound.
            candidate = abs(table[v] - to_target)
            if candidate > result:
                result = candidate
        return result
    return lower_bound


@dataclass
class SearchResult:
    cost: int | None
    path: list
    expanded: int
    pushes: int
    stale_pops: int
    max_heap: int
    heuristic_evaluations: int
    trace: list


def search(grid, start, target, method="manhattan", index=None, record=False):
    grid.check_vertex(start)
    grid.check_vertex(target)
    heuristic = make_heuristic(grid, target, method, index)
    if grid.component[start] != grid.component[target]:
        return SearchResult(None, [], 0, 0, 0, 0, 0, [])
    g = [-1] * grid.n
    parent = [-1] * grid.n
    cached_h = [-1] * grid.n
    closed = bytearray(grid.n)
    g[start], cached_h[start] = 0, heuristic(start)
    heap = [(cached_h[start], 0, start)]
    expanded, pushes, stale, peak, evaluations = 0, 1, 0, 1, 1
    trace = []
    while heap:
        f, minus_g, v = heappop(heap)
        if closed[v] or -minus_g != g[v]:
            stale += 1
            continue
        closed[v] = 1
        expanded += 1  # includes the goal, uniformly across methods
        if record:
            trace.append({"v": v, "g": g[v], "h": cached_h[v], "f": f})
        if v == target:
            path, u = [], target
            while u != -1:
                path.append(u)
                u = parent[u]
            path.reverse()
            return SearchResult(g[v], path, expanded, pushes, stale, peak, evaluations, trace)
        for u in grid.adj[v]:
            next_g = g[v] + 1
            if not closed[u] and (g[u] < 0 or next_g < g[u]):
                g[u], parent[u] = next_g, v
                if cached_h[u] < 0:
                    cached_h[u] = heuristic(u)
                    evaluations += 1
                # Min f, then larger g, then smaller row-major id.
                heappush(heap, (next_g+cached_h[u], -next_g, u))
                pushes += 1
        peak = max(peak, len(heap))
    return SearchResult(None, [], expanded, pushes, stale, peak, evaluations, trace)


def validate_path(grid, start, target, result):
    if result.cost is None:
        if result.path or grid.component[start] == grid.component[target]:
            raise AssertionError("False unreachable result")
        return
    assert result.path[0] == start and result.path[-1] == target
    assert result.cost == len(result.path)-1
    assert len(set(result.path)) == len(result.path)
    for v in result.path:
        grid.check_vertex(v)
    for v, u in zip(result.path, result.path[1:]):
        assert abs(grid.x[v]-grid.x[u]) + abs(grid.y[v]-grid.y[u]) == 1
