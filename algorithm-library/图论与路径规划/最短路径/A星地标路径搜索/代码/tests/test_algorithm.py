"""Independent coordinate BFS/Floyd oracles and heuristic properties."""
from collections import deque
from pathlib import Path
import random
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pathfinding import (Grid, METHODS, STRATEGIES, bfs_distances, build_landmarks,
                         make_heuristic, search, touches_obstacle, validate_path)
from datasets import FAMILIES, generate, load_map, make_queries, micro_case, save_map


def coordinate_bfs(rows, source):
    """Never uses production adjacency, component labels, or BFS."""
    w, h = len(rows[0]), len(rows)
    sx, sy = source % w, source // w
    distances = {(sx, sy): 0}
    queue = deque([(sx, sy)])
    while queue:
        x, y = queue.popleft()
        for xx, yy in ((x-1, y), (x+1, y), (x, y-1), (x, y+1)):
            if 0 <= xx < w and 0 <= yy < h and rows[yy][xx] == "." and (xx, yy) not in distances:
                distances[(xx, yy)] = distances[(x, y)] + 1
                queue.append((xx, yy))
    return {y*w+x: d for (x, y), d in distances.items()}


class AlgorithmTests(unittest.TestCase):
    def all_methods(self, grid, start, target):
        expected = coordinate_bfs(grid.rows, start).get(target)
        results = {}
        for method in METHODS:
            index = build_landmarks(grid, method[4:]) if method.startswith("alt_") else None
            result = search(grid, start, target, method, index, record=True)
            self.assertEqual(result.cost, expected)
            validate_path(grid, start, target, result)
            results[method] = result
        return results

    def test_hand_computed_detour(self):
        grid, s, t = micro_case()
        for result in self.all_methods(grid, s, t).values():
            self.assertEqual(result.cost, 8)

    def test_floyd_all_pairs_on_tiny_map(self):
        grid, _, _ = micro_case()
        vertices = grid.free
        d = {(v, u): (0 if v == u else 1 if abs(grid.x[v]-grid.x[u])+abs(grid.y[v]-grid.y[u]) == 1 else 999)
             for v in vertices for u in vertices}
        for mid in vertices:
            for v in vertices:
                for u in vertices:
                    d[v, u] = min(d[v, u], d[v, mid]+d[mid, u])
        indexes = {strategy: build_landmarks(grid, strategy) for strategy in STRATEGIES}
        for s in vertices:
            for t in vertices:
                for method in METHODS:
                    result = search(grid, s, t, method, indexes.get(method[4:]))
                    self.assertEqual(result.cost, d[s, t])

    def test_single_cell(self):
        for result in self.all_methods(Grid((".",)), 0, 0).values():
            self.assertEqual(result.path, [0])
            self.assertEqual(result.expanded, 1)

    def test_disconnected_components(self):
        grid = Grid(("..#..", "..#.."))
        for result in self.all_methods(grid, 0, 4).values():
            self.assertIsNone(result.cost)
            self.assertEqual(result.expanded, 0)
        self.all_methods(grid, 3, 9)

    def test_many_small_components_share_distance_slots(self):
        grid = Grid((".#.#.#.#.", "#########", "..#..#..#"))
        for strategy in STRATEGIES:
            index = build_landmarks(grid, strategy, 4)
            self.assertEqual(len(index.tables), 2)
            self.assertEqual(index.table_payload_bytes, 2*grid.n*index.tables[0].itemsize)
            for cid, landmarks in enumerate(index.landmarks):
                for j, landmark in enumerate(landmarks):
                    expected = coordinate_bfs(grid.rows, landmark)
                    for v in grid.components[cid]:
                        self.assertEqual(index.tables[j][v], expected[v])

    def test_all_walls(self):
        grid = Grid(("##", "##"))
        self.assertEqual(build_landmarks(grid).tables, ())
        with self.assertRaises(ValueError):
            search(grid, 0, 0)

    def test_invalid_maps(self):
        for rows in ((), ("",), ("..", "."), ("x",), (1,), (".", 1)):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                Grid(rows)

    def test_invalid_endpoints(self):
        grid = Grid((".#", ".."))
        for v in (-1, 4, 1, True, 0.0, "0"):
            with self.subTest(v=v), self.assertRaises(ValueError):
                search(grid, v, 3)
        for xy in ((-1, 0), (1, 0), (2, 1), (True, 1)):
            with self.assertRaises(ValueError):
                grid.vertex(*xy)

    def test_invalid_strategy_or_budget(self):
        grid = Grid(("..",))
        for k in (0, -1, 65, True, 2.5):
            with self.assertRaises(ValueError):
                build_landmarks(grid, k=k)
        with self.assertRaises(ValueError):
            build_landmarks(grid, "unknown")

    def test_mismatched_index_rejected(self):
        grid, _, _ = micro_case()
        index = build_landmarks(grid, "farthest")
        for method, supplied in (("alt_corners", index), ("alt_farthest", None), ("invalid", None)):
            with self.assertRaises(ValueError):
                search(grid, 0, 4, method, supplied)
        with self.assertRaises(ValueError):
            search(Grid(grid.rows), 0, 4, "alt_farthest", index)

    def test_heuristics_admissible_and_consistent_all_tiny_targets(self):
        for grid in (micro_case()[0], Grid((".....", ".###.", "....."))):
            for strategy in STRATEGIES:
                index = build_landmarks(grid, strategy)
                for target in grid.free:
                    exact = coordinate_bfs(grid.rows, target)
                    h = make_heuristic(grid, target, "alt_"+strategy, index)
                    self.assertEqual(h(target), 0)
                    for v in exact:
                        self.assertGreaterEqual(h(v), abs(grid.x[v]-grid.x[target])+abs(grid.y[v]-grid.y[target]))
                        self.assertLessEqual(h(v), exact[v])
                        for u in grid.adj[v]:
                            self.assertLessEqual(h(v), 1+h(u))

    def test_bfs_distances_against_independent_oracle(self):
        grid = Grid(generate("random", 17, 20261105))
        for s in grid.free[::31]:
            expected = coordinate_bfs(grid.rows, s)
            actual = bfs_distances(grid, s)
            for v in range(grid.n):
                self.assertEqual(actual[v], expected.get(v, -1))

    def test_families_and_seeded_queries(self):
        for family in FAMILIES:
            grid = Grid(generate(family, 15, 12))
            for s, t in make_queries(grid, 4, 15):
                self.all_methods(grid, s, t)

    def test_random_maps_multiple_components(self):
        rng = random.Random(72)
        for _ in range(18):
            rows = tuple("".join("#" if rng.random() < .38 else "." for _ in range(8)) for _ in range(7))
            grid = Grid(rows)
            for _ in range(3):
                self.all_methods(grid, *rng.choices(grid.free, k=2))

    def test_boundary_candidates_and_fallback(self):
        empty = Grid((".....",)*5)
        a, b = build_landmarks(empty, "farthest"), build_landmarks(empty, "boundary")
        self.assertEqual(a.landmarks, b.landmarks)
        self.assertEqual(b.boundary_candidates, (0,))
        grid, _, _ = micro_case()
        index = build_landmarks(grid, "boundary")
        self.assertEqual(index.landmarks[0][0], grid.free[0])
        self.assertTrue(all(touches_obstacle(grid, v) for v in index.landmarks[0][1:]))

    def test_landmark_prefix_and_no_duplicates(self):
        grid = Grid(generate("random", 15, 47))
        for strategy in STRATEGIES:
            a, b = build_landmarks(grid, strategy, 4), build_landmarks(grid, strategy, 8)
            for small, large in zip(a.landmarks, b.landmarks):
                self.assertEqual(small, large[:len(small)])
                self.assertEqual(len(large), len(set(large)))

    def test_more_landmarks_never_weaken_bound(self):
        grid = Grid(generate("maze", 15, 87))
        target = grid.free[-1]
        for strategy in STRATEGIES:
            hs = [make_heuristic(grid, target, "alt_"+strategy, build_landmarks(grid, strategy, k)) for k in (1, 4, 8)]
            for v in grid.free:
                self.assertLessEqual(hs[0](v), hs[1](v))
                self.assertLessEqual(hs[1](v), hs[2](v))

    def test_trace_and_deterministic_tie_breaking(self):
        grid, s, t = micro_case()
        for method, result in self.all_methods(grid, s, t).items():
            self.assertEqual(result.expanded, len(result.trace))
            self.assertEqual(len(set(step["v"] for step in result.trace)), result.expanded)
            self.assertEqual(result.trace[-1]["v"], t)
            self.assertTrue(all(step["g"]+step["h"] == step["f"] for step in result.trace))
            self.assertEqual([x["f"] for x in result.trace], sorted(x["f"] for x in result.trace))
            index = build_landmarks(grid, method[4:]) if method.startswith("alt_") else None
            again = search(grid, s, t, method, index, record=True)
            self.assertEqual(result, again)

    def test_seed_reproducibility_and_global_rng_untouched(self):
        state = random.getstate()
        for family in FAMILIES:
            self.assertEqual(generate(family, 15, 18), generate(family, 15, 18))
        grid, _, _ = micro_case()
        self.assertEqual(make_queries(grid, 8, 19), make_queries(grid, 8, 19))
        self.assertEqual(state, random.getstate())

    def test_map_file_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"map.txt"
            grid, _, _ = micro_case()
            save_map(path, grid.rows)
            self.assertEqual(load_map(path).rows, grid.rows)


if __name__ == "__main__":
    unittest.main()
