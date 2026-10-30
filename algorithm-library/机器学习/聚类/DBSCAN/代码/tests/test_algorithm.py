"""Independent graph oracle, boundary regressions, and pruning invariants."""
from collections import deque
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from dbscan import METHODS, RangeIndex, canonical_labels, fit, reference_fit
from datasets import FAMILIES, PARAMETERS, generate, load_csv, save_csv


def oracle(points, eps, min_samples):
    # Deliberately independent: scalar distances, core-component graph, then
    # border assignment to the first component in original core-point order.
    neighbors = [[j for j, q in enumerate(points) if math.hypot(float(p[0]) - float(q[0]), float(p[1]) - float(q[1])) <= eps]
                 for p in points]
    core = np.array([len(row) >= min_samples for row in neighbors])
    labels = np.full(len(points), -1)
    cluster = 0
    for i in range(len(points)):
        if not core[i] or labels[i] >= 0:
            continue
        labels[i] = cluster
        queue = deque([i])
        while queue:
            v = queue.popleft()
            for j in neighbors[v]:
                if core[j] and labels[j] < 0:
                    labels[j] = cluster
                    queue.append(j)
        cluster += 1
    for i, row in enumerate(neighbors):
        if not core[i]:
            possible = [labels[j] for j in row if core[j]]
            if possible:
                labels[i] = min(possible)
    return labels, core, neighbors


class AlgorithmTests(unittest.TestCase):
    def check_case(self, points, eps, m, external=False):
        points = np.asarray(points, dtype=float).reshape(-1, 2)
        labels, core, neighbors = oracle(points, eps, m)
        for mode in METHODS:
            index = RangeIndex(points, eps, mode)
            for i, expected in enumerate(neighbors):
                np.testing.assert_array_equal(index.query(i), expected, err_msg=f"{mode}, point {i}")
            result = fit(points, eps, m, mode)
            np.testing.assert_array_equal(result.labels, labels)
            np.testing.assert_array_equal(result.core, core)
            self.assertEqual(result.stats["queries"], len(points))
        if external and len(points):
            result = reference_fit(points, eps, m)
            np.testing.assert_array_equal(canonical_labels(result.labels), canonical_labels(labels))
            np.testing.assert_array_equal(result.core, core)

    def test_hand_computed_case(self):
        points = [[0, 0], [0.1, 0], [0, 0.1], [0.1, 0.1], [0.28, 0],
                  [2, 2], [2.1, 2], [2, 2.1], [2.1, 2.1], [5, 5]]
        self.check_case(points, 0.2, 4, True)
        result = fit(points, 0.2, 4)
        self.assertEqual(result.labels.tolist(), [0, 0, 0, 0, 0, 1, 1, 1, 1, -1])
        self.assertFalse(result.core[4])

    def test_empty_and_single(self):
        self.check_case([], 1, 1)
        self.check_case([[0, 0]], 1, 1, True)
        self.check_case([[0, 0]], 1, 2, True)

    def test_duplicates_and_self_count(self):
        self.check_case([[0, 0]] * 5 + [[3, 3]], 0.1, 5, True)

    def test_negative_coordinates(self):
        self.check_case([[-1.01, -1], [-1, -1], [-0.99, -1], [0, 0]], 0.03, 3, True)

    def test_exact_circle_boundary(self):
        self.check_case([[0, 0], [3, 4], [-3, -4], [5, 0], [0, -5]], 5, 2, True)

    def test_float_boundary_crossing_two_cells(self):
        # hypot(1 - (-smallest_subnormal), 0) rounds to exactly 1. A naive
        # nine-cell lookup around floor(-tiny / 1) would miss cell +1.
        tiny = np.nextafter(0.0, -np.inf)
        self.check_case([[tiny, 0], [1, 0], [np.nextafter(1., np.inf), 0]], 1, 2)

    def test_one_ulp_inside_and_outside(self):
        for eps in (0.1, 1., 1e-100, 1e100):
            self.check_case([[0, 0], [np.nextafter(eps, 0), 0], [eps, 0], [np.nextafter(eps, np.inf), 0]], eps, 2)

    def test_extreme_scale_fallback(self):
        self.check_case([[1e100, 1e100], [1e100, 1e100], [-1e100, 0]], 1e-100, 2)

    def test_subnormal_eps(self):
        eps = np.nextafter(0., 1.)
        self.check_case([[0, 0], [eps, 0], [2 * eps, 0]], eps, 2)

    def test_all_noise(self):
        self.check_case([[0, 0], [2, 2], [4, 4]], 0.1, 2, True)

    def test_all_mutual_neighbors(self):
        self.check_case(generate("dense", 50, 7), 0.2, 8, True)

    def test_previously_noise_becomes_border(self):
        self.check_case([[0, 0], [0.9, 0], [1., 0], [1.05, 0.02], [1.1, 0]], 1, 4, True)

    def test_ambiguous_border_does_not_join_components(self):
        # Point zero touches one core in each component, but is itself not core.
        points = [[0, 0], [-0.9, 0], [-1.5, 0], [-1.5, 0.1], [-1.5, -0.1],
                  [0.9, 0], [1.5, 0], [1.5, 0.1], [1.5, -0.1]]
        self.check_case(points, 1, 4, True)
        self.assertEqual(fit(points, 1, 4).clusters, 2)

    def test_seeded_graph_oracle(self):
        for seed in range(12):
            points = np.random.default_rng(seed).normal(size=(45, 2))
            for eps in (0.1, 0.5, 1.0):
                self.check_case(points, eps, 4, True)

    def test_families(self):
        for family in FAMILIES:
            self.check_case(generate(family, 120, 812), **{"eps": PARAMETERS[family]["eps"], "m": 8}, external=True)

    def test_all_three_box_actions(self):
        points = np.array([[0.01, 0.01], [0.05, 0.01], [0.95, 0.95], [1.9, 1.9], [1.0, 0]])
        index = RangeIndex(points, 1, "pruned")
        for i in range(len(points)):
            index.query(i)
        for key in ("skipped_cells", "accepted_cells", "distance_evaluations"):
            self.assertGreater(getattr(index.stats, key), 0, key)

    def test_counter_accounting(self):
        points = generate("blobs", 100, 41)
        base = fit(points, 0.075, 8, "brute")
        self.assertEqual(base.stats["distance_evaluations"], 10000)
        grid, pruned = (fit(points, 0.075, 8, m) for m in ("grid", "pruned"))
        self.assertLessEqual(pruned.stats["distance_evaluations"], grid.stats["distance_evaluations"])
        self.assertLessEqual(grid.stats["distance_evaluations"], 10000)
        self.assertEqual(base.stats["neighbor_entries"], pruned.stats["neighbor_entries"])

    def test_trace_finishes_at_result(self):
        result = fit(generate("blobs", 80, 7), 0.12, 4, trace=True)
        self.assertEqual(result.frames[-1]["labels"], result.labels.tolist())
        self.assertEqual(result.frames[-1]["visited"], 80)

    def test_csv_round_trip(self):
        points = generate("moons", 80, 47)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "points.csv"
            save_csv(path, points)
            np.testing.assert_array_equal(load_csv(path), points)

    def test_invalid_input(self):
        for points in ([1, 2], [[1, 2, 3]], [[np.nan, 0]], [[np.inf, 0]]):
            with self.assertRaises(ValueError):
                fit(points, 1)
        for eps in (0, -1, np.nan, np.inf):
            with self.assertRaises(ValueError):
                fit([[0, 0]], eps)
        for m in (0, -1, 1.2, True):
            with self.assertRaises(ValueError):
                fit([[0, 0]], 1, m)
        with self.assertRaises(ValueError):
            fit([[0, 0]], 1, method="unknown")


if __name__ == "__main__":
    unittest.main()
