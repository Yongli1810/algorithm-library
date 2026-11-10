import csv
from dataclasses import replace
import itertools
import math
from pathlib import Path
import random
import tempfile
import unittest

from annealing import (MODES, Problem, SAConfig, acceptance_probability, exact_tsp,
                       nearest_neighbor, solve, tour_length, two_opt_delta, validate_tour)
from datasets import generate_points, load_points, write_points
from experiments import bootstrap_interval, summarize


class AlgorithmTests(unittest.TestCase):
    def setUp(self):
        self.problem = generate_points(9, "uniform", 13)
        self.config = SAConfig(evaluations=500, patience=40, periodic_interval=100,
                               trace_every=7, calibration_samples=30)

    def test_delta_matches_full_recomputation_for_every_segment(self):
        rng = random.Random(29)
        for layout in ("uniform", "clustered"):
            p = generate_points(12, layout, 8)
            for _ in range(8):
                tail = list(range(1, 12))
                rng.shuffle(tail)
                route = [0] + tail
                before = tour_length(route, p.distances)
                for i in range(1, 11):
                    for j in range(i + 1, 12):
                        candidate = route[:i] + list(reversed(route[i:j + 1])) + route[j + 1:]
                        self.assertAlmostEqual(before + two_opt_delta(route, p.distances, i, j),
                                               tour_length(candidate, p.distances), places=9)

    def test_square_manual_example(self):
        p = Problem(((0, 0), (1, 0), (1, 1), (0, 1)))
        self.assertAlmostEqual(tour_length([0, 2, 1, 3], p.distances), 2 + 2 * math.sqrt(2))
        self.assertAlmostEqual(two_opt_delta([0, 2, 1, 3], p.distances, 1, 2), 2 - 2 * math.sqrt(2))
        self.assertEqual(exact_tsp(p)[1], 4)

    def test_exact_solver_matches_exhaustive_permutations(self):
        p = generate_points(7, "uniform", 55)
        exhaustive = min(tour_length([0] + list(tail), p.distances)
                         for tail in itertools.permutations(range(1, 7)))
        route, optimum = exact_tsp(p)
        validate_tour(route, 7)
        self.assertAlmostEqual(optimum, exhaustive, places=10)
        self.assertAlmostEqual(tour_length(route, p.distances), optimum, places=10)

    def test_best_tour_never_loses_initial_or_beats_oracle(self):
        _, optimum = exact_tsp(self.problem)
        for mode in MODES:
            result = solve(self.problem, replace(self.config, mode=mode), 42)
            validate_tour(result["best_route"], 9)
            self.assertGreaterEqual(result["best_length"] + 1e-8, optimum)
            self.assertLessEqual(result["best_length"], result["initial_length"] + 1e-8)
            self.assertAlmostEqual(result["best_length"], tour_length(result["best_route"], self.problem.distances))
            lengths = [row["best_length"] for row in result["trace"]]
            self.assertTrue(all(b <= a + 1e-9 for a, b in zip(lengths, lengths[1:])))

    def test_trace_matches_stored_tours_and_exact_budget(self):
        result = solve(self.problem, self.config)
        self.assertEqual(result["evaluations"], 500)
        self.assertEqual(result["calibration_evaluations"], 30)
        self.assertEqual(result["trace"][0]["evaluation"], 0)
        self.assertEqual(result["trace"][-1]["evaluation"], 500)
        steps = [r["evaluation"] for r in result["trace"]]
        self.assertEqual(steps, sorted(set(steps)))
        for row in result["trace"]:
            self.assertAlmostEqual(row["current_length"], tour_length(row["current_route"], self.problem.distances), places=8)
            self.assertAlmostEqual(row["best_length"], tour_length(row["best_route"], self.problem.distances), places=8)

    def test_seed_reproducibility(self):
        first, second = solve(self.problem, self.config, 8), solve(self.problem, self.config, 8)
        first.pop("elapsed_seconds")
        second.pop("elapsed_seconds")
        self.assertEqual(first, second)

    def test_global_random_state_untouched(self):
        state = random.getstate()
        solve(self.problem, self.config)
        self.assertEqual(state, random.getstate())

    def test_reheating_disabled_is_identical_to_baseline(self):
        results = [solve(self.problem, replace(self.config, mode=mode, max_reheats=0), 19) for mode in MODES]
        for result in results[1:]:
            self.assertEqual(result["trace"], results[0]["trace"])
            self.assertEqual(result["best_route"], results[0]["best_route"])

    def test_calibration_and_initial_state_paired(self):
        results = [solve(self.problem, replace(self.config, mode=mode), 51) for mode in MODES]
        self.assertEqual(len({r["initial_temperature"] for r in results}), 1)
        self.assertEqual(len({tuple(r["initial_route"]) for r in results}), 1)

    def test_zero_distance_problem_and_reheat_limits(self):
        p = Problem(((1, 1),) * 6)
        for mode in MODES:
            c = replace(self.config, mode=mode, max_reheats=2, reheat_until_fraction=0.75)
            r = solve(p, c, 9)
            self.assertEqual(r["best_length"], 0)
            self.assertLessEqual(r["reheats"], 2)
            for e in r["events"]:
                self.assertGreater(e["after"], e["before"])
                self.assertLessEqual(e["after"], r["initial_temperature"])
                self.assertLessEqual(e["evaluation"], int(c.evaluations * 0.75))
                if mode == "stagnation":
                    self.assertGreaterEqual(e["stale_evaluations"], c.patience)
                elif mode == "periodic":
                    self.assertEqual(e["evaluation"] % c.periodic_interval, 0)
            if mode != "baseline":
                self.assertGreater(r["reheats"], 0)

    def test_one_evaluation_terminates(self):
        r = solve(self.problem, replace(self.config, evaluations=1, trace_every=100))
        self.assertEqual([x["evaluation"] for x in r["trace"]], [0, 1])
        self.assertEqual(r["reheats"], 0)

    def test_reheated_schedules_reach_common_final_temperature(self):
        p = Problem(((1, 1),) * 6)
        for mode in MODES:
            r = solve(p, replace(self.config, mode=mode), 9)
            target = r["initial_temperature"] * self.config.final_temperature_ratio
            self.assertAlmostEqual(r["trace"][-1]["next_temperature"] / target, 1, places=9)

    def test_acceptance_rule(self):
        self.assertEqual(acceptance_probability(-1, 0), 1)
        self.assertEqual(acceptance_probability(1, 0), 0)
        self.assertAlmostEqual(acceptance_probability(2, 1), math.exp(-2))
        self.assertEqual(acceptance_probability(1e100, 1e-100), 0)

    def test_invalid_parameters_rejected(self):
        for overrides in ({"evaluations": 0}, {"patience": 0}, {"max_reheats": -1},
                          {"mode": "unknown"}, {"initial_acceptance": 1},
                          {"reheat_multiplier": 1}, {"final_temperature_ratio": float("nan")},
                          {"evaluations": True}, {"trace_every": -1}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                replace(self.config, **overrides).validate()

    def test_bad_points_and_routes_rejected(self):
        for points in (((0, 0),) * 3, ((float("nan"), 0),) * 4,
                       ((float("inf"), 0),) * 4, ((1, 2, 3),) * 4):
            with self.assertRaises(ValueError):
                Problem(points)
        for route in ([0, 1, 1, 3], [1, 0, 2, 3], [0, 1, 2], [0, 1, 2, 3.0]):
            with self.assertRaises(ValueError):
                validate_tour(route, 4)

    def test_csv_roundtrip_and_input_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "points.csv"
            write_points(self.problem, path)
            self.assertEqual(load_points(path).points, self.problem.points)
            for text in ("x,y\n1,2", "id,x,y\n0,1,2\n0,3,4", "id,x,y\n1,0,0\n3,1,1",
                         "id,x,y\n0,no,1", "id,x,y\n0,nan,0\n1,1,1\n2,2,2\n3,3,3"):
                path.write_text(text, encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_points(path)

    def test_synthetic_data_reproducible_and_depot_fixed(self):
        for layout in ("uniform", "clustered"):
            a = generate_points(20, layout, 123)
            self.assertEqual(a.points, generate_points(20, layout, 123).points)
            self.assertEqual(a.points[0], (50, 50))
            self.assertTrue(all(0 <= x <= 100 and 0 <= y <= 100 for x, y in a.points))

    def test_pairing_sign_and_map_level_uncertainty(self):
        rows = []
        for name, base in (("a", 100.0), ("b", 200.0)):
            for seed in (0, 1):
                for mode, ratio in (("baseline", 1), ("periodic", 1.1), ("stagnation", 0.9)):
                    rows.append({"instance": name, "n": 20, "layout": "uniform", "seed": seed,
                                 "mode": mode, "best_length": base * ratio, "elapsed_seconds": 1,
                                 "reheats": int(mode != "baseline")})
        summary = summarize(rows)
        gain = next(r for r in summary["overall"] if r["mode"] == "stagnation")
        self.assertAlmostEqual(gain["mean_paired_gain_percent"], 10)
        self.assertEqual((gain["wins"], gain["ties"], gain["losses"]), (4, 0, 0))
        self.assertEqual(gain["instances"], 2)
        self.assertEqual(bootstrap_interval([10, 10]), [10, 10])


if __name__ == "__main__":
    unittest.main()
