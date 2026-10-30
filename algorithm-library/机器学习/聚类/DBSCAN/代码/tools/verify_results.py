"""Verify saved evidence without rerunning the full timing benchmark."""
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets import load_csv
from dbscan import METHODS, RangeIndex, canonical_labels
from experiments import ALL_METHODS, digest, dump_json, summarize


def verify(results):
    results = Path(results)
    read = lambda name: json.loads((results / name).read_text(encoding="utf-8"))
    manifest, protocol, solutions = read("manifest.json"), read("protocol.json"), read("solutions.json")
    assert digest(results / "protocol.json") == manifest["protocol_sha256"]
    for name, expected in protocol["source_sha256"].items():
        assert digest(results / "source_snapshot" / name) == expected, name
    case_ids = {c["case"] for c in manifest["cases"]}
    assert set(solutions) == case_ids
    queries_checked = 0
    for case in manifest["cases"]:
        assert digest(results / case["file"]) == case["sha256"], case["case"]
        points = load_csv(results / case["file"])
        assert len(points) == case["n"]
        saved = solutions[case["case"]]
        assert set(saved) == set(ALL_METHODS)
        for method in ALL_METHODS:
            assert len(saved[method]["labels"]) == len(points)
            assert len(saved[method]["core"]) == len(points)
            np.testing.assert_array_equal(canonical_labels(saved[method]["labels"]), canonical_labels(saved["brute"]["labels"]))
            np.testing.assert_array_equal(saved[method]["core"], saved["brute"]["core"])
            for field in ("partition_equal", "core_equal", "noise_equal"):
                assert saved[method][field]
        # Recompute selected queries using a scalar independent distance kernel,
        # including first/last and fixed random IDs; do not reuse saved neighbors.
        import math
        ids = sorted(set([0, len(points) - 1, *np.random.default_rng(case["seed"]).integers(0, len(points), 10).tolist()]))
        indexes = {method: RangeIndex(points, case["eps"], method) for method in METHODS}
        for i in ids:
            expected = [j for j, p in enumerate(points) if math.hypot(float(points[i, 0]) - float(p[0]), float(points[i, 1]) - float(p[1])) <= case["eps"]]
            for index in indexes.values():
                np.testing.assert_array_equal(index.query(i), expected)
            assert saved["brute"]["core"][i] == int(len(expected) >= case["min_samples"])
            queries_checked += 1
    with (results / "runs.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    expected_runs = len(case_ids) * len(ALL_METHODS) * protocol["config"]["repeats"]
    assert len(rows) == expected_runs
    keys = [(r["case"], r["method"], r["repeat"]) for r in rows]
    assert len(set(keys)) == expected_runs
    numeric_int = ("n", "seed", "min_samples", "repeat", "order", "clusters", "noise", "core", "queries", "distance_evaluations", "neighbor_entries", "candidate_cells", "box_checks", "skipped_cells", "accepted_cells", "bulk_accepted_points", "fallback_queries")
    for row in rows:
        assert row["case"] in case_ids and row["method"] in ALL_METHODS
        for field in numeric_int:
            if row[field] != "":
                row[field] = int(row[field])
        for field in ("eps", "seconds", "index_seconds", "query_seconds"):
            if row[field] != "":
                row[field] = float(row[field])
        for field in ("partition_equal", "core_equal", "noise_equal"):
            assert row[field] == "True"
            row[field] = True
        assert row["seconds"] > 0
        if row["method"] == "brute":
            assert row["distance_evaluations"] == row["n"] ** 2
        if row["method"] in METHODS:
            assert row["queries"] == row["n"]
    computed = summarize(rows)
    stored = read("summary.json")
    assert computed == stored, "Summary differs from raw measurements"
    audit = {"datasets": len(case_ids), "measured_runs": expected_runs,
             "independently_recomputed_queries": queries_checked,
             "source_and_input_hashes": "pass", "saved_partitions_core_noise": "pass", "raw_summary_recalculation": "pass"}
    dump_json(results / "audit.json", audit)
    print(json.dumps(audit, indent=2))
