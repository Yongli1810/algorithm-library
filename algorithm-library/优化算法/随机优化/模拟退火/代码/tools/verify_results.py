"""Independently audit saved routes, paired budgets, statistics, and hashes."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

parser = argparse.ArgumentParser()
parser.add_argument("directory", type=Path)
args = parser.parse_args()
out = args.directory
load = lambda name: json.loads((out / name).read_text(encoding="utf-8"))
solutions, manifest, summary = load("solutions.json"), load("manifest.json"), load("summary.json")
with (out / "runs.csv").open(encoding="utf-8-sig", newline="") as h:
    rows = list(csv.DictReader(h))
by_key = {(r["instance"], int(r["seed"]), r["mode"]): r for r in rows}
assert len(by_key) == len(rows) == len(solutions) == manifest["completed_runs"]
data = {}
for name, expected in manifest["data_sha256"].items():
    path = out / "data" / name
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
    with path.open(encoding="utf-8-sig", newline="") as h:
        data[path.stem] = [(float(r["x"]), float(r["y"])) for r in csv.DictReader(h)]
for name, expected in manifest["source_sha256"].items():
    assert hashlib.sha256((out / "source_snapshot" / name).read_bytes()).hexdigest() == expected
assert hashlib.sha256((out / "protocol.json").read_bytes()).hexdigest() == manifest["protocol_sha256"]
for solution in solutions:
    points, route = data[solution["instance"]], solution["best_route"]
    assert route[0] == 0 and sorted(route) == list(range(len(points)))
    distance = math.fsum(math.dist(points[route[k]], points[route[(k+1) % len(route)]]) for k in range(len(route)))
    row = by_key[solution["instance"], solution["seed"], solution["mode"]]
    assert math.isclose(distance, float(row["best_length"]), rel_tol=1e-12, abs_tol=1e-9)
    assert math.isclose(distance, solution["best_length"], rel_tol=1e-12, abs_tol=1e-9)
    assert distance <= float(row["initial_length"]) + 1e-9
    assert int(row["reheats"]) == len(solution["events"])
    for mode in ("baseline", "periodic", "stagnation"):
        peer = by_key[solution["instance"], solution["seed"], mode]
        for field in ("initial_length", "initial_temperature", "evaluations", "calibration_evaluations"):
            assert row[field] == peer[field]
for group in summary["overall"]:
    gains = []
    for instance in data:
        values = []
        for row in rows:
            if row["instance"] == instance and row["mode"] == group["mode"]:
                base = float(by_key[instance, int(row["seed"]), "baseline"]["best_length"])
                values.append(100 * (base - float(row["best_length"])) / base)
        gains.append(statistics.fmean(values))
    assert math.isclose(statistics.fmean(gains), group["mean_paired_gain_percent"], abs_tol=1e-10)
print(f"PASS: {len(solutions)} stored tours, {len(data)} input maps, paired budgets, summary means and source/data/protocol hashes.")
