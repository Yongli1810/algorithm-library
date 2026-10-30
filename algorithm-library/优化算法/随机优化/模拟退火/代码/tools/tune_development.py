"""Small, fully recorded parameter study on development maps only.

Run from the project directory. Test maps in configs/full.json are never read.
"""
import csv
from dataclasses import asdict, replace
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from annealing import SAConfig, solve
from datasets import generate_points
from experiments import write_csv, write_json

spec = json.loads((ROOT / "configs/development_v2.json").read_text(encoding="utf-8"))
base = SAConfig(**spec["algorithm"], mode="stagnation", trace_every=0)
candidates = [
    ("gate_001", {"reheat_gate_ratio": 0.01, "reheat_floor_ratio": 0.005, "reheat_until_fraction": 0.9}),
    ("gate_0005", {"reheat_gate_ratio": 0.005, "reheat_floor_ratio": 0.002, "reheat_until_fraction": 0.9}),
    ("gate_0002", {"reheat_gate_ratio": 0.002, "reheat_floor_ratio": 0.001, "max_reheats": 2, "reheat_until_fraction": 0.9}),
    ("gate_0001", {"reheat_gate_ratio": 0.001, "reheat_floor_ratio": 0.0005, "max_reheats": 2, "patience": 500, "reheat_until_fraction": 0.95}),
    ("long_patience", {"reheat_gate_ratio": 0.01, "reheat_floor_ratio": 0.005, "max_reheats": 2, "patience": 2000, "reheat_until_fraction": 0.9}),
]
out = ROOT / "outputs/development_sweep"
if out.exists() and any(out.iterdir()):
    raise SystemExit("Development sweep already exists; retain the original record.")
out.mkdir(parents=True, exist_ok=True)
write_json(out / "protocol.json", {"split": "development only", "data_seed": spec["data_seed"],
                                  "candidates": {name: asdict(replace(base, **overrides)) for name, overrides in candidates}})
with (ROOT / "outputs/development_v2/runs.csv").open(encoding="utf-8-sig", newline="") as h:
    baseline = {(r["instance"], int(r["seed"])): float(r["best_length"]) for r in csv.DictReader(h) if r["mode"] == "baseline"}
rows, summary = [], []
for name, overrides in candidates:
    config = replace(base, **overrides)
    for n in spec["sizes"]:
        for layout in spec["layouts"]:
            for index in range(spec["instances_per_layout"]):
                data_seed = spec["data_seed"] + n * 1009 + (0 if layout == "uniform" else 1000003) + index * 97
                instance = f"{layout}_{n}_{index}"
                problem = generate_points(n, layout, data_seed)
                for seed in spec["seeds"]:
                    r = solve(problem, config, seed)
                    ref = baseline[instance, seed]
                    rows.append({"candidate": name, "instance": instance, "seed": seed,
                                 "best_length": r["best_length"], "baseline_length": ref,
                                 "gain_percent": 100 * (ref - r["best_length"]) / ref,
                                 "reheats": r["reheats"]})
    subset = [r for r in rows if r["candidate"] == name]
    summary.append({"candidate": name, "mean_gain_percent": statistics.fmean(r["gain_percent"] for r in subset),
                    "config": asdict(config), "runs": len(subset)})
    print(name, summary[-1]["mean_gain_percent"], flush=True)
write_csv(out / "runs.csv", rows)
write_json(out / "summary.json", summary)
print("Development study complete. Freeze a configuration before opening held-out results.")
