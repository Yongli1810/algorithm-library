"""Export self-contained teaching artifacts; no server or network required."""
from pathlib import Path
import json
import numpy as np

from datasets import generate, save_csv
from dbscan import METHODS, RangeIndex, fit
from experiments import dump_json

ROOT = Path(__file__).resolve().parent
MICRO_POINTS = np.array([[0, 0], [0.1, 0], [0, 0.1], [0.1, 0.1], [0.28, 0],
                         [2, 2], [2.1, 2], [2, 2.1], [2.1, 2.1], [5, 5]], dtype=float)


def build_micro(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    expected = [0, 0, 0, 0, 0, 1, 1, 1, 1, -1]
    result = fit(MICRO_POINTS, 0.2, 4, trace=True)
    assert result.labels.tolist() == expected
    index = RangeIndex(MICRO_POINTS, 0.2, "pruned")
    queries = []
    for i in range(len(MICRO_POINTS)):
        neighbors, detail = index.query(i, explain=True)
        queries.append({"id": i, "neighbors": neighbors.tolist(), "core": len(neighbors) >= 4, **detail})
    save_csv(output / "points.csv", MICRO_POINTS)
    dump_json(output / "micro.json", {"eps": 0.2, "min_samples": 4, "points": MICRO_POINTS.tolist(),
                                      "expected_labels": expected, "labels": result.labels.tolist(),
                                      "core": result.core.astype(int).tolist(), "queries": queries,
                                      "explanation": "0–3 form cluster 0 core; point 4 is its border; 5–8 form cluster 1 core; point 9 is noise. Self counts in min_samples."})
    return output / "micro.json"


def build_demo(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    specifications = [("moons", "双月形", 260, 0.14, 6), ("blobs", "多个团块", 260, 0.14, 6),
                      ("uniform", "均匀散点", 260, 0.14, 6), ("dense", "高密度退化", 180, 0.2, 8),
                      ("micro", "十点手算", 10, 0.2, 4)]
    cases = []
    for i, (family, title, n, eps, m) in enumerate(specifications):
        points = MICRO_POINTS if family == "micro" else generate(family, n, 20261029 + i)
        results = {mode: fit(points, eps, m, mode, trace=(mode == "pruned")) for mode in METHODS}
        for result in results.values():
            np.testing.assert_array_equal(result.labels, results["brute"].labels)
        queries = {}
        for mode in METHODS:
            index = RangeIndex(points, eps, mode)
            records = []
            for point_id in range(n):
                neighbors, detail = index.query(point_id, explain=True)
                records.append({"neighbors": neighbors.tolist(), **detail})
            queries[mode] = records
        case = {"id": family, "title": title, "n": n, "eps": eps, "min_samples": m,
                "points": points.tolist(), "queries": queries, "frames": results["pruned"].frames,
                "labels": results["pruned"].labels.tolist(), "core": results["pruned"].core.astype(int).tolist(),
                "clusters": results["pruned"].clusters, "noise": int((results["pruned"].labels == -1).sum()),
                "totals": {mode: r.stats for mode, r in results.items()}}
        cases.append(case)
        save_csv(output / f"{family}.csv", points)
    payload = {"cases": cases, "summary": None}
    summary_file = ROOT / "outputs/validation/summary.json"
    if summary_file.exists():
        summary = json.loads(summary_file.read_text(encoding="utf-8"))
        payload["summary"] = {"datasets": summary["datasets"], "measured_runs": summary["measured_runs"],
                              "all_equal": summary["all_equal"], "groups": [r for r in summary["groups"] if r["n"] == 10000]}
    dump_json(output / "demo.json", payload)
    source = (ROOT / "demo_template.html").read_text(encoding="utf-8")
    packed = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    (output / "demo.html").write_text(source.replace("__PAYLOAD__", packed), encoding="utf-8")
    return output / "demo.html"
