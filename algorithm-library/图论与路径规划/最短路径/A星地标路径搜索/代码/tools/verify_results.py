"""Coordinate BFS audits saved paths, bounds, inputs, snapshots and aggregates."""
from collections import deque
from pathlib import Path
import csv
import gzip
import hashlib
import json
import math
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import digest, dump_json, summarize
from pathfinding import Grid, METHODS, build_landmarks, make_heuristic


def independent_bfs(rows, start):
    width, height = len(rows[0]), len(rows)
    distances, queue = {start: 0}, deque([start])
    while queue:
        v = queue.popleft()
        x, y = v % width, v // width
        for xx, yy in ((x-1, y), (x+1, y), (x, y-1), (x, y+1)):
            u = yy*width+xx
            if 0 <= xx < width and 0 <= yy < height and rows[yy][xx] == "." and u not in distances:
                distances[u] = distances[v]+1
                queue.append(u)
    return distances


def verify(output):
    output = Path(output)
    load = lambda name: json.loads((output/name).read_text(encoding="utf-8"))
    protocol, manifest, saved_indexes = load("protocol.json"), load("manifest.json"), load("indexes.json")
    assert digest(output/"protocol.json") == manifest["protocol_sha256"]
    assert load("config.json") == protocol["config"]
    for relative, expected in protocol["source_sha256"].items():
        assert digest(output/"source_snapshot"/relative) == expected, relative
        assert digest(ROOT/relative) == expected, f"Current source differs from timed snapshot: {relative}"
    with (output/"runs.csv").open(encoding="utf-8", newline="") as stream:
        runs = list(csv.DictReader(stream))
    for row in runs:
        for key in row:
            if key == "seconds":
                row[key] = float(row[key])
            elif key not in ("case", "family", "method"):
                row[key] = int(row[key])
        assert math.isfinite(row["seconds"]) and row["seconds"] > 0
    with (output/"preprocessing.csv").open(encoding="utf-8", newline="") as stream:
        preparation = list(csv.DictReader(stream))
    for row in preparation:
        row["seconds"] = float(row["seconds"])
        row["repeat"], row["table_payload_bytes"] = int(row["repeat"]), int(row["table_payload_bytes"])
        assert math.isfinite(row["seconds"]) and row["seconds"] >= 0
    keyed = {(r["case"], r["query"], r["method"], r["repeat"]): r for r in runs}
    assert len(keyed) == len(runs)
    queries_checked, paths_checked, bound_vertices, edge_checks = 0, 0, 0, 0
    for case in manifest["cases"]:
        assert digest(output/case["map"]) == case["map_sha256"]
        assert digest(output/case["queries"]) == case["queries_sha256"]
        rows = (output/case["map"]).read_text(encoding="utf-8").splitlines()
        queries = load(case["queries"])
        solutions = json.loads(gzip.decompress((output/"solutions"/(case["case"]+".json.gz")).read_bytes()))
        assert len(solutions) == len(queries) == protocol["config"]["queries"]
        width = len(rows[0])
        grid = Grid(rows)
        assert len(grid.free) == case["free_cells"] and len(grid.components) == case["components"]
        for qi, ((s, t), solution) in enumerate(zip(queries, solutions)):
            exact = independent_bfs(rows, s)
            assert (s, t) == (solution["start"], solution["target"])
            assert solution["oracle_cost"] == exact[t]
            queries_checked += 1
            assert set(solution["methods"]) == set(METHODS)
            for method, result in solution["methods"].items():
                path = result["path"]
                assert path[0] == s and path[-1] == t and len(path)-1 == result["cost"] == exact[t]
                assert len(path) == len(set(path))
                for v in path:
                    assert 0 <= v < grid.n and rows[v//width][v % width] == "."
                assert all(abs(v % width-u % width)+abs(v//width-u//width) == 1 for v, u in zip(path, path[1:]))
                for rep in range(protocol["config"]["repeats"]):
                    measured = keyed[case["case"], qi, method, rep]
                    assert measured["cost"] == exact[t] and measured["expanded"] == result["expanded"]
                    assert (measured["start"], measured["target"]) == (s, t)
                paths_checked += 1
        # Every reachable vertex for two saved targets, every directed edge.
        for method, saved in saved_indexes[case["case"]].items():
            index = build_landmarks(grid, method[4:], protocol["config"]["landmarks"])
            assert [list(group) for group in index.landmarks] == saved["landmarks"]
            assert [hashlib.sha256(table.tobytes()).hexdigest() for table in index.tables] == saved["table_sha256"]
            assert index.table_payload_bytes == saved["table_payload_bytes"]
            for target in dict.fromkeys(pair[1] for pair in queries[:2]):
                exact = independent_bfs(rows, target)
                h = make_heuristic(grid, target, method, index)
                bounds = {v: h(v) for v in exact}
                assert bounds[target] == 0
                for v in exact:
                    assert 0 <= bounds[v] <= exact[v]
                    bound_vertices += 1
                    x, y = v % width, v//width
                    for xx, yy in ((x-1,y), (x+1,y), (x,y-1), (x,y+1)):
                        u = yy*width+xx
                        if 0 <= xx < width and 0 <= yy < len(rows) and u in exact:
                            assert bounds[v] <= 1+bounds[u]
                            edge_checks += 1
    assert len(runs) == queries_checked*len(METHODS)*protocol["config"]["repeats"]
    recomputed = summarize(runs, preparation)
    assert recomputed == load("summary.json")
    audit = {"status": "passed", "map_query_cases": len(manifest["cases"]), "independent_bfs_queries": queries_checked,
             "saved_paths_checked": paths_checked, "timed_searches_checked": len(runs),
             "admissibility_vertex_checks": bound_vertices, "consistency_directed_edge_checks": edge_checks,
             "inputs_sources_tables_summaries": "all matched",
             "limits": "Stored optimality, path, heuristic and aggregate audit; elapsed wall times are measurements and cannot be reconstructed."}
    dump_json(output/"audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return audit


if __name__ == "__main__":
    verify(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT/"outputs/validation")
