"""Self-contained offline HTML, actual search traces and a hand-worked case."""
from dataclasses import asdict
from pathlib import Path
import json

from datasets import FAMILIES, generate, micro_case, save_map
from experiments import NAMES, dump_json
from pathfinding import Grid, METHODS, bfs_distances, build_landmarks, make_heuristic, search

ROOT = Path(__file__).resolve().parent


def case_payload(name, family, grid, start, target):
    exact = bfs_distances(grid, target)
    methods = {}
    for method in METHODS:
        index = build_landmarks(grid, method[4:], 4) if method.startswith("alt_") else None
        h = make_heuristic(grid, target, method, index)
        result = search(grid, start, target, method, index, record=True)
        landmarks = list(index.landmarks[grid.component[target]]) if index else []
        methods[method] = {**asdict(result), "name": NAMES[method], "landmarks": landmarks,
                           "h": [h(v) if grid.component[v] >= 0 else None for v in range(grid.n)],
                           "landmark_distances": [list(table) for table in index.tables[:len(landmarks)]] if index else []}
    return {"name": name, "family": family, "rows": grid.rows, "width": grid.width, "height": grid.height,
            "start": start, "target": target, "exact": [v if v >= 0 else None for v in exact], "methods": methods}


def build_micro(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    grid, start, target = micro_case()
    save_map(output/"map.txt", grid.rows)
    result = case_payload("手算绕墙", "micro", grid, start, target)
    assert all(method["cost"] == 8 for method in result["methods"].values())
    result["manual"] = {"start_xy": [0, 0], "target_xy": [4, 0], "manhattan_lower_bound": 4,
                         "optimal_path_xy": [[0,0],[1,0],[1,1],[1,2],[2,2],[3,2],[4,2],[4,1],[4,0]],
                         "optimal_cost": 8, "reason": "The wall at x=2, y=0/1 forces reaching y=2 and returning, adding four vertical moves."}
    dump_json(output/"micro.json", result)
    print(output.resolve())
    return result


def build_demo(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    grid, start, target = micro_case()
    cases = [case_payload("手算绕墙 · 5×3", "micro", grid, start, target)]
    titles = {"open": "空地图", "random": "随机障碍", "maze": "迷宫", "corridor": "狭窄通道"}
    for i, family in enumerate(FAMILIES):
        grid = Grid(generate(family, 31, 20261105+i))
        component = max(grid.components, key=lambda c: (len(c), -c[0]))
        start, target = component[0], component[-1]
        cases.append(case_payload(titles[family]+" · 31×31", family, grid, start, target))
        save_map(output/(family+".txt"), grid.rows)
    grid = Grid(("..#..", "..#..", "..#.."))
    cases.append(case_payload("不可达案例 · 5×3", "disconnected", grid, 0, 4))
    summary_path = ROOT/"outputs/validation/summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else None
    data = {"cases": cases, "method_names": NAMES, "formal": summary["cases"] if summary else [],
            "notes": "Traces use 31x31 demonstration maps; formal costs come from separate held-out maps. No live timing in browser."}
    dump_json(output/"demo.json", data)
    encoded = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    html = (ROOT/"demo_template.html").read_text(encoding="utf-8").replace("__DATA__", encoded)
    (output/"demo.html").write_text(html, encoding="utf-8")
    build_micro(ROOT/"outputs/micro")
    print(output/"demo.html")
