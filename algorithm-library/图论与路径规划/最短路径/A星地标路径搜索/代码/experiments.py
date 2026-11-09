"""Frozen paired experiment protocol; query time and setup cost stay separate."""
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from time import perf_counter
import csv
import gzip
import hashlib
import json
import math
import platform
import random
import shutil
import sys

from pathfinding import Grid, METHODS, bfs_distances, build_landmarks, search, validate_path
from datasets import generate, make_queries, save_map

ROOT = Path(__file__).resolve().parent
NAMES = {"dijkstra": "Dijkstra", "manhattan": "曼哈顿 A*", "alt_corners": "角点地标 A*",
         "alt_farthest": "最远点地标 A*", "alt_boundary": "障碍候选地标 A*"}
SOURCE_FILES = ("pathfinding.py", "datasets.py", "experiments.py", "run.py", "tools/verify_results.py", "tests/test_algorithm.py")


def dump_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def geomean(values):
    values = list(values)
    return math.exp(mean(math.log(x) for x in values))


def summarize(runs, preprocessing):
    grouped, setups, commons = defaultdict(list), defaultdict(list), defaultdict(list)
    for row in runs:
        grouped[row["case"], row["method"], row["query"]].append(row)
    for row in preprocessing:
        if row["method"] == "common_grid":
            commons[row["case"]].append(row["seconds"])
        else:
            setups[row["case"], row["method"]].append(row)
    per_query = []
    for (case, method, query), rows in grouped.items():
        first = rows[0]
        assert len({r["cost"] for r in rows}) == 1
        assert len({r["expanded"] for r in rows}) == 1
        per_query.append({"case": case, "family": first["family"], "size": first["size"], "method": method,
                          "query": query, "cost": first["cost"], "expanded": first["expanded"],
                          "median_seconds": median(r["seconds"] for r in rows)})
    paired, query_groups = {}, defaultdict(list)
    for row in per_query:
        query_groups[row["case"], row["method"]].append(row)
    for (case, method), rows in query_groups.items():
        setup = setups[case, method]
        paired[case, method] = {"case": case, "family": rows[0]["family"], "size": rows[0]["size"], "method": method,
                               "queries": len(rows), "mean_query_seconds": mean(r["median_seconds"] for r in rows),
                               "mean_expanded": mean(r["expanded"] for r in rows),
                               "setup_seconds": median(r["seconds"] for r in setup),
                               "common_seconds": median(commons[case]),
                               "table_payload_bytes": setup[0]["table_payload_bytes"]}
    for (case, method), row in paired.items():
        base, farthest = paired[case, "manhattan"], paired[case, "alt_farthest"]
        row["query_speedup_vs_manhattan"] = base["mean_query_seconds"] / row["mean_query_seconds"]
        row["query_speedup_vs_farthest"] = farthest["mean_query_seconds"] / row["mean_query_seconds"]
        row["expanded_reduction_vs_manhattan"] = 1-row["mean_expanded"]/base["mean_expanded"]
        saving = base["mean_query_seconds"]-row["mean_query_seconds"]
        row["break_even_queries"] = math.floor(row["setup_seconds"]/saving)+1 if method.startswith("alt_") and saving > 0 else None
        for queries in (1, 10, 100, 1000):
            row[f"total_speedup_q{queries}"] = (base["common_seconds"]+queries*base["mean_query_seconds"])/(row["common_seconds"]+row["setup_seconds"]+queries*row["mean_query_seconds"])
    cases = list(paired.values())
    overall, families = [], []
    for family in (None, *sorted({row["family"] for row in cases})):
        for method in METHODS:
            rows = [r for r in cases if r["method"] == method and (family is None or r["family"] == family)]
            item = {"method": method, "cases": len(rows),
                    "geomean_query_speedup": geomean(r["query_speedup_vs_manhattan"] for r in rows),
                    "geomean_speedup_vs_farthest": geomean(r["query_speedup_vs_farthest"] for r in rows),
                    "mean_expanded_reduction": mean(r["expanded_reduction_vs_manhattan"] for r in rows),
                    "faster_cases": sum(r["query_speedup_vs_manhattan"] > 1 for r in rows),
                    "faster_than_farthest_cases": sum(r["query_speedup_vs_farthest"] > 1 for r in rows),
                    "amortizable_cases": sum(r["break_even_queries"] is not None for r in rows)}
            for q in (1, 10, 100, 1000):
                item[f"geomean_total_speedup_q{q}"] = geomean(r[f"total_speedup_q{q}"] for r in rows)
            if family is None:
                overall.append(item)
            else:
                families.append({"family": family, **item})
    return {"map_query_cases": len(commons), "unique_queries": len(per_query)//len(METHODS),
            "timed_searches": len(runs), "all_costs_match_bfs": True, "overall": overall,
            "families": families, "cases": cases, "per_query": per_query}


def benchmark(config_path, output):
    config_path, output = Path(config_path), Path(output)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory is not empty; use a new --out")
    output.mkdir(parents=True, exist_ok=True)
    for relative in SOURCE_FILES:
        dst = output/"source_snapshot"/relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/relative, dst)
    shutil.copyfile(config_path, output/"config.json")
    protocol = {"created_utc": datetime.now(timezone.utc).isoformat(), "config": config, "methods": METHODS,
                "problem": "Static undirected 4-neighbour unit-cost grid. Uniform distinct endpoints within largest component; disconnected cases tested separately.",
                "selection": "k landmarks per component; farthest/boundary share row-major first anchor. Selection does not inspect queries. Boundary excludes rectangle exterior; empty candidate pool falls back to all remaining vertices.",
                "timing": "Common Grid adjacency/components timed separately. Landmark setup includes candidate selection and BFS distance tables. Query includes validation, h closure/cache, g/parent allocations, heap search and path reconstruction; excludes trace, I/O, map/query generation and oracle. One map-local warm-up per method. Random query/method order per repeat, fixed order seed.",
                "aggregation": "Per-query median of repeats; arithmetic mean over queries on a map; geometric mean of paired map-level speedups. Predicted total(Q)=common+index+Q*mean_query. Break-even strictly faster: floor(index/(base_query-alt_query))+1, only if saving>0. Finite repeats are descriptive, not a significance test.",
                "tie_break": "(g+h, -g, row-major vertex id); fixed neighbour order up,left,right,down. Expanded count includes goal.",
                "memory": "Exact retained distance array payload bytes only; excludes Python container overhead, common grid and temporary queues. No RSS or peak-memory claim.",
                "environment": {"python": sys.version, "platform": platform.platform(), "processor": platform.processor()},
                "source_sha256": {p: digest(ROOT/p) for p in SOURCE_FILES}}
    dump_json(output/"protocol.json", protocol)
    cases = []
    for fi, family in enumerate(config["families"]):
        for size in config["sizes"]:
            for seed in config["seeds"]:
                map_seed = seed + fi*100000 + size
                query_seed = seed + config["query_seed_offset"] + fi*100000 + size
                case = f"{family}_{size}_{seed}"
                rows = generate(family, size, map_seed)
                grid = Grid(rows)
                map_path, query_path = output/"data"/(case+".txt"), output/"data"/(case+"_queries.json")
                save_map(map_path, rows)
                dump_json(query_path, make_queries(grid, config["queries"], query_seed))
                cases.append({"case": case, "family": family, "size": size, "map_seed": map_seed, "query_seed": query_seed,
                              "free_cells": len(grid.free), "components": len(grid.components), "largest_component": max(map(len, grid.components)),
                              "map": str(map_path.relative_to(output)), "queries": str(query_path.relative_to(output)),
                              "map_sha256": digest(map_path), "queries_sha256": digest(query_path)})
    # All source, settings, maps and endpoint pairs are frozen before timing.
    dump_json(output/"manifest.json", {"protocol_sha256": digest(output/"protocol.json"), "cases": cases})
    rng, runs, preparation, index_records = random.Random(config["order_seed"]), [], [], {}
    for number, case in enumerate(cases, 1):
        rows = (output/case["map"]).read_text(encoding="utf-8").splitlines()
        queries = json.loads((output/case["queries"]).read_text(encoding="utf-8"))
        for rep in range(config["repeats"]):
            tick = perf_counter()
            grid = Grid(rows)
            elapsed = perf_counter()-tick
            preparation.append({"case": case["case"], "method": "common_grid", "repeat": rep, "seconds": elapsed, "table_payload_bytes": 0})
        indexes = {}
        for rep in range(config["repeats"]):
            methods = list(METHODS)
            rng.shuffle(methods)
            for method in methods:
                index, elapsed = None, 0.0
                if method.startswith("alt_"):
                    tick = perf_counter()
                    index = build_landmarks(grid, method[4:], config["landmarks"])
                    elapsed = perf_counter()-tick
                indexes[method] = index
                preparation.append({"case": case["case"], "method": method, "repeat": rep, "seconds": elapsed,
                                    "table_payload_bytes": index.table_payload_bytes if index else 0})
        index_records[case["case"]] = {method: {"landmarks": index.landmarks, "boundary_candidates": index.boundary_candidates,
                                               "table_payload_bytes": index.table_payload_bytes,
                                               "table_sha256": [hashlib.sha256(table.tobytes()).hexdigest() for table in index.tables]}
                                      for method, index in indexes.items() if index is not None}
        expected = [bfs_distances(grid, s)[t] for s, t in queries]
        solutions = [{"start": s, "target": t, "oracle_cost": expected[i], "methods": {}} for i, (s, t) in enumerate(queries)]
        for method in METHODS:
            search(grid, *queries[0], method, indexes[method])
        for rep in range(config["repeats"]):
            order = list(range(len(queries)))
            rng.shuffle(order)
            for qi in order:
                methods = list(METHODS)
                rng.shuffle(methods)
                for mi, method in enumerate(methods):
                    s, t = queries[qi]
                    tick = perf_counter()
                    result = search(grid, s, t, method, indexes[method])
                    elapsed = perf_counter()-tick
                    assert result.cost == expected[qi], (case["case"], qi, method)
                    validate_path(grid, s, t, result)
                    saved = {"cost": result.cost, "path": result.path, "expanded": result.expanded}
                    if rep == 0:
                        solutions[qi]["methods"][method] = saved
                    else:
                        assert solutions[qi]["methods"][method] == saved
                    runs.append({"case": case["case"], "family": case["family"], "size": case["size"], "query": qi,
                                 "start": s, "target": t, "method": method, "repeat": rep, "order": mi,
                                 "seconds": elapsed, "cost": result.cost, "expanded": result.expanded, "pushes": result.pushes,
                                 "stale_pops": result.stale_pops, "max_heap": result.max_heap, "heuristic_evaluations": result.heuristic_evaluations})
        solution_path = output/"solutions"/(case["case"]+".json.gz")
        solution_path.parent.mkdir(exist_ok=True)
        solution_path.write_bytes(gzip.compress(json.dumps(solutions, separators=(",", ":")).encode(), mtime=0))
        print(f"[{number}/{len(cases)}] {case['case']}: {len(queries)} queries, all five methods optimal", flush=True)
    write_csv(output/"runs.csv", runs)
    write_csv(output/"preprocessing.csv", preparation)
    dump_json(output/"indexes.json", index_records)
    summary = summarize(runs, preparation)
    dump_json(output/"summary.json", summary)
    write_csv(output/"case_summary.csv", summary["cases"])
    write_csv(output/"group_summary.csv", summary["families"])
    write_report(output, summary)
    plot_results(output, summary)
    print(json.dumps({"maps": summary["map_query_cases"], "queries": summary["unique_queries"], "searches": summary["timed_searches"],
                      "query_speedups": {r["method"]: round(r["geomean_query_speedup"], 3) for r in summary["overall"]}}, ensure_ascii=False), flush=True)
    return summary


def write_report(output, summary):
    lines = ["# A* 地标选择实验报告", "", f"{summary['map_query_cases']} 个地图查询实例，{summary['unique_queries']} 对端点，{summary['timed_searches']} 次计时搜索，最短路长度全部与 BFS 一致。",
             "", "空地图在不同种子下几何相同，端点不同；不能称为全部地图几何独立。所有端点从最大连通分量均匀抽取，每对端点互异。",
             "", "## 查询与总成本", "", "所有加速比相对曼哈顿 A*，大于 1 表示更快。查询时间不含地标预处理；后四列包含公共地图构建和地标预处理，是按样本平均查询成本外推的总成本比。",
             "", "| 方法 | 查询加速比 | 更快实例 | 扩展减少均值 | 1次总成本 | 10次 | 100次 | 1000次 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in summary["overall"]:
        lines.append(f"| {NAMES[r['method']]} | {r['geomean_query_speedup']:.3f}× | {r['faster_cases']}/{r['cases']} | {r['mean_expanded_reduction']:.1%} | "+" | ".join(f"{r[f'geomean_total_speedup_q{q}']:.3f}×" for q in (1,10,100,1000))+" |")
    lines += ["", "## 各地图族", "", "| 地图族 | 方法 | 查询加速比 | 相对最远点策略 | 比曼哈顿更快 | 扩展减少均值 |", "|---|---|---:|---:|---:|---:|"]
    for r in summary["families"]:
        lines.append(f"| {r['family']} | {NAMES[r['method']]} | {r['geomean_query_speedup']:.3f}× | {r['geomean_speedup_vs_farthest']:.3f}× | {r['faster_cases']}/{r['cases']} | {r['mean_expanded_reduction']:.1%} |")
    lines += ["", "## 逐地图预处理与摊销", "", "均摊阈值只在查询均值确实节省时存在；Q > P/(t_base−t_alt)。负收益写为‘无’，没有用多数案例的收益抵销这些案例。阈值受本机计时波动和未来查询分布影响。",
              "", "| 地图实例 | 策略 | 索引预处理 ms | 查询均值 ms | 查询加速比 | 开始省时的查询数 | 距离表 KiB |", "|---|---|---:|---:|---:|---:|---:|"]
    for r in summary["cases"]:
        if r["method"].startswith("alt_"):
            lines.append(f"| {r['case']} | {r['method'][4:]} | {r['setup_seconds']*1000:.3f} | {r['mean_query_seconds']*1000:.3f} | {r['query_speedup_vs_manhattan']:.3f}× | {r['break_even_queries'] if r['break_even_queries'] is not None else '无'} | {r['table_payload_bytes']/1024:.2f} |")
    boundary = next(r for r in summary["overall"] if r["method"] == "alt_boundary")
    lines += ["", "## 结论的边界", "", f"障碍候选策略相对最远点策略的总体查询加速比为 {boundary['geomean_speedup_vs_farthest']:.3f}×，在 {boundary['faster_than_farthest_cases']}/{boundary['cases']} 个实例更快。应结合各地图族和节点计数解释，不能把所有 ALT 的收益归给候选限制。",
              "", "候选池为空时两种最远策略完全相同；候选池覆盖全部可行节点时也完全相同。此时计时差异主要反映波动，不是算法改进。更强的启发式也有逐节点计算成本，不保证查询更快。",
              "", "额外内存列仅为地标距离数组的准确有效载荷，不含 Python 对象、公共邻接表、队列和临时变量，不是峰值 RSS。",
              "", "每次查询先分配 O(N) 搜索数组；所有实现共用相同分配、堆与平局规则。Dijkstra 是共用堆内核的零启发式参照，单位边权上 BFS 通常更合适；独立 BFS 用于正确性审计，未作为计时方法。",
              "", "正式计时先冻结源码、地图和查询；每查询重复取中位数，再取地图内均值，最后取配对地图比的几何平均。三次重复只提供描述性比较，不宣称统计显著或普遍领先。",
              "", "![查询与扩展](comparison.png)", "", "## 依据", "", "地标和三角不等式启发式来自 [Goldberg 与 Harrelson 的 ALT 研究](https://www.microsoft.com/en-us/research/publication/computing-the-shortest-path-a-search-meets-graph-theory/)。本项目只研究静态无向单位栅格上的单向搜索及具体地标选择；不复刻原论文全部算法。"]
    (output/"实验报告.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def plot_results(output, summary):
    import os
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT/".cache"/"matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    families = sorted({r["family"] for r in summary["families"]})
    labels = {"manhattan": "Manhattan", "alt_corners": "Corners", "alt_farthest": "Farthest", "alt_boundary": "Boundary"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    colors = ("#64748b", "#d69b32", "#237d9a", "#be5f59")
    for j, (method, label) in enumerate(labels.items()):
        rows = [next(r for r in summary["families"] if r["method"] == method and r["family"] == family) for family in families]
        positions = [i+(j-1.5)*.19 for i in range(len(families))]
        axes[0].bar(positions, [r["geomean_query_speedup"] for r in rows], .18, label=label, color=colors[j])
        axes[1].bar(positions, [100*r["mean_expanded_reduction"] for r in rows], .18, label=label, color=colors[j])
    axes[0].axhline(1, color="#333333", lw=.8)
    axes[0].set_ylabel("Query speedup vs Manhattan (geomean)")
    axes[0].set_title("Query time excludes preprocessing")
    axes[1].set_ylabel("Mean reduction in expanded vertices (%)")
    axes[1].set_title("Fewer expansions do not guarantee lower time")
    for ax in axes:
        ax.set_xticks(range(len(families)), families)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=8)
    fig.savefig(output/"comparison.png", dpi=170)
    plt.close(fig)
