"""Paired comparisons, frozen inputs, raw measurements, and reproducibility."""
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
import os
import platform
from pathlib import Path
import shutil
import sys
from time import perf_counter

import numpy as np

from dbscan import METHODS, canonical_labels, fit, reference_fit
from datasets import FAMILIES, PARAMETERS, generate, save_csv

ROOT = Path(__file__).resolve().parent
ALL_METHODS = (*METHODS, "sklearn_kd_tree")
NAMES = {"brute": "全量扫描", "grid": "仅网格", "pruned": "网格加剪枝", "sklearn_kd_tree": "scikit-learn KDTree"}


def dump_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def solve(points, params, method):
    return reference_fit(points, **params) if method == "sklearn_kd_tree" else fit(points, **params, method=method)


def benchmark(config_path, output):
    import scipy
    import sklearn
    config_path, output = Path(config_path), Path(output)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory is not empty; select a new --out to preserve earlier evidence")
    output.mkdir(parents=True, exist_ok=True)
    (output / "source_snapshot").mkdir()
    for filename in ("dbscan.py", "datasets.py", "experiments.py", "run.py"):
        shutil.copyfile(ROOT / filename, output / "source_snapshot" / filename)
    shutil.copyfile(config_path, output / "config.json")
    protocol = {"created_utc": datetime.now(timezone.utc).isoformat(), "config": config,
                "parameters": PARAMETERS, "methods": list(ALL_METHODS),
                "distance_rule": "numpy.hypot(dx,dy) <= eps; self included; sorted original ids",
                "timing": "End-to-end fit, including validation and index/box construction; excludes imports, data generation, saving, comparison and plotting. No trace. Warm-up once per method. Randomized method order, fixed order seed.",
                "external": "sklearn DBSCAN algorithm=kd_tree, leaf_size=40, euclidean, n_jobs=1; implementation comparison, not isolated algorithm effect",
                "environment": {"python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__,
                                "sklearn": sklearn.__version__, "platform": platform.platform(), "processor": platform.processor()},
                "source_sha256": {p.name: digest(p) for p in sorted((output / "source_snapshot").iterdir())}}
    dump_json(output / "protocol.json", protocol)
    cases = []
    for family_index, family in enumerate(config["families"]):
        for n in config["sizes"]:
            for seed in config["seeds"]:
                actual_seed = seed + 100000 * family_index + n
                case_id = f"{family}_{n}_{seed}"
                points = generate(family, n, actual_seed)
                path = output / "data" / f"{case_id}.csv"
                save_csv(path, points)
                cases.append({"case": case_id, "family": family, "n": n, "seed": actual_seed,
                              "file": str(path.relative_to(output)), "sha256": digest(path), **PARAMETERS[family]})
    # Freeze all inputs before any measured validation run.
    dump_json(output / "manifest.json", {"protocol_sha256": digest(output / "protocol.json"), "cases": cases})
    warm = generate("moons", 120, 1234567)
    for method in ALL_METHODS:
        solve(warm, PARAMETERS["moons"], method)
    order_rng = np.random.default_rng(config["order_seed"])
    rows, solutions = [], {}
    for number, case in enumerate(cases, 1):
        points = generate(case["family"], case["n"], case["seed"])
        params = {"eps": case["eps"], "min_samples": case["min_samples"]}
        case_results = {}
        for rep in range(config["repeats"]):
            for order, method in enumerate(order_rng.permutation(ALL_METHODS).tolist()):
                tick = perf_counter()
                result = solve(points, params, method)
                elapsed = perf_counter() - tick
                if method in case_results:
                    np.testing.assert_array_equal(result.labels, case_results[method].labels)
                else:
                    case_results[method] = result
                row = {"case": case["case"], "family": case["family"], "n": case["n"], "seed": case["seed"],
                       "eps": case["eps"], "min_samples": case["min_samples"], "repeat": rep, "order": order,
                       "method": method, "seconds": elapsed, "index_seconds": result.timings.get("index_seconds", ""),
                       "query_seconds": result.timings.get("query_seconds", ""), "clusters": result.clusters,
                       "noise": int((result.labels == -1).sum()), "core": int(result.core.sum())}
                for key in ("queries", "distance_evaluations", "neighbor_entries", "candidate_cells", "box_checks",
                            "skipped_cells", "accepted_cells", "bulk_accepted_points", "fallback_queries"):
                    row[key] = result.stats.get(key, "")
                rows.append(row)
        baseline = case_results["brute"]
        solutions[case["case"]] = {}
        for method, result in case_results.items():
            checks = {"partition_equal": bool(np.array_equal(canonical_labels(result.labels), canonical_labels(baseline.labels))),
                      "core_equal": bool(np.array_equal(result.core, baseline.core)),
                      "noise_equal": bool(np.array_equal(result.labels == -1, baseline.labels == -1))}
            solutions[case["case"]][method] = {"labels": result.labels.tolist(), "core": result.core.astype(int).tolist(), **checks}
            for row in rows:
                if row["case"] == case["case"] and row["method"] == method:
                    row.update(checks)
            if not all(checks.values()):
                dump_json(output / "mismatch.json", solutions[case["case"]])
                raise AssertionError(f"Clustering mismatch in {case['case']}: {method}. Evidence saved.")
        write_csv(output / "runs.csv", rows)
        print(f"[{number}/{len(cases)}] {case['case']} verified; {len(rows)} measured fits", flush=True)
    dump_json(output / "solutions.json", solutions)
    summary = summarize(rows)
    dump_json(output / "summary.json", summary)
    write_csv(output / "case_summary.csv", summary["cases"])
    write_csv(output / "group_summary.csv", summary["groups"])
    make_report(output, protocol, summary)
    plot_summary(output, summary)
    return summary


def summarize(rows):
    by_case = defaultdict(list)
    for row in rows:
        by_case[(row["case"], row["method"])].append(row)
    case_rows = []
    for (case, method), records in by_case.items():
        first = records[0]
        baseline = float(np.median([r["seconds"] for r in by_case[(case, "brute")]]))
        grid_time = float(np.median([r["seconds"] for r in by_case[(case, "grid")]]))
        median = float(np.median([r["seconds"] for r in records]))
        distances = first["distance_evaluations"]
        case_rows.append({"case": case, "family": first["family"], "n": first["n"], "method": method,
                          "median_seconds": median, "min_seconds": min(r["seconds"] for r in records),
                          "max_seconds": max(r["seconds"] for r in records), "speedup_vs_brute": baseline / median,
                          "speedup_vs_grid": grid_time / median, "distance_evaluations": distances,
                          "distance_reduction_percent": 100 * (1 - distances / first["n"] ** 2) if distances != "" else "",
                          "index_median_seconds": float(np.median([r["index_seconds"] for r in records])) if method in METHODS else "",
                          "neighbor_entries": first["neighbor_entries"], "clusters": first["clusters"], "noise": first["noise"],
                          "all_equal": all(r["partition_equal"] and r["core_equal"] and r["noise_equal"] for r in records)})
    groups = []
    for family in dict.fromkeys(r["family"] for r in rows):
        for n in sorted(set(r["n"] for r in rows)):
            for method in ALL_METHODS:
                selected = [r for r in case_rows if r["family"] == family and r["n"] == n and r["method"] == method]
                if not selected:
                    continue
                groups.append({"family": family, "n": n, "method": method, "instances": len(selected),
                               "mean_case_median_seconds": float(np.mean([r["median_seconds"] for r in selected])),
                               "geomean_speedup_vs_brute": math.exp(float(np.mean([math.log(r["speedup_vs_brute"]) for r in selected]))),
                               "geomean_speedup_vs_grid": math.exp(float(np.mean([math.log(r["speedup_vs_grid"]) for r in selected]))),
                               "mean_distance_reduction_percent": float(np.mean([r["distance_reduction_percent"] for r in selected])) if method in METHODS else ""})
    overall = []
    for method in ALL_METHODS:
        selected = [r for r in case_rows if r["method"] == method]
        overall.append({"method": method, "cases": len(selected),
                        "geomean_speedup_vs_brute": math.exp(float(np.mean([math.log(r["speedup_vs_brute"]) for r in selected]))),
                        "geomean_speedup_vs_grid": math.exp(float(np.mean([math.log(r["speedup_vs_grid"]) for r in selected]))),
                        "faster_than_brute_cases": sum(r["speedup_vs_brute"] > 1 for r in selected),
                        "faster_than_grid_cases": sum(r["speedup_vs_grid"] > 1 for r in selected)})
    return {"measured_runs": len(rows), "datasets": len(by_case) // len(ALL_METHODS), "all_equal": all(r["all_equal"] for r in case_rows),
            "cases": case_rows, "groups": groups, "overall": overall}


def make_report(output, protocol, summary):
    lines = ["# DBSCAN 网格与几何剪枝对照实验", "", f"数据集 {summary['datasets']} 份，实测 {summary['measured_runs']} 次；聚类分区、核心点和噪声点全部一致：{summary['all_equal']}。", "",
             "三个手写版本共享距离核与扩展代码。KDTree 为成熟实现的实际性能参照，库底层优化也会影响耗时，不能把其差距全部归因于数据结构。", "",
             "输入为合成二维数据，参数按数据族预先固定。每份输入重复计时后取中位数，再对实例配对加速比取几何平均；大于 1 表示加速，小于 1 表示变慢。有限重复仅描述本机观察，不作统计显著性结论。", "",
             "计时包含输入检查、网格与包围盒构建、查询、排序和聚类扩展，排除首次导入、生成数据、保存文件与绘图。四种方法先预热，每轮随机排列执行顺序；没有在计时中开启演示记录或内存采样。", "",
             "## 总体结果", "", "| 方法 | 相对全量扫描 | 相对仅网格 | 比全量扫描快的实例数 |", "|---|---:|---:|---:|"]
    for row in summary["overall"]:
        lines.append(f"| {NAMES[row['method']]} | {row['geomean_speedup_vs_brute']:.3f}× | {row['geomean_speedup_vs_grid']:.3f}× | {row['faster_than_brute_cases']}/{row['cases']} |")
    lines += ["", "## 按分布与规模", "", "| 分布 | 点数 | 方法 | 时间 秒 | 相对全量扫描 | 距离计算减少 |", "|---|---:|---|---:|---:|---:|"]
    for row in summary["groups"]:
        reduction = f"{row['mean_distance_reduction_percent']:.2f}%" if row["method"] in METHODS else "未获取"
        lines.append(f"| {row['family']} | {row['n']} | {NAMES[row['method']]} | {row['mean_case_median_seconds']:.5f} | {row['geomean_speedup_vs_brute']:.3f}× | {reduction} |")
    lines += ["", "## 解释与适用范围", "",
              "几何剪枝减少点对距离计算，但新增矩形判断与分支。应同时看 pruned 与 grid 的配对时间，不能仅凭距离次数降低宣布改进有效。计数不把包围盒的两次 hypot 当作点对距离；box_checks 另列。", "",
              "dense 数据中全部点互为邻居，输出的邻居总数为 n²。即使整格接纳使点对距离计算趋近于零，构建邻居数组、排序与扩展仍有平方级输出开销。程序逐点查询，不预存完整 n×n 邻接关系；scikit-learn 会批量保存邻域，因此内存行为不同。", "",
              "固定分布和固定 eps 时，随 n 增大平均邻居数也增加，不能从三档规模拟合出普遍的渐近复杂度。该方案限二维欧氏距离，不直接推广至高维或非欧氏距离。", "",
              "邻居查询默认只检查局部网格；浮点边界采用保守的范围扩张，可能比理想的九格多检查边缘格。包围盒临界判断回到逐点距离核，不修改 eps 判定规则。", "",
              "内存另用隔离子进程采样，见 memory.csv 和 memory_protocol.json（运行 memory 命令后生成）。RSS 采样是观察到的峰值下界，不能当作严格峰值证明；数据导入后的常驻内存为基线。", "",
              "## 复现与来源", "",
              "config.json 为本轮实际配置；protocol.json 固定规则与环境；manifest.json 固定输入和哈希；source_snapshot 是开跑时的核心源码；runs.csv 保留每次计时；solutions.json 保存每种实现的标签及核心点。", "",
              "[DBSCAN 官方文档](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html)；"
              "[GriT-DBSCAN 原论文](https://arxiv.org/abs/2210.07580)。本项目借鉴空间索引与几何剪枝思想，未实现论文的 grid tree，不沿用其复杂度结论，也不宣称原创聚类算法。", "",
              f"运行环境：Python {platform.python_version()}；NumPy {protocol['environment']['numpy']}；scikit-learn {protocol['environment']['sklearn']}。"]
    (output / "实验报告.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_summary(output, summary):
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.7), constrained_layout=True)
    colors = {"brute": "#7b8594", "grid": "#277da8", "pruned": "#15977f", "sklearn_kd_tree": "#ba6e21"}
    for ax, family in zip(axes.flat, FAMILIES):
        for method in ALL_METHODS:
            rows = [r for r in summary["groups"] if r["family"] == family and r["method"] == method]
            if rows:
                ax.plot([r["n"] for r in rows], [r["mean_case_median_seconds"] for r in rows], "o-", label=method, color=colors[method])
        ax.set(title=family, xlabel="Number of points", ylabel="End-to-end seconds", yscale="log")
        ax.grid(alpha=0.18)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("DBSCAN | same inputs and parameters | lower is faster", fontsize=15)
    fig.savefig(output / "comparison.png", dpi=160)
    plt.close(fig)
