"""Paired experiments. Parameters are saved before any benchmark run."""

from dataclasses import asdict, replace
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import random
import statistics
import sys
import time

from annealing import LABELS, MODES, SAConfig, nearest_neighbor, solve
from datasets import generate_points, write_points


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def write_csv(path: Path, records: list[dict]) -> None:
    if not records:
        raise ValueError("没有可保存的实验记录。")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bootstrap_interval(values: list[float], repetitions: int = 4000) -> list[float]:
    """Resample map-level means, not individual random seeds as independent maps."""
    if not values:
        raise ValueError("bootstrap 需要至少一个地图均值。")
    rng = random.Random(421)
    draws = sorted(statistics.fmean(rng.choices(values, k=len(values))) for _ in range(repetitions))
    return [draws[int(0.025 * repetitions)], draws[int(0.975 * repetitions)]]


def summarize(rows: list[dict]) -> dict:
    baseline = {(r["instance"], r["seed"]): r for r in rows if r["mode"] == "baseline"}
    groups = []
    for size, layout in sorted({(r["n"], r["layout"]) for r in rows}):
        for mode in MODES:
            subset = [r for r in rows if (r["n"], r["layout"], r["mode"]) == (size, layout, mode)]
            values = [r["best_length"] for r in subset]
            gains = [100 * (baseline[r["instance"], r["seed"]]["best_length"] - r["best_length"])
                     / baseline[r["instance"], r["seed"]]["best_length"] for r in subset]
            within_stds = [statistics.stdev([r["best_length"] for r in subset if r["instance"] == instance])
                           if sum(r["instance"] == instance for r in subset) > 1 else 0.0
                           for instance in sorted({r["instance"] for r in subset})]
            groups.append({"n": size, "layout": layout, "mode": mode, "runs": len(subset),
                           "mean_length": statistics.fmean(values),
                           "mean_within_instance_std": statistics.fmean(within_stds),
                           "mean_paired_gain_percent": statistics.fmean(gains),
                           "mean_seconds": statistics.fmean(r["elapsed_seconds"] for r in subset),
                           "mean_reheats": statistics.fmean(r["reheats"] for r in subset)})
    overall = []
    for mode in MODES:
        subset = [r for r in rows if r["mode"] == mode]
        gains = { (r["instance"], r["seed"]):
                  100 * (baseline[r["instance"], r["seed"]]["best_length"] - r["best_length"])
                  / baseline[r["instance"], r["seed"]]["best_length"] for r in subset }
        maps = sorted({r["instance"] for r in subset})
        instance_gains = [statistics.fmean(v for (instance, _), v in gains.items() if instance == name) for name in maps]
        overall.append({"mode": mode, "runs": len(subset), "instances": len(maps),
                        "mean_paired_gain_percent": statistics.fmean(instance_gains),
                        "map_bootstrap_95_interval": bootstrap_interval(instance_gains),
                        "wins": sum(v > 1e-8 for v in gains.values()),
                        "ties": sum(abs(v) <= 1e-8 for v in gains.values()),
                        "losses": sum(v < -1e-8 for v in gains.values()),
                        "mean_seconds": statistics.fmean(r["elapsed_seconds"] for r in subset),
                        "mean_reheats": statistics.fmean(r["reheats"] for r in subset),
                        "map_mean_gains": dict(zip(maps, instance_gains))})
    return {"groups": groups, "overall": overall,
            "gain_definition": "100 * (baseline_length - method_length) / baseline_length; positive is better",
            "uncertainty": "95% percentile bootstrap of map-level paired mean gains; descriptive, no generalization guarantee",
            "std_definition": "Mean of sample standard deviations across seeds within each map; not across-map length variation"}


def run_benchmark(spec: dict, out: Path) -> dict:
    config = SAConfig(**spec["algorithm"], trace_every=0)
    config.validate()
    for key in ("sizes", "layouts", "seeds"):
        if not spec.get(key) or len(set(spec[key])) != len(spec[key]):
            raise ValueError(f"{key} 必须非空且无重复。")
    if any(type(n) is not int or n < 4 for n in spec["sizes"]):
        raise ValueError("sizes 必须为至少为 4 的整数。")
    if any(layout not in ("uniform", "clustered") for layout in spec["layouts"]):
        raise ValueError("未知地图分布。")
    if any(type(seed) is not int for seed in spec["seeds"]) or type(spec["data_seed"]) is not int:
        raise ValueError("随机种子必须为整数。")
    if type(spec["instances_per_layout"]) is not int or spec["instances_per_layout"] < 1:
        raise ValueError("instances_per_layout 必须为正整数。")
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"输出目录已有结果，请指定新的 --out 目录：{out}")
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "protocol.json", spec)
    started = time.perf_counter()
    root = Path(__file__).parent
    manifest = {"python": sys.version, "platform": platform.platform(), "processor": platform.processor(),
                "source_sha256": {p.name: file_hash(p) for p in sorted(root.glob("*.py"))},
                "protocol_sha256": file_hash(out / "protocol.json"),
                "matrix_setup_timing": "Distance matrix construction excluded equally; solver time includes greedy initialization and temperature calibration.",
                "order": "Deterministically shuffled method order for each paired run.", "data_sha256": {}}
    manifest["resolved_base_config"] = asdict(config)
    manifest["packages"] = {name: importlib.metadata.version(name) for name in ("numpy", "matplotlib")}
    snapshot = out / "source_snapshot"
    snapshot.mkdir()
    for source in sorted(root.glob("*.py")):
        (snapshot / source.name).write_bytes(source.read_bytes())
    rows, solutions = [], []
    total = len(spec["sizes"]) * len(spec["layouts"]) * spec["instances_per_layout"] * len(spec["seeds"]) * len(MODES)
    for size in spec["sizes"]:
        for layout in spec["layouts"]:
            for index in range(spec["instances_per_layout"]):
                data_seed = spec["data_seed"] + size * 1009 + (0 if layout == "uniform" else 1000003) + index * 97
                name = f"{layout}_{size}_{index}"
                problem = generate_points(size, layout, data_seed)
                data_path = out / "data" / f"{name}.csv"
                write_points(problem, data_path)
                manifest["data_sha256"][data_path.name] = file_hash(data_path)
                for seed in spec["seeds"]:
                    modes = list(MODES)
                    random.Random(seed + data_seed).shuffle(modes)
                    for mode in modes:
                        result = solve(problem, replace(config, mode=mode), seed)
                        rows.append({"instance": name, "n": size, "layout": layout, "data_seed": data_seed,
                                     "seed": seed, "mode": mode, "initial_length": result["initial_length"],
                                     "best_length": result["best_length"], "elapsed_seconds": result["elapsed_seconds"],
                                     "evaluations": result["evaluations"], "calibration_evaluations": result["calibration_evaluations"],
                                     "initial_temperature": result["initial_temperature"], "reheats": result["reheats"],
                                     "accepted": result["accepted"], "accepted_worse": result["accepted_worse"]})
                        solutions.append({"instance": name, "seed": seed, "mode": mode,
                                          "best_route": result["best_route"], "best_length": result["best_length"],
                                          "events": result["events"]})
                        if len(rows) % 60 == 0 or len(rows) == total:
                            print(f"实验进度 {len(rows)}/{total}", flush=True)
    summary = summarize(rows)
    manifest["wall_seconds"] = time.perf_counter() - started
    manifest["completed_runs"] = len(rows)
    write_csv(out / "runs.csv", rows)
    write_csv(out / "group_summary.csv", summary["groups"])
    write_json(out / "solutions.json", solutions)
    write_json(out / "summary.json", summary)
    write_json(out / "manifest.json", manifest)
    write_report(out, summary, spec, manifest)
    return summary


def write_report(out: Path, summary: dict, spec: dict, manifest: dict) -> None:
    lines = ["# 对照实验报告", "", "本报告由实际运行结果自动生成。坐标为合成平面坐标，路程单位为坐标单位。",
             "", "实验用途：" + spec.get("purpose", "记录给定配置下的配对比较。"),
             "", f"共 {manifest['completed_runs']} 次求解；每组 {len(spec['seeds'])} 个种子，"
             f"每种规模和分布 {spec['instances_per_layout']} 张独立地图。每次搜索 "
             f"{spec['algorithm']['evaluations']:,} 次候选评估，另有相同预算的温度校准。",
             "", "## 总体配对比较", "", "收益为 100 × (不重热路程 − 当前方法路程) / 不重热路程；正值表示路线更短。",
             "", "| 方法 | 平均收益 | 地图层面 bootstrap 95%区间 | 胜 / 平 / 负 | 平均耗时 | 平均重热次数 |",
             "|---|---:|---:|---:|---:|---:|"]
    for r in summary["overall"]:
        lo, hi = r["map_bootstrap_95_interval"]
        lines.append(f"| {LABELS[r['mode']]} | {r['mean_paired_gain_percent']:.3f}% | [{lo:.3f}%, {hi:.3f}%] | "
                     f"{r['wins']} / {r['ties']} / {r['losses']} | {r['mean_seconds']:.4f} s | {r['mean_reheats']:.2f} |")
    improved = next(r for r in summary["overall"] if r["mode"] == "stagnation")
    verb = "缩短" if improved["mean_paired_gain_percent"] >= 0 else "增加"
    lines.extend(["", f"停滞触发方案相对不重热方案的配对平均路程{verb} "
                  f"{abs(improved['mean_paired_gain_percent']):.3f}%。这个平均值不代表每张地图或每次运行都改善。",
                  "区间以地图均值为重采样单位，同一地图上的多次随机运行不被当成独立地图。"
                  "地图数量较少，区间只描述本组合成实验，不作普遍优越性或显著性结论。",
                  "", "## 分场景结果", "", "标准差列先在同一地图的多次运行内计算样本标准差，再对地图取平均。",
                  "", "| 点数（含仓库） | 分布 | 方法 | 平均路程 | 图内标准差均值 | 配对平均收益 |",
                  "|---:|---|---|---:|---:|---:|"])
    for r in summary["groups"]:
        lines.append(f"| {r['n']} | {r['layout']} | {LABELS[r['mode']]} | {r['mean_length']:.3f} | "
                     f"{r['mean_within_instance_std']:.3f} | {r['mean_paired_gain_percent']:.3f}% |")
    lines.extend(["", "## 比较边界与复现", "",
                  "- 三组使用相同的最近邻初始路线、2-opt 邻域、校准方法、数据和种子；仅重热触发规则不同。",
                  "- periodic 是触发机制对照，baseline 是关闭整个重热组件的消融。两种重热方法共享幅度、温度门槛、次数上限和截止比例。实际触发次数可能不同，已单独报告。",
                  "- 候选提案与接受随机数采用独立随机流，各组每步消耗相同数量的接受随机数。随着路线变化，候选结果仍会不同。",
                  "- 总预算包括相同的搜索评估数和校准评估数；CPU 时间另行报告。矩阵构建不计入求解耗时，各组一致。方法执行顺序按固定种子打乱。",
                  "- 参数在运行前写入 protocol.json，完整展开值见 manifest.json。正式验证使用与开发阶段不同的数据种子和求解种子；最终参数只依据开发实验选择，验证后不再调整。开发结果单独保留。",
                  "- 受控重热后按剩余评估预算重新计算几何降温率，使各组在预算结束时到达相同温度下限。具体模式以 resolved_base_config 为准。",
                  "- routes、CSV 输入和代码哈希分别保存在 solutions.json、data/ 和 manifest.json；逐次记录在 runs.csv。",
                  "- 大规模实例没有精确最优值，因此收益是相对基准的改善率，不是最优误差。微型实例另用 Held–Karp 验证。",
                  "- 重热、2-opt 和模拟退火均为已有方法；本项目考察具体受控触发规则的实验效果，不宣称首次提出这些机制。",
                  "- 本项目仅处理单车、对称欧氏距离、无容量和时间窗约束的 TSP。",
                  "", f"本次全部求解及数据写入耗时约 {manifest['wall_seconds']:.1f} 秒，机器信息见 manifest.json。", ""])
    (out / "实验报告.md").write_text("\n".join(lines), encoding="utf-8")
