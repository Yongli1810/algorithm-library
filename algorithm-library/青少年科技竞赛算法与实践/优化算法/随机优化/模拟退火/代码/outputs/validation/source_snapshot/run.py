"""Command line entry: demo, benchmark, micro, test."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import unittest

from annealing import LABELS, MODES, Problem, SAConfig, exact_tsp, solve, tour_length, two_opt_delta
from datasets import generate_points, load_points, write_points
from experiments import run_benchmark, write_json

ROOT = Path(__file__).resolve().parent


def demo(args) -> None:
    from visualization import plot_demo, write_demo_html
    problem = load_points(args.csv) if args.csv else generate_points(args.n, args.layout, args.data_seed)
    config = SAConfig(evaluations=args.evaluations, patience=args.patience,
                      periodic_interval=args.interval, trace_every=max(1, args.evaluations // 200))
    results = [solve(problem, replace(config, mode=mode), args.seed) for mode in MODES]
    args.out.mkdir(parents=True, exist_ok=True)
    write_points(problem, args.out / "points.csv")
    write_json(args.out / "demo.json", {"points": problem.points, "results": results})
    plot_demo(problem.points, results, args.out)
    write_demo_html(problem.points, results, args.out)
    for result in results:
        print(f"{LABELS[result['mode']]}: 初始 {result['initial_length']:.3f} → 最好 {result['best_length']:.3f}；重热 {result['reheats']} 次")
    print(f"交互演示：{(args.out / 'demo.html').resolve()}")


def micro(args) -> None:
    square = Problem(((0, 0), (1, 0), (1, 1), (0, 1)))
    crossed = [0, 2, 1, 3]
    delta = two_opt_delta(crossed, square.distances, 1, 2)
    route = [0, 1, 2, 3]
    small = generate_points(8, "uniform", 73)
    optimum_route, optimum = exact_tsp(small)
    results = [solve(small, SAConfig(mode=mode, evaluations=3000, patience=250,
                                    periodic_interval=750, trace_every=1), seed=7) for mode in MODES]
    for result in results:
        if result["best_length"] < optimum - 1e-8:
            raise AssertionError("启发式结果小于精确最优值，请检查距离计算。")
    payload = {"square": {"points": square.points, "before": crossed, "after": route,
                          "before_length": tour_length(crossed, square.distances),
                          "after_length": tour_length(route, square.distances), "delta": delta},
               "eight_points": small.points, "exact_route": optimum_route, "exact_length": optimum,
               "results": results}
    write_json(args.out / "micro.json", payload)
    write_points(small, args.out / "eight_points.csv")
    print(f"正方形手算：{payload['square']['before_length']:.6f} + ({delta:.6f}) = 4.000000")
    print(f"8 点精确最优：{optimum:.6f}")
    for result in results:
        print(f"{LABELS[result['mode']]}：{result['best_length']:.6f}，相对最优差距 {(result['best_length']/optimum - 1)*100:.4f}%")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="模拟退火配送路线优化实验（单车对称欧氏 TSP）")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("demo", help="生成路线图、曲线和离线交互演示")
    p.add_argument("--n", type=int, default=50, help="点数，包含仓库")
    p.add_argument("--layout", choices=("uniform", "clustered"), default="clustered")
    p.add_argument("--data-seed", type=int, default=20261022)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--evaluations", type=int, default=12000)
    p.add_argument("--patience", type=int, default=1000)
    p.add_argument("--interval", type=int, default=3000)
    p.add_argument("--csv", type=Path, help="自定义 id,x,y 坐标文件，覆盖合成数据参数")
    p.add_argument("--out", type=Path, default=ROOT / "outputs" / "demo")
    p = sub.add_parser("benchmark", help="运行三组配对实验；已有结果不会被覆盖")
    p.add_argument("--config", type=Path, default=ROOT / "configs" / "full.json")
    p.add_argument("--out", type=Path, default=ROOT / "outputs" / "validation")
    p = sub.add_parser("micro", help="4 点手算及 8 点精确解验证，保留每一步状态")
    p.add_argument("--out", type=Path, default=ROOT / "outputs" / "micro")
    sub.add_parser("test", help="运行标准库 unittest 测试")
    args = parser.parse_args(argv)
    try:
        if args.command == "demo":
            demo(args)
        elif args.command == "micro":
            micro(args)
        elif args.command == "benchmark":
            from visualization import plot_benchmark
            spec = json.loads(args.config.read_text(encoding="utf-8-sig"))
            summary = run_benchmark(spec, args.out)
            plot_benchmark(summary, args.out)
            print(f"实验报告：{(args.out / '实验报告.md').resolve()}")
        else:
            suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
            result = unittest.TextTestRunner(verbosity=2).run(suite)
            return 0 if result.wasSuccessful() else 1
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
