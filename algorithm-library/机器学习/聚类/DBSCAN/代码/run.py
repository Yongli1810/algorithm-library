"""Command-line entry point. Run `python run.py --help`."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="DBSCAN grid and geometric pruning experiments")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("test", help="Run correctness tests")
    bench = sub.add_parser("benchmark", help="Run frozen paired comparisons")
    bench.add_argument("--config", type=Path, default=ROOT / "configs/validation.json")
    bench.add_argument("--out", type=Path, default=ROOT / "outputs/validation")
    demo = sub.add_parser("demo", help="Build an offline interactive teaching demo")
    demo.add_argument("--out", type=Path, default=ROOT / "outputs/demo")
    sub.add_parser("micro", help="Write a hand-checkable ten-point example")
    memory = sub.add_parser("memory", help="Separate-process sampled RSS measurements")
    memory.add_argument("--results", type=Path, default=ROOT / "outputs/validation")
    memory.add_argument("--size", type=int, default=5000)
    verify = sub.add_parser("verify", help="Audit saved inputs, source hashes, and clustering results")
    verify.add_argument("--results", type=Path, default=ROOT / "outputs/validation")
    cluster = sub.add_parser("cluster", help="Cluster your own id,x,y CSV")
    cluster.add_argument("--csv", type=Path, required=True)
    cluster.add_argument("--eps", type=float, required=True)
    cluster.add_argument("--min-samples", type=int, default=5)
    cluster.add_argument("--method", choices=["brute", "grid", "pruned"], default="pruned")
    cluster.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "test":
            return subprocess.call([sys.executable, "-m", "unittest", "discover", "-s", str(ROOT / "tests"), "-v"], cwd=ROOT)
        if args.command == "benchmark":
            from experiments import benchmark
            result = benchmark(args.config, args.out)
            print(json.dumps({"datasets": result["datasets"], "runs": result["measured_runs"], "all_equal": result["all_equal"], "overall": result["overall"]}, indent=2))
        elif args.command in ("demo", "micro"):
            from visualization import build_demo, build_micro
            print(build_demo(args.out) if args.command == "demo" else build_micro(ROOT / "outputs/micro"))
        elif args.command == "memory":
            from tools.measure_memory import measure
            measure(args.results, args.size)
        elif args.command == "verify":
            from tools.verify_results import verify
            verify(args.results)
        elif args.command == "cluster":
            from datasets import load_csv
            from dbscan import fit
            from experiments import dump_json
            if args.out.exists():
                raise ValueError("Output already exists; choose a new --out")
            result = fit(load_csv(args.csv), args.eps, args.min_samples, args.method)
            args.out.parent.mkdir(parents=True, exist_ok=True)
            dump_json(args.out, {"labels": result.labels.tolist(), "core": result.core.astype(int).tolist(),
                                 "stats": result.stats, "timings": result.timings,
                                 "eps": args.eps, "min_samples": args.min_samples, "method": args.method})
            print(args.out.resolve())
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
