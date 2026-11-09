"""Command-line entry point; run from this folder or use run.cmd anywhere."""
import argparse
from dataclasses import asdict
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="A* landmark selection and amortized costs")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("test", help="Run independent-oracle correctness tests")
    bench = commands.add_parser("benchmark", help="Run frozen paired benchmark")
    bench.add_argument("--config", type=Path, default=ROOT/"configs/validation.json")
    bench.add_argument("--out", type=Path, default=ROOT/"outputs/validation")
    verify = commands.add_parser("verify", help="Independently audit stored results")
    verify.add_argument("--results", type=Path, default=ROOT/"outputs/validation")
    for name in ("demo", "micro"):
        cmd = commands.add_parser(name)
        cmd.add_argument("--out", type=Path, default=ROOT/"outputs"/name)
    solve = commands.add_parser("solve", help="Solve a custom rectangular . / # text map")
    solve.add_argument("--map", required=True, type=Path)
    solve.add_argument("--start", nargs=2, type=int, required=True, metavar=("X", "Y"))
    solve.add_argument("--goal", nargs=2, type=int, required=True, metavar=("X", "Y"))
    solve.add_argument("--method", choices=("dijkstra", "manhattan", "alt_corners", "alt_farthest", "alt_boundary"), default="alt_boundary")
    solve.add_argument("--landmarks", type=int, default=4)
    solve.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "test":
        return subprocess.call([sys.executable, "-X", "utf8", "-m", "unittest", "discover", "-s", str(ROOT/"tests"), "-v"], cwd=ROOT)
    if args.command == "benchmark":
        from experiments import benchmark
        benchmark(args.config, args.out)
    elif args.command == "verify":
        from tools.verify_results import verify
        verify(args.results)
    elif args.command in ("demo", "micro"):
        from visualization import build_demo, build_micro
        (build_demo if args.command == "demo" else build_micro)(args.out)
    elif args.command == "solve":
        from datasets import load_map
        from experiments import dump_json
        from pathfinding import build_landmarks, search
        if args.out.exists():
            raise ValueError("Output already exists; use a new --out")
        grid = load_map(args.map)
        index = build_landmarks(grid, args.method[4:], args.landmarks) if args.method.startswith("alt_") else None
        result = search(grid, grid.vertex(*args.start), grid.vertex(*args.goal), args.method, index)
        dump_json(args.out, {"method": args.method, "start_xy": args.start, "goal_xy": args.goal,
                             **asdict(result), "path_xy": [[grid.x[v], grid.y[v]] for v in result.path]})
        print(f"cost={result.cost}, expanded={result.expanded}, output={args.out.resolve()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2)
