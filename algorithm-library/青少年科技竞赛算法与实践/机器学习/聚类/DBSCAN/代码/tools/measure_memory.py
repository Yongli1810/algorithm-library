"""Sample isolated-worker RSS separately from the timing experiment.

Reports an observed peak (sampling can miss brief allocations), never claims
an exact allocator peak. Imports and input loading precede the RSS baseline.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def worker(csv_path, method, eps, min_samples):
    import gc
    import psutil
    from datasets import load_csv
    from experiments import solve
    if method == "sklearn_kd_tree":
        from sklearn.cluster import DBSCAN  # preload before measuring baseline
    points = load_csv(csv_path)
    gc.collect()
    info = psutil.Process().memory_info()
    print(json.dumps({"event": "ready", "worker_pid": os.getpid(), "baseline_rss_bytes": info.rss,
                      "baseline_native_peak_rss_bytes": getattr(info, "peak_wset", None)}), flush=True)
    if sys.stdin.readline().strip() != "GO":
        raise RuntimeError("missing parent handshake")
    result = solve(points, {"eps": eps, "min_samples": min_samples}, method)
    info = psutil.Process().memory_info()
    print(json.dumps({"event": "done", "rss_bytes": info.rss,
                      "native_peak_rss_bytes": getattr(info, "peak_wset", None),
                      "clusters": result.clusters, "noise": int((result.labels == -1).sum())}), flush=True)
    # Keep result alive until the parent receives the completion packet.
    sys.stdin.readline()


def measure(results, size=5000):
    import psutil
    from experiments import ALL_METHODS, dump_json, write_csv
    results = Path(results).resolve()
    destination = results / "memory.csv"
    if destination.exists():
        raise ValueError("memory.csv already exists; preserve earlier measurements")
    manifest = json.loads((results / "manifest.json").read_text(encoding="utf-8"))
    selected, seen = [], set()
    for case in manifest["cases"]:
        if case["n"] == size and case["family"] not in seen:
            selected.append(case)
            seen.add(case["family"])
    if not selected:
        raise ValueError("No cases at the requested size")
    dump_json(results / "memory_protocol.json", {
        "size": size, "instances_per_family": 1, "repeats": 1, "interval_seconds": 0.002,
        "description": "Separate fresh process per method/input. RSS baseline after imports, loading input and GC. Parent polls during fit, including construction. End RSS also sampled. Peak is an observed lower bound, affected by allocator and polling; delta excludes import/input baseline. Not used for timing claims.",
        "psutil": psutil.__version__, "cases": [c["case"] for c in selected]})
    rows = []
    for case in selected:
        for method in ALL_METHODS:
            command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--csv", str(results / case["file"]),
                       "--method", method, "--eps", str(case["eps"]), "--min-samples", str(case["min_samples"])]
            env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                       text=True, encoding="utf-8", env=env, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            stop = threading.Event()
            peak, samples = [0], [0]
            def poll():
                # Windows virtual-environment python.exe can be a launcher.
                # Monitor the PID reported by the actual interpreter, not Popen.pid.
                child = psutil.Process(ready["worker_pid"])
                while not stop.is_set():
                    try:
                        peak[0] = max(peak[0], child.memory_info().rss)
                        samples[0] += 1
                    except psutil.NoSuchProcess:
                        return
                    stop.wait(0.002)
            try:
                packet = process.stdout.readline()
                if not packet:
                    raise RuntimeError(process.stderr.read())
                ready = json.loads(packet)
                peak[0] = ready["baseline_rss_bytes"]
                thread = threading.Thread(target=poll, daemon=True)
                thread.start()
                process.stdin.write("GO\n")
                process.stdin.flush()
                packet = process.stdout.readline()
                if not packet:
                    raise RuntimeError(process.stderr.read())
                done = json.loads(packet)
                stop.set()
                thread.join(timeout=2)
                peak[0] = max(peak[0], done["rss_bytes"])
                process.stdin.write("ACK\n")
                process.stdin.flush()
                process.wait(timeout=30)
                if process.returncode:
                    raise RuntimeError(process.stderr.read())
                rows.append({"case": case["case"], "family": case["family"], "n": size, "method": method,
                             "baseline_rss_bytes": ready["baseline_rss_bytes"], "sampled_peak_rss_bytes": peak[0],
                             "sampled_extra_rss_bytes": max(0, peak[0] - ready["baseline_rss_bytes"]), "samples": samples[0],
                             "worker_pid": ready["worker_pid"], "launcher_pid": process.pid,
                             "native_peak_rss_bytes": done["native_peak_rss_bytes"],
                             "baseline_native_peak_rss_bytes": ready["baseline_native_peak_rss_bytes"]})
                write_csv(destination, rows)
                print(f"Memory {case['family']} {method}: +{rows[-1]['sampled_extra_rss_bytes'] / 2**20:.2f} MiB", flush=True)
            finally:
                stop.set()
                if process.poll() is None:
                    process.kill()
                    process.wait()
                for stream in (process.stdin, process.stdout, process.stderr):
                    stream.close()
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--method")
    parser.add_argument("--eps", type=float)
    parser.add_argument("--min-samples", type=int)
    args = parser.parse_args()
    if args.worker:
        worker(args.csv, args.method, args.eps, args.min_samples)
