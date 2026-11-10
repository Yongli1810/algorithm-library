"""Reproducible synthetic Euclidean instances and strict CSV I/O."""

import csv
from pathlib import Path
import random

from annealing import Problem


def generate_points(n: int, layout: str, seed: int) -> Problem:
    if type(n) is not int or n < 4:
        raise ValueError("点数 n 必须是至少为 4 的整数，并包含仓库。")
    rng = random.Random(seed)
    points = [(50.0, 50.0)]
    if layout == "uniform":
        points.extend((rng.uniform(0, 100), rng.uniform(0, 100)) for _ in range(n - 1))
    elif layout == "clustered":
        centers = [(20, 20), (80, 20), (20, 80), (80, 80)]
        for _ in range(n - 1):
            cx, cy = rng.choice(centers)
            points.append((min(100.0, max(0.0, rng.gauss(cx, 7))),
                           min(100.0, max(0.0, rng.gauss(cy, 7)))))
    else:
        raise ValueError("layout 必须为 uniform 或 clustered。")
    return Problem(tuple(points))


def write_points(problem: Problem, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "x", "y"])
        for i, (x, y) in enumerate(problem.points):
            writer.writerow([i, format(x, ".17g"), format(y, ".17g")])


def load_points(path: Path) -> Problem:
    rows = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not {"id", "x", "y"}.issubset(reader.fieldnames):
            raise ValueError("CSV 必须有 id,x,y 三列；id 为从 0 开始的连续整数。")
        for number, row in enumerate(reader, start=2):
            try:
                node = int(row["id"])
                point = (float(row["x"]), float(row["y"]))
            except (ValueError, TypeError) as exc:
                raise ValueError(f"CSV 第 {number} 行不是有效的 id,x,y。") from exc
            if node in rows:
                raise ValueError(f"CSV 第 {number} 行的 id 重复。")
            rows[node] = point
    if sorted(rows) != list(range(len(rows))):
        raise ValueError("CSV id 必须从 0 开始连续编号。")
    return Problem(tuple(rows[i] for i in range(len(rows))))
