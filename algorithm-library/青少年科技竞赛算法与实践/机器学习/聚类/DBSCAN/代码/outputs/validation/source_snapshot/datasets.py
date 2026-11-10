"""Fixed synthetic families; development and validation use disjoint seeds."""
import csv
from pathlib import Path
import numpy as np

FAMILIES = ("uniform", "moons", "blobs", "dense")
PARAMETERS = {
    "uniform": {"eps": 0.055, "min_samples": 8},
    "moons": {"eps": 0.070, "min_samples": 8},
    "blobs": {"eps": 0.075, "min_samples": 8},
    "dense": {"eps": 0.20, "min_samples": 8},
}


def generate(family, n, seed):
    if family not in FAMILIES or n < 1:
        raise ValueError("unknown family or nonpositive size")
    rng = np.random.default_rng(seed)
    if family == "uniform":
        points = rng.uniform(-1, 1, (n, 2))
    elif family == "moons":
        side = np.arange(n) % 2
        theta = rng.uniform(0, np.pi, n)
        points = np.column_stack((np.where(side == 0, np.cos(theta), 1 - np.cos(theta)),
                                  np.where(side == 0, np.sin(theta), 0.5 - np.sin(theta))))
        points += rng.normal(0, 0.025, (n, 2))
    elif family == "blobs":
        centers = np.array([[-0.7, -0.6], [0.65, -0.5], [-0.25, 0.65], [0.8, 0.7]])
        points = centers[rng.integers(0, 4, n)] + rng.normal(0, 0.08, (n, 2))
        noise = rng.random(n) < 0.08
        points[noise] = rng.uniform(-1.1, 1.1, (noise.sum(), 2))
    else:
        # All points lie in a single cell and are mutual neighbors. This is
        # an intentional quadratic-output stress case, not a favorable demo.
        points = rng.uniform(0.02, 0.06, (n, 2))
    return points[rng.permutation(n)].astype(np.float64)


def save_csv(path, points):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "x", "y"])
        for i, (x, y) in enumerate(points):
            writer.writerow([i, format(x, ".17g"), format(y, ".17g")])


def load_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ["id", "x", "y"]:
            raise ValueError("CSV header must be id,x,y")
        rows = list(reader)
    if [int(r["id"]) for r in rows] != list(range(len(rows))):
        raise ValueError("id must be consecutive from zero in input order")
    return np.asarray([[float(r["x"]), float(r["y"])] for r in rows], dtype=np.float64).reshape(-1, 2)
