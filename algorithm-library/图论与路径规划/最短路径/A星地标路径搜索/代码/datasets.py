"""Map/query generation with separate seeds; all maps remain unweighted."""
from pathlib import Path
import random

from pathfinding import Grid

FAMILIES = ("open", "random", "maze", "corridor")


def generate(family, size, seed):
    if family not in FAMILIES:
        raise ValueError("Unknown family")
    if type(size) is not int or size < 7 or size % 2 != 1:
        raise ValueError("Size must be an odd integer >= 7")
    rng = random.Random(seed)
    rows = [["."] * size for _ in range(size)]
    if family == "random":
        rows = [["#" if rng.random() < .24 else "." for _ in range(size)] for _ in range(size)]
    elif family == "maze":
        rows = [["#"] * size for _ in range(size)]
        rows[1][1] = "."
        stack = [(1, 1)]
        while stack:
            x, y = stack[-1]
            choices = [(nx, ny) for nx, ny in ((x-2, y), (x+2, y), (x, y-2), (x, y+2))
                       if 0 < nx < size-1 and 0 < ny < size-1 and rows[ny][nx] == "#"]
            if not choices:
                stack.pop()
                continue
            nx, ny = rng.choice(choices)
            rows[(y+ny)//2][(x+nx)//2] = rows[ny][nx] = "."
            stack.append((nx, ny))
    elif family == "corridor":
        # Parallel full-height walls, alternating narrow top/bottom passages.
        for i, x in enumerate(range(4, size-2, 6)):
            gap = rng.randrange(1, min(4, size-1))
            if i % 2:
                gap = size-1-gap
            for y in range(size):
                rows[y][x] = "." if y == gap else "#"
    return tuple("".join(row) for row in rows)


def make_queries(grid, count, seed):
    if type(count) is not int or count < 1:
        raise ValueError("Query count must be positive")
    if not grid.components:
        raise ValueError("No free cells")
    component = max(grid.components, key=lambda c: (len(c), -c[0]))
    if len(component) < 2:
        raise ValueError("Need a connected component of at least two cells")
    rng = random.Random(seed)
    return [tuple(rng.sample(component, 2)) for _ in range(count)]


def save_map(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def load_map(path):
    return Grid(Path(path).read_text(encoding="utf-8-sig").splitlines())


def micro_case():
    # S=(0,0), T=(4,0). The wall forces a four-step vertical detour.
    return Grid(("..#..", "..#..", ".....")), 0, 4
