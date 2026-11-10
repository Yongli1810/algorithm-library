"""Symmetric Euclidean TSP with auditable simulated annealing and reheating.

Core implementation uses only the standard library. Node 0 is the depot;
tours store each node once, with the return edge to the depot implicit.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
import random
import time
from typing import Sequence


MODES = ("baseline", "periodic", "stagnation")
LABELS = {"baseline": "不重热", "periodic": "固定间隔重热", "stagnation": "停滞触发重热"}


@dataclass(frozen=True)
class Problem:
    points: tuple[tuple[float, float], ...]
    distances: tuple[tuple[float, ...], ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if len(self.points) < 4:
            raise ValueError("至少需要 4 个点（包括编号 0 的仓库）。")
        try:
            if any(len(p) != 2 for p in self.points):
                raise ValueError("每个点必须包含 x、y 两个坐标。")
            points = tuple((float(p[0]), float(p[1])) for p in self.points)
        except (TypeError, IndexError, OverflowError) as exc:
            raise ValueError("坐标必须是有限数值。") from exc
        if not all(math.isfinite(x) and math.isfinite(y) for x, y in points):
            raise ValueError("坐标必须是有限数值，不能包含 NaN 或无穷大。")
        distances = tuple(tuple(math.hypot(a[0] - b[0], a[1] - b[1])
                                for b in points) for a in points)
        if not all(math.isfinite(d) for row in distances for d in row):
            raise ValueError("坐标尺度过大，距离溢出。")
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "distances", distances)


@dataclass(frozen=True)
class SAConfig:
    mode: str = "stagnation"
    evaluations: int = 12000
    calibration_samples: int = 200
    initial_acceptance: float = 0.5
    final_temperature_ratio: float = 0.0001
    patience: int = 1000
    periodic_interval: int = 3000
    reheat_multiplier: float = 3.0
    reheat_floor_ratio: float = 0.002
    reheat_gate_ratio: float = 0.005
    max_reheats: int = 3
    reheat_until_fraction: float = 0.9
    reheat_cooling: str = "remaining_budget"
    trace_every: int = 50

    def validate(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"mode 必须属于 {MODES}")
        if self.reheat_cooling not in ("fixed", "remaining_budget"):
            raise ValueError("reheat_cooling 必须为 fixed 或 remaining_budget。")
        for name in ("evaluations", "calibration_samples", "patience", "periodic_interval"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} 必须为正整数。")
        for name in ("max_reheats", "trace_every"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} 必须为非负整数。")
        for name in ("initial_acceptance", "final_temperature_ratio", "reheat_floor_ratio",
                     "reheat_gate_ratio", "reheat_until_fraction"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value < 1:
                raise ValueError(f"{name} 必须在 (0, 1) 内。")
        if not math.isfinite(self.reheat_multiplier) or self.reheat_multiplier <= 1:
            raise ValueError("reheat_multiplier 必须大于 1。")
        if self.final_temperature_ratio >= self.reheat_floor_ratio:
            raise ValueError("最低温度比例必须小于重热温度下限比例。")


def validate_tour(tour: Sequence[int], n: int) -> None:
    if len(tour) != n or any(type(x) is not int for x in tour) or sorted(tour) != list(range(n)):
        raise ValueError("路线必须恰好包含每个节点一次。")
    if tour[0] != 0:
        raise ValueError("路线必须以仓库 0 开头。")


def tour_length(tour: Sequence[int], distances: Sequence[Sequence[float]]) -> float:
    return math.fsum(distances[tour[k]][tour[(k + 1) % len(tour)]] for k in range(len(tour)))


def nearest_neighbor(problem: Problem) -> list[int]:
    remaining = set(range(1, len(problem.points)))
    route = [0]
    while remaining:
        node = min(remaining, key=lambda j: (problem.distances[route[-1]][j], j))
        route.append(node)
        remaining.remove(node)
    return route


def two_opt_delta(tour: Sequence[int], distances: Sequence[Sequence[float]], i: int, j: int) -> float:
    """Length change after reversing tour[i:j+1]; symmetric distances only."""
    if not 1 <= i < j < len(tour):
        raise ValueError("2-opt 索引必须满足 1 <= i < j < n。")
    a, b, c, d = tour[i - 1], tour[i], tour[j], tour[(j + 1) % len(tour)]
    return math.fsum((distances[a][c], distances[b][d], -distances[a][b], -distances[c][d]))


def sample_move(rng: random.Random, n: int) -> tuple[int, int]:
    # Uniform over unordered index pairs, excluding reversal of the entire cycle.
    while True:
        i, j = sorted(rng.sample(range(1, n), 2))
        if (i, j) != (1, n - 1):
            return i, j


def acceptance_probability(delta: float, temperature: float) -> float:
    if delta <= 0:
        return 1.0
    if temperature <= 0:
        return 0.0
    return math.exp(-delta / temperature)


def calibrate_temperature(problem: Problem, route: Sequence[int], config: SAConfig, seed: int) -> float:
    """A separate random stream keeps calibration from changing search proposals."""
    rng = random.Random(seed + 104729)
    positive = []
    for _ in range(config.calibration_samples):
        i, j = sample_move(rng, len(route))
        delta = two_opt_delta(route, problem.distances, i, j)
        if delta > 0:
            positive.append(delta)
    scale = math.fsum(positive) / len(positive) if positive else tour_length(route, problem.distances) / len(route)
    return max(scale / -math.log(config.initial_acceptance), 1e-12)


def solve(problem: Problem, config: SAConfig, seed: int = 0,
          initial_tour: Sequence[int] | None = None) -> dict:
    config.validate()
    started = time.perf_counter()
    route = list(initial_tour) if initial_tour is not None else nearest_neighbor(problem)
    validate_tour(route, len(problem.points))
    initial_route = route.copy()
    current = tour_length(route, problem.distances)
    best, best_route = current, route.copy()
    initial_length = current
    t0 = calibrate_temperature(problem, route, config, seed)
    t_min = t0 * config.final_temperature_ratio
    alpha = config.final_temperature_ratio ** (1.0 / config.evaluations)
    initial_alpha = alpha
    temperature = t0
    proposal_rng, acceptance_rng = random.Random(seed), random.Random(seed + 130363)
    accepted = accepted_worse = stale = reheats = 0
    last_reheat = 0
    trace, events = [], []

    def record(evaluation: int, used_temperature: float, delta: float = 0.0,
               probability: float = 1.0, uniform: float = 0.0,
               did_accept: bool = False, move: tuple[int, int] | None = None) -> None:
        trace.append({"evaluation": evaluation, "current_length": current, "best_length": best,
                      "temperature": used_temperature, "next_temperature": temperature,
                      "best_route": best_route.copy(), "current_route": route.copy(),
                      "reheats": reheats, "delta": delta, "probability": probability,
                      "uniform": uniform, "accepted": did_accept, "move": move})

    if config.trace_every:
        record(0, temperature)
    for step in range(1, config.evaluations + 1):
        i, j = sample_move(proposal_rng, len(route))
        # Always draw once, even for an improving move: paired runs share RNG indices.
        uniform = acceptance_rng.random()
        delta = two_opt_delta(route, problem.distances, i, j)
        used_temperature = temperature
        probability = acceptance_probability(delta, temperature)
        did_accept = uniform < probability
        stale += 1
        if did_accept:
            route[i:j + 1] = reversed(route[i:j + 1])
            current += delta
            accepted += 1
            accepted_worse += int(delta > 0)
            if accepted % 256 == 0:
                current = tour_length(route, problem.distances)
            if current < best - 1e-10:
                # Recompute the full objective before storing a new incumbent.
                current = tour_length(route, problem.distances)
                if current < best - 1e-10:
                    best, best_route = current, route.copy()
                    stale = 0
        temperature = max(t_min, temperature * alpha)
        is_periodic = config.mode == "periodic" and step % config.periodic_interval == 0
        is_stagnant = config.mode == "stagnation" and stale >= config.patience and step - last_reheat >= config.patience
        can_reheat = (reheats < config.max_reheats
                      and step <= int(config.evaluations * config.reheat_until_fraction)
                      and temperature <= t0 * config.reheat_gate_ratio)
        event = False
        if can_reheat and (is_periodic or is_stagnant):
            old_t = temperature
            temperature = min(t0, max(config.reheat_multiplier * temperature,
                                      config.reheat_floor_ratio * t0))
            if config.reheat_cooling == "remaining_budget":
                # Reheating must not remove the final cold phase. Reach the same
                # temperature floor at the same evaluation budget in all arms.
                alpha = (t_min / temperature) ** (1.0 / (config.evaluations - step))
            reheats += 1
            events.append({"evaluation": step, "before": old_t, "after": temperature,
                           "stale_evaluations": stale, "reason": config.mode, "cooling_alpha": alpha})
            last_reheat, stale = step, 0
            event = True
        if config.trace_every and (step % config.trace_every == 0 or step == config.evaluations or event):
            record(step, used_temperature, delta, probability, uniform, did_accept, (i, j))
    return {"mode": config.mode, "seed": seed, "config": asdict(config),
            "initial_route": initial_route, "initial_length": initial_length,
            "best_route": best_route, "best_length": tour_length(best_route, problem.distances),
            "current_route": route, "current_length": tour_length(route, problem.distances),
            "evaluations": config.evaluations, "calibration_evaluations": config.calibration_samples,
            "initial_temperature": t0, "alpha": initial_alpha, "final_alpha": alpha, "accepted": accepted,
            "accepted_worse": accepted_worse, "reheats": reheats,
            "elapsed_seconds": time.perf_counter() - started, "events": events, "trace": trace}


def exact_tsp(problem: Problem) -> tuple[list[int], float]:
    """Held–Karp oracle for small teaching instances, O(n^2 2^n) time."""
    n, distances = len(problem.points), problem.distances
    if n > 12:
        raise ValueError("精确求解仅用于至多 12 个点的微型验证。")
    dp: dict[tuple[int, int], tuple[float, int]] = {}
    for node in range(1, n):
        dp[1 << (node - 1), node] = (distances[0][node], 0)
    for mask in range(1, 1 << (n - 1)):
        for end in range(1, n):
            if not mask & (1 << (end - 1)):
                continue
            previous = mask ^ (1 << (end - 1))
            if not previous:
                continue
            dp[mask, end] = min((dp[previous, k][0] + distances[k][end], k)
                                for k in range(1, n) if previous & (1 << (k - 1)))
    mask = (1 << (n - 1)) - 1
    optimum, end = min((dp[mask, node][0] + distances[node][0], node) for node in range(1, n))
    reverse_route = []
    while end:
        reverse_route.append(end)
        parent = dp[mask, end][1]
        mask ^= 1 << (end - 1)
        end = parent
    return [0] + list(reversed(reverse_route)), optimum
