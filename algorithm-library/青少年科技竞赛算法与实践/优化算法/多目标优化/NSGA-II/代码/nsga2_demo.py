#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""可复现的 NSGA-II 多目标优化演示：ZDT1、随机采样与无变异消融。
依赖：numpy、matplotlib；使用示例见文档。
"""
from __future__ import annotations
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np


def zdt1(X: np.ndarray) -> np.ndarray:
    """二维目标最小化；决策变量维数至少为2，边界均为[0, 1]。"""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] < 2:
        raise ValueError("X 的形状必须为 (样本数, 决策变量维数>=2)")
    if not np.all(np.isfinite(X)) or np.any((X < 0) | (X > 1)):
        raise ValueError("所有决策变量必须是有限数值，并处于 [0, 1]")
    f1 = X[:, 0]
    g = 1.0 + 9.0 * np.mean(X[:, 1:], axis=1)
    f2 = g * (1.0 - np.sqrt(f1 / g))
    return np.column_stack((f1, f2))


def nondominated_sort(F: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
    """两两非支配排序，耗时 O(m N^2)、额外空间 O(N^2)。"""
    F = np.asarray(F, dtype=float)
    n = len(F)
    if n == 0:
        return [], np.empty(0, dtype=int)
    dominates = np.all(F[:, None, :] <= F[None, :, :], axis=2) & np.any(
        F[:, None, :] < F[None, :, :], axis=2)
    dominated_count = np.sum(dominates, axis=0).astype(int)
    ranks = np.full(n, -1, dtype=int)
    front = np.flatnonzero(dominated_count == 0)
    fronts: list[np.ndarray] = []
    k = 0
    while len(front):
        fronts.append(front)
        ranks[front] = k
        dominated_count -= np.sum(dominates[front, :], axis=0).astype(int)
        front = np.flatnonzero((dominated_count == 0) & (ranks == -1))
        k += 1
    return fronts, ranks


def crowding_distance(F: np.ndarray, inds: np.ndarray) -> np.ndarray:
    """返回和 inds 同序的拥挤距离；目标的两个边界解记为无穷大。"""
    inds = np.asarray(inds, dtype=int)
    n = len(inds)
    d = np.zeros(n, dtype=float)
    if n <= 2:
        d[:] = np.inf
        return d
    sub = F[inds]
    for j in range(sub.shape[1]):
        order = np.argsort(sub[:, j], kind="mergesort")
        d[order[0]] = np.inf
        d[order[-1]] = np.inf
        spread = sub[order[-1], j] - sub[order[0], j]
        if spread > 0:
            for t in range(1, n - 1):
                d[order[t]] += (sub[order[t + 1], j] - sub[order[t - 1], j]) / spread
    return d


def all_crowding(F: np.ndarray, fronts: list[np.ndarray]) -> np.ndarray:
    out = np.zeros(len(F), dtype=float)
    for inds in fronts:
        out[inds] = crowding_distance(F, inds)
    return out


def tournament(ranks: np.ndarray, distances: np.ndarray, rng: np.random.Generator) -> int:
    i, j = rng.integers(0, len(ranks), size=2)
    if ranks[i] != ranks[j]:
        return int(i if ranks[i] < ranks[j] else j)
    if distances[i] != distances[j]:
        return int(i if distances[i] > distances[j] else j)
    return int(i if rng.random() < 0.5 else j)


def sbx(a: np.ndarray, b: np.ndarray, rng: np.random.Generator,
        pc: float, eta_c: float) -> tuple[np.ndarray, np.ndarray]:
    """模拟二进制交叉，边界外截断到[0,1]。"""
    if rng.random() >= pc:
        return a.copy(), b.copy()
    u = rng.random(a.size)
    beta = np.where(u <= 0.5, (2 * u) ** (1.0 / (eta_c + 1)),
                    (0.5 / np.maximum(1 - u, 1e-15)) ** (1.0 / (eta_c + 1)))
    c1 = 0.5 * ((1 + beta) * a + (1 - beta) * b)
    c2 = 0.5 * ((1 - beta) * a + (1 + beta) * b)
    return np.clip(c1, 0, 1), np.clip(c2, 0, 1)


def polynomial_mutation(x: np.ndarray, rng: np.random.Generator,
                        pm: float, eta_m: float) -> np.ndarray:
    """Deb 有界多项式变异，支持关闭变异 pm=0。"""
    y = x.copy()
    for j in range(len(y)):
        if rng.random() >= pm:
            continue
        r = rng.random()
        if r < 0.5:
            val = 2 * r + (1 - 2 * r) * (1 - y[j]) ** (eta_m + 1)
            delta = val ** (1.0 / (eta_m + 1)) - 1
        else:
            val = 2 * (1 - r) + 2 * (r - 0.5) * y[j] ** (eta_m + 1)
            delta = 1 - val ** (1.0 / (eta_m + 1))
        y[j] = np.clip(y[j] + delta, 0, 1)
    return y


def generate_offspring(X: np.ndarray, F: np.ndarray, rng: np.random.Generator,
                       pc: float, pm: float, eta_c: float, eta_m: float) -> np.ndarray:
    n, d = X.shape
    fronts, ranks = nondominated_sort(F)
    distances = all_crowding(F, fronts)
    offspring = np.empty((n, d), dtype=float)
    for k in range(0, n, 2):
        i = tournament(ranks, distances, rng)
        j = tournament(ranks, distances, rng)
        c1, c2 = sbx(X[i], X[j], rng, pc, eta_c)
        offspring[k] = polynomial_mutation(c1, rng, pm, eta_m)
        if k + 1 < n:
            offspring[k + 1] = polynomial_mutation(c2, rng, pm, eta_m)
    return offspring


def survival(X: np.ndarray, F: np.ndarray, pop_size: int) -> tuple[np.ndarray, np.ndarray]:
    fronts, _ = nondominated_sort(F)
    selected: list[int] = []
    for ids in fronts:
        remaining = pop_size - len(selected)
        if remaining <= 0:
            break
        if len(ids) <= remaining:
            selected.extend(ids.tolist())
        else:
            distances = crowding_distance(F, ids)
            order = np.argsort(-distances, kind="mergesort")[:remaining]
            selected.extend(ids[order].tolist())
    ids = np.asarray(selected, dtype=int)
    return X[ids].copy(), F[ids].copy()


def get_front(F: np.ndarray) -> np.ndarray:
    return nondominated_sort(F)[0][0]


def hypervolume_2d(F: np.ndarray, ref: tuple[float, float] = (1.1, 1.1)) -> float:
    """二维最小化的精确超体积；裁去参考点以外的候选点。"""
    F = np.asarray(F, dtype=float)
    valid = (F[:, 0] < ref[0]) & (F[:, 1] < ref[1])
    P = F[valid]
    if len(P) == 0:
        return 0.0
    P = P[get_front(P)]
    P = P[np.argsort(P[:, 0], kind="mergesort")]
    last_y = float(ref[1])
    area = 0.0
    for x, y in P:
        if y < last_y:
            area += (ref[0] - float(x)) * (last_y - float(y))
            last_y = float(y)
    return float(area)


def igd(F: np.ndarray, n_reference: int = 201) -> float:
    """IGD = 真实 Pareto 前沿各点到近似解集的最近欧氏距离的平均。"""
    u = np.linspace(0.0, 1.0, n_reference)
    actual = np.column_stack((u, 1 - np.sqrt(u)))
    F = np.asarray(F, dtype=float)
    if len(F) == 0:
        return float("inf")
    min_dist = np.sqrt(np.sum((actual[:, None, :] - F[None, :, :]) ** 2, axis=2)).min(axis=1)
    return float(min_dist.mean())


def nsga2(pop: int, generations: int, dims: int, seed: int,
          mutation: bool = True, pc: float = 0.9, eta_c: float = 15,
          eta_m: float = 20) -> dict:
    rng = np.random.default_rng(seed)
    X = rng.random((pop, dims))
    F = zdt1(X)
    pm = 1.0 / dims if mutation else 0.0
    history = [{"generation": 0, "hv": hypervolume_2d(F), "igd": igd(F)}]
    for g in range(1, generations + 1):
        C = generate_offspring(X, F, rng, pc, pm, eta_c, eta_m)
        Fc = zdt1(C)
        X, F = survival(np.vstack((X, C)), np.vstack((F, Fc)), pop)
        history.append({"generation": g, "hv": hypervolume_2d(F), "igd": igd(F)})
    front = get_front(F)
    return {"X": X, "F": F, "pareto": F[front], "history": history,
            "eval_count": (generations + 1) * pop}


def front2d(F: np.ndarray) -> np.ndarray:
    """仅限二维、连续随机样本的快速 Pareto 提取；按 f1、f2 递增扫描。"""
    order = np.lexsort((F[:, 1], F[:, 0]))
    keep = []
    best_f2 = float("inf")
    for idx in order:
        if F[idx, 1] < best_f2:
            keep.append(int(idx))
            best_f2 = float(F[idx, 1])
    return np.asarray(keep, dtype=int)


def random_baseline(pop: int, generations: int, dims: int, seed: int) -> dict:
    # 和 NSGA-II 相同的目标函数评价次数；避免假设同一耗时。
    rng = np.random.default_rng(seed)
    count = (generations + 1) * pop
    X = rng.random((count, dims))
    F = zdt1(X)
    front = front2d(F)
    P = F[front]
    return {"pareto": P, "hv": hypervolume_2d(P), "igd": igd(P), "eval_count": count}


def plot_and_export(results: list[dict], showcase: dict[str, dict], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out.mkdir(parents=True, exist_ok=True)
    with (out / "experiments.csv").open("w", newline="", encoding="utf-8-sig") as f:
        cols = ["method", "seed", "hv", "igd", "seconds", "eval_count", "front_count"]
        wr = csv.DictWriter(f, fieldnames=cols)
        wr.writeheader()
        wr.writerows(results)
    summary = {}
    for method in ("NSGA-II", "No-mutation", "Random"):
        r = [e for e in results if e["method"] == method]
        summary[method] = {
            "hv_mean": float(np.mean([v["hv"] for v in r])),
            "hv_std": float(np.std([v["hv"] for v in r], ddof=1)) if len(r) > 1 else 0.0,
            "igd_mean": float(np.mean([v["igd"] for v in r])),
            "igd_std": float(np.std([v["igd"] for v in r], ddof=1)) if len(r) > 1 else 0.0,
            "seconds_mean": float(np.mean([v["seconds"] for v in r]))
        }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    u = np.linspace(0, 1, 400)
    fig, ax = plt.subplots(figsize=(8.2, 5.7), dpi=145)
    ax.plot(u, 1 - np.sqrt(u), "k--", lw=2, label="True Pareto front")
    for name, style in (("NSGA-II", "o"), ("No-mutation", "x"), ("Random", "+")):
        P = showcase[name]["pareto"]
        ax.scatter(P[:, 0], P[:, 1], s=23, marker=style, alpha=.76, label=f"{name} (n={len(P)})")
    ax.set(xlabel="f1 (min)", ylabel="f2 (min)", title="ZDT1: approximated Pareto front")
    ax.set_xlim(-.04, 1.14)
    ax.set_ylim(-.04, 2.0)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(out / "pareto_front.png", bbox_inches="tight")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8.2, 4.3), dpi=145)
    for name in ("NSGA-II", "No-mutation"):
        H = showcase[name]["history"]
        ax.plot([v["generation"] for v in H], [v["igd"] for v in H], label=name, lw=1.9)
    ax.set(xlabel="Generation", ylabel="IGD (lower is better)", title="Convergence of a representative run")
    ax.legend()
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(out / "igd_history.png", bbox_inches="tight")
    plt.close(fig)
    with (out / "pareto_nsga2.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["f1", "f2"])
        w.writerows(showcase["NSGA-II"]["pareto"].tolist())
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="NSGA-II 在 ZDT1 上的多目标优化演示")
    parser.add_argument("--pop-size", type=int, default=64)
    parser.add_argument("--generations", type=int, default=60)
    parser.add_argument("--dims", type=int, default=12)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("nsga2_results"))
    args = parser.parse_args()
    if args.pop_size < 4 or args.generations < 1 or args.dims < 2 or args.runs < 1:
        parser.error("要求 pop-size>=4、generations>=1、dims>=2、runs>=1")
    results, showcase = [], {}
    for offset in range(args.runs):
        seed = args.seed + offset
        for method in ("NSGA-II", "No-mutation", "Random"):
            start = time.perf_counter()
            if method == "Random":
                r = random_baseline(args.pop_size, args.generations, args.dims, seed)
                hv, v_igd = r["hv"], r["igd"]
            else:
                r = nsga2(args.pop_size, args.generations, args.dims, seed,
                          mutation=(method == "NSGA-II"))
                hv, v_igd = hypervolume_2d(r["pareto"]), igd(r["pareto"])
            seconds = time.perf_counter() - start
            results.append({"method": method, "seed": seed, "hv": round(hv, 8),
                            "igd": round(v_igd, 8), "seconds": round(seconds, 6),
                            "eval_count": r["eval_count"], "front_count": len(r["pareto"])})
            if offset == 0:
                showcase[method] = r
            print(f"{method:12s} seed={seed}  HV={hv:.5f}  IGD={v_igd:.5f}  "
                  f"eval={r['eval_count']}  seconds={seconds:.2f}")
    summary = plot_and_export(results, showcase, args.out)
    print("\nSummary (mean ± sample SD, across independent seeds):")
    for method, r in summary.items():
        print(f"{method:12s}  HV={r['hv_mean']:.5f} ± {r['hv_std']:.5f}, "
              f"IGD={r['igd_mean']:.5f} ± {r['igd_std']:.5f}, "
              f"time={r['seconds_mean']:.2f}s")
    print(f"Files written to: {args.out.resolve()}")

if __name__ == "__main__":
    main()
