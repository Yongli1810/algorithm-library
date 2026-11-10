# -*- coding: utf-8 -*-
"""0/1 背包动态规划教学示例：应急物资装载（全部为模拟数据）
运行: python knapsack_demo.py
可选绘图依赖: pip install matplotlib
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import csv
import json
import time

OUTPUT = Path(__file__).resolve().parent / "results"
OUTPUT.mkdir(parents=True, exist_ok=True)

@dataclass(frozen=True)
class Item:
    name: str
    weight: int
    value: int

ITEMS = [
    Item("简易药包", 2, 3),
    Item("饮用水", 3, 4),
    Item("应急电池", 4, 5),
    Item("急救箱", 5, 8),
    Item("照明设备", 6, 9),
    Item("通信设备", 7, 10),
]
MICRO_ITEMS = ITEMS[:4]


def dp_2d(items: list[Item], capacity: int):
    """标准二维 DP + 路径回溯；输出最优价值、选中物品索引、完整 DP 表。"""
    if capacity < 0 or any(item.weight <= 0 or item.value < 0 for item in items):
        raise ValueError("容量必须非负，物品重量必须为正、价值必须非负")
    n = len(items)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    actions = [[False] * (capacity + 1) for _ in range(n + 1)]
    for i, item in enumerate(items, start=1):
        for c in range(capacity + 1):
            skip = dp[i - 1][c]
            take = (dp[i - 1][c - item.weight] + item.value
                    if c >= item.weight else -1)
            if take > skip:
                dp[i][c] = take
                actions[i][c] = True
            else:
                dp[i][c] = skip
    chosen = []
    c = capacity
    for i in range(n, 0, -1):
        if actions[i][c]:
            chosen.append(i - 1)
            c -= items[i - 1].weight
    chosen.reverse()
    return dp[n][capacity], chosen, dp


def dp_1d(items: list[Item], capacity: int):
    """空间优化：逆序遍历容量，防止同一件物品被重复使用。"""
    f = [0] * (capacity + 1)
    for item in items:
        for c in range(capacity, item.weight - 1, -1):
            f[c] = max(f[c], f[c - item.weight] + item.value)
    return f[capacity]


def greedy_ratio(items: list[Item], capacity: int):
    """按照价值/重量贪心选取，作为对照；不保证最优。"""
    order = sorted(range(len(items)), key=lambda i: items[i].value / items[i].weight,
                   reverse=True)
    chosen = []
    used = 0
    for i in order:
        if used + items[i].weight <= capacity:
            chosen.append(i)
            used += items[i].weight
    chosen.sort()
    return sum(items[i].value for i in chosen), chosen


def exhaustive(items: list[Item], capacity: int):
    """只用于教学小样本的穷举校验，复杂度 O(2^n)。"""
    best_value, best_indices = 0, []
    for mask in range(1 << len(items)):
        ids = [i for i in range(len(items)) if mask & (1 << i)]
        weight = sum(items[i].weight for i in ids)
        value = sum(items[i].value for i in ids)
        if weight <= capacity and value > best_value:
            best_value, best_indices = value, ids
    return best_value, best_indices


def export_table(dp, name):
    with (OUTPUT / name).open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["已处理物品数 / 容量", *range(len(dp[0]))])
        for i, row in enumerate(dp):
            writer.writerow([i, *row])


def make_plot(rows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        cs = [r["capacity"] for r in rows]
        optimal = [r["dp_optimal"] for r in rows]
        greedy = [r["greedy_ratio"] for r in rows]
        fig, ax = plt.subplots(figsize=(8.0, 4.5))
        x = range(len(cs))
        ax.plot(x, optimal, "o-", lw=2.5, label="DP 最优值")
        ax.plot(x, greedy, "s--", lw=2.0, label="价值密度贪心")
        ax.set_xticks(list(x), cs)
        ax.set_xlabel("背包容量 / kg")
        ax.set_ylabel("累计效用值 / 分")
        ax.set_title("容量变化下的最优结果与贪心对比")
        ax.grid(alpha=.22)
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUTPUT / "comparison.png", dpi=180)
        plt.close(fig)
    except ImportError:
        print("提示：未安装 matplotlib，跳过绘图；CSV 与算法不受影响。")


def main():
    capacities = [8, 10, 12, 14, 16]
    rows = []
    for cap in capacities:
        tic = time.perf_counter()
        opt, chosen, table = dp_2d(ITEMS, cap)
        elapsed_ms = (time.perf_counter() - tic) * 1000
        small_mem = dp_1d(ITEMS, cap)
        greedy, greedy_indices = greedy_ratio(ITEMS, cap)
        brute, brute_indices = exhaustive(ITEMS, cap)
        assert opt == small_mem == brute, "DP 与穷举、1D DP 的结果不一致"
        rows.append({
            "capacity": cap,
            "dp_optimal": opt,
            "dp_1d": small_mem,
            "greedy_ratio": greedy,
            "greedy_gap": opt - greedy,
            "brute_force": brute,
            "dp_states": len(ITEMS) * (cap + 1),
            "brute_subsets": 1 << len(ITEMS),
            "dp_runtime_ms": round(elapsed_ms, 4),
            "chosen_items": "/".join(ITEMS[i].name for i in chosen),
            "chosen_weight": sum(ITEMS[i].weight for i in chosen),
            "greedy_items": "/".join(ITEMS[i].name for i in greedy_indices),
        })
        if cap == 12:
            export_table(table, "dp_table_capacity12.csv")
    micro_value, micro_ids, micro_table = dp_2d(MICRO_ITEMS, 8)
    export_table(micro_table, "dp_table_micro.csv")
    assert micro_value == 12 and micro_ids == [1, 3]
    with (OUTPUT / "comparison.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (OUTPUT / "summary.json").write_text(
        json.dumps({"algorithm": "0/1 背包动态规划", "data": "模拟数据", "items": [item.__dict__ for item in ITEMS],
                    "micro_optimal": micro_value, "micro_selected_indices": micro_ids,
                    "experiments": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    make_plot(rows)
    print("0/1 背包动态规划教学演示（模拟数据）")
    print("容量  DP最优  贪心  差额  选取物品")
    for row in rows:
        print(f"{row['capacity']:>4}  {row['dp_optimal']:>5}  {row['greedy_ratio']:>4}  "
              f"{row['greedy_gap']:>4}  {row['chosen_items']}")
    print("微型案例：容量8，最优价值12，选择饮用水+急救箱。")
    print("6种物品，容量12：最优价值18，选择急救箱+通信设备。")
    print("数据已导出到 results/ 目录。")


if __name__ == "__main__":
    main()
