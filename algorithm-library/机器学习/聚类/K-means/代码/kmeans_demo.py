# -*- coding: utf-8 -*-
"""
K-means 聚类算法教学演示
主题：基于学习行为特征的学生群体聚类（模拟数据）
依赖：numpy、matplotlib
运行：python kmeans_demo.py
"""

from pathlib import Path
import csv
import time
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUTPUT_DIR = Path("kmeans_results")
OUTPUT_DIR.mkdir(exist_ok=True)


def make_dataset(seed=42):
    """生成 120 条模拟学习行为数据：学习时长、作业完成率、课堂参与度。"""
    rng = np.random.default_rng(seed)
    centers = np.array([
        [18, 92, 88],   # 学习投入较高
        [11, 70, 62],   # 学习投入中等
        [5, 38, 32],    # 学习投入较低
    ], dtype=float)
    spreads = np.array([
        [3.0, 7.0, 8.0],
        [2.5, 8.0, 10.0],
        [2.0, 9.0, 9.0],
    ], dtype=float)

    groups = [
        rng.normal(center, spread, size=(40, 3))
        for center, spread in zip(centers, spreads)
    ]
    data = np.vstack(groups)
    data[:, 0] = np.clip(data[:, 0], 0, 30)
    data[:, 1:] = np.clip(data[:, 1:], 0, 100)
    return data


def standardize(data):
    """按列进行 Z-score 标准化，避免量纲影响欧氏距离。"""
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    std[std == 0] = 1.0
    return (data - mean) / std, mean, std


def init_kmeans_pp(data, k, rng):
    """K-means++ 初始化：后续中心按到已有中心的平方距离概率抽取。"""
    n_samples = len(data)
    centers = np.empty((k, data.shape[1]), dtype=float)

    first = rng.integers(n_samples)
    centers[0] = data[first]
    closest_sq = np.sum((data - centers[0]) ** 2, axis=1)

    for idx in range(1, k):
        total = closest_sq.sum()
        if total <= 0:
            chosen = rng.integers(n_samples)
        else:
            chosen = rng.choice(n_samples, p=closest_sq / total)
        centers[idx] = data[chosen]
        new_sq = np.sum((data - centers[idx]) ** 2, axis=1)
        closest_sq = np.minimum(closest_sq, new_sq)

    return centers


def run_kmeans(data, k, seed=42, max_iter=100, tol=1e-4):
    """执行一次 Lloyd 迭代，返回标签、质心、SSE、迭代次数和 SSE 历史。"""
    rng = np.random.default_rng(seed)
    centers = init_kmeans_pp(data, k, rng)
    history = []

    for iteration in range(max_iter):
        # 步骤一：计算每个样本到各质心的欧氏距离，并分配到最近质心
        distances = np.linalg.norm(
            data[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2
        )
        labels = np.argmin(distances, axis=1)

        # 步骤二：用簇内样本均值更新质心
        new_centers = centers.copy()
        for cluster_id in range(k):
            members = data[labels == cluster_id]
            if len(members) > 0:
                new_centers[cluster_id] = members.mean(axis=0)
            else:
                # 空簇处理：重新选取一个样本作为该簇中心
                new_centers[cluster_id] = data[rng.integers(len(data))]

        shift = np.linalg.norm(new_centers - centers)
        centers = new_centers

        # 记录本轮更新后的 SSE
        updated_distances = np.linalg.norm(
            data[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2
        )
        updated_labels = np.argmin(updated_distances, axis=1)
        sse = float(np.sum(
            (data - centers[updated_labels]) ** 2
        ))
        history.append(sse)

        if shift < tol:
            break

    final_distances = np.linalg.norm(
        data[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2
    )
    labels = np.argmin(final_distances, axis=1)
    sse = float(np.sum((data - centers[labels]) ** 2))
    return labels, centers, sse, iteration + 1, history


def best_of_restarts(data, k, n_init=10, random_state=42,
                     max_iter=100, tol=1e-4):
    """多次随机初始化，保留 SSE 最小的一次结果。"""
    best_result = None
    for run_id in range(n_init):
        result = run_kmeans(
            data, k, seed=random_state + run_id,
            max_iter=max_iter, tol=tol
        )
        if best_result is None or result[2] < best_result[2]:
            best_result = result
    return best_result


def silhouette_score(data, labels):
    """计算平均轮廓系数；值越接近 1，聚类分离程度通常越好。"""
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return 0.0

    distances = np.linalg.norm(
        data[:, np.newaxis, :] - data[np.newaxis, :, :], axis=2
    )
    scores = []

    for i in range(len(data)):
        same_cluster = labels == labels[i]
        same_cluster[i] = False

        if not np.any(same_cluster):
            scores.append(0.0)
            continue

        a = distances[i, same_cluster].mean()
        other_means = [
            distances[i, labels == other].mean()
            for other in unique_labels if other != labels[i]
        ]
        b = min(other_means)
        denominator = max(a, b)
        scores.append(0.0 if denominator == 0 else (b - a) / denominator)

    return float(np.mean(scores))


def sort_cluster_labels(labels, centers_original):
    """按学习时长质心从低到高排序，使类别编号便于讲解。"""
    order = np.argsort(centers_original[:, 0])
    mapping = {old: new for new, old in enumerate(order)}
    sorted_labels = np.array([mapping[int(label)] for label in labels])
    return sorted_labels, centers_original[order]


def save_cluster_csv(raw_data, labels, centers_original):
    names = ["需关注型", "中等型", "积极型"]
    path = OUTPUT_DIR / "cluster_results.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow([
            "student_id", "study_hours_per_week",
            "homework_completion_percent",
            "class_participation_score", "cluster_id", "cluster_name"
        ])
        for i, (row, label) in enumerate(zip(raw_data, labels), start=1):
            writer.writerow([
                f"S{i:03d}", *[round(float(v), 2) for v in row],
                int(label), names[int(label)]
            ])

    center_path = OUTPUT_DIR / "cluster_centers.csv"
    with center_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow([
            "cluster_id", "cluster_name", "study_hours_per_week",
            "homework_completion_percent", "class_participation_score"
        ])
        for i, center in enumerate(centers_original):
            writer.writerow([i, names[i], *[round(float(v), 2) for v in center]])


def save_metrics_csv(metrics):
    path = OUTPUT_DIR / "k_comparison.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["K", "SSE_inertia", "silhouette", "iterations", "cluster_sizes"])
        for row in metrics:
            writer.writerow([
                row["k"], round(row["sse"], 4),
                round(row["silhouette"], 4), row["iterations"],
                "/".join(map(str, row["cluster_sizes"]))
            ])


def plot_clusters(raw_data, labels, centers_original):
    names = ["需关注型", "中等型", "积极型"]
    plt.figure(figsize=(8, 5.2))
    for cluster_id in range(3):
        subset = raw_data[labels == cluster_id]
        plt.scatter(
            subset[:, 0], subset[:, 1],
            s=35, alpha=0.75, label=names[cluster_id]
        )
    plt.scatter(
        centers_original[:, 0], centers_original[:, 1],
        marker="X", s=180, edgecolors="black",
        linewidths=1.0, label="簇中心"
    )
    plt.xlabel("每周自主学习时长（小时）")
    plt.ylabel("作业完成率（%）")
    plt.title("K-means 学习行为聚类结果（模拟数据）")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "cluster_scatter.png", dpi=180)
    plt.close()


def plot_elbow(metrics):
    ks = [row["k"] for row in metrics]
    sses = [row["sse"] for row in metrics]
    plt.figure(figsize=(7.2, 4.6))
    plt.plot(ks, sses, marker="o", linewidth=2)
    plt.xticks(ks)
    plt.xlabel("聚类数量 K")
    plt.ylabel("SSE（簇内平方误差和）")
    plt.title("不同 K 值下的 SSE 变化")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "elbow_curve.png", dpi=180)
    plt.close()


def main():
    start = time.perf_counter()
    raw_data = make_dataset(seed=42)
    scaled_data, mean, std = standardize(raw_data)

    # 比较 K=2、3、4；每个 K 使用 10 次初始化
    metrics = []
    for k in (2, 3, 4):
        labels, centers, sse, iterations, history = best_of_restarts(
            scaled_data, k, n_init=10, random_state=42
        )
        centers_original = centers * std + mean
        order = np.argsort(centers_original[:, 0])
        sorted_sizes = np.bincount(labels, minlength=k)[order].tolist()
        metrics.append({
            "k": k,
            "sse": sse,
            "silhouette": silhouette_score(scaled_data, labels),
            "iterations": iterations,
            "cluster_sizes": sorted_sizes
        })

    # 本示例选择 K=3，输出可解释的三类学习行为
    labels, centers, sse, iterations, history = best_of_restarts(
        scaled_data, 3, n_init=10, random_state=42
    )
    centers_original = centers * std + mean
    labels, centers_original = sort_cluster_labels(labels, centers_original)

    save_cluster_csv(raw_data, labels, centers_original)
    save_metrics_csv(metrics)
    plot_clusters(raw_data, labels, centers_original)
    plot_elbow(metrics)

    elapsed = time.perf_counter() - start
    counts = np.bincount(labels, minlength=3)

    print("=== K-means 演示完成 ===")
    print(f"样本数：{len(raw_data)}，特征数：{raw_data.shape[1]}")
    print(f"选定 K：3；迭代次数：{iterations}")
    print(f"最终 SSE：{sse:.4f}")
    print(f"轮廓系数：{silhouette_score(scaled_data, labels):.4f}")
    print(f"各簇样本数（低/中/高）：{counts.tolist()}")
    print("质心（原始量纲：小时、%、分）：")
    for idx, center in enumerate(centers_original):
        print(f"  簇{idx}：{center[0]:.2f} 小时，{center[1]:.2f}%，{center[2]:.2f} 分")
    print("\nK 值比较：")
    for row in metrics:
        print(
            f"  K={row['k']}: SSE={row['sse']:.4f}, "
            f"Silhouette={row['silhouette']:.4f}, "
            f"簇大小={row['cluster_sizes']}"
        )
    print(f"\n运行耗时：{elapsed:.3f} 秒（随硬件和环境变化）")
    print(f"结果目录：{OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
