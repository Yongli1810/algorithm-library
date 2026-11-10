# K-means 聚类算法讲解与代码演示

## 演示主题
基于学习行为特征的学生群体聚类。数据为程序生成的模拟数据，不包含真实学生个人信息。

## 文件
- `K-means算法讲解与代码演示.docx`：按用户提供的《算法讲解与代码演示通用模板》填写。
- `kmeans_demo.py`：可独立运行的 Python 源代码。
- `kmeans_results/cluster_results.csv`：120 条样本的聚类标签。
- `kmeans_results/cluster_centers.csv`：各簇质心（原始量纲）。
- `kmeans_results/k_comparison.csv`：K=2、3、4 的对比指标。
- `kmeans_results/cluster_scatter.png`：聚类散点图。
- `kmeans_results/elbow_curve.png`：SSE 与 K 的关系图。

## 环境安装
```bash
pip install numpy matplotlib
```

## 运行
```bash
python kmeans_demo.py
```

## 本次实测结果
=== K-means 演示完成 ===
样本数：120，特征数：3
选定 K：3；迭代次数：5
最终 SSE：46.7934
轮廓系数：0.5705
各簇样本数（低/中/高）：[40, 38, 42]
质心（原始量纲：小时、%、分）：
  簇0：5.24 小时，38.98%，31.74 分
  簇1：10.77 小时，69.07%，60.41 分
  簇2：17.76 小时，90.73%，87.36 分

K 值比较：
  K=2: SSE=109.4492, Silhouette=0.5509, 簇大小=[51, 69]
  K=3: SSE=46.7934, Silhouette=0.5705, 簇大小=[40, 38, 42]
  K=4: SSE=41.5047, Silhouette=0.4527, 簇大小=[40, 38, 17, 25]

运行耗时：0.676 秒（随硬件和环境变化）
结果目录：/mnt/data/kmeans_results


说明：运行耗时受操作系统、CPU 和 Python 环境影响；重新运行时可能有轻微差异。
