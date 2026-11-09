# 第三周：A* 地标路径搜索（代码包）

本项目在经典 A* / ALT 地标启发式基础上，比较角点、最远点和障碍邻域候选地标的选择方式。包含算法实现、独立正确性测试、交互演示和可复现实验。实验应同时考察查询耗时与地标预处理成本；障碍候选策略的收益需要在具体地图上验证。

## 1. 环境与启动

推荐 Python 3.12（本代码包在 Python 3.12.0 下验证）。解压后，在本文件所在目录打开终端。

算法、测试、自定义地图求解和 HTML 演示只使用 Python 标准库。实验绘图还需要 Matplotlib，运行实验前执行：

```powershell
python -m pip install -r requirements.txt
```

Windows 也可将下文的 `python run.py` 换成 `./run.cmd`。启动器优先使用上两级目录的 `.venv` 环境，找不到时使用 PATH 中的 `python`。

## 2. 快速使用

先运行正确性测试：

```powershell
python run.py test
```

生成交互演示：

```powershell
python run.py demo
```

完成后用浏览器打开 `outputs/demo/demo.html`。页面可切换地图和算法，查看逐步搜索过程；不需要启动服务器。未生成正式实验结果时，页面中的实验成本表为空，搜索演示仍可正常使用。

求解附带的绕墙地图：

```powershell
python run.py solve --map examples/detour.txt --start 0 0 --goal 4 0 --method alt_boundary --out outputs/detour_result.json
```

这张地图的最短路径长度应为 8。结果 JSON 包含路径坐标、最短路长度和搜索统计。

检查无法到达的情况：

```powershell
python run.py solve --map examples/disconnected.txt --start 0 0 --goal 4 0 --method alt_boundary --out outputs/disconnected_result.json
```

该地图的起终点不连通，结果中 `cost` 为 `null`，路径为空。

地图文件为等宽文本，每行代表一行网格：`.` 表示可通行格，`#` 表示障碍。坐标使用从 0 开始的 `(x, y)`，x 为列，y 为行；只允许上下左右移动，每步代价为 1。可修改示例地图，并调整起终点坐标。`solve` 的输出文件必须尚不存在；重复运行时请换一个 `--out` 文件名。

## 3. 五种算法参数

| `--method` 参数 | 含义 |
| --- | --- |
| `dijkstra` | Dijkstra，作为无启发式基线 |
| `manhattan` | 曼哈顿距离 A* |
| `alt_corners` | 角点地标 A* |
| `alt_farthest` | 最远点地标 A* |
| `alt_boundary` | 障碍邻域候选地标 A*，本周改进实验方案 |

地标方法可添加 `--landmarks 4` 调整每个连通分量的地标数量。启发函数结合曼哈顿距离和地标距离差下界，保持最短路正确性。

## 4. 对比实验与核验

快速实验（4 张地图、20 个查询、5 种方法，每项计时重复 3 次）：

```powershell
python run.py benchmark --config configs/quick.json --out outputs/quick
python run.py verify --results outputs/quick
```

正式实验（24 张地图、576 个查询，每项计时重复 3 次）：

```powershell
python run.py benchmark --config configs/validation.json --out outputs/validation
python run.py verify --results outputs/validation
python run.py demo
```

完成正式实验后，重新生成的演示页面会读取 `outputs/validation/summary.json` 并显示实验成本表。快速实验保存在 `outputs/quick` 时不会填充该表。

实验输出包括计时数据、配置与随机种子、地图、查询、路径、源码快照、汇总和比较图。`verify` 使用独立 BFS 检查路径与最优距离，并检查启发式下界、一致性、哈希和汇总结果。

`benchmark` 要求输出目录不存在或为空；重复实验请选择新的输出目录，并向 `verify --results` 传入同一目录。核验期间不要修改源码，否则源码哈希检查会失败。不同电脑上的耗时与性能排名可能不同；一次快速实验不能替代正式实验结论。

## 5. 文件导览

```text
03_A星地标路径搜索/
  README.md                  本运行说明
  run.py                     命令行统一入口
  run.cmd                    Windows 启动器
  pathfinding.py             网格、A* 搜索、地标构建与启发函数
  datasets.py                地图生成、加载与查询采样
  experiments.py             对比实验、计时、结果汇总与绘图
  visualization.py           交互演示和微型示例数据生成
  demo_template.html         交互演示页面模板
  requirements.txt           实验绘图依赖
  configs/
    quick.json               快速检查配置
    development.json         开发实验配置
    validation.json          正式实验配置
  examples/
    detour.txt               绕墙示例，最短路长度为 8
    disconnected.txt         不连通示例
  tests/
    test_algorithm.py        20 项正确性测试
  tools/
    verify_results.py        独立结果核验
  文件清单_SHA256.json        包内文件校验清单（不包含自身）
```

建议阅读顺序：先运行 `run.py`，再读 `pathfinding.py`，随后查看测试和实验模块。本包保留已验证源码的原始内容，并添加此说明、示例地图及校验清单；实验输出、Word 讲义和缓存不随包分发，运行相应命令后会自动生成 `outputs`。
