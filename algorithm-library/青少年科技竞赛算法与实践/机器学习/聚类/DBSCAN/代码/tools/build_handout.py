"""Fill the retained four-page teaching layout with verified DBSCAN results.

Run with the bundled document runtime (lxml), not the algorithm environment.
The preceding week's reference is read-only; all opaque package parts survive.
"""
import argparse
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean
from zipfile import ZipFile
from lxml import etree as E

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--reference", type=Path, default=ROOT.parent / "01_模拟退火配送路线优化/docs/模拟退火算法讲解与代码演示.docx")
parser.add_argument("--output", type=Path, default=ROOT / "docs/DBSCAN算法讲解与代码演示.docx")
args = parser.parse_args()
expected_hash = "c0b2eee843b70386a49c106df203381e6d23bb352af50014835097a8282336b9"
assert hashlib.sha256(args.reference.read_bytes()).hexdigest() == expected_hash, "Reference changed; re-inspect its layout."
load = lambda p: json.loads(p.read_text(encoding="utf-8"))
summary = load(ROOT / "outputs/validation/summary.json")
audit = load(ROOT / "outputs/validation/audit.json")
demo = load(ROOT / "outputs/demo/demo.json")["cases"][0]
overall = {r["method"]: r for r in summary["overall"]}
with (ROOT / "outputs/validation/memory.csv").open(encoding="utf-8", newline="") as f:
    memory = list(csv.DictReader(f))
dense_memory = {r["method"]: int(r["sampled_extra_rss_bytes"]) / 2**20 for r in memory if r["family"] == "dense"}
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
M = "http://schemas.openxmlformats.org/officeDocument/2006/math"
ns = {"w": W, "m": M}
q = lambda tag: "{" + W + "}" + tag
with ZipFile(args.reference) as archive:
    infos = archive.infolist()
    original = {i.filename: archive.read(i.filename) for i in infos}
parts = dict(original)
root = E.fromstring(parts["word/document.xml"])
body = root.find("w:body", ns)
paragraphs, tables = body.findall("w:p", ns), body.findall("w:tbl", ns)
geometry = E.tostring(body.find("w:sectPr", ns))
grids = [E.tostring(t.find("w:tblGrid", ns)) for t in tables]


def set_paragraph(p, text):
    candidates = p.findall("w:r", ns)
    representative = max(candidates, key=lambda r: len("".join(r.xpath(".//w:t/text()", namespaces=ns))), default=None)
    props = representative.find("w:rPr", ns) if representative is not None else None
    saved = deepcopy(props) if props is not None else None
    for child in list(p):
        if child.tag != q("pPr"):
            p.remove(child)
    for i, line in enumerate(text.split("\n")):
        r = E.SubElement(p, q("r"))
        if saved is not None:
            r.append(deepcopy(saved))
        if i:
            E.SubElement(r, q("br"))
        t = E.SubElement(r, q("t"))
        t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
        t.text = line


def cell(t, row, col):
    return tables[t].findall("w:tr", ns)[row].findall("w:tc", ns)[col]


def put(t, row, col, text):
    c = cell(t, row, col)
    ps = c.findall("w:p", ns)
    set_paragraph(ps[0], text)
    for p in ps[1:]:
        c.remove(p)


def fill(t, rows):
    assert len(rows) == len(tables[t].findall("w:tr", ns)) - 1
    for i, values in enumerate(rows, 1):
        if isinstance(values, str):
            put(t, i, 1, values)
        else:
            for j, value in enumerate(values):
                put(t, i, j, value)


def formulas(t, row, lines):
    c = cell(t, row, 1)
    old = c.find("w:p", ns)
    props = deepcopy(old.find("w:pPr", ns))
    for p in c.findall("w:p", ns):
        c.remove(p)
    for line in lines:
        p = E.SubElement(c, q("p"))
        if props is not None:
            p.append(deepcopy(props))
        math = E.SubElement(p, "{" + M + "}oMath")
        run = E.SubElement(math, "{" + M + "}r")
        rp = E.SubElement(run, q("rPr"))
        E.SubElement(rp, q("rFonts"), {q("ascii"): "Cambria Math", q("hAnsi"): "Cambria Math"})
        E.SubElement(rp, q("sz"), {q("val"): "17"})
        E.SubElement(run, "{" + M + "}t").text = line


set_paragraph(paragraphs[0], "DBSCAN 网格与几何剪枝")
set_paragraph(paragraphs[1], "算法名称：网格与包围盒加速 DBSCAN        算法类别：密度聚类与空间检索")
set_paragraph(paragraphs[2], "讲解对象：具备 Python 基础的学习者          讲解时长：建议 45 分钟")
fill(0, ["DBSCAN with Grid Index and Bounding Box Pruning；项目简称 DBSCAN-GP。",
         "在经典 DBSCAN 上实现精确邻域加速；借鉴已有空间索引研究，非全新聚类方法。",
         "解释密度可达与剪枝；核对十点案例；复现四组对照；判断减少计算是否真正提速。",
         "二维距离、集合、队列与 Python；核心依赖 NumPy，成熟对照使用 scikit-learn。"])
fill(1, ["二维空间点集聚类；寻找密集区域并标出噪声，可用于模拟配送点的聚集分析。",
         "邻域查询常重复检查远处点；检验网格及几何剪枝是否在保持结果一致时降低耗时。",
         "CSV 含 id,x,y，编号从 0 连续；有限二维坐标。正式规模为 1000、5000、10000 点。",
         "每点簇标签、核心点标记、噪声、距离次数、端到端耗时及扩展过程。",
         "二维欧氏距离；相同 ε 与 min_samples；包含查询点自身；统一原始编号访问顺序。"])
fill(2, ["先缩小候选网格，再用格内点集包围盒排除或接纳整格；临界相交时逐点判断。",
         "十点案例，ε=0.2、min_samples=4：0–4 为簇0，5–8 为簇1，9 为噪声；4 为边界点。",
         "校验数据 → 建索引 → 查邻居 → 判定核心点 → 队列扩展 → 输出标签和指标。",
         "邻居达到阈值才继续扩展；包围盒完全在圆内可整格接纳，完全在圆外可排除。"])
fill(3, ["DBSCAN 的核心点连通结构；网格的局部性与点到矩形距离上下界。",
         "", "核心点扩展尚未入队的邻居；噪声可改为边界点；边界点不继续扩展。",
         "包围盒包含格内全部点，因此上下界可用于精确剪枝；不改变邻域与密度定义。"])
formulas(3, 2, ["Nε(p) = {q : d(p,q) ≤ ε} ;  core(p) ⇔ |Nε(p)| ≥ m",
                "g(p) = (⌊x/ε⌋, ⌊y/ε⌋) ;  dmin(p,B) ≤ d(p,q) ≤ dmax(p,B)",
                "dmin > ε ⇒ 排除 ;  dmax ≤ ε ⇒ 接纳 ;  其余逐点判断"])
fill(4, [("p、q、B", "查询点、候选点、点集包围盒", "距离上下界以二维欧氏距离计算，单位与输入坐标相同"),
         ("ε、m、Nε", "邻域半径、最小点数、邻居集合", "m 包含自身；所有版本共享参数与邻居判定"),
         ("n、kᵢ、cᵢ", "总点数、第 i 次邻居数与候选数", "解释距离计算、邻居输出和排序的成本")])
fill(5, [("数据准备", "读取 CSV，校验有限坐标与编号", "id,x,y → 二维数组"),
         ("初始化", "按边长 ε 分格；剪枝版计算包围盒", "网格键 → 点编号与矩形边界"),
         ("核心计算", "筛选网格、判矩形、计算必要距离", "返回按原始编号排序的精确邻居"),
         ("状态更新", "核心点通过队列扩展；更新边界点归属", "每点至多查询一次；队列防重复"),
         ("终止判断", "当前簇队列为空，寻找下一个未访问点", "所有点均查询后结束"),
         ("结果输出", "保存标签、核心点、计数及时间", "JSON、CSV、比较图与离线演示")])
set_paragraph(paragraphs[15], "建立网格与包围盒，全部点设为未访问\n依编号选择未访问点 p，精确查询 Nε(p)\n    若邻居不足 m：暂标噪声，继续下一点\n    否则创建新簇，将邻居放入队列\n    逐点出队：核心点继续扩展，更新未归属点\n返回全部标签、核心点与计算指标")
fill(6, [("ε", "确定邻域及网格边长", "正式四族 0.055–0.20", "按数据族预先固定；越大通常邻居越多"),
         ("min_samples", "核心点最小邻居数", "正式实验 8", "包含自身；过大可能增加噪声"),
         ("method", "邻域查询策略", "brute / grid / pruned", "只改查询；距离核与聚类扩展共用"),
         ("规模 / 重复", "输入规模与计时次数", "1000/5000/10000；3次", "每族每规模2实例；开发与验证种子分离")])
fill(7, ["Windows CPU；Python 3.12.0；NumPy 2.4.4；scikit-learn 1.9.0。",
         "dbscan.py 核心；datasets.py 数据；experiments.py 对照；visualization.py 演示。",
         "在本算法目录执行 run.cmd test；run.cmd demo；run.cmd benchmark --out outputs/recheck。",
         "拒绝非有限坐标、非正 ε 和非法 min_samples；网格负坐标向下取整。",
         "datasets.py 的 PARAMETERS；configs/validation.json 固定规模、种子和重复次数。",
         "RangeIndex.query 对应邻居；box_action 对应矩形判断；fit 对应密度扩展。",
         "每点查询一次后结束；返回 Result；计时不记录演示帧，也不预存完整邻接矩阵。",
         "结果一致性、距离次数、建索引与总耗时；内存另以独立解释器进程采样。"])
fill(8, [("dbscan.py / query", "候选网格与几何剪枝", "点编号 → 精确邻居及计数"),
         ("dbscan.py / fit", "密度聚类与队列扩展", "点集、参数、查询方法 → 标签和核心点"),
         ("experiments.py / benchmark", "冻结输入与随机顺序配对计时", "配置 → 原始记录、哈希、分组报告")])
fill(9, ["默认双月形260点；ε=0.14、min_samples=6；另含团块、均匀、密集与十点案例。",
         "打开 outputs/demo/demo.html；选择查询点、切换方法、播放或拖动时间轴。",
         "精确邻居数、整格排除/接纳、点对距离次数；留意非核心点不会继续扩展。",
         "run.cmd micro 输出手算核对；run.cmd demo 生成自包含页面，无网络依赖。"])
fill(10, [("初始状态", "260点未分配；点0有7个邻居", "三种查询返回相同集合；查询包含点自身"),
          ("中间状态", "核心点扩展队列，逐步形成密集区域", "边界点可加入已有簇，不继续扩展"),
          ("结束状态", f"{demo['clusters']}个簇、{demo['noise']}个噪声点；260点均访问", "演示参数单独设置，不参与正式计时")])
fill(11, ["4分布×3规模×2实例=24份独立数据；每种实现重复3次，共288次计时。",
          "全量扫描、仅网格、网格加剪枝；成熟参照固定 sklearn KDTree、单工作线程。",
          "分区/核心点/噪声一致性；距离次数；含建索引的总时间；独立进程采样RSS。",
          "先预热，每轮随机方法顺序；实例内取中位数，再对配对加速比取几何平均。"])
rows = []
for mode, title, config in [("brute", "全量扫描", "全部点作候选"), ("grid", "仅网格", "边长 ε；精确距离"), ("pruned", "网格加剪枝", "点集包围盒上下界")]:
    row = overall[mode]
    seconds = mean(r["median_seconds"] for r in summary["cases"] if r["method"] == mode)
    rows.append((title, config, f"相对扫描\n{row['geomean_speedup_vs_brute']:.3f}倍", f"中位数均值\n{seconds:.3f}秒", f"结果全部一致\n{row['faster_than_brute_cases']}/24实例更快" if mode != "brute" else "比较基准"))
fill(12, rows)
set_paragraph(paragraphs[27], f"分析结论：剪枝相对全量扫描约{overall['pruned']['geomean_speedup_vs_brute']:.2f}倍，相对仅网格约{overall['pruned']['geomean_speedup_vs_grid']:.3f}倍，额外收益依赖分布。"
              f"KDTree相对扫描约{overall['sklearn_kd_tree']['geomean_speedup_vs_brute']:.2f}倍。小样本与部分分布中剪枝更慢，不作普遍提速承诺。")
fill(13, ["扫描O(n²)；网格版另有邻居排序，保守上界O(n²log n)；无演示记录时额外空间O(n)。",
          "适合二维欧氏点集和算法教学；邻域稀疏时网格可省去大量远点比较。",
          f"密集时仍输出n²邻居；5000点密集例采样RSS增量：剪枝{dense_memory['pruned']:.2f} MiB，库{dense_memory['sklearn_kd_tree']:.1f} MiB。",
          "CSV须为id,x,y；已有结果目录请换 --out；输出时间受CPU、实现和环境影响。",
          "可在新开发集研究低占用网格跳过矩形判断；本轮验证后未据结果调参。"])
fill(14, [("原理复述", "说明核心点、边界点、网格范围、矩形上下界及临界距离处理。"),
          ("微型案例手工推演", "十点案例得到2簇1噪声，点4为边界点；核对每点邻居列表。"),
          ("独立运行", f"20项测试通过；{audit['independently_recomputed_queries']}次抽样查询独立复算；能运行demo与verify。"),
          ("调整参数并分析", "改变开发数据上的 ε；同时比较结果、次数和时间，保留未改善情况。")])
fill(15, ["README含推导与命令；validation含输入、源码快照、哈希、288次计时、标签和报告。",
          "scikit-learn DBSCAN 官方文档：scikit-learn.org/stable/modules/clustering.html#dbscan。\n"
          "Huang等，GriT-DBSCAN，2022，arXiv:2210.07580；本项目未实现其grid tree。"])
all_text = "".join(root.xpath("//w:t/text() | //m:t/text()", namespaces=ns))
for unwanted in ("模拟退火", "重热", "annealing.py", "Kirkpatrick", "【", "】"):
    assert unwanted not in all_text, unwanted
assert E.tostring(body.find("w:sectPr", ns)) == geometry
assert [E.tostring(t.find("w:tblGrid", ns)) for t in tables] == grids
parts["word/document.xml"] = E.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)
settings = E.fromstring(parts["word/settings.xml"])
update = settings.find("w:updateFields", ns)
if update is None:
    update = E.SubElement(settings, q("updateFields"))
update.set(q("val"), "true")
parts["word/settings.xml"] = E.tostring(settings, xml_declaration=True, encoding="UTF-8", standalone=True)
args.output.parent.mkdir(parents=True, exist_ok=True)
with ZipFile(args.output, "w") as archive:
    for info in infos:
        archive.writestr(info, parts[info.filename])
with ZipFile(args.output) as archive:
    for name, value in original.items():
        if name not in ("word/document.xml", "word/settings.xml"):
            assert archive.read(name) == value, name
assert hashlib.sha256(args.reference.read_bytes()).hexdigest() == expected_hash
qa = ROOT.parents[1] / ".cache/dbscan_doc/fidelity_audit.json"
qa.parent.mkdir(parents=True, exist_ok=True)
qa.write_text(json.dumps({"reference": str(args.reference.resolve()), "reference_sha256": expected_hash,
                          "output": str(args.output.resolve()), "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
                          "parts": {name: {"reference": hashlib.sha256(original[name]).hexdigest(), "output": hashlib.sha256(value).hexdigest()} for name, value in parts.items()},
                          "sections": 1, "tables": len(tables), "geometry_and_grids": "unchanged", "preserve_only_parts": "all identical", "old_content_and_placeholders": "none"}, ensure_ascii=False, indent=2), encoding="utf-8")
print(args.output.resolve())
