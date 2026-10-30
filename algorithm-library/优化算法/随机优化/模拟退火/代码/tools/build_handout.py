"""Fill the retained DOCX template without rewriting unrelated package parts.

Optional document-authoring tool; run with the bundled document Python/lxml.
The algorithm itself does not need lxml or Word.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile
from lxml import etree as E

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--template", type=Path, required=True)
parser.add_argument("--output", type=Path, default=ROOT / "docs/模拟退火算法讲解与代码演示.docx")
args = parser.parse_args()
expected_hash = "a9f624efacf175c63e494ad9924fcbb04ac6fbc52e289db33093bc13534adeef"
assert hashlib.sha256(args.template.read_bytes()).hexdigest() == expected_hash, "Template changed; inspect it before using this slot map."
load = lambda p: json.loads(p.read_text(encoding="utf-8"))
summary = load(ROOT / "outputs/validation/summary.json")
manifest = load(ROOT / "outputs/validation/manifest.json")
demo = load(ROOT / "outputs/demo/demo.json")
micro = load(ROOT / "outputs/micro/micro.json")
by_mode = {r["mode"]: r for r in summary["overall"]}
demo_by_mode = {r["mode"]: r for r in demo["results"]}
best = by_mode["stagnation"]
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
M = "http://schemas.openxmlformats.org/officeDocument/2006/math"
ns = {"w": W, "m": M}
q = lambda tag: "{" + W + "}" + tag

with ZipFile(args.template) as archive:
    parts = {i.filename: archive.read(i.filename) for i in archive.infolist()}
    infos = archive.infolist()
root = E.fromstring(parts["word/document.xml"])
body = root.find("w:body", ns)
paragraphs = body.findall("w:p", ns)
tables = body.findall("w:tbl", ns)


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


def cell(t, row, column):
    return tables[t].findall("w:tr", ns)[row].findall("w:tc", ns)[column]


def put(t, row, column, text):
    c = cell(t, row, column)
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


def math_line(p, text):
    # Native OMML linear math with real Greek letters and Unicode subscripts.
    om = E.SubElement(p, "{" + M + "}oMath")
    run = E.SubElement(om, "{" + M + "}r")
    props = E.SubElement(run, q("rPr"))
    E.SubElement(props, q("rFonts"), {q("ascii"): "Cambria Math", q("hAnsi"): "Cambria Math"})
    E.SubElement(props, q("sz"), {q("val"): "17"})
    E.SubElement(run, "{" + M + "}t").text = text


def formula_cell(t, row, lines):
    c = cell(t, row, 1)
    old = c.find("w:p", ns)
    props = deepcopy(old.find("w:pPr", ns))
    for p in c.findall("w:p", ns):
        c.remove(p)
    for text in lines:
        p = E.SubElement(c, q("p"))
        if props is not None:
            p.append(deepcopy(props))
        math_line(p, text)


set_paragraph(paragraphs[0], "模拟退火配送路线优化")
set_paragraph(paragraphs[1], "算法名称：停滞触发的受控重热模拟退火        算法类别：组合优化")
set_paragraph(paragraphs[2], "讲解对象：具备 Python 基础的学习者          讲解时长：建议 45 分钟")
fill(0, ["Simulated Annealing with Controlled Reheating；项目简称 SA-CR。",
         "经典 SA 与 2-opt 基础上的教学实验变体；参考文献见末页。",
         "解释接受规则与重热；完成手算；复现三组对照；判断改动是否有效。",
         "排列、概率、指数函数、Python 列表；核心求解器只用标准库。"])
fill(1, ["单车配送访问顺序优化；寻找较短的闭合路线。",
         "普通退火后期可能停滞；检验适时升温的收益与搜索预算的代价。",
         "合成二维坐标；CSV 含 id,x,y；20/50/100 点均包含仓库，单位为坐标单位。",
         "节点排列、最好路程、温度轨迹和重热事件；0 为仓库，回程边隐含。",
         "每点访问一次并返回仓库；对称欧氏距离；不含容量、时间窗和路网约束。"])
fill(2, ["用 2-opt 反转一段路线；较差候选按概率接受，历史最好路线始终保留。",
         "正方形四点：交叉路线长 2+2√2；反转中间两点后路长 4，变化为 2−2√2。",
         "最近邻初解 → 温度校准 → 候选与接受 → 降温/受控重热 → 预算耗尽。",
         "低温、持续无改善、剩余预算充足且重热未达上限时，才触发停滞重热。"])
fill(3, ["Metropolis 接受思想与几何降温；2-opt 的四条边增量仅用于对称距离。",
         "公式由下方原生数学对象给出。",
         "T≤0.005T₀、无改善 K 步、重热未满 3 次且 t≤0.9B 时触发；重热后重新降温。",
         "固定预算保证终止；保存的最好值不增加。本有限预算、重热变体不保证全局最优。"])
formula_cell(3, 2, ["L(π) = d(π₀,π₁) + ⋯ + d(πₙ₋₁,π₀)",
                     "Δ = d(a,c) + d(b,d) − d(a,b) − d(c,d)",
                     "P = min(1, exp(−Δ/T))； T′ = min(T₀, max(3T, 0.002T₀))"])
fill(4, [("π、d、Δ", "路线、欧氏距离、候选路程变化", "目标函数与 2-opt 接受判断；距离均为坐标单位"),
         ("T、T₀、Tmin", "当前、初始、最低温度", "Tmin=0.0001T₀；温度与路程使用相同尺度"),
         ("B、K、t、α", "预算、停滞阈值、当前步、降温率", "重热后按剩余 B−t 步重设 α，最终达到相同温度下限")])
fill(5, [("数据准备", "检查坐标和连续编号，计算距离矩阵", "CSV → 有效的对称 TSP"),
         ("初始化", "最近邻路线；200 次候选校准 T₀", "平均正增量 / (−ln 0.5)"),
         ("核心计算", "抽取 2-opt 片段，计算 Δ 与接受概率", "当前路线 → 候选"),
         ("状态更新", "按概率接受；保存最好路线；降温或重热", "更新温度、停滞计数及重热次数"),
         ("终止判断", "达到 12000 次搜索候选评估", "校准另计且三组相同"),
         ("结果输出", "保存路线、指标、事件与轨迹", "JSON、CSV、PNG、离线 HTML")])
set_paragraph(paragraphs[15], "初始化最近邻路线、温度、最好解和计数\n重复 B 次：\n    抽取 2-opt；依据 Δ 与 T 接受或拒绝\n    更新最好解与停滞计数，并几何降温\n    若满足重热条件：升温，按剩余预算重设 α\n返回历史最好路线、指标和完整配置")
fill(6, [("B / 校准数", "搜索与温度校准预算", "12000 / 200", "三组相同；增大预算增加计算量"),
         ("K / 周期间隔", "停滞或定期触发阈值", "1000 / 3000", "过小可能频繁扰动；需开发集验证"),
         ("门槛 / 重热下限", "仅低温触发；避免升温太弱", "0.005T₀ / 0.002T₀", "重热倍率 3；五组开发候选后固定"),
         ("次数 / 截止比例", "控制重热与保留收敛时间", "3 / 0.9B", "最多三次；最后 10% 预算不重热")])
fill(7, ["Windows CPU；Python 3.12.0；NumPy 2.4.4；Matplotlib 3.11.1。",
         "annealing.py 求解；datasets.py 数据；experiments.py 评估；outputs 保存结果。",
         "在本算法目录执行 run.cmd demo；run.cmd benchmark --out outputs/recheck。",
         "检查 id 连续且不重复；拒绝非有限坐标；建立对称欧氏距离矩阵。",
         "SAConfig 与 configs/full.json；完整实际参数见 validation/manifest.json。",
         "two_opt_delta 对应增量；solve 对应接受、重热与降温；exact_tsp 为微型验证。",
         "预算固定结束；返回最好路线及事件；错误输入提示原因并返回非零退出码。",
         "配对收益、图内波动、耗时；路线、收敛与温度曲线；可拖动时间轴的离线演示。"])
fill(8, [("annealing.py / two_opt_delta", "四条边计算 Δ", "路线、反转区间 → 路程变化"),
         ("annealing.py / solve", "接受、停滞与受控重热", "问题、参数、种子 → 结果字典"),
         ("experiments.py / summarize", "配对统计与地图级重采样", "逐次记录 → 分组和总体结果")])
fill(9, ["50 点聚集地图；数据种子 20261022，求解种子 7；每组 12000 次评估。",
         "先用 micro 手算和精确解核对，再打开 outputs/demo/demo.html 拖动时间轴。",
         "当前/最好路程、温度、接受概率与重热次数；单步信息在 outputs/micro/micro.json。",
         "run.cmd micro；run.cmd demo；输出演示页面、路线图、温度曲线和 JSON。"])
dr = demo_by_mode["stagnation"]
first = dr["events"][0] if dr["events"] else None
fill(10, [("初始状态", f"最近邻路程 {dr['initial_length']:.3f}", "校准温度后开始；三组共用初始路线"),
          ("中间状态", f"首个重热步 {first['evaluation']}" if first else "本演示未触发重热", "温度上升；历史最好解继续保留" if first else "无满足触发条件的时刻"),
          ("结束状态", f"最好路程 {dr['best_length']:.3f}；重热 {dr['reheats']} 次", "单次演示不能替代多地图验证")])
fill(11, ["18 张新地图：3 规模×2 分布×3 实例；种子 100–119；共 1080 次求解。",
          "不重热为基准；固定间隔为触发规则对照；停滞触发为拟改进方案。",
          "配对收益=100×(基准路程−方案路程)/基准路程；正值更好；另报波动与耗时。",
          "地图种子 20271123；与开发数据分离。每图 20 次重复；区间按地图均值重采样。"])
labels = {"baseline": "不重热", "periodic": "固定间隔", "stagnation": "停滞触发"}
rows=[]
for mode in ("baseline", "periodic", "stagnation"):
    r=by_mode[mode]
    rows.append((labels[mode], {"baseline":"关闭重热", "periodic":"每 3000 步检查", "stagnation":"K=1000；低温门槛"}[mode],
                 f"平均收益 {r['mean_paired_gain_percent']:+.3f}%", f"{r['mean_seconds']:.4f} 秒/次",
                 f"胜/平/负\n{r['wins']}/{r['ties']}/{r['losses']}"))
fill(12, rows)
lo,hi=best["map_bootstrap_95_interval"]
set_paragraph(paragraphs[27], f"分析结论：停滞触发的平均收益 {best['mean_paired_gain_percent']:+.3f}%，地图重采样区间 [{lo:+.3f}%, {hi:+.3f}%]。"
              "效果依赖地图与预算；不据此声称普遍优越。开发阶段的负结果与全部候选均保留。")
fill(13, ["n 点、B 次评估：矩阵与最近邻 O(n²)，搜索最坏 O(Bn)，空间 O(n²)；轨迹另占 O(Bn/s)。",
          "适合单车欧氏 TSP 教学与过程观察；增量计算 O(1)，但反转和复制仍为 O(n)。",
          "参数与尺度敏感；重热可能损害结果；大实例无最优误差证明；峰值内存未实测。",
          "找不到 Python：选根目录 .venv；CSV 错误：检查 id,x,y；已有实验目录：指定新 --out。",
          "可在新开发集研究 K、门槛和接受率反馈；保留独立验证，避免在验证结果上反复选参。"])
fill(14, [("原理复述", "说明 2-opt、接受较差解、低温门槛、共同终温和固定预算。"),
          ("微型案例手工推演", f"四点得到 4；8 点精确最优值 {micro['exact_length']:.6f}，与程序核对。"),
          ("独立运行", "run.cmd test 通过 18 项检查；能打开 demo.html 并定位结果文件。"),
          ("调整参数并分析", "在开发地图改变 K；保存配置和所有结果，再用新地图检验。")])
fill(15, ["README 含命令与公式；validation 含协议、输入、源码快照、哈希与逐次结果；全部数据本地生成。",
          "Kirkpatrick 等，Science 220, 671–680, 1983，doi:10.1126/science.220.4598.671。\n"
          "Gunawan 与 Iryanto，International Journal of Computing 22(1), 43–50, 2023，doi:10.47839/ijc.22.1.2878。"])
for table in tables:
    for t in table.xpath("./w:tr[1]//w:t", namespaces=ns):
        if t.text == "填写内容":
            t.text = "内容"
remaining = "".join(root.xpath("//w:t/text()", namespaces=ns))
assert "【" not in remaining and "】" not in remaining, "Unfilled template slot"
parts["word/document.xml"] = E.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)
settings = E.fromstring(parts["word/settings.xml"])
update = settings.find("w:updateFields", ns)
if update is None:
    update=E.SubElement(settings, q("updateFields"))
update.set(q("val"), "true")
parts["word/settings.xml"] = E.tostring(settings, xml_declaration=True, encoding="UTF-8", standalone=True)
args.output.parent.mkdir(parents=True, exist_ok=True)
with ZipFile(args.output, "w") as archive:
    for info in infos:
        archive.writestr(info, parts[info.filename])
with ZipFile(args.template) as a, ZipFile(args.output) as b:
    for name in a.namelist():
        if name not in ("word/document.xml", "word/settings.xml"):
            assert a.read(name) == b.read(name), name
assert hashlib.sha256(args.template.read_bytes()).hexdigest() == expected_hash
print(args.output)
