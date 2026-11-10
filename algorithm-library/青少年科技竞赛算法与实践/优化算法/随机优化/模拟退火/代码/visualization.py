"""Export scientific figures and a local, offline timeline demonstration."""

import json
import os
from pathlib import Path

from annealing import LABELS, MODES

COLORS = {"baseline": "#667085", "periodic": "#dc8441", "stagnation": "#087f8c"}


def pyplot():
    os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).parent / ".cache" / "matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
                         "axes.unicode_minus": False, "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": "white", "savefig.dpi": 180})
    return plt


def plot_demo(points, results: list[dict], out: Path) -> None:
    plt = pyplot()
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.4), sharex=True, sharey=True)
    for ax, result in zip(axes, results):
        mode, route = result["mode"], result["best_route"]
        closed = route + [route[0]]
        ax.plot([points[i][0] for i in closed], [points[i][1] for i in closed], color=COLORS[mode], lw=1.3)
        ax.scatter([p[0] for p in points], [p[1] for p in points], s=12, color=COLORS[mode])
        ax.scatter(*points[0], marker="s", s=70, color="#182230", zorder=5, label="仓库")
        ax.set_title(f"{LABELS[mode]}\n最好路程 {result['best_length']:.2f}")
        ax.set_aspect("equal")
        ax.set_xlabel("x / 坐标单位")
    axes[0].set_ylabel("y / 坐标单位")
    fig.suptitle("相同初始路线与评估预算下的最终历史最好路线", fontsize=13)
    fig.tight_layout()
    fig.savefig(out / "routes.png")
    plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)
    for result in results:
        mode, trace = result["mode"], result["trace"]
        x = [r["evaluation"] for r in trace]
        axes[0].plot(x, [r["best_length"] for r in trace], label=LABELS[mode], color=COLORS[mode])
        axes[1].plot(x, [r["next_temperature"] for r in trace], color=COLORS[mode])
        for event in result["events"]:
            axes[1].scatter(event["evaluation"], event["after"], marker="^", color=COLORS[mode], s=30, zorder=5)
    axes[0].set_ylabel("历史最好路程")
    axes[0].legend(loc="upper right", ncols=3)
    axes[1].set_ylabel("下一步温度（对数刻度）")
    axes[1].set_xlabel("搜索候选评估次数（校准另计）")
    axes[1].set_yscale("log")
    for ax in axes:
        ax.grid(alpha=0.15)
    fig.suptitle("搜索过程与重热事件（三角标记）", fontsize=13)
    fig.tight_layout()
    fig.savefig(out / "convergence.png")
    plt.close(fig)


def plot_benchmark(summary: dict, out: Path) -> None:
    plt = pyplot()
    groups = summary["groups"]
    scenes = sorted({(g["n"], g["layout"]) for g in groups})
    fig, ax = plt.subplots(figsize=(10, 5))
    for offset, mode in zip((-0.18, 0.18), ("periodic", "stagnation")):
        values = [next(g["mean_paired_gain_percent"] for g in groups
                       if (g["n"], g["layout"], g["mode"]) == (n, layout, mode)) for n, layout in scenes]
        bars = ax.bar([i + offset for i in range(len(scenes))], values, width=0.34, color=COLORS[mode], label=LABELS[mode])
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
    ax.axhline(0, color="#182230", lw=0.8)
    ax.set_xticks(range(len(scenes)), [f"{n} 点\n{'均匀' if layout == 'uniform' else '聚集'}" for n, layout in scenes])
    ax.set_ylabel("相对不重热方案的配对平均收益 / %")
    ax.set_title("正值表示路线更短，负值表示路线更长")
    ax.legend()
    ax.margins(y=0.22)
    fig.tight_layout()
    fig.savefig(out / "comparison.png")
    plt.close(fig)


def write_demo_html(points, results: list[dict], out: Path) -> None:
    payload = json.dumps({"points": points, "results": results, "labels": LABELS,
                          "colors": COLORS}, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    html = r'''<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>模拟退火配送路线实验</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#f5f7f8;color:#182230;font:15px/1.7 "Microsoft YaHei",sans-serif}
main{max-width:1240px;margin:auto;padding:32px 24px}h1{font-size:29px;line-height:1.4;margin:6px 0}p{color:#596579}
.eyebrow{color:#087f8c;font-size:12px;letter-spacing:2px}.controls{display:flex;gap:16px;flex-wrap:wrap;align-items:center;background:white;padding:18px;margin:24px 0;border:1px solid #dce3e7;border-radius:10px}
button,select{font:inherit;border:1px solid #b7c4cc;border-radius:6px;background:white;color:#182230;padding:7px 14px;cursor:pointer}button:first-child{background:#087f8c;color:white;border-color:#087f8c}
input[type=range]{flex:1;min-width:160px;accent-color:#087f8c}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:18px}.card{border:1px solid #dce3e7;border-radius:10px;background:white;padding:20px}.card h2{font-size:17px;margin:0}.length{font-size:27px;font-weight:650;margin-top:8px}.muted{font-size:12px;color:#596579}.values{display:grid;grid-template-columns:1fr 1fr;font-size:13px;gap:4px}canvas{width:100%;aspect-ratio:1;display:block;margin:14px 0}figure{margin:24px 0;background:white;border:1px solid #dce3e7;border-radius:10px;padding:18px}figure img{width:100%;display:block}figcaption{color:#596579;font-size:13px}.note{font-size:13px}a{color:#087f8c}@media(max-width:800px){.cards{grid-template-columns:1fr}main{padding:20px 14px}.card canvas{max-height:380px}}
</style>
<main><div class="eyebrow">每周算法 · 01 / 可复现实验</div><h1>停滞后升温，能找到更短的路线吗？</h1>
<p>在同一张地图上比较三种温度策略。拖动时间轴查看中间状态；路线图默认显示历史最好结果。</p>
<div class="controls"><button id="play">播放</button><button id="reset">回到起点</button><label for="speed">速度</label><select id="speed"><option value="1">1 倍</option><option value="3">3 倍</option><option value="6">6 倍</option></select><input id="timeline" aria-label="候选评估进度" type="range" min="0" value="0"><strong id="progress"></strong><label><input type="checkbox" id="current">显示当前路线</label></div>
<div id="cards" class="cards"></div>
<p class="note">仓库为黑色方块。当前路线可能暂时变差，历史最好路线始终保留。温度显示为下一次决策使用的温度。此页为单张地图、单个种子的过程演示，整体效果请查看完整实验报告。</p>
<figure><img src="convergence.png" alt="三组算法的历史最好路程与温度变化曲线"><figcaption>完整运行的曲线；三角标记为实际发生的重热事件。时间轴控制上方路线图。</figcaption></figure>
<p class="note">数据与轨迹：<a href="demo.json">demo.json</a> · 原始坐标：<a href="points.csv">points.csv</a> · 最终路线图：<a href="routes.png">routes.png</a></p></main>
<script>
const data=__PAYLOAD__;
const slider=document.getElementById('timeline'), max=data.results[0].evaluations;
slider.max=max; slider.step=Math.max(1,Math.floor(max/200));
document.getElementById('cards').innerHTML=data.results.map(r=>`<section class="card"><h2 style="color:${data.colors[r.mode]}">${data.labels[r.mode]}</h2><div class="length" id="length-${r.mode}"></div><div class="muted">历史最好路程 / 坐标单位</div><canvas id="canvas-${r.mode}" role="img" aria-label="${data.labels[r.mode]}路线图"></canvas><div class="values" id="values-${r.mode}"></div></section>`).join('');
function draw(canvas,route,color){const rect=canvas.getBoundingClientRect(), size=Math.max(240,rect.width), dpr=window.devicePixelRatio||1;canvas.width=size*dpr;canvas.height=size*dpr;const c=canvas.getContext('2d');c.scale(dpr,dpr);c.clearRect(0,0,size,size);const points=data.points,xs=points.map(p=>p[0]),ys=points.map(p=>p[1]);const loX=Math.min(...xs),loY=Math.min(...ys),span=Math.max(Math.max(...xs)-loX,Math.max(...ys)-loY,1),pad=22;const xy=p=>[pad+(p[0]-loX)/span*(size-2*pad),size-pad-(p[1]-loY)/span*(size-2*pad)];c.strokeStyle='#e7ecef';c.strokeRect(pad,pad,size-2*pad,size-2*pad);c.strokeStyle=color;c.lineWidth=1.7;c.beginPath();[...route,route[0]].forEach((id,i)=>{const [x,y]=xy(points[id]);i?c.lineTo(x,y):c.moveTo(x,y)});c.stroke();points.forEach((p,i)=>{const[x,y]=xy(p);c.fillStyle=i?color:'#182230';c.beginPath();if(i){c.arc(x,y,3.1,0,Math.PI*2);c.fill()}else{c.fillRect(x-5,y-5,10,10)}})}
function render(){const step=Number(slider.value),showCurrent=document.getElementById('current').checked;document.getElementById('progress').textContent=`${step.toLocaleString()} / ${max.toLocaleString()}`;data.results.forEach(r=>{let t=r.trace[0];for(const row of r.trace){if(row.evaluation>step)break;t=row}document.getElementById('length-'+r.mode).textContent=t.best_length.toFixed(2);document.getElementById('values-'+r.mode).innerHTML=`<span>当前路程 ${t.current_length.toFixed(2)}</span><span>温度 ${t.next_temperature.toPrecision(3)}</span><span>已重热 ${t.reheats} 次</span><span>记录步数 ${t.evaluation}</span>`;draw(document.getElementById('canvas-'+r.mode),showCurrent?t.current_route:t.best_route,data.colors[r.mode])})}
let timer=null;function stop(){clearInterval(timer);timer=null;document.getElementById('play').textContent='播放'}document.getElementById('play').onclick=()=>{if(timer){stop();return}if(Number(slider.value)>=max)slider.value=0;document.getElementById('play').textContent='暂停';timer=setInterval(()=>{slider.value=Math.min(max,Number(slider.value)+Number(slider.step)*Number(document.getElementById('speed').value));render();if(Number(slider.value)>=max)stop()},60)};
document.getElementById('reset').onclick=()=>{stop();slider.value=0;render()};slider.oninput=render;document.getElementById('current').onchange=render;window.addEventListener('resize',render);render();
</script></html>'''
    (out / "demo.html").write_text(html.replace("__PAYLOAD__", payload), encoding="utf-8")
