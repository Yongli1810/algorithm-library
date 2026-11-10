# -*- coding: utf-8 -*-
"""按教学示例的浅色卡片、章节标题、底部字幕结构生成动态教学视频。
需安装 Pillow，机器需有 espeak / ffmpeg。运行: python make_video.py
"""
from __future__ import annotations
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import subprocess
import wave
import shutil
import csv
import json
import math
import textwrap

ROOT = Path(__file__).resolve().parent
WORK = ROOT / 'video_work'
WORK.mkdir(parents=True, exist_ok=True)
OUT = ROOT / '0-1_Knapsack_DP_Demo_20261106.mp4'
S = (1280, 720)
BG = '#F7F9F7'
WHITE = '#FFFFFF'
DARK = '#173D34'
GREEN = '#247E69'
TINT = '#E1F0E9'
MUTED = '#60726B'
ORANGE = '#BE8137'
OTINT = '#FAEEDC'
BORDER = '#DCE5E1'
INK = '#1C302B'
PURPLE = '#675F9D'
FREG = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
FBOLD = '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'

def font(size=22, bold=False):
    return ImageFont.truetype(FBOLD if bold else FREG, size)

def rect(d, coords, fill=WHITE, outline=BORDER, radius=14, width=1):
    d.rounded_rectangle(coords, radius=radius, fill=fill, outline=outline, width=width)

def txt(d, pos, content, size=22, fill=INK, bold=False, anchor=None):
    d.text(pos, str(content), font=font(size,bold), fill=fill, anchor=anchor)

def wrap(text, maximum_chars):
    lines=[]
    for para in text.split('\n'):
        while len(para)>maximum_chars:
            pos=para.rfind('，', 0,maximum_chars)
            if pos < maximum_chars//2:
                pos=maximum_chars
            else:
                pos+=1
            lines.append(para[:pos]);para=para[pos:]
        lines.append(para)
    return lines

def multi(d, x,y, content, size=22,fill=INK, bold=False, line=1.55, maxchars=None):
    lines=wrap(content,maxchars) if maxchars else content.split('\n')
    for j,line_txt in enumerate(lines):
        txt(d,(x,y+j*size*line),line_txt,size,fill,bold)
    return y+len(lines)*size*line

def card(d,x,y,w,h,title,subtitle=None,color=GREEN,bg=WHITE):
    rect(d,(x,y,x+w,y+h),fill=bg)
    d.rounded_rectangle((x+17,y+18,x+22,y+45),radius=2,fill=color)
    txt(d,(x+37,y+17),title,23,DARK,True)
    if subtitle:
        multi(d,x+38,y+65,subtitle,size=19,fill=MUTED,maxchars=max(17,int((w-75)/20)),line=1.45)

def base(title, chapter, subtitle, idx, desc='教学演示 / 04'):
    im=Image.new('RGB',S,BG);d=ImageDraw.Draw(im)
    txt(d,(56,29),chapter,17,GREEN,True)
    txt(d,(1221,30),f'{desc}  ·  {idx:02d}',15,MUTED,False,anchor='ra')
    txt(d,(56,70),title,37,DARK,True)
    d.line((56,132,1224,132),fill=BORDER,width=2)
    rect(d,(30,666,1250,713),fill=DARK,outline=DARK,radius=10)
    # Fit subtitle to bottom banner width
    fs = 22 if len(subtitle)<=50 else 19
    txt(d,(640,688),subtitle,fs,WHITE,False,anchor='mm')
    return im,d

NAMES=['简易药包','饮用水','应急电池','急救箱','照明设备','通信设备']
WEIGHTS=[2,3,4,5,6,7]
VALUES=[3,4,5,8,9,10]


def table_draw(d, box, table, high=None, reveal=None, title_text='dp[i][c]（前 i 件物品，容量 c）'):
    x,y,w,h=box
    n=len(table); cols=len(table[0]); cellw=min(99, (w-45)//(cols+1)); cellh=min(53,(h-48)//(n+1))
    left=x+(w-cellw*(cols+1))//2;top=y+55
    txt(d,(x+22,y+13),title_text,20,DARK,True)
    for i in range(n+1):
        for c in range(cols+1):
            a=left+c*cellw;b=top+i*cellh
            if i==0 and c==0: value='i / c'; bgcolor=DARK;fg=WHITE
            elif i==0:value=str(c-1);bgcolor=DARK;fg=WHITE
            elif c==0:value=str(i-1);bgcolor=TINT;fg=DARK
            else:
                ti=i-1;tc=c-1
                avail= reveal is None or ti==0 or (ti-1,tc) in reveal
                value=str(table[ti][tc]) if avail else '·'
                bgcolor='#F0F4F2' if ((i+c)%2) else WHITE
                fg=INK if avail else '#B7C4BD'
                if high==(ti-1,tc):bgcolor='#F5DEAF';fg=DARK
            d.rectangle((a,b,a+cellw-2,b+cellh-2),fill=bgcolor)
            txt(d,(a+cellw/2,b+cellh/2),value,16 if value!='i / c' else 14,fg,True,anchor='mm')


def make_slide(kind, idx, subtitle, extras=None):
    chapters={'title':'00 / 本周算法演示','items':'01 / 问题与数据','goal':'01 / 问题与数据','micro':'01 / 问题与数据',
              'greedy':'01 / 问题与数据','formula':'02 / 核心原理','decision':'02 / 核心原理',
              'grid':'02 / 动态过程','trace':'02 / 动态过程','code':'03 / 代码演示',
              'compress':'03 / 代码演示','result':'04 / 实验验证','plot':'04 / 实验验证',
              'compare':'04 / 实验验证','complexity':'04 / 局限与扩展','reproduce':'04 / 复现方法','end':'04 / 本周总结'}
    titles={'title':'0/1 背包动态规划','items':'6 件物资：有限容量下怎样选择？','goal':'目标：在限制内获得更高效用',
            'micro':'先用 4 件物品手工推演','greedy':'为什么“按性价比选”会失败？',
            'formula':'动态规划：把问题拆成小问题','decision':'每个格子，只需做一次二选一',
            'grid':'可视化填写动态规划表','trace':'从答案反推：到底选哪几件？',
            'code':'代码演示：二维 DP 的核心循环','compress':'进阶：用一维数组节省内存',
            'result':'正式实验：6 件物资，容量 12kg','plot':'容量改变时，算法表现怎样？',
            'compare':'完整验证：DP、贪心、穷举','complexity':'时间与空间复杂度','reproduce':'跟着代码，复现本周算法','end':'从问题到代码，完成一次完整演示'}
    im,d=base(titles[kind],chapters[kind],subtitle,idx)
    if kind=='title':
        txt(d,(75,217),'有限容量，最优组合',46,DARK,True)
        multi(d,82,294,'建模约束  →  填写状态表  →  回溯组合  →  代码验证',23,MUTED,maxchars=36)
        for j,(v,t) in enumerate([('6 件','可选物资'),('12 kg','最大容量'),('18 分','最优效用')]):
            x=80+j*204
            rect(d,(x,398,x+177,531),fill=TINT if j<2 else OTINT)
            txt(d,(x+89,442),v,35,GREEN if j<2 else ORANGE,True,anchor='mm')
            txt(d,(x+89,499),t,20,MUTED,False,anchor='mm')
        card(d,777,201,413,359,'演示流程','看懂一个小案例\n手工推算每一步\n用 Python 输出结果\n比较两种选法',GREEN)
    elif kind=='items':
        for i,(name,weight,value) in enumerate(zip(NAMES,WEIGHTS,VALUES)):
            x=58+(i%3)*403;y=171+(i//3)*230
            rect(d,(x,y,x+370,y+205))
            txt(d,(x+23,y+20),f'{i+1:02d}  {name}',26,DARK,True)
            txt(d,(x+24,y+97),f'{weight} kg',31,GREEN,True)
            txt(d,(x+180,y+102),f'{value} 分',29,ORANGE,True)
            txt(d,(x+23,y+163),'每件最多选一次',17,MUTED)
    elif kind=='goal':
        rect(d,(70,187,1210,362))
        txt(d,(101,230),'最大化  Σ (效用ᵢ × 是否选入ᵢ)',36,DARK,True)
        txt(d,(102,305),'约束：Σ (重量ᵢ × 是否选入ᵢ) ≤ 背包容量',27,GREEN)
        card(d,73,403,356,195,'重量限制','总重量不能超过 12 kg',GREEN)
        card(d,456,403,356,195,'每件只选一次','选择变量 xᵢ ∈ {0,1}',GREEN)
        card(d,839,403,356,195,'优化目标','让总效用尽可能高',ORANGE)
    elif kind=='micro':
        txt(d,(75,159),'微型案例容量：8 kg',25,GREEN,True)
        for j,(name,w,v) in enumerate(zip(['A','B','C','D'],WEIGHTS[:4],VALUES[:4])):
            x=70+j*295
            rect(d,(x,216,x+270,400),fill=WHITE)
            txt(d,(x+135,255),f'物品 {name}',25,DARK,True,anchor='mm')
            txt(d,(x+135,318),f'{w} kg',31,GREEN,True,anchor='mm')
            txt(d,(x+135,364),f'{v} 分',26,ORANGE,True,anchor='mm')
        rect(d,(73,435,1205,606),fill=TINT)
        txt(d,(106,476),'最优组合：B + D',34,DARK,True)
        txt(d,(108,543),'重量 3 + 5 = 8 kg       效用 4 + 8 = 12 分',25,GREEN)
    elif kind=='greedy':
        card(d,73,180,548,382,'按价值/重量排序','先选 D（5kg，8分）\n再选 A（2kg，3分）\n余下 1kg，无法继续选',ORANGE)
        txt(d,(345,486),'共 11 分',38,ORANGE,True,anchor='mm')
        card(d,650,180,548,382,'动态规划找到更优方案','选择 B（3kg，4分）\n选择 D（5kg，8分）\n恰好占满 8kg',GREEN)
        txt(d,(925,486),'共 12 分',38,GREEN,True,anchor='mm')
        txt(d,(640,601),'最优方案比贪心多 1 分',24,DARK,True,anchor='mm')
    elif kind=='formula':
        rect(d,(65,174,1215,345),fill=WHITE)
        txt(d,(95,205),'dp[i][c]：前 i 件物品，容量上限 c 的最大效用',30,DARK,True)
        txt(d,(95,282),'dp[i][c] = max( 不选第 i 件 , 选第 i 件 )',32,GREEN,True)
        card(d,72,390,360,210,'情况 A：不选','dp[i−1][c]\n直接继承上一行的值',GREEN)
        card(d,457,390,360,210,'情况 B：选','dp[i−1][c−wᵢ] + vᵢ\n必须满足 c ≥ wᵢ',ORANGE)
        card(d,841,390,360,210,'边界条件','dp[0][c] = 0\n没有物品时总效用为零',GREEN)
    elif kind=='decision':
        txt(d,(78,181),'第 4 件 D：重量 5kg，效用 8 分',28,DARK,True)
        rect(d,(70,256,565,462),fill=WHITE)
        txt(d,(104,296),'不选 D',26,MUTED,True)
        txt(d,(104,371),'dp[3][8] = 9',33,DARK,True)
        rect(d,(598,256,1200,462),fill=TINT)
        txt(d,(640,296),'选择 D',26,GREEN,True)
        txt(d,(640,371),'dp[3][3] + 8 = 12',32,DARK,True)
        txt(d,(635,521),'取更大值 → dp[4][8] = 12',34,GREEN,True,anchor='mm')
    elif kind=='grid':
        tbl=extras.get('table')
        current=extras.get('current')
        reveal=extras.get('reveal')
        rect(d,(57,160,1223,610),fill=WHITE)
        table_draw(d,(73,166,915,425),tbl, high=current,reveal=reveal,title_text='微型案例：4 件物品 × 容量 0~8')
        cx,cy=current
        itemidx=min(cx,3)
        rect(d,(992,195,1190,544),fill=TINT)
        txt(d,(1010,221),'当前状态',21,DARK,True)
        txt(d,(1010,278),f'i = {cx+1}',23,GREEN,True)
        txt(d,(1010,319),f'c = {cy}',23,GREEN,True)
        txt(d,(1010,364),f'物品 {"ABCD"[itemidx]}',24,DARK,True)
        txt(d,(1010,430),f'最优值 {tbl[cx+1][cy]}',25,ORANGE,True)
    elif kind=='trace':
        rect(d,(66,176,1214,360))
        txt(d,(98,211),'从 dp[4][8] = 12 反向检查决策',32,DARK,True)
        txt(d,(100,291),'D 被选中  →  容量 8−5=3  →  B 被选中',30,GREEN,True)
        card(d,76,402,350,185,'物品 B','3 kg / 4 分',GREEN,bg=TINT)
        card(d,457,402,350,185,'物品 D','5 kg / 8 分',GREEN,bg=TINT)
        card(d,839,402,350,185,'最终结果','8 kg / 12 分',ORANGE,bg=OTINT)
    elif kind=='code':
        rect(d,(70,169,795,600),fill='#153A31',outline='#153A31')
        lines=[
            'for i, item in enumerate(items, 1):',
            '    for c in range(capacity + 1):',
            '        dp[i][c] = dp[i - 1][c]',
            '        if c >= item.weight:',
            '            dp[i][c] = max(',
            '                dp[i][c],',
            '                dp[i-1][c-item.weight]',
            '                + item.value)',
        ]
        for j,line in enumerate(lines):
            txt(d,(101,208+47*j),line,19,WHITE if j not in (4,6,7) else '#AEE3B8')
        card(d,830,190,365,183,'循环的含义','外层枚举物品\n内层枚举可用容量',GREEN)
        card(d,830,401,365,183,'关键约束','只能引用上一行\n不能重复选入物品',ORANGE)
    elif kind=='compress':
        rect(d,(68,186,1211,331),fill=WHITE)
        txt(d,(107,228),'for c in range(C, weight − 1, −1):',30,DARK,True)
        txt(d,(107,281),'    f[c] = max(f[c], f[c−weight] + value)',29,GREEN,True)
        card(d,70,372,544,216,'为什么要倒序？','防止同一件物品在当前轮被重复使用\n只保存当前容量数组',GREEN)
        card(d,639,372,544,216,'空间开销对比','二维 DP：O(n×C)\n一维 DP：O(C)',ORANGE)
    elif kind=='result':
        card(d,74,189,539,360,'普通贪心：15 分','依次挑选高价值密度物品\n最终选中：药包、饮用水、急救箱',ORANGE)
        card(d,637,189,539,360,'动态规划：18 分','选择：急救箱 + 通信设备\n总重量 5 + 7 = 12 kg',GREEN)
        txt(d,(343,477),'15 分',49,ORANGE,True,anchor='mm')
        txt(d,(909,477),'18 分',49,GREEN,True,anchor='mm')
        txt(d,(644,600),'同样是 12 kg 容量，DP 多获得 3 分',24,DARK,True,anchor='mm')
    elif kind=='plot':
        import matplotlib.pyplot as plt
        # The chart is generated by the runnable Python source, not fabricated.
        chart=Image.open(ROOT/'results/comparison.png').convert('RGB')
        chart.thumbnail((1015,450))
        rect(d,(71,162,1209,622),fill=WHITE)
        im.paste(chart,(640-chart.width//2,165+(452-chart.height)//2))
        txt(d,(640,621),'5 组整数容量对照；DP 与 1D DP、穷举的最优值一致',18,MUTED,anchor='mm')
    elif kind=='compare':
        headers=['容量 (kg)','DP 最优值','贪心值','差额']
        data=[('8','12','11','1'),('10','15','15','0'),('12','18','15','3'),('14','21','20','1'),('16','24','24','0')]
        xx=[95,386,682,963];yy=192
        rect(d,(65,174,1215,593),fill=WHITE)
        d.rectangle((78,187,1201,244),fill=DARK)
        for x,head in zip(xx,headers):txt(d,(x,200),head,23,WHITE,True)
        for j,row in enumerate(data):
            y=254+j*63
            d.rectangle((78,y,1201,y+58),fill='#E1F0E9' if j==2 else ('#F0F4F2' if j%2 else WHITE))
            for x,cell in zip(xx,row):txt(d,(x,y+12),cell,24,ORANGE if j==2 and x>=682 else DARK,True)
    elif kind=='complexity':
        card(d,75,185,535,373,'二维动态规划','时间复杂度：O(n×C)\n空间复杂度：O(n×C)\n优点：可直接回溯物品组合\n局限：大容量会增加状态数',GREEN)
        card(d,640,185,535,373,'空间压缩与边界','一维空间：O(C)\n容量必须逆序遍历\n重量与容量需整数化\n极大容量不宜直接建大表',ORANGE)
    elif kind=='reproduce':
        rect(d,(70,178,1210,319),fill='#153A31',outline='#153A31')
        txt(d,(104,221),'python knapsack_demo.py',34,WHITE,True)
        txt(d,(104,270),'可选绘图：pip install matplotlib',20,'#ABD6C6')
        card(d,76,367,355,210,'代码','knapsack_demo.py\n二维 DP、回溯、一维优化',GREEN)
        card(d,455,367,355,210,'实验结果','comparison.csv\nDP 状态表与图表',GREEN)
        card(d,834,367,355,210,'讲解文档','按六页教学模板填写\n数学推导与结果分析',ORANGE)
    elif kind=='end':
        txt(d,(92,191),'把一个复杂选择问题，拆成清楚的小格子。',37,DARK,True)
        for j,(n,one) in enumerate([('01','定义目标与容量约束'),('02','枚举“选 / 不选”'),('03','填表、回溯、验证')]):
            y=300+j*94
            rect(d,(90,y,1170,y+76),fill=WHITE)
            txt(d,(121,y+17),n,27,GREEN,True)
            txt(d,(219,y+20),one,25,DARK,True)
        txt(d,(640,626),'完整 Word、代码、数据、视频已整理在复现压缩包中',19,MUTED,anchor='mm')
    return im

# spoken text independent from bottom subtitles
STORY=[
('title','这周的算法是零一背包动态规划。我们用一个应急物资装载问题，依次讲解建模、状态转移、代码实现和实验结果。','这节课用一套物资数据，演示从问题到算法再到代码的全过程。',16),
('items','假设车辆只能装下有限的重量。这里有六件不同物资，每一件都有重量和效用分数。我们需要在不超载的前提下，让总效用尽可能高。','每种物资都有重量和效用；只有容量限制，没有预先指定答案。',16),
('goal','数学目标是最大化被选物品的效用总和。所有被选物品的重量不能超过背包容量，而且每件物品最多只能取一次。这正是零一背包问题。','选择变量只有零或一：要么不选，要么选一件。',16),
('micro','先看一个更容易手算的例子。四件物品的重量分别是二、三、四、五公斤，效用分别是三、四、五、八分。容量只有八公斤。','先用四件物品、八公斤容量，做一次完整的手工推演。',17),
('greedy','如果只按每公斤的效用排序，可能会先选急救箱和药包，得到十一分。换成饮用水加急救箱，却能刚好装满八公斤，得到十二分。','贪心方案得到 11 分，真正的最优方案可以得到 12 分。',20),
('formula','动态规划不用只看眼前的性价比。我们定义状态，表示前 i 件物品在容量 c 时的最优效用。对每件物品，比较不选它和选它两种情况，保存更大的结果。','状态转移的核心只有一件事：比较选与不选。',20),
('decision','比如计算第四件物品、容量八的格子。若不选第四件，原来的最优值是九。若选它，剩余容量是三，原来的最优值是四，再加新物品的八分，总共十二分。','例子：容量 8 时，比较不选得到 9 与选入得到 12。',18),
('grid','现在我们把这套规则真的填进表格。每一行对应已经处理的物品数量，每一列对应可用容量。高亮格子就是当前正在计算的状态。','逐格计算：横向看容量，纵向看已经处理的物品数量。',27),
('trace','表格最后一行的答案是十二。接下来从表格右下角反向查看，判断每件物品是否被选中。最终得到饮用水和急救箱，重量八公斤，效用十二分。','回溯表格右下角，找到物品 B 和 D。',18),
('code','Python 代码中的外层循环逐件处理物品。内层循环遍历所有容量。若物品放得下，就比较选与不选两种情况。每次都从上一行取值，所以不会重复使用同一件物品。','核心代码：两个循环加一个 max 比较。',22),
('compress','如果只需要最优值，不一定保存完整的二维表。只用一个一维数组就能压缩空间，但容量必须从大到小倒序更新，否则同一件物品可能被反复计算。','一维 DP 更省空间；记住容量必须倒序更新。',19),
('result','回到六件物资、十二公斤容量的正式实验。普通贪心的结果是十五分，而动态规划找到急救箱加通信设备，刚好十二公斤，总效用十八分，领先三分。','在相同容量限制下，DP 找到 18 分的更优组合。',19),
('plot','把背包容量改成八、十、十二、十四和十六公斤，再运行程序。可以看到最优值会随容量变化。贪心有时与最优值相同，有时会落后。','五组容量对照数据由 Python 程序实际运行生成。',19),
('compare','我们还用穷举搜索做正确性检查。二维动态规划、一维空间优化和穷举，在五组容量下全部得到相同的最优值。这能帮助我们确认代码结果可靠。','二维 DP、一维 DP 与穷举的结果一致；贪心不一定最优。',18),
('complexity','二维动态规划的时间复杂度是物品数量乘以容量，空间复杂度也一样。采用一维压缩后，额外空间可以降低为容量大小。不过当整数容量很大时，状态表也会变大。','时间 O(nC)，空间可以从 O(nC) 压缩到 O(C)。',21),
('reproduce','解压完整文件后，在项目文件夹运行 Python 脚本。它会输出最优物品组合、比较数据和动态规划状态表。若安装 matplotlib，还能直接生成实验曲线。','运行源代码即可复现实验；不依赖外部真实数据。',20),
('end','这次演示讲清了一个常用的动态规划模型。从问题约束，到状态转移，再到代码验证，希望大家能真正理解为什么选与不选的比较可以得到最优解。','本周算法演示到这里，文档、程序、数据和视频都在压缩包里。',17)
]


def generate_voice(text,i):
    voice=WORK/f'voice_{i:02d}.wav'
    subprocess.run(['espeak','-v','zh','-s','153','-a','170','-w',str(voice),text],
                   check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    with wave.open(str(voice),'rb') as w:
        return voice, w.getnframes()/w.getframerate()


def main():
    import importlib.util
    assert shutil.which('espeak') and shutil.which('ffmpeg'),'需要 espeak 和 ffmpeg'
    from knapsack_demo import dp_2d, MICRO_ITEMS
    best, ids, micro_table=dp_2d(MICRO_ITEMS,8)
    assert best==12 and ids==[1,3]
    image_entries=[]
    audio_entries=[]
    slides=[]
    for i,(kind,narration,subtitle,min_duration) in enumerate(STORY,1):
        audio,spoken=generate_voice(narration,i)
        if kind=='grid':
            length=max(float(min_duration),spoken+1.2,36*0.54)
            each=length/36.0
            reveal=set()
            for n in range(36):
                row=n//9;col=n%9
                reveal.add((row,col))
                im=make_slide(kind,i,subtitle,{'table':micro_table,'current':(row,col),'reveal':reveal})
                image=WORK/f'slide_{i:02d}_step_{n:02d}.png';im.save(image,optimize=True)
                image_entries.append((image,each))
        else:
            duration=max(float(min_duration),spoken+1.2)
            im=make_slide(kind,i,subtitle)
            image=WORK/f'slide_{i:02d}.png';im.save(image,optimize=True)
            image_entries.append((image,duration))
        duration=sum(t for p,t in image_entries if p.name.startswith(f'slide_{i:02d}'))
        padded=WORK/f'audio_{i:02d}.wav'
        subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-i',str(audio),
                        '-af','apad','-t',f'{duration:.3f}','-ar','22050','-ac','1',str(padded)],check=True)
        audio_entries.append(padded)
        slides.append({'kind':kind,'duration':round(duration,2),'narration':narration,'subtitle':subtitle})
    # Concat demuxer for image hold durations, plus a matching concatenated audio track.
    pictures=WORK/'pictures.txt'
    with pictures.open('w',encoding='utf-8') as out:
        for path,duration in image_entries:
            out.write(f"file '{path.as_posix()}'\n")
            out.write(f'duration {duration:.6f}\n')
        out.write(f"file '{image_entries[-1][0].as_posix()}'\n")
    audios=WORK/'audios.txt'
    with audios.open('w',encoding='utf-8') as out:
        for f in audio_entries:out.write(f"file '{f.as_posix()}'\n")
    sound=WORK/'narration_all.wav'
    subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','concat','-safe','0',
                    '-i',str(audios),'-c:a','pcm_s16le',str(sound)],check=True)
    subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','concat','-safe','0',
                    '-i',str(pictures),'-i',str(sound),'-c:v','libx264','-preset','ultrafast',
                    '-crf','25','-r','24','-pix_fmt','yuv420p',
                    '-c:a','aac','-b:a','96k','-shortest','-movflags','+faststart',str(OUT)],check=True)
    (ROOT/'video_script.json').write_text(json.dumps(slides,ensure_ascii=False,indent=2),encoding='utf-8')
    print('视频生成完成：',OUT)
    print('章节数:',len(STORY),'总时长（预估秒）:',round(sum(s['duration'] for s in slides),1))

if __name__=='__main__':main()
