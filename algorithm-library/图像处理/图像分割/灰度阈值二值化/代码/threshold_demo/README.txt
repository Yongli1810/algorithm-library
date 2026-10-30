灰度阈值二值化 Python 教学演示 v1.0
====================================

一、这份演示做什么
对每个像素判断：灰度大于阈值 T 则变白（255），否则变黑（0）。
它是最基础的图像分割方法之一，适合刚学 Python 与计算机视觉的同学。
默认自动生成一张含三种形状的图片，无需下载数据、训练模型或使用 GPU。
配套 Word 按用户提供模板编写，涵盖原理、流程、代码、演示和实测分析。

二、文件
threshold_demo.py       可直接运行的 Python 主程序，含中文注释
requirements.txt        已验证的依赖版本
README.txt              本说明
results/                预先生成的默认演示结果
压缩包顶层的 .docx       配套讲解文档

三、快速运行
建议安装 Python 3.11 或更高版本。已验证环境为 Windows 11 / Python 3.12.14。
只需要 NumPy 2.3.5 和 Pillow 12.3.0。

1. 解压整个压缩包，进入 threshold_demo 文件夹。
2. 在该文件夹打开 PowerShell 或命令提示符，执行：

   python -m pip install -r requirements.txt
   python threshold_demo.py

3. 打开 results/comparison.png 查看对比。程序默认保存结果，不要求弹出窗口。
   若希望运行完自动打开图片，可执行：

   python threshold_demo.py --show

也可以在 VS Code / PyCharm 中打开 threshold_demo.py 运行。
如果系统使用 py 命令，请把上面和下文的 python 换成 py。

可选：若需隔离依赖，可在文件夹中运行以下命令，无需激活环境：

   python -m venv .venv
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   .\.venv\Scripts\python.exe threshold_demo.py

四、现场演示顺序（约 20 分钟）
0～4 分钟：说明灰度 0～255，以及前景白、背景黑的约定。
4～7 分钟：手算 [[20,100,127],[128,180,240]] 在 T=127 时的输出。
           答案为 [[0,0,0],[255,255,255]]；强调等于阈值仍为黑色。
7～11 分钟：阅读 binary_threshold 函数，理解 gray>T 和 np.where。
11～15 分钟：运行默认演示，观察三个阈值下的误检和漏检。
15～18 分钟：换阈值或换图片，说明一个阈值的适用前提。
18～20 分钟：解释 IoU，完成 Word 文档中的检验任务。

五、调节阈值与更换图片
阈值范围为 0～255，默认 127。每次还会输出 60、200 两个对比阈值。
若指定值与 60 或 200 相同，则合并重复值。

   python threshold_demo.py --threshold 160 --output-dir results_160
   python threshold_demo.py --threshold 220 --output-dir results_220
   python threshold_demo.py --input "自己的图片.jpg" --threshold 127 --output-dir my_results

路径含空格或中文时加双引号。默认结果目录位于脚本同级，与终端当前目录无关。
建议每个实验使用单独的 --output-dir；复用目录时，同名文件会覆盖，旧阈值图片可能保留。
查看 run_info.json 可以确定当前运行所用输入、阈值及生成的指标。

自选图片支持普通 8 位彩色/灰度图；照片方向按 EXIF 校正。
透明图片先在白色背景上合成，再转换为灰度。
16 位整数图、浮点图不在本演示输入范围内，请先转换为 RGB PNG/JPEG。
读取多帧图片时演示仅使用第一帧。处理大图会占用更多内存。

六、输出说明
original.png             本次输入图
gray.png                 转换后的灰度图
ground_truth.png         仅默认合成数据生成的独立真值掩膜
binary_T60.png           阈值 60 的结果
binary_T127.png          默认阈值 127 的结果，改参数后文件名随之改变
binary_T200.png          阈值 200 的结果
comparison.png           原图、灰度图和各阈值结果对照
metrics.csv              白色比例、IoU、误检漏检像素数、耗时
run_info.json            软件、系统、参数、计时次数与原始指标

对照图使用英文标签以避免字体安装问题：Original=原图；Grayscale=灰度；
T=阈值；selected=命令行指定的阈值。默认场景本身采用灰阶，因此原图与灰度图相同。

七、默认结果与解释
默认图片为 320×240，共 76800 个像素，seed=42。
背景为灰度渐变 45～110 叠加 -8～8 整数噪声；三个形状灰度分别约为 180、215、235。
随机数使用 np.random.default_rng(42)，生成器版本记录在 run_info.json。
真实掩膜由形状的几何位置先定义，不是由阈值结果倒推。

阈值 60： 白色比例 81.89%，IoU=0.2967，误检 44227 像素、漏检 0。
阈值 127：白色比例 24.30%，IoU=1.0000，误检 0、漏检 0。
阈值 200：白色比例 13.78%，IoU=0.5673，误检 0、漏检 8075 像素。

IoU = 预测前景与真实前景的交集像素数 / 并集像素数，越接近 1 越好。
若两者都为空，代码约定 IoU=1；默认图片的真值不为空。
白色比例是前景面积比例，不是准确率，也不是越高越好。
误检：把背景预测为前景；漏检：把前景预测为背景。
本例亮度区间分离，127 可以完整分割；它不是所有图片的通用最优阈值。
自选图片无独立真值，IoU 显示 N/A，CSV 对应指标留空，JSON 对应值为 null。

计时策略：每个阈值先预热一次，再计时 30 次，记录中位数、25/75 百分位，单位 ms。
只统计 binary_threshold 函数，包括输入检查与数组创建，不含读图、可视化、写盘。
很小的耗时受系统负载和计时波动影响，不用它判断不同阈值的计算速度优劣。

八、常见问题
1. 提示 No module named numpy 或 No module named PIL：
   使用运行代码的同一个 Python 执行 python -m pip install -r requirements.txt。
   pip 包叫 Pillow，导入名称是 PIL。
2. 图片路径错误：检查真实路径、文件扩展名以及双引号。
3. 输出全黑/全白：检查阈值是否过高/过低，先看 gray.png。
   T=255 时必为全黑；T=0 时灰度为 0 的像素仍为黑色。
4. 原图目标较暗：当前规则把亮像素当作前景，可改为 gray<threshold。
   改变规则时也要同步修改评估中的前景约定。
5. 没有弹出窗口：这是正常现象，直接打开 comparison.png，或增加 --show。
6. 无法安装依赖：确认 Python 版本和网络；请使用 Python 3.11 或更高版本。

九、延伸方向
下一步可以学习 Otsu 自动阈值、自适应阈值、去噪、形态学操作。
本程序只演示固定全局阈值，不包含上述扩展，也不进行物体类别识别。

十、官方参考
NumPy np.where：
https://numpy.org/doc/stable/reference/generated/numpy.where.html
Pillow Image.convert：
https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.convert
