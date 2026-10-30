Sobel 边缘检测 Python 教学演示 v1.0
==================================

一、演示目标
在图像中找到亮度变化明显的位置，例如物体轮廓。
算法先看相邻像素的亮度变化，再用阈值把较强变化标为白色边缘。
它与前一份灰度阈值演示的区别是：这里判断的是“局部变化有多大”，不是“像素有多亮”。
不需要训练模型、下载数据或使用 GPU。

二、快速运行
建议使用 Python 3.11 或更高版本。已验证：Windows 11 / Python 3.12.14，
NumPy 2.3.5 / Pillow 12.3.0。无需安装 OpenCV，两个 Sobel 核直接用 NumPy 实现。

1. 解压完整压缩包，在 sobel_demo 文件夹打开终端。
2. 执行：

   python -m pip install -r requirements.txt
   python sobel_demo.py

3. 打开 results/comparison.png 查看结果。
   需要自动打开系统图片查看器时，运行 python sobel_demo.py --show。
   系统若使用 py 命令，把上述 python 换成 py 即可。

可选的独立环境（无需激活）：
   python -m venv .venv
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   .\.venv\Scripts\python.exe sobel_demo.py

三、调参及换图
   python sobel_demo.py --threshold 200 --output-dir results_200
   python sobel_demo.py --threshold 600 --output-dir results_600
   python sobel_demo.py --input "自己的图片.jpg" --threshold 100 --output-dir my_results

--threshold  原始梯度幅值的阈值，默认100，可取任意非负有限数；不是原图灰度阈值。
--input      可选本地图片路径，支持普通8位PNG/JPEG；省略时自动生成示例。
--seed       生成示例噪声的非负随机种子，默认42；自选图片不使用此参数。
--output-dir 输出目录，默认脚本同级的 results 文件夹。
--show       额外打开系统图片查看器，默认只保存文件，不要求图形界面。

每次运行还会自动比较 T=30、T=600；重复阈值会合并。
建议每次实验使用独立输出目录；同名文件会覆盖，复用目录时旧阈值图片可能保留。
当前运行参数、文件对应的阈值及指标以 run_info.json 为准。

四、核心原理
图片转成二维8位灰度 I；x 向右，y 向下。
Sobel 使用两个3×3权重核，对每个像素周围的3×3邻域逐项乘权重后求和：

       -1  0  1              -1 -2 -1
Kx =   -2  0  2       Ky =    0  0  0
       -1  0  1               1  2  1

Gx 表示左右方向的灰度变化，因此竖直边缘通常响应强。
Gy 表示上下方向的灰度变化，因此水平边缘通常响应强。
本程序不翻转核，采用互相关实现；若按严格卷积翻转两个核，梯度符号会反转，幅值不变。
边界采用 np.pad(..., mode="edge")，复制最近的边缘像素补一圈。
先把 uint8 转成 float32，保留负梯度，避免减法溢出。

幅值 M = sqrt(Gx^2 + Gy^2)。若 M>T，输出255，否则输出0；相等仍为0。
全图使用同一个梯度阈值。无需训练、迭代或收敛判断。
Gx、Gy 和 M 的原始数值可超过255。
为了保存显示图，单独将绝对值除以4并截断到0～255；此显示处理不参与阈值计算。
例如显示图亮度100可能对应原始梯度幅值约400，不能把显示图当作原始幅值。

五、手工计算
3×3输入邻域为：
  0  0  100
  0  0  100
  0  0  100

只计算中心像素：
Gx = 100 + 2×100 + 100 = 400。
Gy = (-100) + 100 = 0。
M = sqrt(400^2 + 0^2) = 400。
T=100时，400>100，中心是白色边缘。
T=400时，400>400不成立，中心是黑色。
左右镜像后Gx=-400、Gy=0，M仍为400；这说明边缘强度与变化方向的正负不同。

六、文件说明
sobel_demo.py             主程序，含中文注释
requirements.txt          已验证的依赖版本
README.txt                本说明
results/original.png      输入图
results/gray.png          灰度图
results/gx_display.png    左右变化的绝对值显示图，除以4并截断
results/gy_display.png    上下变化的绝对值显示图，除以4并截断
results/magnitude_display.png 合成幅值显示图，除以4并截断
results/edges_T30.png     低阈值边缘图
results/edges_T100.png    默认阈值边缘图，改参数后文件名随之改变
results/edges_T600.png    高阈值边缘图
results/comparison.png    原图、两个方向响应与三个阈值结果
results/gradients.npz     原始float32 Gx、Gy、magnitude，可用np.load读取
results/metrics.csv       边缘像素数、占比、耗时统计
results/run_info.json     实际环境、核的约定、阈值、幅值范围和指标

对照图标签：Original为原图；display表示仅用于显示；selected为命令行指定阈值。
默认场景是灰阶图，以便专注于亮度变化；也支持彩色图片，读入后转灰度。
照片按EXIF校正方向；透明区域先合成白底；多帧图片只读第一帧。
16位整数图、浮点图不在本演示输入范围内，应先转换为普通RGB PNG/JPEG。

七、默认实测与解释
默认图320×240，共76800像素；seed=42。
背景灰度35，矩形220、圆150、三角形90；加-8～8整数噪声后保存为uint8。
T=30： 18175个边缘像素，占23.67%，背景和物体内部有大量噪点。
T=100：1990个边缘像素，占2.59%，三个物体轮廓较清晰。
T=600：709个边缘像素，占0.92%，主要保留对比度最大的矩形，弱轮廓消失。

边缘占比 = 被判为边缘的像素数 / 总像素数。
它描述结果稀疏程度，不是准确率，也不是越高越好或越低越好。
本演示没有人工标注的真实边缘，不报告准确率、召回率、F1或IoU。
默认T=100是为了这张演示图设置的观察值，不是所有图片都适用的最佳阈值。
Sobel通常产生有厚度的响应；本程序未做非极大值抑制，也未连接断裂边缘。

计时：每个阈值先预热1次，再重复30次，报告中位数及25/75百分位，单位ms。
计时范围包含 Sobel 和阈值化，不包含读图、灰度预处理、绘图、压缩和写文件。
阈值改变不会改变本算法的渐近复杂度，微小耗时差别不可据此解释为速度改进。

八、课堂演示建议（约25分钟）
0～4分钟：说明亮度变化与边缘的关系，对比亮度阈值分割。
4～10分钟：画出两个3×3核，手算中心像素Gx=400、Gy=0。
10～15分钟：阅读 sobel_gradients，解释float32、边界复制、切片累加。
15～20分钟：运行默认演示，观察方向响应与T=30/100/600的区别。
20～25分钟：换自己的图片，尝试T=200或400，讨论噪声、弱边缘和边缘厚度。

九、常见问题与下一步
缺少numpy或PIL：用运行代码的同一个Python执行依赖安装命令；包名是Pillow，导入名是PIL。
图片路径含空格/中文：使用双引号；图片打不开时检查文件名及扩展名。
结果全黑：阈值可能高于原始梯度最大值；查看run_info.json中的magnitude_max。
结果噪点很多：阈值可能偏低，也可在下一阶段学习先平滑再检测。
没有弹窗：默认只保存结果；打开comparison.png，或追加--show。
方向理解混淆：Gx看左右变化，通常突出竖边；Gy看上下变化，通常突出横边。
下一步可学习高斯去噪、Scharr、Canny、非极大值抑制与双阈值连接。

十、官方参考
OpenCV Sobel原理（仅用于核及原理说明，本程序不依赖OpenCV）：
https://docs.opencv.org/4.x/d2/d2c/tutorial_sobel_derivatives.html
NumPy pad的边界模式：
https://numpy.org/doc/stable/reference/generated/numpy.pad.html
