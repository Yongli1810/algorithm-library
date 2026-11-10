均值滤波去噪 Python 教学演示 v1.0
================================

一、这个演示做什么
把每个像素周围 k×k 个像素取平均，用平均值替换中心像素。
这是基础的图像平滑算法，能减弱随机亮度噪声，也会模糊物体边缘和细线。
默认自动生成干净图片和加噪图片，比较3×3、5×5、9×9窗口，不需下载数据或训练模型。
配套Word沿用算法讲解模板，包含原理、公式、手算、代码和实际结果。

二、文件与快速运行
mean_filter_demo.py    主程序，含中文注释
requirements.txt      已验证的依赖版本
README.txt            本说明
results/              默认运行结果，可不运行代码直接查看

建议使用 Python 3.11 或更高版本。已验证环境为 Windows 11 / Python 3.12.14，
NumPy 2.3.5 / Pillow 12.3.0；普通CPU即可，不需要GPU或OpenCV。

1. 解压完整压缩包，进入 mean_filter_demo 文件夹，在该文件夹打开终端。
2. 执行：

   python -m pip install -r requirements.txt
   python mean_filter_demo.py

3. 打开 results/comparison.png 观察结果。
   需要运行后自动打开系统图片查看器时，执行 python mean_filter_demo.py --show。
   如果系统使用 py 命令，将上述 python 换成 py 即可。

可选独立环境，无需激活：
   python -m venv .venv
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   .\.venv\Scripts\python.exe mean_filter_demo.py

三、调节参数与换图
   python mean_filter_demo.py --kernel-size 7 --output-dir results_k7
   python mean_filter_demo.py --sigma 30 --output-dir results_sigma30
   python mean_filter_demo.py --input "自己的图片.jpg" --kernel-size 3 --output-dir my_results

--kernel-size  窗口边长，1～31的奇数，默认3；k=1保持输入不变。
--sigma        合成高斯噪声的标准差，非负有限数，默认18；单位为灰度级。
--seed         合成噪声的非负整数种子，默认42。
--input        可选本地图片；设置后直接对其灰度图滤波，不再加噪。
--output-dir   输出目录，默认脚本同级的results文件夹。
--show         额外使用系统图片查看器打开对照图。

每次固定比较k=3、5、9，并额外加入指定的窗口；重复值自动合并。
指定7时会有3、5、7、9四个结果，对照图会相应增加一行。
建议每个实验使用独立输出目录；复用目录会覆盖同名文件，旧窗口结果可能保留。
当前输入、窗口、指标以run_info.json为准。
自选图片时sigma和seed不参与处理，运行信息中这两个字段为null。
路径含空格或中文时使用双引号。

四、核心算法
对每个位置取k×k邻域，将所有像素值加起来，再除以k²。
3×3核为：

             1  1  1
K = (1/9) × 1  1  1
             1  1  1

所有权重相同且总和为1。输出由原始输入的邻域计算，不把刚更新的像素继续传给下一位置。
程序先转float64，避免uint8累加时溢出，并保留小数平均值。
窗口半径r=k//2，使用np.pad(...,mode="edge")复制最近边缘像素补r圈。
因此输出与输入大小相同，图像四周的窗口也有k²个值。
直接实现时间复杂度为O(HWk²)；k固定时随像素总数线性增长。
补边数组占O((H+k)(W+k))空间，另有O(HW)输出数组；没有构造完整四维滑窗副本。

mean_filter(gray,k) 接受非空二维uint8数组，返回同形状float64数组。
to_uint8()在保存前用np.rint取最近整数、截断到0～255，再转uint8。
np.rint在恰好x.5时采用“取偶数”的规则，例如2.5变2、3.5变4。
本演示MSE与PSNR都基于最后保存的uint8图片，保证文件与指标一致。

五、手算一个像素
3×3邻域为：
   10  20  30
   40 200  60
   70  80  90

只计算中心：
总和 = 10+20+30+40+200+60+70+80+90 = 600。
平均值 = 600/9 = 66.6667。
保存为8位图片后，中心为67。
中心原来是200，周围像素较低，取平均后这个较亮值被压低。
这可以减弱孤立亮度扰动，也说明真实的细亮线为何会被模糊。

六、默认数据与结果文件
默认图为320×240像素，背景灰度70，矩形200、圆150、三角形210；
左下角有五条宽2像素的细线，用于观察细节损失。
用np.random.default_rng(42)生成均值0、标准差18的高斯噪声，加入干净图后截断并取整。
均值0和标准差18指截断取整前的噪声分布，实际保存的噪声统计会略有变化。
所有窗口使用同一张噪声输入；没有为不同窗口重新抽样。

results/clean.png          干净参考图，仅默认合成数据生成
results/input.png          实际参与滤波的灰度输入
results/original.png       仅自选图片时额外保存校正后的RGB输入
results/filtered_k3.png    3×3均值滤波结果
results/filtered_k5.png    5×5均值滤波结果
results/filtered_k9.png    9×9均值滤波结果
results/comparison.png     原图与多个窗口结果对照
results/metrics.csv        输入基线与各窗口的MSE、PSNR、耗时统计
results/run_info.json      实际参数、边界模式、软件版本和完整指标

对照图默认从左到右、从上到下：干净图、噪声图、k=3、k=5、k=9、指定窗口的局部放大。
Clean reference=干净参考；Noisy input=噪声输入；selected=指定窗口；detail (zoom)=局部放大。
放大格默认查看左下细线区域，采用最近邻显示以便观察像素，不代表滤波后的额外处理。
标签使用英文以避免额外中文字体依赖。

支持普通8位灰度或彩色PNG/JPEG输入；彩色图转灰度后滤波。
照片先校正EXIF方向；透明图先合成白底；多帧图仅使用第一帧。
16位整数图、浮点图不在本入门演示范围内，请先转换为普通RGB PNG/JPEG。

七、评价与默认实测
MSE = 所有像素的(结果-干净参考)²的平均值，越小越好，单位为灰度级平方。
PSNR = 10×log10(255²/MSE)，单位dB；对于同一干净参考，越大通常表示像素误差越小。
二者均在整张图片上计算，包括复制补边处理过的边缘位置。
本实验并未测量主观视觉质量或专门的边缘保持指标。

默认seed=42、sigma=18的实际结果：
未滤波输入：MSE=325.565，PSNR=23.004 dB。
k=3：MSE=102.625，PSNR=28.018 dB。
k=5：MSE=152.020，PSNR=26.312 dB。
k=9：MSE=227.035，PSNR=24.570 dB。

这张图上，三个窗口的MSE都低于噪声输入，但k=3的整体误差最小。
较大窗口在平坦区平均更多噪声，同时把边缘两侧的灰度混合，细线也被抹平。
因此窗口越大不保证PSNR越高；本例的k=3也不是所有图片的最佳窗口。
在理想平坦区且噪声独立、零均值、未经过截断取整时，平均k²个噪声可使方差降为原来的1/k²。
该性质不能直接用于真实边缘处，因为邻域中的真实灰度并不相同；复制边界还会重复使用边缘样本。

MSE=0时PSNR数学上为正无穷。终端显示+inf；JSON用psnr_db=null且exact_match=true表示。
没有干净参考的自选图片不计算MSE/PSNR，终端为N/A，JSON中mse、psnr_db、exact_match均为null。
CSV首行case=input为未滤波基线，其后case=filtered为各窗口。
CSV中对应的空指标表示没有参考或无限PSNR，结合exact_match列判断原因。

每个窗口先预热1次，再测20次，记录中位数和第25、75百分位，单位ms。
计时仅包含mean_filter函数的输入检查、转换、补边、累加及除法。
不包含转回uint8、读图、绘图、压缩或保存；运行时间会随机器和系统负载变化。

八、建议讲解顺序（约20分钟）
0～3分钟：看干净图与噪声图，理解局部随机亮度扰动。
3～7分钟：画3×3平均核，手算中心66.6667，说明保存后为67。
7～11分钟：读mean_filter，解释浮点累加、复制补边、切片与原输入保持不变。
11～16分钟：运行默认演示，比较k=3/5/9，观察平坦区、边缘及细线。
16～20分钟：调整窗口或噪声强度，讨论MSE、PSNR与细节损失之间的关系。

九、常见问题与下一步
1. 提示缺少numpy/PIL：用运行代码的同一个Python安装requirements.txt；安装包名Pillow，导入名PIL。
2. 图片打不开：检查实际路径、扩展名与双引号。
3. 参数被拒绝：窗口只能是1～31的奇数；sigma必须是非负有限数；seed必须非负。
4. 没有弹出窗口：默认只保存文件；打开comparison.png，或增加--show。
5. 效果太模糊：尝试减小窗口；重复滤波也会继续混合边缘，不要把它当作自动提升质量。
6. 自选图片没有PSNR：没有与它对应的干净参考，无法据此计算恢复误差。
7. 结果仍有噪声：均值滤波只是一种基础平滑方法，较大窗口也有细节损失。
后续可学习高斯滤波、中值滤波、双边滤波，或用可分离盒式滤波/积分图加速均值计算。

十、官方参考
OpenCV平滑与平均核原理（程序不依赖OpenCV）：
https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html
NumPy pad边界模式：
https://numpy.org/doc/stable/reference/generated/numpy.pad.html
