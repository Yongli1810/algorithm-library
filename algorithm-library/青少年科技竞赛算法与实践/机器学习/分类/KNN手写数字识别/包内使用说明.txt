单个手写数字识别算法实验复现

功能
输入已经分割且按 UCI 协议归一化的 8×8 单个数字，输出 0—9 类别。
采用自主编写的 NumPy 距离加权 K 近邻，属于经典监督学习方法。

环境与运行
实测 Python 3.12.14，NumPy 2.3.5，Windows 10。
安装：python -m pip install numpy==2.3.5
运行：python run_experiment.py
绘制报告样本图需额外安装 Pillow，然后运行 python draw_samples.py。
算法运行不依赖 Pillow。

文件
knn_core.py：距离搜索、加权投票、recognize 函数。
run_experiment.py：数据读取、固定划分、验证集选参、测试集评价、自检。
data/optdigits.tra：官方训练集 3823 条。
data/optdigits.tes：官方测试集 1797 条。
data/optdigits.names：官方数据说明。
results/metrics.json：此次实测指标、耗时、文件哈希。
results/predictions.csv：全部测试预测和投票占比。
results/split_indices.npz：训练子集和验证集的零起始行下标。
results/visual_data.npz：样本图绘制输入与错误行下标。

协议
在官方训练集各类中按 floor(样本数×0.2) 选取验证数据，seed=42。
3062 条用于选参阶段拟合，761 条用于验证。
从 k∈{1,3,5,7,9} 中按验证准确率选参，同分选更小 k。
选定 k=3 后用全部官方训练数据作为样本库，评价官方测试集。
测试集未用于选参；5 次重复用于估计计时波动，识别结果相同。
算法仅输出投票占比，该值未经概率校准。
源码稳定排序解决等距离选择；同分预测较小数字。
运行时不可使用 python -O，因为自检使用 assert。

数据来源和署名
E. Alpaydin and C. Kaynak. Optical Recognition of Handwritten Digits.
UCI Machine Learning Repository, 1998. DOI: 10.24432/C50P49.
https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits
许可证：Creative Commons Attribution 4.0，CC BY 4.0。
https://creativecommons.org/licenses/by/4.0/
此包保留官方训练、测试 CSV 格式文件；读取时仅将 64 个像素除以 16。
分层划分由实验代码生成，数据集作者未参与本实验。
官方完整下载压缩包地址：
https://archive.ics.uci.edu/static/public/80/optical+recognition+of+handwritten+digits.zip

方法资料
https://scikit-learn.org/stable/modules/neighbors.html
资料用于核对近邻分类和距离权重定义，源码未调用 scikit-learn。

适用边界
仅评价这一公开数据集，未评价手机拍摄、连续多位数字或中文识别。
投票占比为 1 的样本也存在错误；实际系统需要独立校验与人工复核机制。
耗时与机器及数值库构建有关，跨机器不保证相同。
