NSGA-II 多目标优化演示（ZDT1）

内容：
  1. NSGA-II多目标遗传优化算法_讲解与代码演示.docx：依照用户模板编排的授课文档。
  2. nsga2_demo.py：可直接运行的完整 Python 程序。
  3. nsga2_results/：实际执行生成的 CSV、JSON 和 PNG。

环境：Python 3.11+，numpy==2.3.5，matplotlib==3.10.8。
安装：python -m pip install numpy==2.3.5 matplotlib==3.10.8
运行：python nsga2_demo.py --pop-size 80 --generations 180 --dims 10 --runs 3 --seed 42 --out nsga2_results

对比方法：完整 NSGA-II / 关闭变异消融 / 等评价次数随机搜索。
种子：42、43、44；每次方法的函数评价次数为 14,480。
指标：IGD 越小越好；二维 HV 越大越好，参考点 (1.1,1.1)。
注意：不同机器耗时不同；参数或环境变化可能导致数字不完全一致。
