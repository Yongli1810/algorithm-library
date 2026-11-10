0/1 背包动态规划 —— 本周算法演示完整复现包
日期：2026-11-06

一、交付内容
1. Knapsack_DP_Algorithm_Demo.docx — 按《算法讲解与代码演示通用模板》填写的可编辑 Word 文档（6页）。
2. Knapsack_DP_Algorithm_Demo.pdf  — 对应 Word 的 PDF 阅读版（6页）。
3. Knapsack_DP_Demo_Video.mp4       — 1280×720 的 MP4 动态讲解视频，带中文字幕和离线合成普通话配音。
4. knapsack_demo.py               — 完整 Python 程序（二维DP+回溯、一维DP、贪心对照、穷举验证）。
5. results/                       — 真实运行的 DP 状态表、对比实验 CSV / JSON / PNG。
6. video_narration_script.txt     — 视频每节讲解词，便于替换为真人配音。
7. make_video.py                  — 视频制作源码；需要额外安装 Pillow、espeak 与 ffmpeg。

二、如何运行算法
安装 Python 3.10+，打开本压缩包解压后的文件夹。
运行：
    python knapsack_demo.py
运行结束后，请查看 results/ 目录下的数据与图表。
如果要重新生成 comparison.png，请先运行：
    pip install matplotlib

三、教学案例
微型手算：4 件物品、背包容量 8kg，最优选择 B+D，重量 8kg，效用 12 分。
正式演示：6 件物资、背包容量 12kg。
二维 DP / 一维 DP / 穷举：最优值 18 分，选择“急救箱+通信设备”（5+7=12kg）。
价值密度贪心：15 分。DP 比贪心高 3 分。
所有物品数据与效用分值均为模拟教学数据，不代表真实应急物资优先级。

四、视频形式
参考用户提供的 A* 演示视频的教学表达方式，采用浅色卡片画面、章节说明、底部中文字幕、代码分屏说明以及动态逐格填表演示。
配音使用离线语音合成（非真人录音），实际结果数值来自本次程序运行。
视频是一份独立可播放的 MP4 文件；不需要运行脚本即可观看。

五、文件使用提示
若下载时浏览器不支持中文文件名，可优先下载英文文件名的整包 ZIP。
如压缩软件显示文件名乱码，请使用 Windows 资源管理器或 7-Zip。
