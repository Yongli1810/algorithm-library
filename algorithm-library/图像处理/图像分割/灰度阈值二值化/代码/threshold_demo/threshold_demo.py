# -*- coding: utf-8 -*-
"""灰度阈值二值化教学演示。

运行：python threshold_demo.py
调参：python threshold_demo.py --threshold 160
自选图：python threshold_demo.py --input "自己的图片.jpg" --threshold 127
仅依赖 NumPy 和 Pillow；默认生成数据并保存图片，不要求桌面窗口。
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import platform
from time import perf_counter_ns

import numpy as np
from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError, __version__ as pillow_version


def binary_threshold(gray: np.ndarray, threshold: int = 127) -> np.ndarray:
    """对二维 uint8 灰度图逐像素判断：大于 T 为 255，否则为 0。"""
    if gray.ndim != 2 or gray.dtype != np.uint8 or gray.size == 0:
        raise ValueError("输入必须是非空的二维 uint8 灰度数组")
    if not isinstance(threshold, (int, np.integer)) or not 0 <= threshold <= 255:
        raise ValueError("阈值必须是 0～255 的整数")
    # 核心算法：条件为真选择 255，条件为假选择 0。
    return np.where(gray > threshold, 255, 0).astype(np.uint8)


def make_sample(seed: int = 42) -> tuple[Image.Image, np.ndarray]:
    """生成 320×240 的亮目标暗背景图，并返回独立的真实前景掩膜。"""
    height, width = 240, 320
    yy, xx = np.indices((height, width))
    # 三种几何形状的位置先定义，真值不依赖阈值分割结果。
    rectangle = (xx >= 30) & (xx < 115) & (yy >= 40) & (yy < 135)
    circle = (xx - 220) ** 2 + (yy - 85) ** 2 <= 45 ** 2
    triangle = (yy >= 155) & (yy < 220) & (np.abs(xx - 160) <= (yy - 155))
    truth = rectangle | circle | triangle
    gray = 45.0 + 65.0 * xx / (width - 1)
    gray[rectangle] = 180.0
    gray[circle] = 215.0
    gray[triangle] = 235.0
    noise = np.random.default_rng(seed).integers(-8, 9, size=(height, width))
    gray = np.clip(np.rint(gray) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(gray).convert("RGB"), truth


def load_image(path: Path) -> Image.Image:
    """读取普通 8 位图像；校正照片方向，并将透明区域合成到白底。"""
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened)
        if image.mode in ("I", "F") or image.mode.startswith("I;16"):
            raise ValueError("本入门演示仅接受普通 8 位图片，请先转换为 RGB PNG/JPEG")
        rgba = image.convert("RGBA")
        base = Image.new("RGBA", rgba.size, "white")
        return Image.alpha_composite(base, rgba).convert("RGB")


def evaluate(binary: np.ndarray, truth: np.ndarray | None) -> dict:
    """统计白色比例；只有存在独立真值时才计算 IoU、误检和漏检。"""
    predicted = binary == 255
    values = {"white_ratio": float(np.mean(predicted))}
    if truth is None:
        values.update(iou=None, false_positive=None, false_negative=None)
    else:
        intersection = int(np.count_nonzero(predicted & truth))
        union = int(np.count_nonzero(predicted | truth))
        values.update(
            iou=intersection / union if union else 1.0,
            false_positive=int(np.count_nonzero(predicted & ~truth)),
            false_negative=int(np.count_nonzero(~predicted & truth)),
        )
    return values


def make_comparison(original: Image.Image, gray: np.ndarray,
                    masks: dict[int, np.ndarray], selected: int) -> Image.Image:
    """输出原图、灰度图与不同阈值的对照图，图内英文避免字体依赖。"""
    panels = [("Original", original), ("Grayscale (0-255)", Image.fromarray(gray))]
    panels += [(f"T = {t}" + (" (selected)" if t == selected else ""), Image.fromarray(b))
               for t, b in masks.items()]
    columns, tile_width, tile_height = 3, 340, 286
    rows = (len(panels) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * tile_width, rows * tile_height), "#edf1f5")
    draw = ImageDraw.Draw(sheet)
    for index, (title, picture) in enumerate(panels):
        x, y = (index % columns) * tile_width, (index // columns) * tile_height
        draw.text((x + 10, y + 9), title, fill="#152e43")
        preview = ImageOps.contain(picture.convert("RGB"), (320, 240))
        sheet.paste(preview, (x + 10 + (320 - preview.width) // 2,
                             y + 32 + (240 - preview.height) // 2))
    return sheet


def show_micro_example(threshold: int) -> None:
    """逐步打印六个像素，便于将代码与手工推演对应。"""
    micro = np.array([[20, 100, 127], [128, 180, 240]], dtype=np.uint8)
    result = binary_threshold(micro, threshold)
    print(f"\n微型案例，T = {threshold}：\n{micro}\n输出：\n{result}")
    for (row, col), value in np.ndenumerate(micro):
        print(f"  ({row}, {col})：{int(value)} > {threshold} = "
              f"{bool(value > threshold)} -> {int(result[row, col])}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="可选的本地图片路径；省略时自动生成示例")
    parser.add_argument("--threshold", type=int, default=127, help="整数阈值 0～255，默认 127")
    parser.add_argument("--seed", type=int, default=42, help="合成数据的非负随机种子，默认 42")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "results",
                        help="输出目录，默认在脚本同级的 results 文件夹")
    parser.add_argument("--show", action="store_true", help="额外用系统图片查看器打开对照图")
    args = parser.parse_args()
    if not 0 <= args.threshold <= 255:
        parser.error("--threshold 必须在 0～255 之间")
    if args.seed < 0:
        parser.error("--seed 必须为非负整数")
    return args


def main() -> int:
    args = parse_arguments()
    try:
        if args.input is None:
            original, truth = make_sample(args.seed)
            source = "generated geometric sample"
        else:
            original, truth = load_image(args.input), None
            source = str(args.input.resolve())
        # Pillow 的 L 模式是 8 位灰度，不直接转 1 模式以避免默认抖动。
        gray = np.array(original.convert("L"), dtype=np.uint8)
        show_micro_example(args.threshold)
        output = args.output_dir.resolve()
        output.mkdir(parents=True, exist_ok=True)
        original.save(output / "original.png")
        Image.fromarray(gray).save(output / "gray.png")
        if truth is not None:
            Image.fromarray((truth * 255).astype(np.uint8)).save(output / "ground_truth.png")

        masks, records = {}, []
        thresholds = sorted({60, args.threshold, 200})
        for threshold in thresholds:
            # 先预热，再计时 30 次；仅计阈值函数，排除读图、画图、写盘。
            binary = binary_threshold(gray, threshold)
            elapsed = []
            for _ in range(30):
                started = perf_counter_ns()
                binary = binary_threshold(gray, threshold)
                elapsed.append((perf_counter_ns() - started) / 1_000_000)
            masks[threshold] = binary
            Image.fromarray(binary).save(output / f"binary_T{threshold}.png")
            record = {"threshold": threshold, **evaluate(binary, truth),
                      "median_ms": float(np.median(elapsed)),
                      "p25_ms": float(np.percentile(elapsed, 25)),
                      "p75_ms": float(np.percentile(elapsed, 75))}
            records.append(record)
            iou = "N/A（自选图片无真值）" if record["iou"] is None else f'{record["iou"]:.4f}'
            print(f'T={threshold:3d} | 白色比例={record["white_ratio"]:.2%} | '
                  f'IoU={iou} | 中位耗时={record["median_ms"]:.4f} ms')

        sheet = make_comparison(original, gray, masks, args.threshold)
        sheet.save(output / "comparison.png")
        # UTF-8 BOM 便于在 Windows Excel 中打开。
        with (output / "metrics.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        metadata = {"source": source, "shape": list(gray.shape), "threshold": args.threshold,
                    "seed": args.seed if truth is not None else None, "has_ground_truth": truth is not None,
                    "python": platform.python_version(), "numpy": np.__version__,
                    "pillow": pillow_version, "platform": platform.platform(),
                    "processor": platform.processor(), "timing_repeats": 30, "results": records}
        (output / "run_info.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n完成。请打开：{output / 'comparison.png'}")
        if args.show:
            sheet.show()
        return 0
    except (OSError, ValueError, UnidentifiedImageError) as error:
        print(f"运行失败：{error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
