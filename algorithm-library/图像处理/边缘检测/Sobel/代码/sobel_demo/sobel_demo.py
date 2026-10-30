# -*- coding: utf-8 -*-
"""Sobel 边缘检测教学演示：用两个 3×3 核计算亮度变化。

python sobel_demo.py
python sobel_demo.py --threshold 200 --output-dir results_200
python sobel_demo.py --input "自己的图片.jpg" --threshold 100
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import platform
from time import perf_counter_ns

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps, __version__ as pillow_version

# 不翻转核，按下列权重作互相关；x 向右，y 向下。
KX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
KY = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)


def sobel_gradients(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """输入二维 uint8 灰度图；返回有符号 Gx、Gy 与非负幅值 M。"""
    if not isinstance(gray, np.ndarray) or gray.ndim != 2 or gray.dtype != np.uint8 or gray.size == 0:
        raise ValueError("输入必须是非空的二维 uint8 灰度数组")
    # 先转浮点，避免 uint8 做减法时溢出；边缘复制补一圈。
    padded = np.pad(gray.astype(np.float32), 1, mode="edge")
    height, width = gray.shape
    gx = np.zeros((height, width), dtype=np.float32)
    gy = np.zeros_like(gx)
    # 九个核位置逐个累加，但每次用切片同时处理整张图。
    for row in range(3):
        for col in range(3):
            neighbor = padded[row:row + height, col:col + width]
            gx += KX[row, col] * neighbor
            gy += KY[row, col] * neighbor
    magnitude = np.hypot(gx, gy)  # sqrt(gx**2 + gy**2)
    return gx, gy, magnitude


def detect_edges(gray: np.ndarray, threshold: float = 100.0) -> np.ndarray:
    """在原始梯度幅值上判断 M>T；255 为边缘，0 为非边缘。"""
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("阈值必须是非负有限数")
    _, _, magnitude = sobel_gradients(gray)
    return np.where(magnitude > threshold, 255, 0).astype(np.uint8)


def make_sample(seed: int = 42) -> Image.Image:
    """生成不同对比度的矩形、圆和三角形；加入 -8～8 的整数噪声。"""
    height, width = 240, 320
    yy, xx = np.indices((height, width))
    gray = np.full((height, width), 35.0)
    gray[(xx >= 30) & (xx < 115) & (yy >= 40) & (yy < 135)] = 220
    gray[(xx - 220) ** 2 + (yy - 85) ** 2 <= 45 ** 2] = 150
    triangle = (yy >= 155) & (yy < 220) & (np.abs(xx - 160) <= (yy - 155))
    gray[triangle] = 90
    noise = np.random.default_rng(seed).integers(-8, 9, size=gray.shape)
    return Image.fromarray(np.clip(gray + noise, 0, 255).astype(np.uint8)).convert("RGB")


def load_image(path: Path) -> Image.Image:
    """读取普通 8 位图片，校正照片方向，透明区域先合成到白底。"""
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened)
        if image.mode in ("I", "F") or image.mode.startswith("I;16"):
            raise ValueError("请先把 16 位整数图或浮点图转换为普通 RGB PNG/JPEG")
        rgba = image.convert("RGBA")
        return Image.alpha_composite(Image.new("RGBA", rgba.size, "white"), rgba).convert("RGB")


def display_gradient(values: np.ndarray) -> Image.Image:
    """只用于显示：绝对值除以固定 4 后截断到 0～255，不参与阈值判断。"""
    return Image.fromarray(np.clip(np.abs(values) / 4, 0, 255).astype(np.uint8))


def comparison_sheet(original: Image.Image, gx: np.ndarray, gy: np.ndarray,
                     masks: dict[float, np.ndarray], selected: float) -> Image.Image:
    panels = [("Original", original), ("|Gx| / 4 (display)", display_gradient(gx)),
              ("|Gy| / 4 (display)", display_gradient(gy))]
    panels += [(f"T={t:g}" + (" (selected)" if t == selected else ""), Image.fromarray(mask))
               for t, mask in masks.items()]
    columns, tw, th = 3, 340, 282
    sheet = Image.new("RGB", (columns * tw, ((len(panels) + columns - 1) // columns) * th), "#edf1f5")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=18)
    for index, (label, picture) in enumerate(panels):
        x, y = (index % columns) * tw, (index // columns) * th
        draw.text((x + 10, y + 6), label, font=font, fill="#17364b")
        preview = ImageOps.contain(picture.convert("RGB"), (320, 240))
        sheet.paste(preview, (x + 10 + (320 - preview.width) // 2,
                             y + 32 + (240 - preview.height) // 2))
    return sheet


def show_micro_example(threshold: float) -> None:
    patch = np.array([[0, 0, 100], [0, 0, 100], [0, 0, 100]], dtype=np.float32)
    gx, gy = float(np.sum(patch * KX)), float(np.sum(patch * KY))
    magnitude = float(np.hypot(gx, gy))
    print("\n手算案例：只计算这个 3×3 邻域的中心像素")
    print(patch.astype(int))
    print("Kx:\n", KX.astype(int), "\nKy:\n", KY.astype(int))
    print(f"Gx=100+200+100={gx:g}，Gy={gy:g}，M={magnitude:g}")
    print(f"M>{threshold:g} 为 {magnitude > threshold}，中心输出 "
          f"{255 if magnitude > threshold else 0}\n")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="可选图片路径；不设置则自动生成示例")
    parser.add_argument("--threshold", type=float, default=100, help="原始梯度幅值阈值，非负，默认100")
    parser.add_argument("--seed", type=int, default=42, help="示例噪声的非负随机种子，默认42")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--show", action="store_true", help="运行后用系统图片查看器打开对照图")
    args = parser.parse_args()
    if not np.isfinite(args.threshold) or args.threshold < 0:
        parser.error("--threshold 必须是非负有限数；它不是 0～255 的原图灰度阈值")
    if args.seed < 0:
        parser.error("--seed 必须是非负整数")
    return args


def main() -> int:
    args = parse_arguments()
    try:
        original = make_sample(args.seed) if args.input is None else load_image(args.input)
        gray = np.array(original.convert("L"), dtype=np.uint8)
        show_micro_example(args.threshold)
        gx, gy, magnitude = sobel_gradients(gray)
        output = args.output_dir.resolve()
        output.mkdir(parents=True, exist_ok=True)
        original.save(output / "original.png")
        Image.fromarray(gray).save(output / "gray.png")
        display_gradient(gx).save(output / "gx_display.png")
        display_gradient(gy).save(output / "gy_display.png")
        display_gradient(magnitude).save(output / "magnitude_display.png")
        # PNG 只用于可视化，NPZ 保留未缩放、未截断的浮点计算结果。
        np.savez_compressed(output / "gradients.npz", gx=gx, gy=gy, magnitude=magnitude)
        masks, records = {}, []
        for threshold in sorted({30.0, args.threshold, 600.0}):
            mask = np.where(magnitude > threshold, 255, 0).astype(np.uint8)
            masks[threshold] = mask
            detect_edges(gray, threshold)  # 先预热一次
            elapsed = []
            for _ in range(30):
                started = perf_counter_ns()
                detect_edges(gray, threshold)
                elapsed.append((perf_counter_ns() - started) / 1_000_000)
            edge_pixels = int(np.count_nonzero(mask))
            record = {"threshold": threshold, "edge_pixels": edge_pixels,
                      "edge_ratio": edge_pixels / gray.size,
                      "median_ms": float(np.median(elapsed)),
                      "p25_ms": float(np.percentile(elapsed, 25)),
                      "p75_ms": float(np.percentile(elapsed, 75))}
            records.append(record)
            Image.fromarray(mask).save(output / f"edges_T{threshold:g}.png")
            print(f'T={threshold:g} | 边缘像素={edge_pixels} | '
                  f'占比={record["edge_ratio"]:.2%} | 中位耗时={record["median_ms"]:.3f} ms')
        sheet = comparison_sheet(original, gx, gy, masks, args.threshold)
        sheet.save(output / "comparison.png")
        with (output / "metrics.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        info = {"source": "generated geometric sample" if args.input is None else str(args.input.resolve()),
                "shape": list(gray.shape), "seed": args.seed if args.input is None else None,
                "selected_threshold": args.threshold, "padding": "edge", "kernel_operation": "correlation",
                "display_scale": 4, "timing_repeats": 30, "timing_scope": "Sobel + threshold; excludes IO",
                "magnitude_max": float(magnitude.max()), "gx_min": float(gx.min()), "gx_max": float(gx.max()),
                "gy_min": float(gy.min()), "gy_max": float(gy.max()),
                "python": platform.python_version(), "numpy": np.__version__, "pillow": pillow_version,
                "platform": platform.platform(), "processor": platform.processor(), "results": records}
        (output / "run_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n完成。请打开：{output / 'comparison.png'}")
        if args.show:
            sheet.show()
        return 0
    except (OSError, ValueError) as error:
        print(f"运行失败：{error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
