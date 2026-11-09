# -*- coding: utf-8 -*-
"""均值滤波去噪教学演示：把邻域像素的平均值写到中心位置。

python mean_filter_demo.py
python mean_filter_demo.py --kernel-size 7 --output-dir results_k7
python mean_filter_demo.py --input "自己的图片.jpg" --kernel-size 3
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


def validate_kernel_size(k: int) -> None:
    """本入门实现接受 1～31 的奇数边长；k=1 表示保持原图。"""
    if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)):
        raise ValueError("窗口边长 k 必须是 1～31 的奇数整数")
    if not 1 <= k <= 31 or k % 2 == 0:
        raise ValueError("窗口边长 k 必须是 1～31 的奇数整数")


def mean_filter(gray: np.ndarray, k: int = 3) -> np.ndarray:
    """输入非空二维 uint8 灰度图；返回同尺寸 float64 均值结果。"""
    if not isinstance(gray, np.ndarray) or gray.ndim != 2 or gray.dtype != np.uint8 or gray.size == 0:
        raise ValueError("输入必须是非空的二维 uint8 灰度数组")
    validate_kernel_size(k)
    radius = k // 2
    # 转浮点后累加，避免 uint8 求和溢出；复制边界像素补齐窗口。
    padded = np.pad(gray.astype(np.float64), radius, mode="edge")
    height, width = gray.shape
    total = np.zeros((height, width), dtype=np.float64)
    # k² 个窗口位置分别累加，每次切片同时处理全部像素。
    for row in range(k):
        for col in range(k):
            total += padded[row:row + height, col:col + width]
    return total / (k * k)


def to_uint8(values: np.ndarray) -> np.ndarray:
    """保存图片前四舍六入五成双到整数，再截断到 0～255。"""
    return np.clip(np.rint(values), 0, 255).astype(np.uint8)


def make_sample(seed: int = 42, sigma: float = 18.0) -> tuple[np.ndarray, np.ndarray]:
    """先生成干净几何图，再加高斯噪声，返回 clean 与 noisy 两个 uint8 数组。"""
    if not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("随机种子必须是非负整数")
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError("噪声标准差 sigma 必须是非负有限数")
    height, width = 240, 320
    yy, xx = np.indices((height, width))
    clean = np.full((height, width), 70, dtype=np.uint8)
    clean[(xx >= 30) & (xx < 115) & (yy >= 40) & (yy < 135)] = 200
    clean[(xx - 220) ** 2 + (yy - 85) ** 2 <= 45 ** 2] = 150
    clean[(yy >= 155) & (yy < 220) & (np.abs(xx - 160) <= yy - 155)] = 210
    # 左下角的细线用来展示：平滑噪声时，细节也可能被抹平。
    for x in [30, 43, 56, 69, 82]:
        clean[165:220, x:x + 2] = 200
    noise = np.random.default_rng(seed).normal(0.0, sigma, size=clean.shape)
    noisy = to_uint8(clean.astype(np.float64) + noise)
    return clean, noisy


def load_image(path: Path) -> Image.Image:
    """读取普通 8 位图片，校正照片方向，透明区域先合成到白底。"""
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened)
        if image.mode in ("I", "F") or image.mode.startswith("I;16"):
            raise ValueError("请先把 16 位整数图或浮点图转换为普通 RGB PNG/JPEG")
        rgba = image.convert("RGBA")
        white = Image.new("RGBA", rgba.size, "white")
        return Image.alpha_composite(white, rgba).convert("RGB")


def evaluate(result: np.ndarray, reference: np.ndarray | None) -> dict:
    """对实际保存的 uint8 图片评价；无干净参考时不猜测恢复质量。"""
    if reference is None:
        return {"mse": None, "psnr_db": None, "exact_match": None}
    if result.shape != reference.shape:
        raise ValueError("评价图片与干净参考必须同尺寸")
    difference = result.astype(np.float64) - reference.astype(np.float64)
    mse = float(np.mean(difference ** 2))
    # MSE=0 时 PSNR 为正无穷，JSON 用 null + exact_match=true 合法表示。
    return {"mse": mse, "psnr_db": None if mse == 0 else float(10 * np.log10(255.0 ** 2 / mse)),
            "exact_match": mse == 0}


def make_comparison(clean: np.ndarray | None, input_gray: np.ndarray,
                    original: Image.Image, filtered: dict[int, np.ndarray],
                    selected: int) -> Image.Image:
    """默认六格：干净图、噪声图、三个窗口结果、指定窗口的细节放大。"""
    if clean is None:
        panels = [("Original", original), ("Grayscale input", Image.fromarray(input_gray))]
    else:
        panels = [("Clean reference", Image.fromarray(clean)), ("Noisy input", Image.fromarray(input_gray))]
    panels += [(f"k={k}" + (" (selected)" if k == selected else ""), Image.fromarray(array))
               for k, array in filtered.items()]
    selected_image = Image.fromarray(filtered[selected])
    if selected_image.width >= 100 and selected_image.height >= 100:
        # 默认图恰好放大左下角细线；其它图片用相对位置的同类区域。
        box = (int(selected_image.width * 0.075), int(selected_image.height * 0.65),
               int(selected_image.width * 0.30), int(selected_image.height * 0.95))
    else:
        box = (0, 0, selected_image.width, selected_image.height)
    panels.append((f"k={selected}: detail (zoom)", selected_image.crop(box)))
    columns, tile_w, tile_h = 3, 340, 282
    sheet = Image.new("RGB", (columns * tile_w, ((len(panels) + columns - 1) // columns) * tile_h), "#edf1f5")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=18)
    for index, (label, picture) in enumerate(panels):
        x, y = (index % columns) * tile_w, (index // columns) * tile_h
        draw.text((x + 10, y + 6), label, font=font, fill="#17364b")
        preview = ImageOps.contain(picture.convert("RGB"), (320, 240), Image.Resampling.NEAREST)
        sheet.paste(preview, (x + 10 + (320 - preview.width) // 2,
                             y + 32 + (240 - preview.height) // 2))
    return sheet


def print_metric(label: str, metric: dict) -> None:
    if metric["exact_match"] is True:
        quality = "MSE=0，PSNR=+inf（与参考完全相同）"
    elif metric["mse"] is None:
        quality = "MSE/PSNR=N/A（没有干净参考图）"
    else:
        quality = f'MSE={metric["mse"]:.3f}，PSNR={metric["psnr_db"]:.3f} dB'
    timing = "" if "median_ms" not in metric else f'，核心中位耗时={metric["median_ms"]:.3f} ms'
    print(f"{label}：{quality}{timing}")


def show_micro_example() -> None:
    patch = np.array([[10, 20, 30], [40, 200, 60], [70, 80, 90]], dtype=np.uint8)
    filtered = mean_filter(patch, 3)
    print("\n手算案例：只计算下面 3×3 邻域的中心像素")
    print(patch)
    print(f"邻域和=600，均值=600/9={filtered[1, 1]:.4f}，保存为 {to_uint8(filtered)[1, 1]}\n")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel-size", type=int, default=3, help="窗口边长：1～31 的奇数，默认3")
    parser.add_argument("--sigma", type=float, default=18.0, help="仅控制合成图的噪声标准差，默认18")
    parser.add_argument("--seed", type=int, default=42, help="合成噪声的非负随机种子，默认42")
    parser.add_argument("--input", type=Path, help="可选本地图片；直接滤波，不再人为加噪")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--show", action="store_true", help="额外用系统图片查看器打开对照图")
    args = parser.parse_args()
    try:
        validate_kernel_size(args.kernel_size)
    except ValueError as error:
        parser.error(str(error))
    if not np.isfinite(args.sigma) or args.sigma < 0:
        parser.error("--sigma 必须是非负有限数")
    if args.seed < 0:
        parser.error("--seed 必须是非负整数")
    return args


def main() -> int:
    args = parse_arguments()
    try:
        if args.input is None:
            clean, input_gray = make_sample(args.seed, args.sigma)
            original = Image.fromarray(input_gray).convert("RGB")
            source = "generated geometric sample"
        else:
            original = load_image(args.input)
            input_gray = np.array(original.convert("L"), dtype=np.uint8)
            clean, source = None, str(args.input.resolve())
        show_micro_example()
        output = args.output_dir.resolve()
        output.mkdir(parents=True, exist_ok=True)
        Image.fromarray(input_gray).save(output / "input.png")
        if clean is not None:
            Image.fromarray(clean).save(output / "clean.png")
        else:
            original.save(output / "original.png")
        baseline = evaluate(input_gray, clean)
        print_metric("未滤波输入", baseline)
        filtered, records = {}, []
        repeats = 20
        for k in sorted({3, 5, 9, args.kernel_size}):
            floating = mean_filter(input_gray, k)  # 预热一次
            elapsed = []
            for _ in range(repeats):
                started = perf_counter_ns()
                floating = mean_filter(input_gray, k)
                elapsed.append((perf_counter_ns() - started) / 1_000_000)
            result = to_uint8(floating)
            filtered[k] = result
            Image.fromarray(result).save(output / f"filtered_k{k}.png")
            record = {"kernel_size": k, **evaluate(result, clean),
                      "median_ms": float(np.median(elapsed)),
                      "p25_ms": float(np.percentile(elapsed, 25)),
                      "p75_ms": float(np.percentile(elapsed, 75))}
            records.append(record)
            print_metric(f"k={k}", record)
        sheet = make_comparison(clean, input_gray, original, filtered, args.kernel_size)
        sheet.save(output / "comparison.png")
        fieldnames = ["case", "kernel_size", "mse", "psnr_db", "exact_match", "median_ms", "p25_ms", "p75_ms"]
        with (output / "metrics.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow({"case": "input", **baseline})
            writer.writerows({"case": "filtered", **record} for record in records)
        info = {"source": source, "shape": list(input_gray.shape), "has_reference": clean is not None,
                "seed": args.seed if clean is not None else None, "sigma": args.sigma if clean is not None else None,
                "selected_kernel_size": args.kernel_size, "padding": "edge", "compute_dtype": "float64",
                "save_rounding": "numpy.rint (ties to even), clip to 0..255, cast uint8",
                "timing_repeats": repeats, "timing_scope": "mean_filter only; excludes uint8 conversion, IO and visualization",
                "metric_scope": "saved uint8 result against clean reference; full image including borders",
                "psnr_zero_mse_encoding": "psnr_db=null and exact_match=true means positive infinity",
                "python": platform.python_version(), "numpy": np.__version__, "pillow": pillow_version,
                "platform": platform.platform(), "processor": platform.processor(),
                "baseline": baseline, "results": records}
        (output / "run_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        print(f"\n完成。请打开：{output / 'comparison.png'}")
        if args.show:
            sheet.show()
        return 0
    except (OSError, ValueError) as error:
        print(f"运行失败：{error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
