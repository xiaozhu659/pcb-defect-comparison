"""把定稿结果图整理到 docs/figures/,供文档与仓库首页引用。

为什么不能直接把 outputs/ 提交上去
--------------------------------
outputs/ 是每次运行的工作目录:混淆矩阵、逐类统计、中间图都会被下一次运行覆盖,
体积也偏大(并排对比图有 4 MB)。仓库里只保留一份"定稿图",既能说明结果,
又不会让仓库变脏。分工是:

    outputs/        工作目录,不入库
    docs/figures/   定稿图,入库并在文档中引用

用法:
    uv run scripts/collect_figures.py
"""

from pathlib import Path

from PIL import Image

from pcb_defect_comparison.common import PROJECT_ROOT

OUTPUTS = PROJECT_ROOT / "outputs"
RUNS = PROJECT_ROOT / "runs" / "detect" / "yolo11n_1024"
TARGET = PROJECT_ROOT / "docs" / "figures"

# 源文件 -> 入库文件名
FIGURES = {
    OUTPUTS / "fig_class_distribution.png": "class_distribution.png",
    OUTPUTS / "fig_box_size.png": "box_size.png",
    RUNS / "results.png": "yolo_training.png",
    RUNS / "confusion_matrix.png": "yolo_confusion_matrix.png",
    OUTPUTS / "fig_method_comparison.png": "method_comparison.png",
    OUTPUTS / "fig_side_by_side.png": "side_by_side.png",
    OUTPUTS / "fig_sam_segmentation.png": "sam_segmentation.png",
}

MAX_WIDTH = 2400


def save_compact(source: Path, target: Path) -> tuple[int, int]:
    """缩放并压缩后保存,返回 (原始KB, 入库KB)。"""
    with Image.open(source) as image:
        image = image.convert("RGB")
        if image.width > MAX_WIDTH:
            height = round(image.height * MAX_WIDTH / image.width)
            image = image.resize((MAX_WIDTH, height), Image.LANCZOS)
        image.save(target, format="PNG", optimize=True)
    return source.stat().st_size // 1024, target.stat().st_size // 1024


def main() -> None:
    TARGET.mkdir(parents=True, exist_ok=True)
    copied = 0
    for source, name in FIGURES.items():
        if not source.exists():
            print(f"[跳过] 找不到 {source.relative_to(PROJECT_ROOT)}")
            continue
        before, after = save_compact(source, TARGET / name)
        print(f"  {source.name:28s} -> docs/figures/{name:26s} {before:5d} -> {after:4d} KB")
        copied += 1
    total = sum(p.stat().st_size for p in TARGET.glob("*.png")) / 1024
    print(f"\n已整理 {copied} 张图,合计 {total:.0f} KB")


if __name__ == "__main__":
    main()
