"""PCB 缺陷数据集统计分析:类别分布、图像尺寸、缺陷框尺寸。

为什么先做这一步:缺陷框的大小直接决定后面的对比结论——
如果缺陷在原图中占比极小,大模型很可能"看不出来",而传统检测模型
经过训练后仍能定位到它们。这正是两种方案适用场景的分水岭。

用法:
    uv run scripts/analyze_dataset.py
"""

import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pcb_defect_comparison.common import (  # noqa: E402
    CLASS_ORDER,
    DATA_DIR,
    FIGURE_DIR,
    class_label,
    normalize_class,
    setup_chinese_font,
)


def parse_annotations() -> pd.DataFrame:
    """遍历所有 XML 标注,把每个缺陷框解析成一行。"""
    records = []
    for xml_path in sorted(DATA_DIR.rglob("*.xml")):
        try:
            root = ET.parse(xml_path).getroot()
        except ET.ParseError:
            continue
        size = root.find("size")
        width = int(size.findtext("width", "0"))
        height = int(size.findtext("height", "0"))
        image_name = root.findtext("filename", xml_path.stem + ".jpg")
        for obj in root.findall("object"):
            box = obj.find("bndbox")
            if box is None:
                continue
            x1 = float(box.findtext("xmin", "0"))
            y1 = float(box.findtext("ymin", "0"))
            x2 = float(box.findtext("xmax", "0"))
            y2 = float(box.findtext("ymax", "0"))
            records.append({
                "image": image_name,
                "folder": xml_path.parent.name,
                "class_name": normalize_class(obj.findtext("name", "unknown")),
                "img_width": width,
                "img_height": height,
                "box_w": abs(x2 - x1),
                "box_h": abs(y2 - y1),
            })
    return pd.DataFrame(records)


def plot_class_distribution(df: pd.DataFrame) -> None:
    images = df.groupby("class_name")["image"].nunique().reindex(CLASS_ORDER)
    boxes = df.groupby("class_name").size().reindex(CLASS_ORDER)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    labels = [class_label(c) for c in images.index]

    axes[0].bar(labels, images.values, color="#5b9bd5")
    axes[0].set_ylabel("包含该类缺陷的图片数")
    axes[0].set_title("各类缺陷的图片数")
    for i, v in enumerate(images.values):
        axes[0].text(i, v + 1, str(v), ha="center", fontsize=9)

    axes[1].bar(labels, boxes.values, color="#f0a04b")
    axes[1].set_ylabel("缺陷框数量")
    axes[1].set_title("各类缺陷的标注框数量")
    for i, v in enumerate(boxes.values):
        axes[1].text(i, v + 1, str(v), ha="center", fontsize=9)

    for ax in axes:
        ax.tick_params(axis="x", labelsize=8)
    fig.suptitle("PCB 缺陷数据集:类别分布", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIGURE_DIR / "fig_class_distribution.png", dpi=130)
    plt.close(fig)
    print("已保存:outputs/fig_class_distribution.png")


def plot_box_size(df: pd.DataFrame) -> None:
    """缺陷框的绝对尺寸与相对占比——这决定了任务有多难。"""
    df = df.copy()
    df["box_max_side"] = df[["box_w", "box_h"]].max(axis=1)
    df["img_max_side"] = df[["img_width", "img_height"]].max(axis=1)
    df["ratio"] = df["box_max_side"] / df["img_max_side"] * 100  # 占长边的百分比

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    axes[0].hist(df["box_max_side"], bins=40, color="#5b9bd5", edgecolor="white")
    axes[0].set_xlabel("缺陷框最长边(像素)")
    axes[0].set_ylabel("缺陷框数量")
    axes[0].set_title(f"缺陷框绝对尺寸(中位数 {df['box_max_side'].median():.0f} 像素)")

    axes[1].hist(df["ratio"], bins=40, color="#f0a04b", edgecolor="white")
    axes[1].axvline(df["ratio"].median(), color="black", linestyle="--",
                    label=f"中位数 {df['ratio'].median():.2f}%")
    axes[1].set_xlabel("缺陷框最长边 / 图像最长边(%)")
    axes[1].set_ylabel("缺陷框数量")
    axes[1].set_title("缺陷在整张图中的相对占比")
    axes[1].legend()

    fig.suptitle("PCB 缺陷规模分布:缺陷有多大?", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIGURE_DIR / "fig_box_size.png", dpi=130)
    plt.close(fig)
    print("已保存:outputs/fig_box_size.png")
    return df


def main() -> None:
    setup_chinese_font()
    FIGURE_DIR.mkdir(exist_ok=True)
    df = parse_annotations()
    if df.empty:
        raise SystemExit("没有解析到任何标注,请检查 data/raw 目录")

    print(f"标注框总数:{len(df)}")
    print(f"图片总数  :{df['image'].nunique()}")
    print(f"缺陷类别  :{sorted(df['class_name'].unique())}\n")

    per_class = df.groupby("class_name").agg(
        图片数=("image", "nunique"),
        缺陷框数=("box_w", "size"),
        框宽中位数=("box_w", "median"),
        框高中位数=("box_h", "median"),
    ).sort_values("缺陷框数", ascending=False)
    print(per_class.to_string())

    boxes_per_image = df.groupby("image").size()
    print(f"\n每张图的缺陷框数量:平均 {boxes_per_image.mean():.1f},"
          f"最少 {boxes_per_image.min()},最多 {boxes_per_image.max()}")
    print(f"图像尺寸: {df['img_width'].unique()[:5]} x {df['img_height'].unique()[:5]}"
          f"({df['img_width'].nunique()} 种不同尺寸)")

    plot_class_distribution(df)
    sized = plot_box_size(df)

    sized["box_max_side"] = sized[["box_w", "box_h"]].max(axis=1)
    sized["ratio"] = sized["box_max_side"] / sized[["img_width", "img_height"]].max(axis=1) * 100
    print(f"\n缺陷框最长边:中位数 {sized['box_max_side'].median():.0f} 像素")
    print(f"缺陷占图像长边比例:中位数 {sized['ratio'].median():.2f}%,"
          f"最小 {sized['ratio'].min():.2f}%")

    per_class.to_csv(FIGURE_DIR / "class_stats.csv", encoding="utf-8-sig")
    print("已保存:outputs/class_stats.csv")


if __name__ == "__main__":
    main()
