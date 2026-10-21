"""并排展示:同一张 PCB 图上,真实标注、传统检测模型、视觉大模型三者的判断。

大模型用的是表现最好的一档(高强度思考),这样对比对它最公平。
样本特意挑三类典型情况:大模型判断正确的、完全漏检的、过度报错的——
一张图就能把"它会什么、不会什么"讲清楚。

用法:
    uv run scripts/visualize_side_by_side.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from pcb_defect_comparison.common import (  # noqa: E402
    CLASS_NAMES_ZH,
    CLASS_ORDER,
    PROJECT_ROOT,
    setup_chinese_font,
)

OUT_DIR = PROJECT_ROOT / "outputs"
LABEL_DIR = PROJECT_ROOT / "data" / "yolo" / "labels" / "val"
ROWS = 6

COLOR_TRUTH = "#2e7d32"     # 真实标注:绿色
COLOR_YOLO = "#1565c0"      # 传统方案:蓝色
COLOR_LLM = "#c62828"       # 大模型:红色


def load_jsonl(path: Path) -> dict[str, dict]:
    return {json.loads(line)["image"]: json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def load_truth(name: str) -> tuple[list, list]:
    """返回 (缺陷类型列表, 归一化框列表 xywh)。"""
    path = LABEL_DIR / f"{Path(name).stem}.txt"
    types, boxes = [], []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            types.append(CLASS_ORDER[int(parts[0])].lower())
            boxes.append(tuple(float(v) for v in parts[1:5]))
    return types, boxes


def pick_samples(llm: dict, yolo: dict) -> list[str]:
    """挑三类典型样本:判断对的、完全漏检的、过度报错的(最严重的)。"""
    right, missed, over = [], [], []
    for name, record in llm.items():
        truth_types, _ = load_truth(name)
        truth_set = set(truth_types)
        predicted = {str(d.get("type", "")).lower() for d in record["defects"]}
        if not predicted:
            missed.append(name)
        elif predicted & truth_set and len(predicted - truth_set) <= 1:
            right.append((len(predicted - truth_set), name))
        else:
            over.append((len(predicted - truth_set), name))

    right.sort()
    over.sort(reverse=True)
    picked = [n for _, n in right[:2]] + missed[:2] + [n for _, n in over[:2]]
    return picked[:ROWS]


def draw_row(axes, name: str, yolo: dict, llm: dict) -> None:
    image_path = next((p for p in (PROJECT_ROOT / "data" / "raw").rglob(name)), None)
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    truth_types, truth_boxes = load_truth(name)

    # 第一列:真实标注
    axes[0].imshow(image)
    axes[0].set_title(f"真实标注:{len(truth_boxes)} 处 "
                      f"{CLASS_NAMES_ZH.get(truth_types[0].capitalize(), truth_types[0]) if truth_types else ''}",
                      fontsize=9)
    for cx, cy, bw, bh in truth_boxes:
        axes[0].add_patch(patches.Rectangle(
            ((cx - bw / 2) * width, (cy - bh / 2) * height),
            bw * width, bh * height, fill=False, edgecolor=COLOR_TRUTH, linewidth=1.6))
    axes[0].axis("off")

    # 第二列:传统方案 YOLO
    yolo_record = yolo.get(name, {"defects": []})
    axes[1].imshow(image)
    axes[1].set_title(f"传统检测 YOLO:检出 {len(yolo_record['defects'])} 处 "
                      f"({yolo_record.get('seconds', 0) * 1000:.0f} 毫秒)", fontsize=9)
    for defect in yolo_record["defects"]:
        cx, cy = defect["center"]
        axes[1].plot(cx * width, cy * height, marker="o", markersize=9,
                     markerfacecolor="none", markeredgecolor=COLOR_YOLO, markeredgewidth=1.6)
    axes[1].axis("off")

    # 第三列:视觉大模型
    llm_record = llm.get(name, {"defects": []})
    axes[2].imshow(image)
    predicted = llm_record["defects"]
    summary = "未发现缺陷" if not predicted else "、".join(
        sorted({CLASS_NAMES_ZH.get(str(d.get("type", "")).capitalize(), d.get("type", "?"))
                for d in predicted}))
    axes[2].set_title(f"视觉大模型:报出 {len(predicted)} 处 · {summary}"
                      f"({llm_record.get('seconds', 0):.0f} 秒)", fontsize=9)
    for defect in predicted:
        center = defect.get("center")
        if isinstance(center, (list, tuple)) and len(center) == 2:
            try:
                cx, cy = float(center[0]), float(center[1])
            except (TypeError, ValueError):
                continue
            axes[2].plot(cx * width, cy * height, marker="x", markersize=10,
                         color=COLOR_LLM, markeredgewidth=2)
    axes[2].axis("off")


def main() -> None:
    setup_chinese_font()
    yolo = load_jsonl(OUT_DIR / "yolo_results.jsonl")
    llm = load_jsonl(OUT_DIR / "llm_results_high.jsonl")
    picked = pick_samples(llm, yolo)
    print(f"挑选样本:{picked}\n")

    fig, axes = plt.subplots(len(picked), 3, figsize=(15.5, 3.5 * len(picked)))
    axes = np.atleast_2d(axes)
    for row, name in enumerate(picked):
        draw_row(axes[row], name, yolo, llm)
        truth_types, truth_boxes = load_truth(name)
        predicted = {str(d.get("type", "")).lower() for d in llm[name]["defects"]}
        print(f"{name}: 真实 {len(truth_boxes)} 处 {sorted(set(truth_types))}"
              f" | 大模型报出 {sorted(predicted) or '无'}")

    fig.suptitle("同一批图片上的三方对比:真实标注 / 传统检测模型 / 视觉大模型",
                 fontsize=15)
    fig.text(0.5, 0.005,
             "绿色框 = 真实缺陷位置    蓝色圆圈 = 传统检测模型 YOLO 的检出位置"
             "    红色叉号 = 视觉大模型报出的缺陷位置",
             ha="center", fontsize=11, color="#444444")
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    out_path = OUT_DIR / "fig_side_by_side.png"
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"\n已保存:{out_path}")


if __name__ == "__main__":
    main()
