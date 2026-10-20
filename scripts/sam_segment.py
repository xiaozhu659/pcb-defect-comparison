"""组合检测与分割:用 YOLO 检出的缺陷框作为提示,让 SAM 生成精确掩膜。

这一步的价值
------------
检测框只能告诉你"缺陷在这一带",而分割掩膜能算出**缺陷的精确面积**——
有了面积就可以客观地划分严重程度(比如"占板面 1% 以上算严重"),
这是视觉大模型完全做不到的事:它只能给"轻微/明显"这种主观描述。

用法:
    uv run scripts/sam_segment.py --limit 8
    uv run scripts/sam_segment.py            # 跑完整验证集
"""

import argparse
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
VAL_LIST = PROJECT_ROOT / "data" / "yolo" / "val.txt"
DEFAULT_YOLO = PROJECT_ROOT / "runs" / "detect" / "yolo11n_1024" / "weights" / "best.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO + SAM 组合推理")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--weights", default=str(DEFAULT_YOLO))
    parser.add_argument("--sam", default="sam2.1_t.pt")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--preview", type=int, default=4, help="可视化展示几张")
    return parser.parse_args()


def main() -> None:
    from ultralytics import SAM, YOLO

    args = parse_args()
    setup_chinese_font()
    images = [Path(p) for p in VAL_LIST.read_text(encoding="utf-8").splitlines() if p.strip()]
    if args.limit:
        images = images[: args.limit]

    detector = YOLO(args.weights)
    segmenter = SAM(args.sam)
    print(f"检测模型:{Path(args.weights).name}")
    print(f"分割模型:{args.sam}")
    print(f"待处理:{len(images)} 张\n")

    records = []
    previews = []
    for index, image in enumerate(images, 1):
        detection = detector.predict(str(image), imgsz=args.imgsz,
                                    conf=args.conf, verbose=False)[0]
        boxes = detection.boxes.xyxy.tolist()
        classes = [CLASS_ORDER[int(c)].lower() for c in detection.boxes.cls]
        if not boxes:
            records.append({"image": image.name, "defects": []})
            continue

        segmentation = segmenter.predict(str(image), bboxes=boxes, verbose=False)[0]
        masks = segmentation.masks.data.cpu().numpy() if segmentation.masks is not None else []
        total_pixels = float(detection.orig_shape[0] * detection.orig_shape[1])

        defects = []
        for cls, box, mask in zip(classes, boxes, masks):
            area = float(mask.sum())
            defects.append({
                "type": cls,
                "box": [round(v, 1) for v in box],
                "area_px": int(area),
                "area_ratio": round(area / total_pixels * 100, 4),
            })
        records.append({"image": image.name, "defects": defects})
        if len(previews) < args.preview:
            previews.append((Image.open(image).convert("RGB"), masks, defects))
        if index % 20 == 0 or index == len(images):
            print(f"[{index}/{len(images)}] 已处理")

    out_path = OUT_DIR / "sam_segmentation.jsonl"
    with out_path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"\n结果已保存:{out_path}")

    areas = [d["area_ratio"] for r in records for d in r["defects"]]
    if areas:
        print(f"缺陷面积占整图比例:中位数 {np.median(areas):.3f}%,"
              f"最大 {max(areas):.3f}%,最小 {min(areas):.3f}%")

    # 可视化:检测框 + 分割掩膜
    if previews:
        fig, axes = plt.subplots(1, len(previews), figsize=(4.6 * len(previews), 4.4))
        axes = np.atleast_1d(axes)
        for ax, (preview_image, masks, defects) in zip(axes, previews):
            ax.imshow(preview_image)
            for mask, defect in zip(masks, defects):
                overlay = np.zeros((*mask.shape, 4))
                overlay[mask > 0.5] = [1.0, 0.2, 0.2, 0.45]
                ax.imshow(overlay)
                row = (defect["box"][0], defect["box"][1])
                ax.add_patch(patches.Rectangle(
                    row, defect["box"][2] - defect["box"][0],
                    defect["box"][3] - defect["box"][1],
                    fill=False, edgecolor="#1565c0", linewidth=1.2))
            total_area = sum(d["area_ratio"] for d in defects)
            ax.set_title(f"检出 {len(defects)} 处,合计占板面 {total_area:.3f}%\n"
                         f"单处最小 {min(d['area_ratio'] for d in defects):.3f}%",
                         fontsize=9)
            ax.axis("off")
        fig.suptitle("YOLO 检测 + SAM 分割:不仅找到位置,还能量出缺陷面积", fontsize=14)
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        fig_path = OUT_DIR / "fig_sam_segmentation.png"
        fig.savefig(fig_path, dpi=120)
        plt.close(fig)
        print(f"已保存可视化:{fig_path}")


if __name__ == "__main__":
    main()
