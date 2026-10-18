"""用训练好的 YOLO 在验证集上推理,结果存成 JSONL(格式与大模型方案对齐)。

输出每条记录包含:图片名、检出的缺陷类别与置信度、框中心归一化坐标、耗时。
这样后面就能和大模型的结果放在同一张表里比较。

用法:
    uv run scripts/yolo_infer.py --weights runs/detect/yolo11n_1024/weights/best.pt
"""

import argparse
import json
import time
from pathlib import Path

from pcb_defect_comparison.common import CLASS_ORDER, PROJECT_ROOT

VAL_LIST = PROJECT_ROOT / "data" / "yolo" / "val.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO 验证集推理")
    parser.add_argument("--weights",
                        default="runs/detect/yolo11n_1024/weights/best.pt")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--out", default="outputs/yolo_results.jsonl")
    return parser.parse_args()


def main() -> None:
    from ultralytics import YOLO

    args = parse_args()
    weights = Path(args.weights)
    if not weights.is_absolute():
        weights = PROJECT_ROOT / weights
    if not weights.exists():
        raise SystemExit(f"找不到权重:{weights}")

    images = [Path(p) for p in VAL_LIST.read_text(encoding="utf-8").splitlines() if p.strip()]
    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(exist_ok=True)

    model = YOLO(str(weights))
    print(f"权重:{weights}")
    print(f"验证集:{len(images)} 张,输入尺寸 {args.imgsz},置信度阈值 {args.conf}\n")

    total_seconds = 0.0
    total_boxes = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for index, image in enumerate(images, 1):
            started = time.time()
            result = model.predict(str(image), imgsz=args.imgsz,
                                   conf=args.conf, iou=args.iou, verbose=False)[0]
            elapsed = time.time() - started
            total_seconds += elapsed

            height, width = result.orig_shape[:2]
            defects = []
            for box in result.boxes:
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
                defects.append({
                    "type": CLASS_ORDER[int(box.cls)].lower(),
                    "center": [round((x1 + x2) / 2 / width, 4),
                               round((y1 + y2) / 2 / height, 4)],
                    "conf": round(float(box.conf), 4),
                })
            total_boxes += len(defects)

            fh.write(json.dumps({
                "image": image.name,
                "defects": defects,
                "seconds": round(elapsed, 3),
                "error": "",
            }, ensure_ascii=False) + "\n")
            if index % 20 == 0 or index == len(images):
                print(f"[{index}/{len(images)}] 已完成,累计检出 {total_boxes} 个框")

    print(f"\n推理完成:平均每张 {total_seconds / len(images) * 1000:.1f} 毫秒")
    print(f"检出框总数:{total_boxes}")
    print(f"结果文件:{out_path}")


if __name__ == "__main__":
    main()
