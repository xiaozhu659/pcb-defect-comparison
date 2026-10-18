"""训练 YOLO 检测模型(传统深度学习方案)。

为什么要用 1024 的输入尺寸:缺陷框最长边中位数只有 78 像素、只占图像长边的 2.85%,
属于典型的小目标检测。YOLO 默认的 640 会把缺陷压缩到十几个像素,漏检会明显增加,
所以这里把输入尺寸提到 1024。

用法:
    uv run scripts/train_yolo.py --model yolo11n.pt --epochs 100 --imgsz 1024 --batch 8
"""

import argparse
from pathlib import Path

from pcb_defect_comparison.common import PROJECT_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练 YOLO 做 PCB 缺陷检测")
    parser.add_argument("--model", default="yolo11n.pt", help="预训练权重,如 yolo11n.pt / yolo11s.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--patience", type=int, default=20, help="多少轮没提升就早停")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--name", default="yolo11n_1024")
    return parser.parse_args()


def main() -> None:
    from ultralytics import YOLO

    args = parse_args()
    data_yaml = PROJECT_ROOT / "data" / "yolo" / "data.yaml"
    if not data_yaml.exists():
        raise SystemExit(f"找不到 {data_yaml},请先运行 scripts/prepare_yolo_dataset.py")

    print(f"模型:{args.model}  输入尺寸:{args.imgsz}  批大小:{args.batch}  轮次:{args.epochs}")
    model = YOLO(args.model)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        workers=args.workers,
        project=str(PROJECT_ROOT / "runs" / "detect"),
        name=args.name,
        exist_ok=True,
        seed=0,
        # 小目标检测常用的一点增强:轻微缩放与平移,但不开水平翻转的强变形
        degrees=5.0,
        translate=0.1,
        scale=0.3,
        fliplr=0.5,
        mosaic=1.0,
        verbose=True,
    )

    save_dir = PROJECT_ROOT / "runs" / "detect" / args.name
    print(f"\n训练完成,结果目录:{save_dir}")
    print(f"最佳权重:{save_dir / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
