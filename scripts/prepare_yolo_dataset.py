"""把 PASCAL VOC 标注转成 YOLO 格式,并划分训练/验证集。

两个必须遵守的约定
------------------
1. YOLO 的标注是一张图一个 txt,每行 `类别编号 中心x 中心y 宽 高`,
   四个坐标都归一化到 0~1;txt 必须与图片同名。
2. ultralytics 通过"把图片路径里的 images 换成 labels"来找标注,
   所以目录必须长成 `images/<split>/xxx.jpg` 与 `labels/<split>/xxx.txt` 的成对结构。
   最初的版本把图片清单直接写进 data.yaml,结果它推导不出标注位置而报错。

图片不复制、只建硬链接:同一磁盘卷内硬链接几乎不占额外空间,也几乎瞬时完成。

用法:
    uv run scripts/prepare_yolo_dataset.py
"""

import os
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

from pcb_defect_comparison.common import CLASS_ORDER, DATA_DIR, PROJECT_ROOT

YOLO_DIR = PROJECT_ROOT / "data" / "yolo"
VAL_RATIO = 0.2
SEED = 0
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def find_image(xml_path: Path) -> Path | None:
    """按本地实际文件名配对图片。

    XML 里记录的 <filename> 是原始数据集的名字(如 01_missing_hole_01.jpg),
    而本地文件被重命名成了 missing_hole01.jpg,两者对不上;
    可靠做法是去掉 "_annotation" 后缀,在同目录里找同名图片。
    """
    base = xml_path.stem.replace("_annotation", "")
    for candidate in sorted(xml_path.parent.iterdir()):
        if candidate.suffix.lower() in IMAGE_SUFFIXES and candidate.stem == base:
            return candidate
    return None


def convert(xml_path: Path) -> tuple[str, str]:
    """返回 (YOLO 格式的标注内容, 是否解析到有效框)。"""
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    width = float(size.findtext("width"))
    height = float(size.findtext("height"))

    lower_to_canonical = {c.lower(): c for c in CLASS_ORDER}
    lines = []
    for obj in root.findall("object"):
        box = obj.find("bndbox")
        if box is None:
            continue
        canonical = lower_to_canonical.get(obj.findtext("name", "").strip().lower())
        if canonical is None:
            continue
        x1, y1 = float(box.findtext("xmin")), float(box.findtext("ymin"))
        x2, y2 = float(box.findtext("xmax")), float(box.findtext("ymax"))
        cx = min(max((x1 + x2) / 2 / width, 0.0), 1.0)
        cy = min(max((y1 + y2) / 2 / height, 0.0), 1.0)
        bw = min(max(abs(x2 - x1) / width, 0.0), 1.0)
        bh = min(max(abs(y2 - y1) / height, 0.0), 1.0)
        lines.append(f"{CLASS_ORDER.index(canonical)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return "\n".join(lines), bool(lines)


def link_or_copy(source: Path, target: Path) -> str:
    """优先建硬链接(不占额外空间),失败则复制。"""
    if target.exists():
        return "已存在"
    try:
        os.link(source, target)
        return "硬链接"
    except OSError:
        shutil.copy2(source, target)
        return "复制"


def main() -> None:
    pairs = []
    for xml_path in sorted(DATA_DIR.rglob("*.xml")):
        image_path = find_image(xml_path)
        if image_path is None:
            continue
        content, has_box = convert(xml_path)
        if has_box:
            pairs.append((image_path, content, content.count("\n") + 1))

    total_boxes = sum(p[2] for p in pairs)
    print(f"解析到 {len(pairs)} 张图、{total_boxes} 个缺陷框")

    random.Random(SEED).shuffle(pairs)
    val_count = max(1, int(len(pairs) * VAL_RATIO))
    splits = {"val": pairs[:val_count], "train": pairs[val_count:]}

    for split, subset in splits.items():
        image_dir = YOLO_DIR / "images" / split
        label_dir = YOLO_DIR / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

        method = ""
        for image_path, content, _ in subset:
            method = link_or_copy(image_path, image_dir / image_path.name)
            (label_dir / f"{image_path.stem}.txt").write_text(content, encoding="utf-8")
        print(f"{split}: {len(subset)} 张图,{sum(p[2] for p in subset)} 个框"
              f"(图片处理方式:{method})")

    yaml_path = YOLO_DIR / "data.yaml"
    names_block = "\n".join(f"  {i}: {name}" for i, name in enumerate(CLASS_ORDER))
    yaml_path.write_text(
        "# PCB 缺陷检测数据集(6 类缺陷,由 PASCAL VOC 标注转换而来)\n"
        f"path: {YOLO_DIR.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        f"nc: {len(CLASS_ORDER)}\n"
        f"names:\n{names_block}\n",
        encoding="utf-8",
    )
    print(f"\n已写入 {yaml_path}")
    for i, name in enumerate(CLASS_ORDER):
        print(f"  {i}: {name}")


if __name__ == "__main__":
    main()
