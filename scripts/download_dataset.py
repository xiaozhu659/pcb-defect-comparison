"""下载 PCB 缺陷数据集(HRIPCB 格式:6 类缺陷,带 PASCAL VOC 标注)。

数据来源:Hugging Face 数据集 RobotHuman/PCB_defect(MIT 许可)。
走国内镜像 hf-mirror.com,不需要代理。

用法:
    uv run scripts/download_dataset.py
"""

import os
from pathlib import Path

# 必须在导入 huggingface_hub 之前设置,才能走国内镜像
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from huggingface_hub import snapshot_download  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
REPO_ID = "RobotHuman/PCB_defect"


def main() -> None:
    print(f"镜像端点:{os.environ['HF_ENDPOINT']}")
    print(f"正在下载数据集 {REPO_ID} 到 {DATA_DIR} ...")
    path = snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=str(DATA_DIR),
    )
    print(f"下载完成:{path}")

    files = [p for p in Path(path).rglob("*") if p.is_file()]
    total_mb = sum(p.stat().st_size for p in files) / 1024**2
    print(f"文件数:{len(files)},总大小:{total_mb:.1f} MB")

    suffixes = {}
    for f in files:
        suffixes[f.suffix.lower()] = suffixes.get(f.suffix.lower(), 0) + 1
    print("按扩展名统计:", dict(sorted(suffixes.items(), key=lambda kv: -kv[1])))

    print("\n顶层结构:")
    for item in sorted(Path(path).iterdir()):
        kind = "目录" if item.is_dir() else "文件"
        print(f"  [{kind}] {item.name}")


if __name__ == "__main__":
    main()
