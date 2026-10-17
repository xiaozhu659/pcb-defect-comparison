"""环境自检:确认 PyTorch、CUDA、ultralytics、OpenCV 与 Hugging Face 镜像都可用。

用法:
    uv run scripts/check_env.py
"""

import platform
import sys


def main() -> None:
    print("=" * 60)
    print("Python  :", sys.version.split()[0], f"({platform.architecture()[0]})")
    print("解释器  :", sys.executable)

    import cv2
    import numpy as np
    import pandas as pd
    import torch
    import ultralytics

    print("-" * 60)
    print("numpy       :", np.__version__)
    print("pandas      :", pd.__version__)
    print("opencv      :", cv2.__version__)
    print("ultralytics :", ultralytics.__version__)
    print("torch       :", torch.__version__)
    print("编译时 CUDA :", torch.version.cuda)

    print("-" * 60)
    available = torch.cuda.is_available()
    print("CUDA 可用   :", available)
    if available:
        print("GPU 名称    :", torch.cuda.get_device_name(0))
        props = torch.cuda.get_device_properties(0)
        print("计算能力    : sm_{}{}".format(props.major, props.minor))
        print("显存        : {:.1f} GB".format(props.total_memory / 1024**3))
        x = torch.randn(4, 3, 32, 32, device="cuda")
        conv = torch.nn.Conv2d(3, 8, 3, padding=1).to("cuda")
        y = conv(x)
        torch.cuda.synchronize()
        print("GPU 前向测试: 通过,输出形状", tuple(y.shape))
    else:
        print("!! 未检测到可用 GPU")

    print("-" * 60)
    import os

    from huggingface_hub import HfApi

    endpoint = os.environ.get("HF_ENDPOINT", "(默认 huggingface.co)")
    print("HF 镜像端点 :", endpoint)
    try:
        info = HfApi().dataset_info("RobotHuman/PCB_defect")
        print("数据集连通性: 正常,共", len(info.siblings), "个文件")
    except Exception as exc:  # noqa: BLE001
        print("数据集连通性: 失败 ->", type(exc).__name__, str(exc)[:80])
    print("=" * 60)


if __name__ == "__main__":
    main()
