"""把传统检测模型与视觉大模型放在同一批验证图片上做量化对比。

对比维度(每一项都对应考核要求里的"在相同输入上比较两种方案的表现"):
1. 类型判断准确率 —— 每张图真实只含一种缺陷,看方法是否正确指出该类型;
2. 检出率 —— 是否至少检出/报出一个缺陷(漏检是致命的);
3. 数量误差 —— 报出的缺陷个数与真实个数相差多少;
4. 定位误差 —— 报出的缺陷中心与真实框中心的距离(按图像对角线归一化);
5. 耗时与成本 —— 单张耗时、整批花费。

用法:
    uv run scripts/compare_methods.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pcb_defect_comparison.common import (  # noqa: E402
    CLASS_ORDER,
    PROJECT_ROOT,
    setup_chinese_font,
)

LABEL_DIR = PROJECT_ROOT / "data" / "yolo" / "labels" / "val"
OUT_DIR = PROJECT_ROOT / "outputs"

METHODS = [
    ("传统检测 (YOLO)", "yolo_results.jsonl", "#5b9bd5"),
    ("大模型 · 关闭思考", "llm_results_none.jsonl", "#7fb069"),
    ("大模型 · 低强度思考", "llm_results_low.jsonl", "#f0a04b"),
    ("大模型 · 高强度思考", "llm_results_high.jsonl", "#c0392b"),
]


def load_truth() -> dict[str, dict]:
    """读取验证集标注:{图片名: {类型集合, 数量, 中心列表}}。"""
    truth = {}
    for label_path in sorted(LABEL_DIR.glob("*.txt")):
        names, centers = [], []
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            names.append(CLASS_ORDER[int(parts[0])].lower())
            centers.append((float(parts[1]), float(parts[2])))
        truth[f"{label_path.stem}.jpg"] = {
            "types": set(names),
            "count": len(names),
            "centers": centers,
        }
    return truth


def center_error(predicted: list, actual: list) -> float:
    """预测中心到最近真实中心的平均距离(逐点最近邻匹配)。"""
    if not predicted or not actual:
        return float("nan")
    errors = []
    for point in predicted:
        distances = [np.hypot(point[0] - a[0], point[1] - a[1]) for a in actual]
        errors.append(min(distances))
    return float(np.mean(errors))


def evaluate(path: Path, truth: dict[str, dict]) -> dict:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
               if line.strip()]
    type_hits = detected = 0
    count_errors, center_errors, type_fp = [], [], []
    total_seconds = 0.0
    total_tokens = 0

    for record in records:
        name = record["image"]
        if name not in truth:
            continue
        actual = truth[name]
        types = {str(d.get("type", "")).lower() for d in record["defects"]}
        centers = [d["center"] for d in record["defects"]
                   if isinstance(d.get("center"), (list, tuple)) and len(d["center"]) == 2]

        if actual["types"] & types:
            type_hits += 1
        if record["defects"]:
            detected += 1
        count_errors.append(abs(len(record["defects"]) - actual["count"]))
        # 只统计"预测出图里根本不存在的那类缺陷",衡量误报
        type_fp.append(len(types - actual["types"]))
        error = center_error(centers, actual["centers"])
        if not np.isnan(error):
            center_errors.append(error)
        total_seconds += record.get("seconds", 0.0)
        total_tokens += record.get("prompt_tokens", 0) + record.get("completion_tokens", 0)

    n = len(records)
    return {
        "样本数": n,
        "类型判断准确率": type_hits / n * 100,
        "检出率": detected / n * 100,
        "数量平均绝对误差": float(np.mean(count_errors)),
        "平均误报类型数": float(np.mean(type_fp)),
        "定位误差(归一化)": float(np.mean(center_errors)) if center_errors else float("nan"),
        "单张耗时(秒)": total_seconds / n,
        "整批token": total_tokens,
    }


def plot(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    bars = df["类型判断准确率"].tolist()
    axes[0].bar(df["方法"], bars, color=df["颜色"])
    axes[0].set_ylabel("准确率(%)")
    axes[0].set_title("缺陷类型判断准确率")
    for i, v in enumerate(bars):
        axes[0].text(i, v + 1.5, f"{v:.1f}%", ha="center", fontsize=9)

    misses = (100 - df["检出率"]).tolist()
    axes[1].bar(df["方法"], misses, color=df["颜色"])
    axes[1].set_ylabel("漏检率(%)")
    axes[1].set_title("漏检率(越低越好)")
    for i, v in enumerate(misses):
        axes[1].text(i, v + 1.5, f"{v:.1f}%", ha="center", fontsize=9)

    seconds = df["单张耗时(秒)"].tolist()
    axes[2].bar(df["方法"], seconds, color=df["颜色"])
    axes[2].set_yscale("log")
    axes[2].set_ylabel("单张耗时(秒,对数坐标)")
    axes[2].set_title("速度对比")
    for i, v in enumerate(seconds):
        axes[2].text(i, v * 1.15, f"{v:.2f}s", ha="center", fontsize=9)

    for ax in axes:
        ax.tick_params(axis="x", labelsize=8, rotation=12)
    fig.suptitle("PCB 缺陷检测:传统模型 vs 视觉大模型", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT_DIR / "fig_method_comparison.png", dpi=130)
    plt.close(fig)
    print("已保存:outputs/fig_method_comparison.png")


def main() -> None:
    setup_chinese_font()
    truth = load_truth()
    print(f"验证集真实标注:{len(truth)} 张\n")

    rows = []
    for label, filename, color in METHODS:
        path = OUT_DIR / filename
        if not path.exists():
            print(f"[跳过] {label}:找不到 {filename}")
            continue
        finished = sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
        if finished == 0:
            print(f"[跳过] {label}:结果文件还是空的(该档位可能正在跑)\n")
            continue
        metrics = evaluate(path, truth)
        rows.append({"方法": label, "颜色": color, **metrics})
        print(f"--- {label} ---")
        if finished < len(truth):
            print(f"    注意:该档位目前只有 {finished}/{len(truth)} 张的结果,还在跑")
        for key, value in metrics.items():
            print(f"    {key}: {value:.3f}" if isinstance(value, float) else f"    {key}: {value}")
        print()

    if not rows:
        raise SystemExit("没有任何结果文件可对比,请先运行推理脚本")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "comparison.csv", index=False, encoding="utf-8-sig")
    plot(df)
    print("\n===== 汇总表 =====")
    print(df.drop(columns=["颜色"]).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\n已保存:outputs/comparison.csv")


if __name__ == "__main__":
    main()
