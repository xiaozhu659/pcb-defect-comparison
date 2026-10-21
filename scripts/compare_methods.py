"""把传统检测模型与视觉大模型放在同一批验证图片上做量化对比。

对比维度(每一项都对应考核要求里的"在相同输入上比较两种方案的表现"):
1. 类型判断 —— 每张图真实只含一种缺陷;同时给出"宽松命中"(包含真实类型)
   与"严格命中"(类型集合完全一致),以及类型精确率 / 召回率;
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
    type_hits = detected = strict_hits = 0
    tp = fp = fn = 0
    count_errors, same_type_errors, center_errors, type_fp = [], [], [], []
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
        # 严格口径:预测出的类型集合必须和真实集合完全一致。
        # 只用"包含真实类型"太宽松——把 6 类全报一遍也会被算成命中,会严重高估。
        if types == actual["types"]:
            strict_hits += 1
        tp += len(types & actual["types"])
        fp += len(types - actual["types"])
        fn += len(actual["types"] - types)
        if record["defects"]:
            detected += 1
        count_errors.append(abs(len(record["defects"]) - actual["count"]))
        # 只统计"类型正确"的缺陷数量。原始数量误差会被"每类各报一个"的低成本策略
        # 占便宜——报 6 类得到 6 个,而真实平均 4.3 个,误差反而显得小。
        same_type_errors.append(
            abs(sum(1 for d in record["defects"]
                    if str(d.get("type", "")).lower() in actual["types"]) - actual["count"]))
        # 只统计"预测出图里根本不存在的那类缺陷",衡量误报
        type_fp.append(len(types - actual["types"]))
        error = center_error(centers, actual["centers"])
        if not np.isnan(error):
            center_errors.append(error)
        total_seconds += record.get("seconds", 0.0)
        total_tokens += record.get("prompt_tokens", 0) + record.get("completion_tokens", 0)

    n = len(records)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "样本数": n,
        "类型命中率(宽松)": type_hits / n * 100,
        "类型命中率(严格)": strict_hits / n * 100,
        "类型精确率": precision * 100,
        "类型召回率": recall * 100,
        "检出率": detected / n * 100,
        "数量平均绝对误差": float(np.mean(count_errors)),
        "同类型数量误差": float(np.mean(same_type_errors)),
        "平均误报类型数": float(np.mean(type_fp)),
        "定位误差(归一化)": float(np.mean(center_errors)) if center_errors else float("nan"),
        "单张耗时(秒)": total_seconds / n,
        "整批token": total_tokens,
    }


def plot(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    x = np.arange(len(df))
    width = 0.38
    axes[0].bar(x - width / 2, df["类型命中率(宽松)"], width,
                color=df["颜色"], alpha=0.55, label="宽松(包含真实类型)")
    axes[0].bar(x + width / 2, df["类型命中率(严格)"], width,
                color=df["颜色"], label="严格(类型集合完全一致)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(df["方法"], rotation=12, fontsize=8)
    axes[0].legend(fontsize=8)
    axes[0].set_ylabel("准确率(%)")
    axes[0].set_title("缺陷类型判断:两种口径")
    for i, (loose, strict) in enumerate(zip(df["类型命中率(宽松)"], df["类型命中率(严格)"])):
        axes[0].text(i - width / 2, loose + 1.5, f"{loose:.1f}", ha="center", fontsize=8)
        axes[0].text(i + width / 2, strict + 1.5, f"{strict:.1f}", ha="center", fontsize=8)

    misses = (100 - df["检出率"]).tolist()
    axes[1].bar(df["方法"], misses, color=df["颜色"])
    axes[1].tick_params(axis="x", rotation=12, labelsize=8)
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

    axes[2].tick_params(axis="x", labelsize=8, rotation=12)
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
