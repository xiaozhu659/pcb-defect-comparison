"""用视觉大模型跑同一批验证集图片,结果存成 JSONL 便于对比。

用法:
    uv run scripts/llm_infer.py                    # 跑验证集全部图片
    uv run scripts/llm_infer.py --limit 20         # 先小批量试跑
    uv run scripts/llm_infer.py --workers 4        # 并发数
"""

import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from pcb_defect_comparison.common import PROJECT_ROOT
from pcb_defect_comparison.llm import ask_one

VAL_LIST = PROJECT_ROOT / "data" / "yolo" / "val.txt"
OUT_DIR = PROJECT_ROOT / "outputs"

# DeepSeek 价格(每百万 token,美元):deepseek-flash 未命中缓存的输入与输出
PRICE_INPUT = 0.15
PRICE_OUTPUT = 0.60

print_lock = threading.Lock()


def load_done(out_path: Path) -> set[str]:
    """已经跑过的图片,支持断点续跑。"""
    if not out_path.exists():
        return set()
    done = set()
    with out_path.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                done.add(json.loads(line)["image"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="用视觉大模型跑 PCB 缺陷识别")
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 张")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--effort", default="low", choices=["none", "low", "high", "max"],
                        help="思考强度:none 关闭思考,其余对应 reasoning_effort")
    parser.add_argument("--tag", default=None, help="结果文件后缀,默认用 effort 值")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_path = OUT_DIR / f"llm_results_{args.tag or args.effort}.jsonl"
    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise SystemExit("请先设置环境变量 DEEPSEEK_API_KEY")
    if not VAL_LIST.exists():
        raise SystemExit(f"找不到 {VAL_LIST},请先运行 scripts/prepare_yolo_dataset.py")

    images = [Path(p) for p in VAL_LIST.read_text(encoding="utf-8").splitlines() if p.strip()]
    if args.limit:
        images = images[: args.limit]
    done = load_done(out_path)
    todo = [p for p in images if p.name not in done]
    print(f"验证集共 {len(images)} 张,已完成 {len(done)} 张,本次待跑 {len(todo)} 张")
    print(f"思考强度:{args.effort}  并发数:{args.workers}")
    print(f"结果文件:{out_path}\n")

    out_path.parent.mkdir(exist_ok=True)
    counters = {"ok": 0, "error": 0, "defects": 0, "seconds": 0.0,
                "prompt_tokens": 0, "completion_tokens": 0}

    with out_path.open("a", encoding="utf-8") as fh, \
            ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(ask_one, path, None, 180, args.effort): path for path in todo}
        for index, future in enumerate(as_completed(futures), 1):
            result = future.result()
            record = {
                "image": result.image,
                "defects": result.defects,
                "summary": result.summary,
                "seconds": round(result.seconds, 2),
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "error": result.error,
                "raw": result.raw[:2000],
            }
            with print_lock:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                fh.flush()
                if result.error:
                    counters["error"] += 1
                    print(f"[{index}/{len(todo)}] {result.image} 失败:{result.error[:70]}")
                else:
                    counters["ok"] += 1
                    counters["defects"] += len(result.defects)
                    counters["seconds"] += result.seconds
                    counters["prompt_tokens"] += result.prompt_tokens
                    counters["completion_tokens"] += result.completion_tokens
                    types = ", ".join(d.get("type", "?") for d in result.defects) or "未发现"
                    print(f"[{index}/{len(todo)}] {result.image}  "
                          f"{len(result.defects)} 处缺陷({types})  {result.seconds:.1f}s")

    ok = counters["ok"] or 1
    cost = (counters["prompt_tokens"] * PRICE_INPUT
            + counters["completion_tokens"] * PRICE_OUTPUT) / 1e6
    print("\n===== 汇总 =====")
    print(f"成功 {counters['ok']} 张,失败 {counters['error']} 张")
    print(f"平均每张耗时 {counters['seconds'] / ok:.1f} 秒")
    print(f"每张平均报出缺陷 {counters['defects'] / ok:.1f} 处")
    print(f"token 用量:输入 {counters['prompt_tokens']},输出 {counters['completion_tokens']}")
    print(f"本次花费约 ${cost:.4f}")
    print(f"结果文件:{out_path}")


if __name__ == "__main__":
    main()
