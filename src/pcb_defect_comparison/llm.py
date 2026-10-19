"""视觉大模型方案:调用 DeepSeek 的 deepseek-flash 判断 PCB 缺陷。

设计要点:
1. 让模型输出结构化 JSON,才能和传统检测模型的结果做机器对比;
2. 除了缺陷类型,还要求它给出归一化坐标(0~1)与所在区域,
   这样"定位能力"这个维度也能量化——这是两种方案最关键的差异之一;
3. 图片统一缩放到长边 1536 再发送:太小会丢掉小缺陷,太大浪费 token。
"""

import base64
import io
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests
from PIL import Image

API_URL = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-flash"
MAX_SIDE = 1536

# 思考强度:DeepSeek 默认是 high,会花大量 token 做链式推理。
# 实测 high 时单张图要 50~125 秒、上万个 token,对批量实验太慢太贵;
# 这里默认降到 low("none" 表示完全关闭思考模式)。
DEFAULT_EFFORT = "low"

CLASS_KEYS = ["missing_hole", "mouse_bite", "open_circuit",
              "short", "spur", "spurious_copper"]

PROMPT = """这是一张印刷电路板(PCB)的照片。请检查图中是否存在下列 6 类缺陷:

1. missing_hole:缺孔,该有孔的地方没有孔
2. mouse_bite:鼠咬,板边被咬掉一块
3. open_circuit:开路,导线断开
4. short:短路,不该连的导线连在一起
5. spur:毛刺,多出一小段铜
6. spurious_copper:多余铜,不该有铜的地方有铜

要求:
1. 逐个列出你发现的每一个缺陷,同类缺陷若有多处请分别列出;
2. 每个缺陷给出大致位置(左上/上中/右上/左中/中央/右中/左下/下中/右下);
3. 给出该缺陷中心的归一化坐标 center,以左上角为 (0,0)、右下角为 (1,1),
   例如 [0.83, 0.79] 表示横向 83%、纵向 79% 的位置;
4. 给出严重程度 severity,取值只能是 轻微 或 明显;
5. 只输出 JSON,不要任何解释文字。

输出格式:
{"defects": [{"type": "missing_hole", "location": "左上", "center": [0.83, 0.79], "severity": "明显"}], "summary": "一句话总结"}"""


@dataclass
class LLMResult:
    """单张图片的调用结果。"""

    image: str
    defects: list[dict] = field(default_factory=list)
    summary: str = ""
    raw: str = ""
    seconds: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: str = ""


def encode_image(path: Path, max_side: int = MAX_SIDE) -> tuple[str, tuple[int, int]]:
    """把图片缩放并编码成 base64,同时返回原始尺寸。"""
    with Image.open(path) as im:
        original = im.size
        image = im.convert("RGB")
        if max(image.size) > max_side:
            scale = max_side / max(image.size)
            image = image.resize(
                (max(1, int(image.width * scale)), max(1, int(image.height * scale))),
                Image.LANCZOS,
            )
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode(), original


def parse_json_reply(text: str) -> dict:
    """从模型回复里抠出 JSON(有时会被 ```json 包裹)。"""
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", cleaned)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    return {}


def ask_one(path: Path, api_key: str | None = None, timeout: int = 180,
            effort: str = DEFAULT_EFFORT) -> LLMResult:
    """对单张图片发起一次视觉问答。"""
    key = api_key or os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise RuntimeError("未设置环境变量 DEEPSEEK_API_KEY")

    result = LLMResult(image=path.name)
    b64, original = encode_image(path)
    result.original_size = original  # type: ignore[attr-defined]
    payload = {
        "model": MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ],
        }],
        # 注意:思考模式下不支持 temperature,所以这里不能设
        "thinking": {"type": "disabled" if effort == "none" else "enabled"},
    }
    if effort != "none":
        payload["reasoning_effort"] = effort

    started = time.time()
    try:
        response = requests.post(
            API_URL,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {key}"},
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        result.raw = content
        result.prompt_tokens = int(usage.get("prompt_tokens", 0))
        result.completion_tokens = int(usage.get("completion_tokens", 0))
        parsed = parse_json_reply(content)
        result.defects = [d for d in parsed.get("defects", []) if isinstance(d, dict)]
        result.summary = parsed.get("summary", "")
        for defect in result.defects:
            raw_type = str(defect.get("type", "")).strip().lower().replace(" ", "_")
            defect["type"] = raw_type
    except Exception as exc:  # noqa: BLE001
        result.error = f"{type(exc).__name__}: {exc}"[:200]
    result.seconds = time.time() - started
    return result
