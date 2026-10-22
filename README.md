# PCB 缺陷检测:传统检测模型 vs 视觉大模型

灵境竞赛组 · 后端 AI 方向 · 第二阶段考核项目

同一个视觉问题、同一组输入,用两种技术路线分别解决并做对比:

- **传统深度学习方案**:YOLO 检测缺陷位置与类别,SAM 由检测框提示生成精确掩膜;
- **视觉 / 多模态大模型方案**:调用 DeepSeek 的 `deepseek-flash` 零样本判断缺陷类型与位置。

## 数据集

PCB 缺陷数据集(HRIPCB 格式),共 **693 张** PCB 板图、**6 类缺陷**、
**2953 个标注框**,标注为 PASCAL VOC 格式,MIT 许可。

| 类别 | 含义 |
| --- | --- |
| Missing_hole | 缺孔 |
| Mouse_bite | 鼠咬(板边缺损) |
| Open_circuit | 开路 |
| Short | 短路 |
| Spur | 毛刺 |
| Spurious_copper | 多余铜 |

数据通过 Hugging Face 镜像下载,不需要代理:

```bash
uv run scripts/download_dataset.py
```

## 主要结果

两种方案在同一批 **138 张验证集图片**上的对比:

| 方法 | 严格命中 | 类型精确率 | 漏检率 | 定位误差 | 单张耗时 |
| --- | --- | --- | --- | --- | --- |
| **传统检测(YOLO)** | **90.6%** | **93.1%** | **2.2%** | **0.003** | **0.054 秒** |
| 大模型 · 关闭思考 | 5.1% | 14.3% | 53.6% | 0.218 | 1.85 秒 |
| 大模型 · 低强度思考 | 1.4% | 17.0% | 47.8% | 0.249 | 49.3 秒 |
| 大模型 · 高强度思考 | **0.7%** | **16.9%** | 23.9% | 0.251 | 82.4 秒 |

> "严格命中"要求预测的缺陷类型集合与真实集合完全一致。若只统计"包含真实类型",
> 高强度档看似有 65.9%,但它在 65 张图上把 6 类缺陷全报了一遍,
> 属于用大量误报换召回,因此这里统一采用严格口径与精确率。

![方法对比](docs/figures/method_comparison.png)

![并排对比](docs/figures/side_by_side.png)

**结论:两者是互补而非替代。** 专用模型负责"看见并量准"——缺陷面积中位数
只占整图 0.037%,它能稳定检出并量出面积;通用大模型负责"看懂并解释",
但在这种微小目标上零样本表现有限,且思考越多耗时越长。

完整的指标定义、逐项分析与适用场景见 [docs/实验结果.md](docs/实验结果.md)。

## 环境

| 项目 | 版本 |
| --- | --- |
| Python | 3.12(.python-version 固定) |
| 包管理 | uv |
| PyTorch | 2.11 + CUDA 12.8 |
| 检测框架 | ultralytics 8.4 |
| 显卡 | NVIDIA RTX 5060 Laptop(8 GB) |

```bash
uv sync
uv run scripts/check_env.py     # 确认 CUDA、显卡、依赖都可用
```

## 复现步骤

### 前提条件

| 项目 | 要求 |
| --- | --- |
| 显卡 | NVIDIA 独显,建议 ≥ 8 GB 显存(需支持 CUDA 12.8 的驱动) |
| 网络 | 数据集走 hf-mirror 镜像,无需代理;YOLO 与 SAM 的权重从 GitHub Releases 下载,网络不稳时需自备代理 |
| API | DeepSeek API key(`deepseek-flash` 支持图片输入),全部实验约花费 1.75 美元 |

### 完整流程

总耗时约 1.5 小时(在 RTX 5060 上实测),其中训练 15 分钟、大模型三档约 50 分钟。

```bash
uv sync                                   # 安装依赖(含 CUDA 12.8 版 PyTorch)
uv run scripts/check_env.py               # 环境自检
uv run scripts/download_dataset.py        # 下载 916 MB 数据集(约 8~10 分钟)
uv run scripts/analyze_dataset.py         # 数据统计
uv run scripts/prepare_yolo_dataset.py    # 标注转换与训练/验证划分
uv run scripts/train_yolo.py --epochs 100 --imgsz 1024 --batch 8   # 约 15 分钟
uv run scripts/yolo_infer.py              # 传统方案推理(约 10 秒)
```

大模型方案需要先设置 API key(**只放在环境变量里,不要写进代码**):

```powershell
$env:DEEPSEEK_API_KEY = "你的 key"        # PowerShell
```

```bash
export DEEPSEEK_API_KEY="你的 key"        # bash
```

```bash
uv run scripts/llm_infer.py --workers 6 --effort none   # 约 3 分钟,花费 < $0.05
uv run scripts/llm_infer.py --workers 6 --effort low    # 约 20 分钟,约 $0.60
uv run scripts/llm_infer.py --workers 6 --effort high   # 约 30 分钟,约 $1.12
uv run scripts/sam_segment.py             # YOLO + SAM 分割(首次会下载 74.5 MB 权重)
uv run scripts/compare_methods.py         # 量化对比
uv run scripts/visualize_side_by_side.py  # 并排可视化
```

大模型那三条命令支持**断点续跑**:中断后重新执行会跳过已完成的图片。

### 结果位置

| 产物 | 路径 |
| --- | --- |
| 量化对比表 | `outputs/comparison.csv` |
| 方法对比图 | `outputs/fig_method_comparison.png` |
| 并排可视化 | `outputs/fig_side_by_side.png` |
| 分割结果与面积 | `outputs/sam_segmentation.jsonl` |

## 目录结构

```
.
├── pyproject.toml          # 依赖与 PyTorch CUDA 源配置
├── uv.lock                 # 锁定精确版本,保证环境可复现
├── data/                   # 数据集与转换后的标注(不入库)
├── docs/                   # 项目计划、实验结果、图表
├── outputs/                # 每次运行的工作目录(不入库)
├── runs/                   # 训练产物(不入库)
├── scripts/
│   ├── check_env.py            # 环境自检
│   ├── download_dataset.py     # 数据集下载
│   ├── analyze_dataset.py      # 类别与尺寸统计
│   ├── prepare_yolo_dataset.py # VOC → YOLO 标注转换与划分
│   ├── train_yolo.py           # 训练检测模型
│   ├── yolo_infer.py           # 传统方案推理
│   ├── llm_infer.py            # 大模型方案推理
│   ├── sam_segment.py          # YOLO + SAM 组合分割
│   ├── compare_methods.py      # 量化对比
│   ├── visualize_side_by_side.py # 并排可视化
│   └── collect_figures.py      # 定稿图整理入库
└── src/
    └── pcb_defect_comparison/  # 核心模块(公共配置、大模型调用)
```

## 当前进度

- [x] 环境搭建与自检
- [x] 数据集下载与统计分析
- [x] 传统方案:YOLO 训练 + SAM 分割
- [x] 大模型方案:视觉 API 调用与三档思考强度
- [x] 并排对比与结论
- [x] 说明文档与个人感悟(见 [docs/项目说明.md](docs/项目说明.md))
- [x] 演示视频与交付打包
