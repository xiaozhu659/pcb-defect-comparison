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

两种方案在同一批 **138 张验证集图片**上的对比(大模型取表现最好的一档):

| 方法 | 类型判断准确率 | 漏检率 | 数量误差 | 定位误差 | 单张耗时 |
| --- | --- | --- | --- | --- | --- |
| **传统检测(YOLO)** | **97.8%** | **2.2%** | 0.33 | **0.003** | **0.054 秒** |
| 大模型 · 关闭思考 | 9.4% | 53.6% | 3.25 | 0.218 | 1.85 秒 |
| 大模型 · 低强度思考 | 44.2% | 47.8% | 2.92 | 0.249 | 49.3 秒 |
| 大模型 · 高强度思考 | 65.9% | 23.9% | 2.90 | 0.251 | 82.4 秒 |

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
- [ ] 说明文档与个人感悟
- [ ] 演示视频与交付打包
