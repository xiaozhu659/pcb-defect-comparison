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
├── data/                   # 数据集(不入库)
├── docs/                   # 项目计划与说明文档
├── outputs/                # 图表与统计结果(不入库)
├── runs/                   # 训练产物(不入库)
├── scripts/                # 命令行入口
│   ├── check_env.py        # 环境自检
│   ├── download_dataset.py # 数据集下载
│   └── analyze_dataset.py  # 数据统计分析
└── src/
    └── pcb_defect_comparison/   # 核心模块
```

## 当前进度

- [x] 环境搭建与自检
- [x] 数据集下载与统计分析
- [ ] 传统方案:YOLO 训练 + SAM 分割
- [ ] 大模型方案:视觉 API 调用
- [ ] 并排对比与结论
- [ ] 说明文档与演示视频
