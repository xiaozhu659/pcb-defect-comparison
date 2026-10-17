"""PCB 缺陷检测:传统检测模型与视觉大模型的对比。

模块分工:
    common      路径、类别名与绘图字体等公共配置
    llm         视觉大模型(DeepSeek deepseek-flash)调用与结构化解析

各功能的命令行入口在 scripts/ 目录下。
"""


def main() -> None:
    """控制台入口:打印模块说明。"""
    print(__doc__)


if __name__ == "__main__":
    main()
