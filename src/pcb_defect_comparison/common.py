"""项目公共配置:路径、类别名、绘图字体设置。"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
FIGURE_DIR = PROJECT_ROOT / "outputs"

# 数据集中 XML 的类别名是"小写下划线"形式,这里统一成规范写法并附中文
CLASS_ALIASES = {
    "missing_hole": "Missing_hole",
    "mouse_bite": "Mouse_bite",
    "open_circuit": "Open_circuit",
    "short": "Short",
    "spur": "Spur",
    "spurious_copper": "Spurious_copper",
}

CLASS_NAMES_ZH = {
    "Missing_hole": "缺孔",
    "Mouse_bite": "鼠咬",
    "Open_circuit": "开路",
    "Short": "短路",
    "Spur": "毛刺",
    "Spurious_copper": "多余铜",
}

# 固定顺序,保证各处图表排序一致
CLASS_ORDER = ["Missing_hole", "Mouse_bite", "Open_circuit",
               "Short", "Spur", "Spurious_copper"]


def normalize_class(raw: str) -> str:
    """把 XML 里的类别名规范成统一写法。"""
    return CLASS_ALIASES.get(raw.strip().lower(), raw.strip())


def class_label(name: str) -> str:
    """返回“英文名(中文)”形式的标签,图表里用。"""
    return f"{name}\n({CLASS_NAMES_ZH.get(name, name)})"


def setup_chinese_font() -> None:
    """让 matplotlib 正常显示中文,否则中文会变成方框。"""
    import matplotlib

    matplotlib.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "DejaVu Sans",
    ]
    matplotlib.rcParams["axes.unicode_minus"] = False
