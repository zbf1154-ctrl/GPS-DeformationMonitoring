"""GNSS 坐标时间序列形变分析工具。

对单站 GNSS 坐标时间序列（East / North / Up）建立形变模型：

    长期线性趋势 + 周年/半周年季节项 + 同震阶跃 + 震后松弛（或速率变化）

并给出参数估计、形式精度、拟合优度与图表输出。

典型用法::

    from gnss_deformation import AnalysisConfig, load_station_data, analyze_station

    data = load_station_data("data/CKSV站点坐标数据.csv")
    analysis = analyze_station(data, AnalysisConfig())
    print(analysis.summary())
"""

from __future__ import annotations

from ._version import __version__
from .analysis import StationAnalysis, analyze_station
from .config import HUALIEN_2024, MODEL_NAMES, AnalysisConfig, Earthquake
from .data import StationData, load_station_data
from .models import ModelFit, ModelSpec, fit_best_model, fit_model

__all__ = [
    "AnalysisConfig",
    "Earthquake",
    "HUALIEN_2024",
    "MODEL_NAMES",
    "ModelFit",
    "ModelSpec",
    "StationAnalysis",
    "StationData",
    "analyze_station",
    "fit_best_model",
    "fit_model",
    "load_station_data",
    "__version__",
]
