"""pytest 公共配置与 fixture。"""

from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")

from pathlib import Path

import pytest

from helpers import synthetic_series, write_station_csv

from gnss_deformation.analysis import analyze_station
from gnss_deformation.config import AnalysisConfig
from gnss_deformation.data import StationData, load_station_data


@pytest.fixture
def station_csv(tmp_path: Path) -> Path:
    """一个 6 历元、含空行与重复历元的测试 CSV。"""
    rows = [
        {"time": 2022.0, "east": 0.100, "north": 0.500, "up": 1.000},
        {"time": 2022.1, "east": 0.101, "north": 0.499, "up": 1.002},
        {"time": 2022.2, "east": 0.102, "north": 0.498, "up": 1.004},
        {"time": 2022.2, "east": 0.102, "north": 0.498, "up": 1.005},  # 重复历元
        {"time": 2022.3, "east": 0.103, "north": 0.497, "up": 1.006},
        {"time": 2022.4, "east": 0.104, "north": 0.496, "up": 1.008},
    ]
    return write_station_csv(tmp_path / "TEST站点坐标数据.csv", rows)


@pytest.fixture
def synthetic_station(tmp_path: Path) -> StationData:
    """约 3.5 年、日采样的合成站点数据。"""
    t, y = synthetic_series(n=1250)
    rows = [
        {
            "site": "SYN",
            "time": float(ti),
            "east": float(yi) / 1000.0,
            "north": float(-0.5 * yi) / 1000.0,
            "up": float(0.2 * yi) / 1000.0,
        }
        for ti, yi in zip(t, y)
    ]
    path = write_station_csv(tmp_path / "SYN站点坐标数据.csv", rows)
    return load_station_data(path, year_range=None)


@pytest.fixture
def synthetic_analysis(synthetic_station: StationData):
    config = AnalysisConfig(model="step+exp", weighted=False)
    return analyze_station(synthetic_station, config)
