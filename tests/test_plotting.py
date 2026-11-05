"""绘图与配置测试。"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pytest

from gnss_deformation.config import AnalysisConfig, Earthquake
from gnss_deformation.plotting import configure_style, plot_station

from datetime import date


def test_plot_station_writes_png(synthetic_analysis, tmp_path) -> None:
    target = tmp_path / "figure.png"
    fig = plot_station(synthetic_analysis, target, dpi=90)

    assert target.is_file()
    assert target.stat().st_size > 5_000
    assert len(fig.axes) == 6  # 3 分量 × (序列, 残差)
    plt.close(fig)


def test_plot_without_output_returns_figure(synthetic_analysis) -> None:
    fig = plot_station(synthetic_analysis, None, dpi=70)
    assert len(fig.axes) == 6
    plt.close(fig)


def test_configure_style_returns_font_name() -> None:
    font = configure_style()
    assert isinstance(font, str) and font


def test_earthquake_decimal_year() -> None:
    event = Earthquake("Hualien", date(2024, 4, 3))
    assert event.decimal_year == pytest.approx(2024 + 94 / 366.0, abs=1e-12)

    non_leap = Earthquake("New year", date(2023, 1, 1))
    assert non_leap.decimal_year == pytest.approx(2023 + 1 / 365.0, abs=1e-12)


def test_config_validation() -> None:
    with pytest.raises(ValueError, match="未知模型"):
        AnalysisConfig(model="bogus")
    with pytest.raises(ValueError, match="tau_bounds"):
        AnalysisConfig(tau_bounds=(0.0, 1.0))
    with pytest.raises(ValueError, match="year_range"):
        AnalysisConfig(year_range=(2025.0, 2020.0))
    with pytest.raises(ValueError, match="reference"):
        AnalysisConfig(reference="median")


def test_config_serialisation() -> None:
    config = AnalysisConfig()
    payload = config.to_dict()
    assert payload["earthquake"]["date"] == "2024-04-03"
    assert payload["components"] == ["East", "North", "Up"]
