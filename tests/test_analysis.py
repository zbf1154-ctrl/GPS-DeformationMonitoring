"""分析流程与结果输出测试。"""

from __future__ import annotations

import json

import numpy as np
import pytest

from gnss_deformation.analysis import TABLE_COLUMNS, analyze_station
from gnss_deformation.config import AnalysisConfig, Earthquake
from datetime import date

from gnss_deformation.reporting import (
    console_report,
    markdown_summary,
    write_outputs,
)

from helpers import T_EQ


def test_analysis_produces_all_components(synthetic_station) -> None:
    analysis = analyze_station(synthetic_station, AnalysisConfig(model="step+exp"))

    assert set(analysis.fits) == {"East", "North", "Up"}
    assert analysis.t_eq == pytest.approx(T_EQ, abs=1e-9)
    for fit in analysis.fits.values():
        assert fit.spec.name == "step+exp"
        assert fit.success


def test_table_has_expected_columns(synthetic_analysis) -> None:
    table = synthetic_analysis.table()
    assert list(table.columns) == TABLE_COLUMNS
    assert len(table) == 3
    assert np.isfinite(table["velocity_mm_yr"]).all()


def test_velocity_and_seasonal_terms_are_recovered(synthetic_analysis) -> None:
    east = synthetic_analysis.fits["East"]
    assert east.velocity == pytest.approx(10.0, abs=0.2)
    assert east.annual_amplitude == pytest.approx(np.hypot(2.0, -1.0), abs=0.2)


def test_auto_model_selection(synthetic_station) -> None:
    analysis = analyze_station(synthetic_station, AnalysisConfig(model="auto"))
    for comp, fit in analysis.fits.items():
        assert fit.spec.name in analysis.candidates[comp]
        assert fit.bic == min(c.bic for c in analysis.candidates[comp].values())


def test_weighted_analysis(synthetic_station) -> None:
    analysis = analyze_station(
        synthetic_station, AnalysisConfig(model="step+exp", weighted=True)
    )
    for fit in analysis.fits.values():
        assert fit.chi2_red is not None


def test_out_of_range_event_raises(synthetic_station) -> None:
    config = AnalysisConfig(
        earthquake=Earthquake("Future event", date(2030, 1, 1))
    )
    with pytest.raises(ValueError, match="不在数据时间范围"):
        analyze_station(synthetic_station, config)


def test_markdown_and_json_outputs(synthetic_analysis, tmp_path) -> None:
    markdown = markdown_summary(synthetic_analysis)
    assert "# SYN 形变分析结果" in markdown
    assert "| East |" in markdown
    assert "step+exp" in markdown

    report = console_report(synthetic_analysis)
    assert "长期速度" in report
    assert "RMS" in report

    paths = write_outputs(synthetic_analysis, tmp_path / "out")
    for key in ("parameters", "timeseries", "summary", "results"):
        assert paths[key].is_file()
        assert paths[key].stat().st_size > 0

    payload = json.loads(paths["results"].read_text(encoding="utf-8"))
    assert payload["station"] == "SYN"
    assert set(payload["results"]) == {"East", "North", "Up"}

    import pandas as pd

    table = pd.read_csv(paths["parameters"], encoding="utf-8-sig")
    assert list(table.columns) == TABLE_COLUMNS


def test_warning_when_tau_hits_bound(synthetic_station) -> None:
    analysis = analyze_station(
        synthetic_station,
        AnalysisConfig(model="step+exp", tau_bounds=(0.02, 0.021)),
    )
    assert any("触及搜索边界" in message for message in analysis.warnings)
