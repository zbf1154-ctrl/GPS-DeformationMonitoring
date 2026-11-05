"""数据读取与清洗测试。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gnss_deformation.data import load_station_data, resolve_data_path

from helpers import write_station_csv


def test_loads_cleans_and_converts_units(station_csv: Path) -> None:
    data = load_station_data(station_csv, year_range=None)

    # 6 行观测 + 1 行空行；重复历元保留最后一条
    assert data.n_raw == 7
    assert data.n_used == 5
    assert data.dropped_rows == 2
    assert data.n_epochs == 5

    # 以首个历元归零，m -> mm
    assert data.east[0] == pytest.approx(0.0)
    assert data.east[1] == pytest.approx(1.0, abs=1e-6)
    assert data.north[1] == pytest.approx(-1.0, abs=1e-6)
    # 重复历元取后者：up = 1.005 - 1.000 = 5 mm
    assert data.up[2] == pytest.approx(5.0, abs=1e-6)

    assert data.station == "TEST"
    assert data.time.dtype == np.float64
    assert np.all(np.diff(data.time) > 0)


def test_sigma_is_converted_to_mm(station_csv: Path) -> None:
    data = load_station_data(station_csv, year_range=None)
    sigma = data.sigma_for("East")
    assert sigma is not None
    assert sigma[0] == pytest.approx(0.8, rel=1e-6)
    assert data.sigma_for("Up") is not None


def test_year_range_filter(tmp_path: Path) -> None:
    rows = [
        {"time": 2021.5, "east": 0.0, "north": 0.0, "up": 0.0},
        {"time": 2022.0, "east": 0.001, "north": 0.001, "up": 0.001},
        {"time": 2023.0, "east": 0.002, "north": 0.002, "up": 0.002},
        {"time": 2024.0, "east": 0.003, "north": 0.003, "up": 0.003},
    ]
    path = write_station_csv(tmp_path / "range.csv", rows)
    data = load_station_data(path, year_range=(2022.0, 2023.0))
    assert data.time_span == pytest.approx((2022.0, 2023.0))


def test_mean_reference(tmp_path: Path) -> None:
    rows = [
        {"time": 2022.0, "east": 0.000, "north": 0.000, "up": 0.000},
        {"time": 2023.0, "east": 0.010, "north": 0.010, "up": 0.010},
        {"time": 2024.0, "east": 0.020, "north": 0.020, "up": 0.020},
    ]
    path = write_station_csv(tmp_path / "mean.csv", rows)
    data = load_station_data(path, year_range=None, reference="mean")
    assert np.mean(data.east) == pytest.approx(0.0, abs=1e-9)
    assert data.east[-1] == pytest.approx(10.0, abs=1e-6)


def test_missing_component_column_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text(
        "site,yyyy.yyyy,_north(m),____up(m)\nTEST,2022.0,0.1,0.2\n",
        encoding="utf-8-sig",
    )
    with pytest.raises(ValueError, match="East"):
        load_station_data(path, year_range=None)


def test_missing_time_column_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad_time.csv"
    path.write_text(
        "site,epoch,__east(m),_north(m),____up(m)\nTEST,2022.0,0.1,0.2,0.3\n",
        encoding="utf-8-sig",
    )
    with pytest.raises(ValueError, match="时间列"):
        load_station_data(path, year_range=None)


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_station_data(tmp_path / "nope.csv")


def test_resolve_data_path_prefers_data_folder(tmp_path: Path, station_csv: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    preferred = data_dir / "CKSV站点坐标数据.csv"
    preferred.write_bytes(station_csv.read_bytes())
    (data_dir / "other.csv").write_text("x", encoding="utf-8")

    resolved = resolve_data_path(None, root=tmp_path)
    assert resolved.name == "CKSV站点坐标数据.csv"

    with pytest.raises(FileNotFoundError):
        resolve_data_path(tmp_path / "absent.csv")
