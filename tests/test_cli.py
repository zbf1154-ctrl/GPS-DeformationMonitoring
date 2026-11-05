"""命令行接口测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

from gnss_deformation.cli import main

from helpers import synthetic_series, write_station_csv


def _make_csv(tmp_path: Path, name: str = "SYN站点坐标数据.csv") -> Path:
    t, y = synthetic_series(n=900)
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
    return write_station_csv(tmp_path / name, rows)


def test_cli_runs_end_to_end(tmp_path: Path, capsys) -> None:
    csv_path = _make_csv(tmp_path)
    outdir = tmp_path / "outputs"

    code = main(
        [
            "--csv",
            str(csv_path),
            "--output-dir",
            str(outdir),
            "--model",
            "step+exp",
        ]
    )

    assert code == 0
    captured = capsys.readouterr().out
    assert "长期速度" in captured
    assert "输出文件" in captured

    expected = [
        outdir / "SYN_deformation.png",
        outdir / "SYN_parameters.csv",
        outdir / "SYN_timeseries.csv",
        outdir / "SYN_summary.md",
        outdir / "SYN_results.json",
    ]
    for path in expected:
        assert path.is_file(), path
        assert path.stat().st_size > 0

    png = expected[0].read_bytes()
    assert png.startswith(b"\x89PNG")


def test_cli_no_plot_skips_figure(tmp_path: Path) -> None:
    csv_path = _make_csv(tmp_path)
    outdir = tmp_path / "out"
    code = main(
        [
            "--csv",
            str(csv_path),
            "--output-dir",
            str(outdir),
            "--no-plot",
            "--quiet",
        ]
    )
    assert code == 0
    assert not (outdir / "SYN_deformation.png").exists()
    assert (outdir / "SYN_results.json").exists()


def test_cli_missing_file_returns_2(tmp_path: Path, capsys) -> None:
    code = main(["--csv", str(tmp_path / "absent.csv")])
    assert code == 2
    assert "找不到数据文件" in capsys.readouterr().err


def test_cli_bad_date_returns_2(tmp_path: Path, capsys) -> None:
    csv_path = _make_csv(tmp_path)
    code = main(["--csv", str(csv_path), "--eq-date", "2024/04/03"])
    assert code == 2
    assert "eq-date" in capsys.readouterr().err


def test_cli_custom_event_and_range(tmp_path: Path) -> None:
    csv_path = _make_csv(tmp_path)
    outdir = tmp_path / "out"
    code = main(
        [
            "--csv",
            str(csv_path),
            "--output-dir",
            str(outdir),
            "--eq-date",
            "2024-01-01",
            "--eq-name",
            "Test event",
            "--year-range",
            "2022",
            "2025",
            "--harmonics",
            "1",
            "--no-plot",
            "--quiet",
        ]
    )
    assert code == 0
    summary = (outdir / "SYN_summary.md").read_text(encoding="utf-8")
    assert "Test event" in summary


def test_cli_weighted_and_robust(tmp_path: Path) -> None:
    csv_path = _make_csv(tmp_path)
    code = main(
        [
            "--csv",
            str(csv_path),
            "--output-dir",
            str(tmp_path / "out"),
            "--weighted",
            "--robust",
            "--no-plot",
            "--quiet",
        ]
    )
    assert code == 0


def test_parser_defaults() -> None:
    from gnss_deformation.cli import build_parser

    args = build_parser().parse_args([])
    assert args.model == "auto"
    assert args.year_range is None
    assert args.harmonics == 2
    assert args.tau_bounds == (0.02, 5.0)
