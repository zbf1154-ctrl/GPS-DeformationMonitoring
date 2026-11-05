"""命令行入口。"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt

from ._version import __version__
from .analysis import analyze_station
from .config import MODEL_NAMES, AnalysisConfig, Earthquake
from .data import load_station_data, resolve_data_path
from .plotting import configure_style, plot_station
from .reporting import console_report, write_outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gnss-deformation",
        description="GNSS 坐标时间序列形变分析：长期趋势、季节项、同震阶跃与震后松弛。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--csv", type=Path, help="站点坐标 CSV 文件；缺省时自动在 data/ 中查找")
    parser.add_argument("--station", help="站点名（缺省从 site 列或文件名推断）")
    parser.add_argument(
        "--year-range",
        nargs=2,
        type=float,
        metavar=("START", "END"),
        default=None,
        help="时间筛选区间（十进制年，闭区间）；缺省使用全部历元",
    )
    parser.add_argument("--all-years", action="store_true", help="兼容选项：等同于不筛选年份")
    parser.add_argument(
        "--eq-date",
        default="2024-04-03",
        help="同震事件日期，格式 YYYY-MM-DD",
    )
    parser.add_argument(
        "--eq-name",
        default="2024 Mw 7.4 Hualien, Taiwan",
        help="同震事件名称（仅用于输出标注）",
    )
    parser.add_argument(
        "--model",
        choices=("auto",) + MODEL_NAMES,
        default="auto",
        help="形变模型；auto 表示按 BIC 在候选模型中自动选择",
    )
    parser.add_argument("--harmonics", type=int, default=2, choices=(0, 1, 2), help="季节项谐波个数")
    parser.add_argument(
        "--tau-bounds",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        default=(0.02, 5.0),
        help="松弛时间常数搜索区间（年）",
    )
    parser.add_argument("--weighted", action="store_true", help="使用数据自带形式中误差加权")
    parser.add_argument("--robust", action="store_true", help="使用 Huber 稳健损失")
    parser.add_argument(
        "--reference",
        choices=("first", "mean"),
        default="first",
        help="参考基准：首个历元或全序列均值",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="结果输出目录")
    parser.add_argument("--figure", type=Path, help="图片输出路径（默认 <output-dir>/<station>_deformation.png）")
    parser.add_argument("--dpi", type=int, default=200, help="图片分辨率")
    parser.add_argument("--no-plot", action="store_true", help="不生成图片")
    parser.add_argument("--show", action="store_true", help="弹出交互窗口显示图片")
    parser.add_argument("--quiet", action="store_true", help="不打印文字报告")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def run(args: argparse.Namespace) -> int:
    try:
        eq_date = date.fromisoformat(args.eq_date)
    except ValueError as exc:
        raise ValueError(f"--eq-date 无法解析：{args.eq_date}（应为 YYYY-MM-DD）") from exc

    earthquake = Earthquake(name=args.eq_name, date=eq_date)
    year_range = None if args.all_years or args.year_range is None else tuple(args.year_range)

    config = AnalysisConfig(
        earthquake=earthquake,
        year_range=year_range,
        model=args.model,
        harmonics=args.harmonics,
        tau_bounds=tuple(args.tau_bounds),
        weighted=args.weighted,
        reference=args.reference,
        robust=args.robust,
    )

    csv_path = resolve_data_path(args.csv)
    data = load_station_data(
        csv_path,
        year_range=year_range,
        reference=args.reference,
        station=args.station,
    )
    analysis = analyze_station(data, config)

    if not args.quiet:
        print(console_report(analysis))

    figure_path: Optional[Path] = None
    if not args.no_plot:
        configure_style()
        figure_path = args.figure or (args.output_dir / f"{analysis.station}_deformation.png")
        figure = plot_station(analysis, figure_path, show=args.show, dpi=args.dpi)
        plt.close(figure)

    paths = write_outputs(analysis, args.output_dir, figure=figure_path)

    if not args.quiet:
        print("\n输出文件：")
        if figure_path is not None:
            print(f"  [图片] {figure_path.resolve()}")
        for label, path in paths.items():
            print(f"  [{label}] {path.resolve()}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run(args)
    except FileNotFoundError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:  # pragma: no cover
        print("已中断", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
