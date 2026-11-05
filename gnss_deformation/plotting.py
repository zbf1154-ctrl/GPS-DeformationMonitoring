"""图表绘制。

默认输出“左列：观测与模型，右列：残差”的三行双栏图，适合直接用于报告
或论文插图。所有中文/英文标签都做了字体回退，避免缺字形警告。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from matplotlib.transforms import blended_transform_factory

from .analysis import StationAnalysis

#: 分量配色（对色盲友好）。
COMPONENT_COLORS = {
    "East": "#1f6feb",
    "North": "#d97706",
    "Up": "#15803d",
}

_CJK_FONTS = (
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "PingFang SC",
    "WenQuanYi Zen Hei",
    "Arial Unicode MS",
)


def configure_style() -> str:
    """统一图表风格，返回实际选用的中文字体名（未找到则为 DejaVu Sans）。"""
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 200,
            "font.size": 10.5,
            "axes.linewidth": 0.9,
            "axes.unicode_minus": False,
            "axes.grid": True,
            "grid.color": "#d9d9d9",
            "grid.linewidth": 0.6,
            "grid.linestyle": "-",
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 4,
            "ytick.major.size": 4,
            "legend.frameon": False,
            "figure.constrained_layout.use": False,
        }
    )
    available = {font.name for font in font_manager.fontManager.ttflist}
    chosen = next((name for name in _CJK_FONTS if name in available), "DejaVu Sans")
    plt.rcParams["font.sans-serif"] = [chosen, "DejaVu Sans", "Arial"]
    plt.rcParams["font.family"] = "sans-serif"
    return chosen


def _annotate_fit(ax: Axes, analysis: StationAnalysis, comp: str) -> None:
    fit = analysis.fits[comp]
    lines = [f"v = {fit.velocity:.2f} ± {fit.velocity_error:.2f} mm/yr"]
    if fit.spec.step:
        lines.append(f"C = {fit.step:.1f} ± {fit.step_error:.1f} mm")
    if fit.spec.relaxation:
        lines.append(f"τ = {fit.relaxation_time:.2f} yr, D = {fit.relaxation_amplitude:.1f} mm")
    if fit.spec.post_rate:
        lines.append(f"Δv = {fit.post_rate:.2f} ± {fit.post_rate_error:.2f} mm/yr")
    lines.append(f"RMS = {fit.rms:.2f} mm")
    ax.text(
        0.03,
        0.05,
        "\n".join(lines),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.5,
        linespacing=1.35,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.9},
    )


def plot_station(
    analysis: StationAnalysis,
    output: Optional[str | Path] = None,
    *,
    show: bool = False,
    dpi: int = 200,
    figsize: Sequence[float] = (13.0, 10.5),
    title: Optional[str] = None,
) -> Figure:
    """绘制三分量形变时间序列与残差。

    Parameters
    ----------
    analysis:
        :func:`gnss_deformation.analysis.analyze_station` 的结果。
    output:
        图片保存路径；``None`` 表示不保存。
    show:
        是否调用 ``plt.show()``。
    dpi, figsize, title:
        输出分辨率、画布尺寸与自定义标题。

    Returns
    -------
    matplotlib.figure.Figure
        生成但未关闭的图对象（由调用方负责 ``plt.close``）。
    """
    configure_style()

    data = analysis.data
    t = data.time
    t_eq = analysis.t_eq
    components = list(analysis.fits)
    n_rows = len(components)

    fig, axes = plt.subplots(
        n_rows,
        2,
        figsize=(figsize[0], figsize[1] * n_rows / 3.0),
        sharex="col",
        gridspec_kw={"width_ratios": [2.2, 1.0], "hspace": 0.12, "wspace": 0.12},
    )
    axes = np.atleast_2d(axes)

    t_max = float(t[-1])
    dense = np.linspace(float(t[0]), t_max, 2000)

    for row, comp in enumerate(components):
        fit = analysis.fits[comp]
        color = COMPONENT_COLORS.get(comp, "#333333")
        observed = data.component(comp)

        ax_series: Axes = axes[row, 0]
        ax_resid: Axes = axes[row, 1]

        ax_series.axvspan(t_eq, t_max, color="#7f7f7f", alpha=0.07, lw=0, zorder=0)
        ax_series.scatter(
            t,
            observed,
            s=3,
            color=color,
            alpha=0.55,
            linewidths=0,
            zorder=2,
            rasterized=True,
            label="Observed",
        )
        ax_series.plot(
            dense,
            fit.predict(dense),
            color="#c1121f",
            lw=1.6,
            zorder=3,
            label=f"Model ({fit.spec.name})",
        )
        ax_series.axvline(t_eq, color="#555555", ls="--", lw=1.0, zorder=1)
        ax_series.set_ylabel(f"{comp} (mm)", fontsize=10.5)
        eq_label_x = blended_transform_factory(ax_series.transData, ax_series.transAxes)
        ax_series.text(
            t_eq + 0.012,
            0.94,
            "EQ",
            transform=eq_label_x,
            color="#555555",
            fontsize=8.5,
            va="top",
            ha="left",
            bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "edgecolor": "none", "alpha": 0.8},
        )
        ax_series.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
        ax_series.tick_params(axis="both", labelsize=9.5)
        _annotate_fit(ax_series, analysis, comp)
        if row == 0:
            ax_series.legend(loc="upper left", fontsize=9, markerscale=0.8)

        ax_resid.axhline(0.0, color="#333333", lw=0.9, zorder=1)
        ax_resid.axvspan(t_eq, t_max, color="#7f7f7f", alpha=0.07, lw=0, zorder=0)
        ax_resid.scatter(
            t,
            fit.residuals,
            s=3,
            color=color,
            alpha=0.55,
            linewidths=0,
            zorder=2,
            rasterized=True,
        )
        band = fit.wrms
        ax_resid.axhspan(-band, band, color=color, alpha=0.10, lw=0, zorder=0)
        ax_resid.axvline(t_eq, color="#555555", ls="--", lw=1.0, zorder=1)
        ax_resid.set_ylabel("Residual (mm)", fontsize=9.5)
        ax_resid.text(
            0.04,
            0.94,
            f"WRMS = {fit.wrms:.2f} mm",
            transform=ax_resid.transAxes,
            fontsize=8.5,
            va="top",
        )
        ax_resid.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
        ax_resid.tick_params(axis="both", labelsize=9.5)

        if row == n_rows - 1:
            ax_series.set_xlabel("Time (year)", fontsize=10.5)
            ax_resid.set_xlabel("Time (year)", fontsize=9.5)

    station = analysis.station
    event = analysis.config.earthquake
    fig.suptitle(
        title or f"{station}  ·  GNSS coordinate time series",
        fontsize=15,
        y=0.995,
    )
    fig.text(
        0.5,
        0.962,
        f"Co-seismic offset and post-seismic deformation after {event.name} "
        f"({event.date.isoformat()}, t_eq = {event.decimal_year:.4f})",
        ha="center",
        fontsize=9.5,
        color="#444444",
    )
    fig.subplots_adjust(top=0.93, bottom=0.055, left=0.075, right=0.985)

    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")

    if show:
        plt.show()
    return fig
