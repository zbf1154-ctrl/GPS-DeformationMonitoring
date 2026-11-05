"""结果整理与输出（控制台、Markdown、CSV、JSON）。"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .analysis import StationAnalysis


def _is_nan(value: Optional[float]) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def _fmt(value: Optional[float], digits: int = 2) -> str:
    if _is_nan(value):
        return "—"
    return f"{value:.{digits}f}"


def _fmt_pm(value: float, sigma: Optional[float], digits: int = 2) -> str:
    if _is_nan(value):
        return "—"
    if _is_nan(sigma):
        return f"{value:.{digits}f}"
    return f"{value:.{digits}f} ± {sigma:.{digits}f}"


def console_report(analysis: StationAnalysis) -> str:
    """生成适合直接打印的纯文本报告。"""
    data = analysis.data
    event = analysis.config.earthquake
    lo, hi = data.time_span
    lines: List[str] = []
    rule = "=" * 78

    lines.append(rule)
    lines.append(f"站点 {analysis.station} | {event.name} ({event.date.isoformat()})")
    lines.append(
        f"数据 {data.source.name} | {data.n_used} 历元 | "
        f"{lo:.4f}–{hi:.4f} | 采样间隔中位数 {data.median_interval_days:.2f} d"
    )
    lines.append(
        f"同震时刻 t_eq = {event.decimal_year:.4f} | 参考历元 {data.reference_epoch:.4f} | "
        f"{'加权' if analysis.config.weighted else '等权'}最小二乘"
    )
    lines.append(rule)

    for comp, fit in analysis.fits.items():
        lines.append(
            f"\n[{comp}]  模型 {fit.spec.name}  (BIC {fit.bic:.1f}, "
            f"RMS {fit.rms:.2f} mm, WRMS {fit.wrms:.2f} mm)"
        )
        lines.append(f"  长期速度 v      : {_fmt_pm(fit.velocity, fit.velocity_error, 3)} mm/yr")
        if fit.spec.step:
            lines.append(f"  同震阶跃 C      : {_fmt_pm(fit.step, fit.step_error, 2)} mm")
        if fit.spec.relaxation:
            lines.append(
                f"  松弛振幅 D      : "
                f"{_fmt_pm(fit.relaxation_amplitude, fit.relaxation_amplitude_error, 2)} mm"
            )
            lines.append(
                f"  松弛时间 τ      : {_fmt_pm(fit.relaxation_time, fit.relaxation_time_error, 3)} yr"
                + ("  ← 触及边界，未有效约束" if fit.tau_at_bound else "")
            )
            lines.append(f"  震后视速率 D/τ  : {fit.post_seismic_slope:.2f} mm/yr")
        if fit.spec.post_rate:
            lines.append(
                f"  震后速率变化 Δv : {_fmt_pm(fit.post_rate, fit.post_rate_error, 2)} mm/yr"
            )
        lines.append(
            f"  季节项振幅      : 周年 {fit.annual_amplitude:.2f} mm, "
            f"半周年 {fit.semiannual_amplitude:.2f} mm"
        )
        lines.append(
            f"  参数个数 {fit.effective_params} | 自由度 {fit.dof} | "
            f"雅可比条件数 {fit.condition_number:.2e}"
        )

    if analysis.warnings:
        lines.append("\n" + "-" * 78)
        for message in analysis.warnings:
            lines.append(f"! {message}")

    lines.append("\n" + rule)
    return "\n".join(lines)


def _table_markdown(table: pd.DataFrame) -> str:
    headers = [
        "分量",
        "模型",
        "速度 v (mm/yr)",
        "阶跃 C (mm)",
        "松弛 D (mm)",
        "τ (yr)",
        "Δv (mm/yr)",
        "RMS (mm)",
        "BIC",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join([":---:"] * len(headers)) + " |",
    ]
    for _, row in table.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["component"]),
                    str(row["model"]),
                    _fmt_pm(row["velocity_mm_yr"], row["velocity_sigma"], 2),
                    _fmt_pm(row["step_mm"], row["step_sigma"], 1),
                    _fmt_pm(row["relax_amp_mm"], row["relax_amp_sigma"], 1),
                    _fmt_pm(row["relax_tau_yr"], row["relax_tau_sigma"], 2),
                    _fmt_pm(row["post_rate_mm_yr"], row["post_rate_sigma"], 2),
                    _fmt(row["rms_mm"], 2),
                    _fmt(row["bic"], 1),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def markdown_summary(analysis: StationAnalysis) -> str:
    """生成 Markdown 格式的摘要，可直接粘贴到报告或 README。"""
    data = analysis.data
    event = analysis.config.earthquake
    lo, hi = data.time_span
    table = analysis.table()
    model_label = (
        "按 BIC 自动选择"
        if analysis.config.model == "auto"
        else f"固定为 `{analysis.config.model}`"
    )

    parts: List[str] = []
    parts.append(f"# {analysis.station} 形变分析结果\n\n")
    parts.append(
        f"- **事件**：{event.name}，{event.date.isoformat()}（t_eq = {event.decimal_year:.4f}）\n"
        f"- **数据**：`{data.source.name}`，{data.n_used} 个历元，{lo:.4f}–{hi:.4f}\n"
        f"- **预处理**：原始 {data.n_raw} 行，剔除/去重 {data.dropped_rows} 行；"
        f"坐标由 m 换算为 mm，并以 {data.reference_epoch:.4f} 历元为参考归零\n"
        f"- **估计方法**：{'加权' if analysis.config.weighted else '等权'}最小二乘，模型{model_label}\n"
    )
    parts.append("\n## 参数估计\n\n")
    parts.append(_table_markdown(table))
    parts.append("\n\n> 表中误差为最小二乘形式精度（1σ）。\n")

    parts.append("\n## 模型\n\n")
    parts.append(
        "```text\n"
        "y(t) = a + v·(t - t0) + Σ_k [ s_k·sin(2πk t) + c_k·cos(2πk t) ]\n"
        "     + H(t - t_eq)·[ C + D·(1 - exp(-(t - t_eq)/τ)) ] + H(t - t_eq)·Δv·(t - t_eq)\n"
        "```\n\n"
        "其中 `H(·)` 为 Heaviside 阶跃函数，k 取 1、2 分别对应周年与半周年项。\n"
    )

    parts.append("\n## 说明\n\n")
    parts.append(
        "- 长期速度 v 由全部历元联合估计，同震阶跃与震后项同时解算，避免两步法带来的误差传播。\n"
        "- 震后观测窗口仅两三年时，D 与 τ 强相关，τ 往往只能得到弱约束；"
        "此时应以 D/τ（震后视速率）或 Δv 描述震后变形。\n"
        "- 形式精度仅反映白噪声假设下的统计不确定性；实际 GNSS 序列存在有色噪声，"
        "真实不确定度通常更大（可用 CATS / Hector 等工具做噪声估计）。\n"
    )
    if analysis.warnings:
        parts.append("\n## 告警\n\n")
        for message in analysis.warnings:
            parts.append(f"- {message}\n")
    return "".join(parts)


def _timeseries_frame(analysis: StationAnalysis) -> pd.DataFrame:
    data = analysis.data
    frame = pd.DataFrame({"time": data.time})
    for comp, fit in analysis.fits.items():
        key = comp.lower()
        frame[f"{key}_obs_mm"] = fit.observed
        frame[f"{key}_model_mm"] = fit.predicted
        frame[f"{key}_resid_mm"] = fit.residuals
        sigma = data.sigma_for(comp)
        if sigma is not None:
            frame[f"{key}_sigma_mm"] = sigma
    return frame


def write_outputs(
    analysis: StationAnalysis,
    outdir: str | Path,
    *,
    figure: Optional[str | Path] = None,
) -> Dict[str, Path]:
    """把分析结果写入目录。

    输出文件：``*_parameters.csv``、``*_timeseries.csv``、``*_summary.md``、
    ``*_results.json``。

    Returns
    -------
    dict
        各输出文件的路径。
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stem = analysis.station
    paths: Dict[str, Path] = {}

    table_path = outdir / f"{stem}_parameters.csv"
    analysis.table().round(4).to_csv(table_path, index=False, encoding="utf-8-sig")
    paths["parameters"] = table_path

    ts_path = outdir / f"{stem}_timeseries.csv"
    _timeseries_frame(analysis).round(4).to_csv(ts_path, index=False, encoding="utf-8-sig")
    paths["timeseries"] = ts_path

    md_path = outdir / f"{stem}_summary.md"
    md_path.write_text(markdown_summary(analysis), encoding="utf-8")
    paths["summary"] = md_path

    payload = analysis.to_dict()
    if figure is not None:
        payload["figure"] = str(figure)
    json_path = outdir / f"{stem}_results.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=float)
    paths["results"] = json_path

    return paths
