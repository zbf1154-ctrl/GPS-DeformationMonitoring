"""分析流程编排：逐分量拟合、模型选择与结果汇总。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .config import AnalysisConfig
from .data import StationData
from .models import ModelFit, fit_best_model

#: 结果表字段顺序（CSV / Markdown 共用）。
TABLE_COLUMNS = [
    "component",
    "model",
    "velocity_mm_yr",
    "velocity_sigma",
    "step_mm",
    "step_sigma",
    "relax_amp_mm",
    "relax_amp_sigma",
    "relax_tau_yr",
    "relax_tau_sigma",
    "post_rate_mm_yr",
    "post_rate_sigma",
    "annual_amp_mm",
    "semiannual_amp_mm",
    "rms_mm",
    "wrms_mm",
    "chi2_red",
    "bic",
    "n_epochs",
]


@dataclass
class StationAnalysis:
    """单站分析结果。"""

    data: StationData
    config: AnalysisConfig
    fits: Dict[str, ModelFit]
    candidates: Dict[str, Dict[str, ModelFit]]
    warnings: List[str] = field(default_factory=list)
    created: str = field(
        default_factory=lambda: datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    )

    @property
    def station(self) -> str:
        return self.data.station

    @property
    def t_eq(self) -> float:
        return self.config.earthquake.decimal_year

    def selected_models(self) -> Dict[str, str]:
        return {comp: fit.spec.name for comp, fit in self.fits.items()}

    def table(self) -> pd.DataFrame:
        """把各分量主要参数整理为 DataFrame。"""
        rows = []
        for comp, fit in self.fits.items():
            rows.append(
                {
                    "component": comp,
                    "model": fit.spec.name,
                    "velocity_mm_yr": fit.velocity,
                    "velocity_sigma": fit.velocity_error,
                    "step_mm": fit.step,
                    "step_sigma": fit.step_error,
                    "relax_amp_mm": fit.relaxation_amplitude,
                    "relax_amp_sigma": fit.relaxation_amplitude_error,
                    "relax_tau_yr": fit.relaxation_time,
                    "relax_tau_sigma": fit.relaxation_time_error,
                    "post_rate_mm_yr": fit.post_rate,
                    "post_rate_sigma": fit.post_rate_error,
                    "annual_amp_mm": fit.annual_amplitude,
                    "semiannual_amp_mm": fit.semiannual_amplitude,
                    "rms_mm": fit.rms,
                    "wrms_mm": fit.wrms,
                    "chi2_red": fit.chi2_red,
                    "bic": fit.bic,
                    "n_epochs": fit.n,
                }
            )
        return pd.DataFrame(rows, columns=TABLE_COLUMNS)

    def residuals(self) -> Dict[str, np.ndarray]:
        return {comp: fit.residuals for comp, fit in self.fits.items()}

    def to_dict(self) -> dict:
        return {
            "station": self.station,
            "created": self.created,
            "data": self.data.to_dict(),
            "config": self.config.to_dict(),
            "results": {comp: fit.summary() for comp, fit in self.fits.items()},
            "parameters": {
                comp: {
                    "values": {k: round(v, 6) for k, v in fit.params.items()},
                    "stderr": {k: round(v, 6) for k, v in fit.stderr.items()},
                }
                for comp, fit in self.fits.items()
            },
            "model_candidates": {
                comp: {
                    name: {"bic": round(fit.bic, 2), "rms_mm": round(fit.rms, 4)}
                    for name, fit in fits.items()
                }
                for comp, fits in self.candidates.items()
            },
            "warnings": list(self.warnings),
        }


def _noise_advisory(fit: ModelFit) -> Optional[str]:
    """当残差明显大于形式中误差时给出提示。"""
    if fit.chi2_red is not None and fit.chi2_red > 4.0:
        return (
            f"{fit.spec.name} 模型的 χ²_red={fit.chi2_red:.1f}，残差显著大于形式中误差，"
            "说明观测噪声被低估或仍存在未建模信号（有色噪声、其他地震/天线变化等）"
        )
    return None


def analyze_station(
    data: StationData,
    config: Optional[AnalysisConfig] = None,
) -> StationAnalysis:
    """对站点三个分量分别建模并做模型选择。

    Parameters
    ----------
    data:
        :func:`gnss_deformation.data.load_station_data` 的返回值。
    config:
        分析参数，缺省使用 :class:`AnalysisConfig` 默认值。

    Returns
    -------
    StationAnalysis
        包含最优拟合、全部候选模型、告警与元数据。
    """
    config = config or AnalysisConfig()
    t_eq = config.earthquake.decimal_year
    t0 = float(data.time[0])

    if not (data.time[0] < t_eq < data.time[-1]):
        raise ValueError(
            f"同震时刻 t_eq={t_eq:.4f} 不在数据时间范围 "
            f"[{data.time[0]:.4f}, {data.time[-1]:.4f}] 内"
        )

    fits: Dict[str, ModelFit] = {}
    candidates: Dict[str, Dict[str, ModelFit]] = {}
    warnings: List[str] = []

    for comp in config.components:
        y = data.component(comp)
        sigma = data.sigma_for(comp) if config.weighted else None
        if config.weighted and sigma is None:
            warnings.append(f"{comp} 分量缺少形式中误差，已改用等权拟合")

        finite = np.isfinite(data.time) & np.isfinite(y)
        if int(finite.sum()) < config.min_epochs:
            raise ValueError(
                f"{comp} 分量有效历元仅 {int(finite.sum())} 个，少于阈值 {config.min_epochs}"
            )

        best, all_fits = fit_best_model(
            data.time,
            y,
            t_eq=t_eq,
            model=config.model,
            t0=t0,
            harmonics=config.harmonics,
            sigma=sigma,
            tau_bounds=config.tau_bounds,
            robust=config.robust,
        )
        fits[comp] = best
        candidates[comp] = all_fits

        if best.tau_at_bound:
            lo, hi = config.tau_bounds
            warnings.append(
                f"{comp} 分量松弛时间常数 τ={best.relaxation_time:.3f} yr 触及搜索边界 "
                f"[{lo}, {hi}] yr：震后观测窗口不足以约束 τ，"
                "D 与 τ 高度相关，建议以 D/τ（震后视速率）解释结果"
            )
        advisory = _noise_advisory(best)
        if advisory:
            warnings.append(f"{comp} 分量：{advisory}")

    return StationAnalysis(
        data=data,
        config=config,
        fits=fits,
        candidates=candidates,
        warnings=warnings,
    )
