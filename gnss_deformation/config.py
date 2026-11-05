"""分析配置与震例定义。"""

from __future__ import annotations

import calendar
from dataclasses import dataclass
from datetime import date
from typing import Optional, Sequence, Tuple

#: 支持的模型名称。``step`` 为最简模型，``step+exp+rate`` 为最完整模型。
MODEL_NAMES: Tuple[str, ...] = ("step", "step+exp", "step+rate", "step+exp+rate")

#: 默认参与分析的分量。
DEFAULT_COMPONENTS: Tuple[str, ...] = ("East", "North", "Up")


@dataclass(frozen=True)
class Earthquake:
    """一次地震事件，用于定义同震时刻 ``t_eq``。

    Parameters
    ----------
    name:
        事件名称，仅用于输出与图表标注。
    date:
        事件发生日期（UTC 近似）。
    """

    name: str
    date: date

    @property
    def decimal_year(self) -> float:
        """十进制年表示，按“年内第几天 / 当年天数”换算。

        与 GNSS 处理软件输出的 ``yyyy.yyyy`` 口径一致，2024-04-03 对应
        ``2024 + 94 / 366 = 2024.2568``。
        """
        doy = self.date.timetuple().tm_yday
        days = 366 if calendar.isleap(self.date.year) else 365
        return self.date.year + doy / days

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "date": self.date.isoformat(),
            "decimal_year": round(self.decimal_year, 6),
        }


#: 2024-04-03 台湾花莲 Mw 7.4 地震（本数据集的主要同震事件）。
HUALIEN_2024 = Earthquake(
    name="2024 Mw 7.4 Hualien, Taiwan",
    date=date(2024, 4, 3),
)


@dataclass
class AnalysisConfig:
    """分析参数集合。

    Parameters
    ----------
    earthquake:
        同震事件，决定阶跃/松弛项的发生时刻。
    components:
        参与分析的分量，默认 East / North / Up。
    year_range:
        时间筛选区间（十进制年，闭区间），``None`` 表示不筛选。
    model:
        ``"auto"`` 表示按 BIC 在三类候选模型中自动选择；
        也可显式指定 :data:`MODEL_NAMES` 中的任意一项。
    harmonics:
        季节项谐波个数：1 = 仅周年项，2 = 周年 + 半周年（默认）。
    tau_bounds:
        松弛时间常数的搜索区间（年），须为正且下界小于上界。
    weighted:
        是否使用数据自带形式中误差（``sig_*``）作为权重。
    reference:
        ``"first"`` 以首个历元为参考，``"mean"`` 以全部历元均值为参考。
    min_epochs:
        单分量最少可用历元数，低于该值直接报错。
    """

    earthquake: Earthquake = HUALIEN_2024
    components: Sequence[str] = DEFAULT_COMPONENTS
    year_range: Optional[Tuple[float, float]] = None
    model: str = "auto"
    harmonics: int = 2
    tau_bounds: Tuple[float, float] = (0.02, 5.0)
    weighted: bool = False
    robust: bool = False
    reference: str = "first"
    min_epochs: int = 30

    def __post_init__(self) -> None:
        if self.model not in MODEL_NAMES and self.model != "auto":
            raise ValueError(
                f"未知模型 {self.model!r}，可选：'auto', {', '.join(MODEL_NAMES)}"
            )
        if self.harmonics < 0:
            raise ValueError("harmonics 必须为非负整数")
        lo, hi = self.tau_bounds
        if not (0.0 < lo < hi):
            raise ValueError(f"tau_bounds 必须满足 0 < 下界 < 上界，实际为 {self.tau_bounds}")
        if self.year_range is not None and self.year_range[0] >= self.year_range[1]:
            raise ValueError(f"year_range 非法：{self.year_range}")
        if self.reference not in ("first", "mean"):
            raise ValueError("reference 只能是 'first' 或 'mean'")
        if self.min_epochs < 2:
            raise ValueError("min_epochs 至少为 2")
        unknown = [c for c in self.components if c not in DEFAULT_COMPONENTS]
        if unknown:
            raise ValueError(f"未知分量：{unknown}，可选：{list(DEFAULT_COMPONENTS)}")

    def to_dict(self) -> dict:
        return {
            "earthquake": self.earthquake.to_dict(),
            "components": list(self.components),
            "year_range": list(self.year_range) if self.year_range else None,
            "model": self.model,
            "harmonics": self.harmonics,
            "tau_bounds": list(self.tau_bounds),
            "weighted": self.weighted,
            "robust": self.robust,
            "reference": self.reference,
        }
