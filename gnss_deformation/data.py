"""数据读取、清洗与单位换算。

源文件为 GNSS 处理软件导出的站点坐标时间序列（UTF-8 BOM、逗号分隔），
坐标单位为米，时间列为十进制年 ``yyyy.yyyy``。本模块负责：

* 兼容 BOM、空行、重复历元与非数值内容；
* 依据表头自动识别 East / North / Up 及其中误差列；
* 统一换算为毫米并按参考历元归零；
* 记录原始坐标与数据质量信息，便于溯源。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import DEFAULT_COMPONENTS

#: 分量 -> 坐标列名候选（规范化后的形式，见 :func:`_normalise`）。
_COMPONENT_ALIASES: Dict[str, Tuple[str, ...]] = {
    "East": ("eastm", "east", "__eastm"),
    "North": ("northm", "north", "__northm"),
    "Up": ("upm", "up", "__upm"),
}

#: 分量 -> 中误差列名候选。
_SIGMA_ALIASES: Dict[str, Tuple[str, ...]] = {
    "East": ("sigem", "sige", "sigeast", "eastsigma"),
    "North": ("signm", "sign", "signorth", "northsigma"),
    "Up": ("sigum", "sigu", "sup", "upsigma"),
}

#: 时间列名候选。
_TIME_ALIASES: Tuple[str, ...] = ("yyyy.yyyy", "yyyyyyyy", "year", "decimalyear", "t")


def _normalise(name: str) -> str:
    """把表头规范化为仅含小写字母、数字与小数点的形式。"""
    return re.sub(r"[^a-z0-9.]", "", str(name).strip().lower())


def _find_column(columns: Sequence[str], aliases: Iterable[str]) -> Optional[str]:
    """在表头中查找别名对应的列，找不到返回 ``None``。"""
    lookup = {}
    for col in columns:
        lookup.setdefault(_normalise(col), col)
    for alias in aliases:
        if alias in lookup:
            return lookup[alias]
    return None


@dataclass(frozen=True)
class StationData:
    """清洗后的单站形变时间序列（单位：毫米）。

    ``east`` / ``north`` / ``up`` 均已减去参考历元的坐标值，表示相对形变。
    """

    station: str
    time: np.ndarray
    east: np.ndarray
    north: np.ndarray
    up: np.ndarray
    sigma: Mapping[str, np.ndarray]
    source: Path
    n_raw: int
    n_used: int
    reference_epoch: float
    reference_values: Mapping[str, float]
    mjd: Optional[np.ndarray] = None
    dropped_rows: int = 0

    def component(self, name: str) -> np.ndarray:
        """返回指定分量的形变序列（mm）。"""
        return {"East": self.east, "North": self.north, "Up": self.up}[name]

    def sigma_for(self, name: str) -> Optional[np.ndarray]:
        """返回指定分量的形式中误差（mm），缺失时返回 ``None``。"""
        values = self.sigma.get(name)
        return None if values is None else np.asarray(values, dtype=float)

    @property
    def components(self) -> Tuple[str, ...]:
        return tuple(DEFAULT_COMPONENTS)

    @property
    def n_epochs(self) -> int:
        return int(self.time.size)

    @property
    def time_span(self) -> Tuple[float, float]:
        return float(self.time[0]), float(self.time[-1])

    @property
    def median_interval_days(self) -> float:
        """相邻历元的间隔中位数（天），可用于判断采样率。"""
        if self.time.size < 2:
            return float("nan")
        return float(np.median(np.diff(self.time)) * 365.25)

    def to_dict(self) -> dict:
        return {
            "station": self.station,
            "source": str(self.source),
            "n_raw": self.n_raw,
            "n_used": self.n_used,
            "dropped_rows": self.dropped_rows,
            "time_span": [round(v, 4) for v in self.time_span],
            "reference_epoch": round(self.reference_epoch, 6),
            "reference_values": {k: round(float(v), 6) for k, v in self.reference_values.items()},
            "median_interval_days": round(self.median_interval_days, 3),
        }


def load_station_data(
    path: str | Path,
    *,
    year_range: Optional[Tuple[float, float]] = (2022.0, 2026.0),
    reference: str = "first",
    station: Optional[str] = None,
) -> StationData:
    """读取并清洗 GNSS 站点坐标 CSV。

    Parameters
    ----------
    path:
        CSV 文件路径（支持 UTF-8 BOM）。
    year_range:
        时间筛选区间（十进制年，闭区间），``None`` 表示不筛选。
    reference:
        ``"first"`` 以首个历元为参考，``"mean"`` 以均值归零。
    station:
        站点名，缺省时从 ``site`` 列或文件名推断。

    Returns
    -------
    StationData
        形变序列（mm）及元数据。

    Raises
    ------
    FileNotFoundError
        文件不存在。
    ValueError
        缺少必要列、有效历元不足或时间列无法解析。
    """
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"找不到数据文件：{path}")

    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:  # 兼容 GBK 编码的导出文件
        df = pd.read_csv(path, encoding="gbk")

    n_raw = len(df)
    df = df.dropna(how="all")
    if df.empty:
        raise ValueError(f"数据文件为空：{path}")

    columns = list(df.columns)
    time_col = _find_column(columns, _TIME_ALIASES)
    if time_col is None:
        raise ValueError(
            f"未找到时间列（期望 yyyy.yyyy），现有列：{columns}"
        )

    comp_cols: Dict[str, str] = {}
    for comp in DEFAULT_COMPONENTS:
        col = _find_column(columns, _COMPONENT_ALIASES[comp])
        if col is None:
            raise ValueError(
                f"未找到 {comp} 分量列（期望形如 __east(m) / _north(m) / ____up(m)），"
                f"现有列：{columns}"
            )
        comp_cols[comp] = col

    sig_cols: Dict[str, str] = {}
    for comp in DEFAULT_COMPONENTS:
        col = _find_column(columns, _SIGMA_ALIASES[comp])
        if col is not None:
            sig_cols[comp] = col

    site_col = _find_column(columns, ("site", "station", "name"))
    mjd_col = _find_column(columns, ("mjd",))

    for col in [time_col, *comp_cols.values(), *sig_cols.values()]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=[time_col])
    # 三个分量全为空的行（常见于文件末尾的 "," 分隔残留）直接丢弃
    df = df.dropna(subset=list(comp_cols.values()), how="all")

    if year_range is not None:
        lo, hi = year_range
        df = df[(df[time_col] >= lo) & (df[time_col] <= hi)]

    df = df.sort_values(time_col, kind="mergesort")
    df = df.drop_duplicates(subset=[time_col], keep="last")

    if df.empty:
        raise ValueError(
            f"筛选后没有可用历元（year_range={year_range}）：{path}"
        )

    time = df[time_col].to_numpy(dtype=float)
    values_m: Dict[str, np.ndarray] = {
        comp: df[col].to_numpy(dtype=float) for comp, col in comp_cols.items()
    }

    # 以参考历元的有效观测归零（参考历元本身缺失时取最近的有效值）
    if reference == "mean":
        ref_values = {
            comp: float(np.nanmean(vals)) for comp, vals in values_m.items()
        }
        ref_epoch = float(time[0])
    else:
        ref_values = {comp: float(vals[0]) for comp, vals in values_m.items()}
        ref_epoch = float(time[0])

    if station is None and site_col is not None:
        sites = df[site_col].dropna().astype(str).str.strip()
        station = sites.iloc[0] if not sites.empty else None
    if not station:
        station = path.stem.split("站点")[0] or "UNKNOWN"

    sigma: Dict[str, np.ndarray] = {}
    for comp, col in sig_cols.items():
        arr = df[col].to_numpy(dtype=float) * 1000.0
        if np.all(np.isfinite(arr)) and np.all(arr > 0):
            sigma[comp] = arr

    n_used = len(df)
    if n_used < 2:
        raise ValueError(f"有效历元数不足（{n_used}），无法建模：{path}")

    return StationData(
        station=str(station),
        time=time,
        east=(values_m["East"] - ref_values["East"]) * 1000.0,
        north=(values_m["North"] - ref_values["North"]) * 1000.0,
        up=(values_m["Up"] - ref_values["Up"]) * 1000.0,
        sigma=sigma,
        source=path,
        n_raw=n_raw,
        n_used=n_used,
        reference_epoch=ref_epoch,
        reference_values=ref_values,
        mjd=df[mjd_col].to_numpy(dtype=float) if mjd_col else None,
        dropped_rows=n_raw - n_used,
    )


def resolve_data_path(path: Optional[str | Path] = None, root: Optional[Path] = None) -> Path:
    """定位数据文件：显式路径 > ``data/*.csv`` > 当前目录 ``*.csv``。"""
    if path is not None:
        candidate = Path(path).expanduser()
        if not candidate.is_file():
            raise FileNotFoundError(f"找不到数据文件：{candidate}")
        return candidate

    root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    search_dirs = [root / "data", root, Path.cwd() / "data", Path.cwd()]
    for directory in search_dirs:
        if not directory.is_dir():
            continue
        matches = sorted(directory.glob("*.csv"))
        preferred = [p for p in matches if "站点坐标数据" in p.name]
        for candidate in preferred or matches:
            return candidate
    raise FileNotFoundError(
        "未找到数据文件，请使用 --csv 指定，或把 CSV 放入 data/ 目录"
    )
