"""测试辅助函数（合成数据与 CSV 生成）。

放在独立模块中，供各测试文件以 ``from helpers import ...`` 方式复用；
``tests`` 目录会被 pytest 自动加入 ``sys.path``。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

T_EQ = 2024.0 + 94.0 / 366.0

CSV_HEADER = (
    "site,YYMMMDD,yyyy.yyyy,__MJD,week,d,reflon,_e0(m),__east(m),"
    "____n0(m),_north(m),u0(m),____up(m),_ant(m),sig_e(m),sig_n(m),sig_u(m),"
    "__corr_en,__corr_eu,__corr_nu,_latitude(deg),_longitude(deg),__height(m)"
)


def synthetic_series(
    n: int = 1460,
    start: float = 2022.0,
    v: float = 10.0,
    step: float = 5.0,
    relax: float = 8.0,
    tau: float = 0.4,
    seasonal: tuple = (2.0, -1.0, 0.5, 0.3),
    noise: float = 0.4,
    t_eq: float = T_EQ,
    seed: int = 0,
) -> tuple:
    """生成含趋势、季节项、同震阶跃与指数松弛的合成序列（mm）。"""
    rng = np.random.default_rng(seed)
    t = start + np.arange(n) / 365.25
    s1, c1, s2, c2 = seasonal
    y = (
        v * (t - start)
        + s1 * np.sin(2 * np.pi * t)
        + c1 * np.cos(2 * np.pi * t)
        + s2 * np.sin(4 * np.pi * t)
        + c2 * np.cos(4 * np.pi * t)
    )
    post = t >= t_eq
    y = y + np.where(post, step + relax * (1.0 - np.exp(-(t - t_eq) / tau)), 0.0)
    return t, y + rng.normal(0.0, noise, n)


def write_station_csv(
    path: Path,
    rows: list,
    *,
    header: str = CSV_HEADER,
    bom: bool = True,
    blank_row: bool = True,
) -> Path:
    """按 GNSS 导出格式写一个最小可用的 CSV。

    ``rows`` 中每项需包含 ``time``、``east``、``north``、``up``（单位 m）。
    ``blank_row=True`` 时会在数据中间插入一行全空记录，用于测试清洗逻辑。
    """
    lines = [header]
    for index, row in enumerate(rows):
        lines.append(
            ",".join(
                [
                    str(row.get("site", "TEST")),
                    str(row.get("date", f"22-Jan-{index + 1:02d}")),
                    f"{row['time']:.4f}",
                    str(60000 + index),
                    "2200",
                    "1",
                    "120.2",
                    "2051",
                    f"{row['east']:.6f}",
                    "2544410",
                    f"{row['north']:.6f}",
                    "59",
                    f"{row['up']:.6f}",
                    "0",
                    "0.0008",
                    "0.0008",
                    "0.0035",
                    "-0.05",
                    "-0.05",
                    "-0.02",
                    "23.0",
                    "120.2",
                    f"{row['up']:.6f}",
                ]
            )
        )
        if blank_row and index == 0:
            lines.append("," * 22)
    path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8-sig" if bom else "utf-8",
    )
    return path
