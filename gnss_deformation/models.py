"""形变模型定义与参数估计。

模型形式（以任一坐标分量为例）::

    y(t) = a + v·(t - t0)                                # 长期线性趋势
         + Σ_{k=1..K} [ s_k·sin(2πk t) + c_k·cos(2πk t) ] # 周年/半周年季节项
         + H(t - t_eq)·[ C                                # 同震阶跃
                        + D·(1 - e^{-(t-t_eq)/τ}) ]       # 震后指数松弛
         + H(t - t_eq)·dν·(t - t_eq)                      # 震后速率变化（可选）

其中 ``H(·)`` 为 Heaviside 阶跃函数。松弛项的时间常数 ``τ`` 以对数参数化，
保证其始终为正；参数估计采用带边界的非线性最小二乘，并以多起点搜索避免
陷入由 ``D`` 与 ``τ`` 强相关造成的局部极小。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import least_squares

from .config import MODEL_NAMES

#: 模型名 -> (是否含同震阶跃, 是否含指数松弛, 是否含震后速率变化)
_SPEC_TABLE: Dict[str, Tuple[bool, bool, bool]] = {
    "step": (True, False, False),
    "step+exp": (True, True, False),
    "step+rate": (True, False, True),
    "step+exp+rate": (True, True, True),
}

#: 自动选模型时的候选集合。
#:
#: 震后观测窗口通常只有两三年，此时 ``D`` 与 ``τ`` 高度相关（只能约束 ``D/τ``），
#: 若再叠加速率变化项则模型过参数化，因此默认不把 ``step+exp+rate`` 纳入自动
#: 选择，需要时可用 ``model="step+exp+rate"`` 显式指定。
AUTO_CANDIDATES: Tuple[str, ...] = ("step", "step+exp", "step+rate")


@dataclass(frozen=True)
class ModelSpec:
    """模型结构描述。"""

    name: str
    step: bool
    relaxation: bool
    post_rate: bool

    @property
    def n_linear(self) -> int:
        """线性参数个数（不含谐波项）。"""
        return 2 + int(self.step) + int(self.relaxation) + int(self.post_rate)


def spec_from_name(name: str) -> ModelSpec:
    if name not in _SPEC_TABLE:
        raise ValueError(f"未知模型 {name!r}，可选：{', '.join(MODEL_NAMES)}")
    return ModelSpec(name, *_SPEC_TABLE[name])


def design_matrix(
    t: np.ndarray,
    spec: ModelSpec,
    *,
    tau: Optional[float],
    t_eq: float,
    t0: float,
    harmonics: int = 2,
) -> Tuple[np.ndarray, List[str]]:
    """构建设计矩阵，返回 ``(X, 参数名列表)``。"""
    t = np.asarray(t, dtype=float)
    columns: List[np.ndarray] = [np.ones_like(t), t - t0]
    names: List[str] = ["a", "v"]

    for k in range(1, harmonics + 1):
        columns.append(np.sin(2.0 * np.pi * k * t))
        columns.append(np.cos(2.0 * np.pi * k * t))
        names.extend([f"s{k}", f"c{k}"])

    heaviside = (t >= t_eq).astype(float)
    if spec.step:
        columns.append(heaviside)
        names.append("C")
    if spec.relaxation:
        if tau is None or tau <= 0:
            raise ValueError("指数松弛模型需要正的 tau")
        elapsed = np.maximum(t - t_eq, 0.0)
        columns.append(heaviside * (1.0 - np.exp(-elapsed / tau)))
        names.append("D")
    if spec.post_rate:
        columns.append(heaviside * (t - t_eq))
        names.append("rate")

    return np.column_stack(columns), names


@dataclass
class ModelFit:
    """单个分量的模型拟合结果。"""

    spec: ModelSpec
    t_eq: float
    t0: float
    harmonics: int
    param_names: Tuple[str, ...]
    params: Dict[str, float]
    stderr: Dict[str, float]
    covariance: np.ndarray
    t: np.ndarray
    observed: np.ndarray
    predicted: np.ndarray
    residuals: np.ndarray
    n: int
    dof: int
    rms: float
    wrms: float
    chi2_red: Optional[float]
    bic: float
    aic: float
    success: bool
    message: str
    tau_at_bound: bool = False
    condition_number: float = float("nan")

    # ------------------------------------------------------------------ 便捷属性
    @property
    def velocity(self) -> float:
        return self.params["v"]

    @property
    def velocity_error(self) -> float:
        return self.stderr["v"]

    @property
    def step(self) -> float:
        return self.params.get("C", float("nan"))

    @property
    def step_error(self) -> float:
        return self.stderr.get("C", float("nan"))

    @property
    def relaxation_amplitude(self) -> float:
        return self.params.get("D", float("nan"))

    @property
    def relaxation_amplitude_error(self) -> float:
        return self.stderr.get("D", float("nan"))

    @property
    def relaxation_time(self) -> float:
        return self.params.get("tau", float("nan"))

    @property
    def relaxation_time_error(self) -> float:
        return self.stderr.get("tau", float("nan"))

    @property
    def post_rate(self) -> float:
        return self.params.get("rate", float("nan"))

    @property
    def post_rate_error(self) -> float:
        return self.stderr.get("rate", float("nan"))

    @property
    def annual_amplitude(self) -> float:
        if self.harmonics < 1:
            return float("nan")
        return float(np.hypot(self.params.get("s1", 0.0), self.params.get("c1", 0.0)))

    @property
    def semiannual_amplitude(self) -> float:
        if self.harmonics < 2:
            return float("nan")
        return float(np.hypot(self.params.get("s2", 0.0), self.params.get("c2", 0.0)))

    @property
    def post_seismic_slope(self) -> float:
        """震后瞬态在 ``t -> t_eq+`` 处的视速率（mm/yr）。"""
        if not self.spec.relaxation:
            return float("nan")
        if self.relaxation_time <= 0:
            return float("nan")
        slope = self.relaxation_amplitude / self.relaxation_time
        if self.spec.post_rate:
            slope += self.post_rate
        return float(slope)

    @property
    def effective_params(self) -> int:
        return len(self.param_names)

    def predict(self, t: np.ndarray) -> np.ndarray:
        """在任意时刻计算模型值。"""
        tau = self.params.get("tau") if self.spec.relaxation else None
        X, names = design_matrix(
            t, self.spec, tau=tau, t_eq=self.t_eq, t0=self.t0, harmonics=self.harmonics
        )
        beta = np.array([self.params[name] for name in names])
        return X @ beta

    def summary(self) -> dict:
        return {
            "model": self.spec.name,
            "velocity_mm_yr": self.velocity,
            "velocity_sigma": self.velocity_error,
            "step_mm": self.step,
            "step_sigma": self.step_error,
            "relax_amplitude_mm": self.relaxation_amplitude,
            "relax_amplitude_sigma": self.relaxation_amplitude_error,
            "relax_tau_yr": self.relaxation_time,
            "relax_tau_sigma": self.relaxation_time_error,
            "post_rate_mm_yr": self.post_rate,
            "post_rate_sigma": self.post_rate_error,
            "annual_amplitude_mm": self.annual_amplitude,
            "semiannual_amplitude_mm": self.semiannual_amplitude,
            "post_seismic_slope_mm_yr": self.post_seismic_slope,
            "rms_mm": self.rms,
            "wrms_mm": self.wrms,
            "chi2_red": self.chi2_red,
            "bic": self.bic,
            "aic": self.aic,
            "n_epochs": self.n,
            "dof": self.dof,
            "tau_at_bound": self.tau_at_bound,
            "condition_number": self.condition_number,
            "success": self.success,
            "message": self.message,
        }


def _initial_tau_grid(tau_bounds: Tuple[float, float]) -> np.ndarray:
    lo, hi = tau_bounds
    starts = np.geomspace(max(lo * 1.01, 1e-4), hi * 0.99, 7)
    starts = np.unique(np.append(starts, min(max(0.5, lo), hi)))
    return starts


def _robust_scale(residuals: np.ndarray) -> float:
    median = np.median(residuals)
    mad = np.median(np.abs(residuals - median))
    scale = 1.4826 * mad
    return float(max(scale, 1e-6))


def fit_model(
    t: Sequence[float],
    y: Sequence[float],
    spec: ModelSpec | str,
    *,
    t_eq: float,
    t0: Optional[float] = None,
    harmonics: int = 2,
    sigma: Optional[Sequence[float]] = None,
    tau_bounds: Tuple[float, float] = (0.02, 5.0),
    robust: bool = False,
    max_nfev: int = 20000,
) -> ModelFit:
    """对单个分量做最小二乘拟合。

    Parameters
    ----------
    t, y:
        时间（十进制年）与形变序列（mm）。
    spec:
        模型名称或 :class:`ModelSpec`。
    t_eq:
        同震时刻（十进制年）。
    t0:
        线性趋势的参考时刻，缺省为 ``t[0]``。
    harmonics:
        季节项谐波个数。
    sigma:
        形式中误差（mm），给定时作加权最小二乘。
    tau_bounds:
        松弛时间常数搜索区间（年）。
    robust:
        是否使用 Huber（soft-L1）稳健损失。

    Raises
    ------
    ValueError
        数据量不足，或该模型在给定时间窗口下不可辨识。
    """
    if isinstance(spec, str):
        spec = spec_from_name(spec)

    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(t) & np.isfinite(y)
    if sigma is not None:
        sigma = np.asarray(sigma, dtype=float)
        finite &= np.isfinite(sigma) & (sigma > 0)
    t, y = t[finite], y[finite]
    if sigma is not None:
        sigma = sigma[finite]

    n = t.size
    n_linear_params = spec.n_linear + 2 * harmonics
    n_params = n_linear_params + int(spec.relaxation)
    dof = n - n_params
    if dof <= 0:
        raise ValueError(f"有效历元太少（{n}）以拟合参数个数为 {n_params} 的模型")
    if t0 is None:
        t0 = float(t[0])

    if spec.step:
        n_pre = int(np.sum(t < t_eq))
        n_post = int(np.sum(t >= t_eq))
        if n_pre < 3 or n_post < 3:
            raise ValueError(
                f"同震时刻 t_eq={t_eq:.4f} 两侧历元不足（震前 {n_pre}，震后 {n_post}），"
                "无法估计阶跃项"
            )
    if spec.relaxation:
        post_span = float(t[-1] - t_eq)
        if post_span <= 0.05:
            raise ValueError("震后时间跨度太短，无法估计松弛时间常数")

    weights = np.ones_like(t) if sigma is None else 1.0 / sigma
    loss = "soft_l1" if robust else "linear"
    f_scale: float = 1.0
    if robust:
        f_scale = 1.345 if sigma is not None else _robust_scale(y - np.median(y))

    lower_linear = np.full(n_linear_params, -np.inf)
    upper_linear = np.full(n_linear_params, np.inf)

    def residuals_of(params: np.ndarray) -> np.ndarray:
        tau = float(np.exp(params[-1])) if spec.relaxation else None
        X, _ = design_matrix(t, spec, tau=tau, t_eq=t_eq, t0=t0, harmonics=harmonics)
        return (X @ params[:n_linear_params] - y) * weights

    def initial_linear(tau: Optional[float]) -> np.ndarray:
        X, _ = design_matrix(t, spec, tau=tau, t_eq=t_eq, t0=t0, harmonics=harmonics)
        beta, *_ = np.linalg.lstsq(X * weights[:, None], y * weights, rcond=None)
        return beta

    def solve(params0: np.ndarray, bounds: Tuple[np.ndarray, np.ndarray]):
        return least_squares(
            residuals_of,
            params0,
            bounds=bounds,
            loss=loss,
            f_scale=f_scale,
            max_nfev=max_nfev,
            xtol=1e-12,
            ftol=1e-12,
            gtol=1e-12,
        )

    if spec.relaxation:
        lo, hi = tau_bounds
        if not (0.0 < lo < hi):
            raise ValueError(f"tau_bounds 非法：{tau_bounds}")
        lower = np.concatenate([lower_linear, [np.log(lo)]])
        upper = np.concatenate([upper_linear, [np.log(hi)]])
        best = None
        for tau0 in _initial_tau_grid(tau_bounds):
            params0 = np.concatenate([initial_linear(float(tau0)), [np.log(float(tau0))]])
            candidate = solve(params0, (lower, upper))
            if best is None or candidate.cost < best.cost:
                best = candidate
        result = best
    else:
        lower = lower_linear
        upper = upper_linear
        result = solve(initial_linear(None), (lower, upper))

    assert result is not None
    params = result.x
    tau_hat = float(np.exp(params[-1])) if spec.relaxation else None
    X, names = design_matrix(t, spec, tau=tau_hat, t_eq=t_eq, t0=t0, harmonics=harmonics)
    beta = params[:n_linear_params]
    model = X @ beta
    residuals = model - y

    tau_bound_hit = False
    if spec.relaxation:
        lo, hi = tau_bounds
        tau_bound_hit = tau_hat <= lo * 1.05 or tau_hat >= hi * 0.95

    # 协方差：(JᵀJ)⁻¹ · σ0²，J 为加权残差对参数的雅可比
    jac = result.jac
    jtj = jac.T @ jac
    condition_number = float(np.linalg.cond(jtj))
    try:
        cov = np.linalg.inv(jtj) * (2.0 * result.cost / dof)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(jtj) * (2.0 * result.cost / dof)

    stderr_vector = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    param_values = {name: float(value) for name, value in zip(names, beta)}
    param_errors = {name: float(value) for name, value in zip(names, stderr_vector)}
    if spec.relaxation:
        param_values["tau"] = tau_hat
        param_errors["tau"] = float(tau_hat * stderr_vector[-1])

    ssr = float(np.sum(residuals**2))
    rms = float(np.sqrt(ssr / n))
    wrms = float(np.sqrt(np.sum((residuals * weights) ** 2) / np.sum(weights**2)))
    chi2_red = None
    if sigma is not None:
        chi2_red = float(np.sum((residuals / sigma) ** 2) / dof)

    safe_ssr = max(ssr, 1e-12)
    bic = float(n * np.log(safe_ssr / n) + n_params * np.log(n))
    aic = float(n * np.log(safe_ssr / n) + 2.0 * n_params)

    return ModelFit(
        spec=spec,
        t_eq=float(t_eq),
        t0=float(t0),
        harmonics=int(harmonics),
        param_names=tuple(names),
        params=param_values,
        stderr=param_errors,
        covariance=cov,
        t=t,
        observed=y,
        predicted=model,
        residuals=residuals,
        n=n,
        dof=dof,
        rms=rms,
        wrms=wrms,
        chi2_red=chi2_red,
        bic=bic,
        aic=aic,
        success=bool(result.success),
        message=str(result.message),
        tau_at_bound=tau_bound_hit,
        condition_number=condition_number,
    )


def fit_best_model(
    t: Sequence[float],
    y: Sequence[float],
    *,
    t_eq: float,
    model: str = "auto",
    t0: Optional[float] = None,
    harmonics: int = 2,
    sigma: Optional[Sequence[float]] = None,
    tau_bounds: Tuple[float, float] = (0.02, 5.0),
    robust: bool = False,
) -> Tuple[ModelFit, Dict[str, ModelFit]]:
    """拟合并按 BIC 选择模型。

    Returns
    -------
    best:
        BIC 最小的拟合结果。
    candidates:
        所有候选模型的拟合结果（键为模型名）。
    """
    if model == "auto":
        names: Iterable[str] = AUTO_CANDIDATES
    else:
        names = (model,)

    candidates: Dict[str, ModelFit] = {}
    for name in names:
        spec = spec_from_name(name)
        try:
            candidates[name] = fit_model(
                t,
                y,
                spec,
                t_eq=t_eq,
                t0=t0,
                harmonics=harmonics,
                sigma=sigma,
                tau_bounds=tau_bounds,
                robust=robust,
            )
        except ValueError:
            if model != "auto":
                raise
            continue

    if not candidates:
        raise ValueError("没有可用的候选模型，请检查时间窗口与数据量")
    best = min(candidates.values(), key=lambda fit: fit.bic)
    return best, candidates
