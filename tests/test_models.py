"""模型拟合与模型选择测试。"""

from __future__ import annotations

import numpy as np
import pytest

from gnss_deformation.models import (
    AUTO_CANDIDATES,
    fit_best_model,
    fit_model,
    spec_from_name,
)

from helpers import T_EQ, synthetic_series


def test_recovers_exponential_parameters() -> None:
    t, y = synthetic_series(v=10.0, step=5.0, relax=8.0, tau=0.4, noise=0.3)
    fit = fit_model(t, y, "step+exp", t_eq=T_EQ, harmonics=2)

    assert fit.velocity == pytest.approx(10.0, abs=0.15)
    assert fit.step == pytest.approx(5.0, abs=0.3)
    assert fit.relaxation_amplitude == pytest.approx(8.0, abs=0.8)
    assert fit.relaxation_time == pytest.approx(0.4, abs=0.08)
    assert fit.rms < 0.6
    assert 0 < fit.dof < fit.n


def test_uncertainties_are_positive() -> None:
    t, y = synthetic_series()
    fit = fit_model(t, y, "step+exp", t_eq=T_EQ)
    assert fit.velocity_error > 0
    assert fit.step_error > 0
    assert fit.relaxation_time_error > 0
    assert fit.annual_amplitude > 0


def test_predict_matches_fitted_series() -> None:
    t, y = synthetic_series()
    fit = fit_model(t, y, "step+exp", t_eq=T_EQ)
    np.testing.assert_allclose(fit.predict(t), fit.predicted, rtol=1e-10, atol=1e-10)


def test_auto_selection_prefers_relaxation() -> None:
    t, y = synthetic_series(relax=8.0, tau=0.4, noise=0.3)
    best, candidates = fit_best_model(t, y, t_eq=T_EQ, model="auto")

    assert set(candidates) == set(AUTO_CANDIDATES)
    assert best.spec.name == "step+exp"
    assert best.bic <= min(fit.bic for fit in candidates.values())


def test_rate_model_recovers_velocity_change() -> None:
    rng = np.random.default_rng(3)
    t = 2022.0 + np.arange(1460) / 365.25
    y = 3.0 + 2.0 * (t - 2022.0)
    y = y + np.where(t >= T_EQ, 4.0 + 9.0 * (t - T_EQ), 0.0)
    y = y + rng.normal(0.0, 0.2, t.size)

    fit = fit_model(t, y, "step+rate", t_eq=T_EQ)
    assert fit.step == pytest.approx(4.0, abs=0.2)
    assert fit.post_rate == pytest.approx(9.0, abs=0.2)
    assert fit.velocity == pytest.approx(2.0, abs=0.1)


def test_weighted_fit_uses_sigma() -> None:
    t, y = synthetic_series()
    sigma = np.full(t.size, 0.5)
    fit = fit_model(t, y, "step+exp", t_eq=T_EQ, sigma=sigma)
    assert fit.chi2_red is not None
    assert fit.chi2_red > 0
    assert fit.wrms > 0
    # 加权后方差应与形式中误差量级一致
    assert 0.1 < fit.chi2_red < 10


def test_robust_loss_runs_and_fits() -> None:
    t, y = synthetic_series(noise=0.3)
    y = y.copy()
    y[[50, 500, 900]] += 25.0  # 人为粗差
    fit = fit_model(t, y, "step+exp", t_eq=T_EQ, robust=True)
    assert fit.velocity == pytest.approx(10.0, abs=0.2)
    assert fit.rms < 1.5


def test_invalid_inputs() -> None:
    t, y = synthetic_series()
    with pytest.raises(ValueError, match="tau_bounds"):
        fit_model(t, y, "step+exp", t_eq=T_EQ, tau_bounds=(2.0, 1.0))

    with pytest.raises(ValueError, match="两侧历元不足"):
        fit_model(t, y, "step", t_eq=2030.0)

    with pytest.raises(ValueError, match="有效历元太少"):
        fit_model(t[:8], y[:8], "step+exp", t_eq=T_EQ)


def test_spec_lookup() -> None:
    spec = spec_from_name("step+exp+rate")
    assert spec.step and spec.relaxation and spec.post_rate
    assert spec.n_linear == 5  # a, v, C, D, rate

    with pytest.raises(ValueError):
        spec_from_name("not-a-model")


def test_model_without_relaxation_has_no_tau() -> None:
    t, y = synthetic_series()
    fit = fit_model(t, y, "step", t_eq=T_EQ)
    assert np.isnan(fit.relaxation_time)
    assert "tau" not in fit.params
    assert not fit.spec.relaxation
