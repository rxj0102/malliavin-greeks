"""
tests/test_greeks_gbm.py — cross-method Greek verification for GBM.

All four methods (Malliavin, FiniteDiff, Pathwise, LR) are tested against
Black-Scholes analytics for European calls and puts at n_paths = 500 000.

Key checks
----------
1. Each method agrees with BS for delta, gamma, vega, theta, rho (European call).
2. Digital call delta:
   - Malliavin / LR : finite-variance estimate converges.
   - FiniteDiff     : noisy (high std_error) — documents the limitation.
   - Pathwise       : raises NotImplementedError (Dirac delta in derivative).
3. Put-call parity for Greeks:
   - delta(call) - delta(put) = e^{-qT}
   - gamma(call) = gamma(put)
   - vega(call)  = vega(put)
4. all_greeks (Malliavin): single simulation produces all five Greeks.
5. higher_order (Malliavin gamma matches first-order gamma estimate).
6. Analytical module: spot-check bs_digital_delta, bs_barrier_delta, geometric_asian_greeks.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from mgreeks.greeks import (
    MalliavinGreeks,
    FiniteDifferenceGreeks,
    PathwiseGreeks,
    LikelihoodRatioGreeks,
    analytical,
)
from mgreeks.models.gbm import GeometricBrownianMotion
from mgreeks.payoffs.european import EuropeanCall, EuropeanPut, DigitalCall
from mgreeks.simulation import MonteCarloEngine
from mgreeks.utils import bs_greeks, bs_price


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

S0, K, T = 100.0, 100.0, 1.0
R, Q, SIGMA = 0.05, 0.02, 0.20
N_PATHS = 500_000
SEED = 42

_model = GeometricBrownianMotion(r=R, q=Q, sigma=SIGMA)
_engine = MonteCarloEngine(_model, n_paths=N_PATHS, n_steps=1, rng_seed=SEED)
_disc = float(np.exp(-R * T))

_call = EuropeanCall(K)
_put = EuropeanPut(K)
_digital = DigitalCall(K)

_bs_call = bs_greeks(S0, K, R, Q, SIGMA, T, "call")
_bs_put = bs_greeks(S0, K, R, Q, SIGMA, T, "put")

_mall = MalliavinGreeks(_model, _engine)
_fd = FiniteDifferenceGreeks(_model, _engine, bump_size=0.01, bump_type="relative")
_pw = PathwiseGreeks(_model, _engine)
_lr = LikelihoodRatioGreeks(_model, _engine)


def _within(estimate: float, truth: float, se: float, n_se: float = 4.0) -> bool:
    """True if |estimate - truth| < n_se * std_error + tiny_tolerance."""
    return abs(estimate - truth) < n_se * se + 1e-6


def _se(samples: np.ndarray) -> float:
    return float(samples.std(ddof=1) / np.sqrt(len(samples)))


# ---------------------------------------------------------------------------
# 1. Malliavin vs Black-Scholes (European call)
# ---------------------------------------------------------------------------

class TestMalliavinVsBS:
    def test_delta(self):
        res = _mall.delta(_call, S0, T)
        truth = _bs_call["delta"]
        assert _within(res["value"], truth, res["std_error"]), (
            f"Malliavin delta={res['value']:.5f} vs BS={truth:.5f} se={res['std_error']:.5f}"
        )

    def test_gamma(self):
        res = _mall.gamma(_call, S0, T)
        truth = _bs_call["gamma"]
        assert _within(res["value"], truth, res["std_error"]), (
            f"Malliavin gamma={res['value']:.6f} vs BS={truth:.6f} se={res['std_error']:.6f}"
        )

    def test_vega(self):
        res = _mall.vega(_call, S0, T)
        truth = _bs_call["vega"]
        assert _within(res["value"], truth, res["std_error"]), (
            f"Malliavin vega={res['value']:.4f} vs BS={truth:.4f} se={res['std_error']:.4f}"
        )

    def test_theta(self):
        res = _mall.theta(_call, S0, T)
        truth = _bs_call["theta"]
        assert _within(res["value"], truth, res["std_error"]), (
            f"Malliavin theta={res['value']:.4f} vs BS={truth:.4f} se={res['std_error']:.4f}"
        )

    def test_rho(self):
        res = _mall.rho(_call, S0, T)
        truth = _bs_call["rho"]
        assert _within(res["value"], truth, res["std_error"]), (
            f"Malliavin rho={res['value']:.4f} vs BS={truth:.4f} se={res['std_error']:.4f}"
        )

    def test_result_keys(self):
        res = _mall.delta(_call, S0, T)
        for key in ("value", "std_error", "ci_lower", "ci_upper", "n_paths"):
            assert key in res, f"Missing key: {key}"

    def test_ci_structure(self):
        res = _mall.delta(_call, S0, T)
        assert res["ci_lower"] < res["value"] < res["ci_upper"]

    def test_n_paths(self):
        res = _mall.delta(_call, S0, T)
        assert res["n_paths"] == N_PATHS


# ---------------------------------------------------------------------------
# 2. Finite differences vs Black-Scholes (European call)
# ---------------------------------------------------------------------------

class TestFiniteDiffVsBS:
    def test_delta(self):
        res = _fd.delta(_call, S0, T)
        truth = _bs_call["delta"]
        assert _within(res["value"], truth, res["std_error"]), (
            f"FD delta={res['value']:.5f} vs BS={truth:.5f} se={res['std_error']:.5f}"
        )

    def test_gamma(self):
        res = _fd.gamma(_call, S0, T)
        truth = _bs_call["gamma"]
        assert _within(res["value"], truth, res["std_error"]), (
            f"FD gamma={res['value']:.6f} vs BS={truth:.6f} se={res['std_error']:.6f}"
        )

    def test_vega(self):
        res = _fd.vega(_call, S0, T)
        truth = _bs_call["vega"]
        assert _within(res["value"], truth, res["std_error"]), (
            f"FD vega={res['value']:.4f} vs BS={truth:.4f} se={res['std_error']:.4f}"
        )

    def test_theta(self):
        res = _fd.theta(_call, S0, T)
        truth = _bs_call["theta"]
        assert _within(res["value"], truth, res["std_error"]), (
            f"FD theta={res['value']:.4f} vs BS={truth:.4f} se={res['std_error']:.4f}"
        )

    def test_rho(self):
        res = _fd.rho(_call, S0, T)
        truth = _bs_call["rho"]
        assert _within(res["value"], truth, res["std_error"]), (
            f"FD rho={res['value']:.4f} vs BS={truth:.4f} se={res['std_error']:.4f}"
        )

    def test_result_keys(self):
        res = _fd.delta(_call, S0, T)
        for key in ("value", "std_error", "ci_lower", "ci_upper", "n_paths"):
            assert key in res


# ---------------------------------------------------------------------------
# 3. Pathwise vs Black-Scholes (European call)
# ---------------------------------------------------------------------------

class TestPathwiseVsBS:
    def test_delta(self):
        res = _pw.delta(_call, S0, T)
        truth = _bs_call["delta"]
        assert _within(res["value"], truth, res["std_error"]), (
            f"Pathwise delta={res['value']:.5f} vs BS={truth:.5f} se={res['std_error']:.5f}"
        )

    def test_vega(self):
        res = _pw.vega(_call, S0, T)
        truth = _bs_call["vega"]
        assert _within(res["value"], truth, res["std_error"]), (
            f"Pathwise vega={res['value']:.4f} vs BS={truth:.4f} se={res['std_error']:.4f}"
        )

    def test_digital_delta_raises(self):
        """Pathwise delta for digital call must raise — Dirac derivative is undefined."""
        with pytest.raises(NotImplementedError):
            _pw.delta(_digital, S0, T)

    def test_gamma_without_deriv2_raises(self):
        """EuropeanCall has no payoff_deriv2 — gamma should raise."""
        with pytest.raises(NotImplementedError):
            _pw.gamma(_call, S0, T)

    def test_result_keys(self):
        res = _pw.delta(_call, S0, T)
        for key in ("value", "std_error", "ci_lower", "ci_upper", "n_paths"):
            assert key in res


# ---------------------------------------------------------------------------
# 4. Likelihood Ratio vs Black-Scholes (European call)
# ---------------------------------------------------------------------------

class TestLikelihoodRatioVsBS:
    def test_delta(self):
        res = _lr.delta(_call, S0, T)
        truth = _bs_call["delta"]
        assert _within(res["value"], truth, res["std_error"]), (
            f"LR delta={res['value']:.5f} vs BS={truth:.5f} se={res['std_error']:.5f}"
        )

    def test_gamma(self):
        res = _lr.gamma(_call, S0, T)
        truth = _bs_call["gamma"]
        assert _within(res["value"], truth, res["std_error"]), (
            f"LR gamma={res['value']:.6f} vs BS={truth:.6f} se={res['std_error']:.6f}"
        )

    def test_vega(self):
        res = _lr.vega(_call, S0, T)
        truth = _bs_call["vega"]
        assert _within(res["value"], truth, res["std_error"]), (
            f"LR vega={res['value']:.4f} vs BS={truth:.4f} se={res['std_error']:.4f}"
        )

    def test_theta(self):
        res = _lr.theta(_call, S0, T)
        truth = _bs_call["theta"]
        assert _within(res["value"], truth, res["std_error"]), (
            f"LR theta={res['value']:.4f} vs BS={truth:.4f} se={res['std_error']:.4f}"
        )

    def test_rho(self):
        res = _lr.rho(_call, S0, T)
        truth = _bs_call["rho"]
        assert _within(res["value"], truth, res["std_error"]), (
            f"LR rho={res['value']:.4f} vs BS={truth:.4f} se={res['std_error']:.4f}"
        )

    def test_lr_equals_malliavin_for_terminal_payoff(self):
        """For GBM terminal payoffs, LR and Malliavin share the same weight."""
        model2 = GeometricBrownianMotion(r=R, q=Q, sigma=SIGMA)
        eng2 = MonteCarloEngine(model2, n_paths=100_000, n_steps=1, rng_seed=SEED)
        mall2 = MalliavinGreeks(model2, eng2)
        lr2 = LikelihoodRatioGreeks(model2, eng2)

        r_mall = mall2.delta(_call, S0, T)
        r_lr = lr2.delta(_call, S0, T)

        # Same seed → same simulation → same result
        assert abs(r_mall["value"] - r_lr["value"]) < 1e-10


# ---------------------------------------------------------------------------
# 5. Digital call — the critical differentiator
# ---------------------------------------------------------------------------

class TestDigitalCallComparison:
    """
    For the digital call payoff 1_{S_T > K}:
      - Malliavin / LR : finite-variance estimator (weight independent of payoff)
      - Finite diff    : large std_error due to discontinuity near K
      - Pathwise       : undefined — raises NotImplementedError
    """

    # Analytic truth
    _d1, _d2, _sqrt_T = analytical._d1d2(S0, K, T, R, Q, SIGMA)
    _digital_delta_truth = float(np.exp(-R * T) * norm.pdf(_d2) / (S0 * SIGMA * np.sqrt(T)))

    def test_malliavin_delta_converges(self):
        res = _mall.delta(_digital, S0, T)
        # Check finite std_error and agreement with analytic
        assert np.isfinite(res["std_error"])
        assert _within(res["value"], self._digital_delta_truth, res["std_error"]), (
            f"Malliavin digital delta={res['value']:.6f} vs truth={self._digital_delta_truth:.6f} "
            f"se={res['std_error']:.6f}"
        )

    def test_lr_delta_converges(self):
        res = _lr.delta(_digital, S0, T)
        assert np.isfinite(res["std_error"])
        assert _within(res["value"], self._digital_delta_truth, res["std_error"])

    def test_pathwise_delta_raises(self):
        with pytest.raises(NotImplementedError):
            _pw.delta(_digital, S0, T)

    def test_fd_has_larger_std_error_than_malliavin(self):
        """
        FD std_error for digital call is much larger than Malliavin's.

        At the optimal bump h ~ n^{-1/4} ≈ 0.027 for n=500k, FD std_error
        should be visibly larger than Malliavin's.  This documents the
        variance explosion for discontinuous payoffs.
        """
        # Use a small bump to expose the variance issue
        fd_tight = FiniteDifferenceGreeks(
            _model, _engine, bump_size=0.005, bump_type="relative"
        )
        res_fd = fd_tight.delta(_digital, S0, T)
        res_mall = _mall.delta(_digital, S0, T)

        assert res_fd["std_error"] > res_mall["std_error"], (
            f"Expected FD se ({res_fd['std_error']:.4f}) > Malliavin se ({res_mall['std_error']:.4f})"
        )

    def test_malliavin_digital_std_error_is_finite(self):
        """Malliavin std_error for digital call must be finite and positive."""
        res = _mall.delta(_digital, S0, T)
        assert 0 < res["std_error"] < 1.0


# ---------------------------------------------------------------------------
# 6. Put-call parity for Greeks
# ---------------------------------------------------------------------------

class TestPutCallParity:
    """
    Put-call parity implies:
        delta(call) - delta(put) = e^{-qT}
        gamma(call) = gamma(put)
        vega(call)  = vega(put)
        theta(call) - theta(put) = -r K e^{-rT} + q S0 e^{-qT}   (less commonly tested)
    """

    def _parity_tol(self, se_call: float, se_put: float) -> float:
        return 5.0 * np.sqrt(se_call**2 + se_put**2) + 1e-4

    def test_delta_parity_malliavin(self):
        dc = _mall.delta(_call, S0, T)
        dp = _mall.delta(_put, S0, T)
        diff = dc["value"] - dp["value"]
        expected = np.exp(-Q * T)
        tol = self._parity_tol(dc["std_error"], dp["std_error"])
        assert abs(diff - expected) < tol, (
            f"delta(call)-delta(put)={diff:.5f} vs e^{{-qT}}={expected:.5f}"
        )

    def test_gamma_parity_malliavin(self):
        gc = _mall.gamma(_call, S0, T)
        gp = _mall.gamma(_put, S0, T)
        diff = gc["value"] - gp["value"]
        tol = self._parity_tol(gc["std_error"], gp["std_error"])
        assert abs(diff) < tol, (
            f"gamma(call)-gamma(put)={diff:.6f}, expected 0"
        )

    def test_vega_parity_malliavin(self):
        vc = _mall.vega(_call, S0, T)
        vp = _mall.vega(_put, S0, T)
        diff = vc["value"] - vp["value"]
        tol = self._parity_tol(vc["std_error"], vp["std_error"])
        assert abs(diff) < tol, (
            f"vega(call)-vega(put)={diff:.4f}, expected 0"
        )

    def test_delta_parity_fd(self):
        dc = _fd.delta(_call, S0, T)
        dp = _fd.delta(_put, S0, T)
        diff = dc["value"] - dp["value"]
        expected = np.exp(-Q * T)
        tol = self._parity_tol(dc["std_error"], dp["std_error"])
        assert abs(diff - expected) < tol

    def test_delta_parity_pathwise(self):
        dc = _pw.delta(_call, S0, T)
        dp = _pw.delta(_put, S0, T)
        diff = dc["value"] - dp["value"]
        expected = np.exp(-Q * T)
        tol = self._parity_tol(dc["std_error"], dp["std_error"])
        assert abs(diff - expected) < tol

    def test_delta_parity_lr(self):
        dc = _lr.delta(_call, S0, T)
        dp = _lr.delta(_put, S0, T)
        diff = dc["value"] - dp["value"]
        expected = np.exp(-Q * T)
        tol = self._parity_tol(dc["std_error"], dp["std_error"])
        assert abs(diff - expected) < tol


# ---------------------------------------------------------------------------
# 7. MalliavinGreeks.all_greeks — single simulation
# ---------------------------------------------------------------------------

class TestAllGreeks:
    def test_returns_all_five(self):
        result = _mall.all_greeks(_call, S0, T)
        for name in ("delta", "gamma", "vega", "rho", "theta"):
            assert name in result, f"Missing {name}"

    def test_each_result_has_keys(self):
        result = _mall.all_greeks(_call, S0, T)
        for name, res in result.items():
            for key in ("value", "std_error", "ci_lower", "ci_upper", "n_paths"):
                assert key in res, f"{name} missing key {key}"

    def test_delta_agrees_with_bs(self):
        result = _mall.all_greeks(_call, S0, T)
        res = result["delta"]
        truth = _bs_call["delta"]
        assert _within(res["value"], truth, res["std_error"])

    def test_vega_agrees_with_bs(self):
        result = _mall.all_greeks(_call, S0, T)
        res = result["vega"]
        truth = _bs_call["vega"]
        assert _within(res["value"], truth, res["std_error"])


# ---------------------------------------------------------------------------
# 8. MalliavinGreeks.higher_order (gamma)
# ---------------------------------------------------------------------------

class TestHigherOrder:
    def test_gamma_agrees_with_bs(self):
        """higher_order gamma = ∂²V/∂S² should match BS gamma."""
        res = _mall.higher_order(_call, S0, T, greek_name="gamma")
        truth = _bs_call["gamma"]
        assert _within(res["value"], truth, res["std_error"]), (
            f"higher_order gamma={res['value']:.6f} vs BS={truth:.6f} se={res['std_error']:.6f}"
        )

    def test_gamma_consistent_with_first_order(self):
        """higher_order gamma and first-order gamma should agree closely."""
        res_ho = _mall.higher_order(_call, S0, T, greek_name="gamma")
        res_fo = _mall.gamma(_call, S0, T)
        # Both target the same quantity — combined SE test
        combined_se = np.sqrt(res_ho["std_error"]**2 + res_fo["std_error"]**2)
        assert abs(res_ho["value"] - res_fo["value"]) < 5.0 * combined_se + 1e-4

    def test_vanna_is_finite(self):
        res = _mall.higher_order(_call, S0, T, greek_name="vanna")
        assert np.isfinite(res["value"])
        assert np.isfinite(res["std_error"])

    def test_volga_is_finite(self):
        res = _mall.higher_order(_call, S0, T, greek_name="volga")
        assert np.isfinite(res["value"])
        assert np.isfinite(res["std_error"])


# ---------------------------------------------------------------------------
# 9. sim_result sharing
# ---------------------------------------------------------------------------

class TestSimResultSharing:
    """Passing sim_result avoids re-simulation — results must be reproducible."""

    def test_delta_and_vega_share_simulation(self):
        model2 = GeometricBrownianMotion(r=R, q=Q, sigma=SIGMA)
        eng2 = MonteCarloEngine(model2, n_paths=50_000, n_steps=1, rng_seed=7)
        mall2 = MalliavinGreeks(model2, eng2)

        sim = eng2._rng()  # pre-draw rng
        # Simulate once externally
        out = model2.simulate(S0, T, 1, 50_000, return_full_paths=True,
                              rng=np.random.default_rng(7))

        r_delta = mall2.delta(_call, S0, T, sim_result=out)
        r_vega = mall2.vega(_call, S0, T, sim_result=out)

        # Both used the same simulation — results should be finite and positive
        assert np.isfinite(r_delta["value"])
        assert np.isfinite(r_vega["value"])
        assert r_delta["value"] > 0
        assert r_vega["value"] > 0


# ---------------------------------------------------------------------------
# 10. Analytical module spot-checks
# ---------------------------------------------------------------------------

class TestAnalytical:
    def test_bs_delta_call(self):
        delta = analytical.bs_delta(S0, K, T, R, Q, SIGMA, "call")
        assert abs(delta - _bs_call["delta"]) < 1e-10

    def test_bs_gamma(self):
        gamma = analytical.bs_gamma(S0, K, T, R, Q, SIGMA)
        assert abs(gamma - _bs_call["gamma"]) < 1e-10

    def test_bs_vega(self):
        vega = analytical.bs_vega(S0, K, T, R, Q, SIGMA)
        assert abs(vega - _bs_call["vega"]) < 1e-10

    def test_bs_theta_call(self):
        theta = analytical.bs_theta(S0, K, T, R, Q, SIGMA, "call")
        assert abs(theta - _bs_call["theta"]) < 1e-10

    def test_bs_rho_call(self):
        rho = analytical.bs_rho(S0, K, T, R, Q, SIGMA, "call")
        assert abs(rho - _bs_call["rho"]) < 1e-10

    def test_bs_delta_put(self):
        delta = analytical.bs_delta(S0, K, T, R, Q, SIGMA, "put")
        assert abs(delta - _bs_put["delta"]) < 1e-10

    def test_bs_digital_delta_formula(self):
        """e^{-rT} phi(d2) / (S0 sigma sqrt(T))."""
        _, d2, sqrt_T = analytical._d1d2(S0, K, T, R, Q, SIGMA)
        expected = float(np.exp(-R * T) * norm.pdf(d2) / (S0 * SIGMA * sqrt_T))
        result = analytical.bs_digital_delta(S0, K, T, R, Q, SIGMA)
        assert abs(result - expected) < 1e-12

    def test_bs_digital_gamma_sign(self):
        """Digital gamma is negative for ATM (d1 > 0 → negative gamma)."""
        gamma = analytical.bs_digital_gamma(S0, K, T, R, Q, SIGMA)
        # For ATM with positive drift, d1 > 0, so gamma < 0
        d1, _, _ = analytical._d1d2(S0, K, T, R, Q, SIGMA)
        if d1 > 0:
            assert gamma < 0

    def test_bs_barrier_delta_above_barrier(self):
        """Down-and-out call delta with B << S0 should be close to vanilla delta."""
        B = 80.0  # well below S0=100
        barrier_delta = analytical.bs_barrier_delta(S0, K, B, T, R, Q, SIGMA)
        vanilla_delta = analytical.bs_delta(S0, K, T, R, Q, SIGMA, "call")
        # When barrier is far OTM, knock-out probability is small → delta ≈ vanilla
        assert abs(barrier_delta - vanilla_delta) < 0.05

    def test_bs_barrier_delta_at_barrier_is_zero(self):
        """When S0 = B (on the barrier), option is worthless → delta=0."""
        B = S0
        delta = analytical.bs_barrier_delta(S0, K, B, T, R, Q, SIGMA)
        assert delta == 0.0

    def test_geometric_asian_greeks_structure(self):
        result = analytical.geometric_asian_greeks(S0, K, T, R, Q, SIGMA, 52)
        for key in ("price", "delta", "gamma", "vega"):
            assert key in result
        assert result["price"] > 0
        assert 0 < result["delta"] < 1

    def test_geometric_asian_delta_less_than_european(self):
        """Geometric Asian delta < European call delta (lower effective vol)."""
        asian_delta = analytical.geometric_asian_greeks(S0, K, T, R, Q, SIGMA, 52)["delta"]
        euro_delta = analytical.bs_delta(S0, K, T, R, Q, SIGMA, "call")
        assert asian_delta < euro_delta


# ---------------------------------------------------------------------------
# 11. Non-GBM model raises
# ---------------------------------------------------------------------------

class TestNonGBMRaises:
    def test_malliavin_raises_for_non_gbm(self):
        from mgreeks.models.local_vol import LocalVolModel

        def flat_vol(t, S):
            return np.full_like(S, 0.2)

        lv_model = LocalVolModel(r=R, q=Q, sigma_func=flat_vol)
        lv_engine = MonteCarloEngine(lv_model, n_paths=1000, n_steps=10, rng_seed=0)
        mall = MalliavinGreeks(lv_model, lv_engine)
        with pytest.raises(NotImplementedError):
            mall.delta(_call, S0, T)

    def test_lr_raises_for_non_gbm(self):
        from mgreeks.models.local_vol import LocalVolModel

        def flat_vol(t, S):
            return np.full_like(S, 0.2)

        lv_model = LocalVolModel(r=R, q=Q, sigma_func=flat_vol)
        lv_engine = MonteCarloEngine(lv_model, n_paths=1000, n_steps=10, rng_seed=0)
        lr = LikelihoodRatioGreeks(lv_model, lv_engine)
        with pytest.raises(NotImplementedError):
            lr.delta(_call, S0, T)
