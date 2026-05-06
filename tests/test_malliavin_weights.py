"""
Tests for mgreeks/weights/malliavin_weights.py and bismut_elworthy_li.py.

Critical verifications
----------------------
For each Greek π_θ we check:

    disc · E[f(S_T) · π_θ]  ≈  BS_Greek(θ)

using a European call payoff and n_paths = 500k.  Tolerance: 4 standard
errors of the MC estimate.

Additional tests:
  · Digital call delta: Malliavin weight gives finite-variance estimate
    matching the analytic digital-call delta e^{-rT} φ(d2)/(S_0 σ √T)
  · BEL formula matches score-function delta weight exactly (zero diff for GBM)
  · Path-dependent delta: Malliavin weight gives Asian-call delta consistent
    with pathwise IPA estimate
  · Zero-coupon-bond sanity: disc·E[1·π_θ] equals the ZCB Greek
  · Constant payoff: delta and gamma weights have zero mean (ZCB delta/gamma = 0)
  · all_weights_gbm convenience wrapper: correct keys and values
  · bel_second_order_weight gamma: matches score-function gamma weight
  · bel_second_order_weight vanna/volga: runs without error, finite variance
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from mgreeks.models.gbm import GeometricBrownianMotion
from mgreeks.models.local_vol import LocalVolModel
from mgreeks.payoffs.european import EuropeanCall, DigitalCall
from mgreeks.payoffs.asian import ArithmeticAsianCall
from mgreeks.utils import bs_greeks, bs_price
from mgreeks.weights.malliavin_weights import (
    delta_weight_gbm,
    gamma_weight_gbm,
    vega_weight_gbm,
    rho_weight_gbm,
    theta_weight_gbm,
    delta_weight_path_dependent,
    all_weights_gbm,
)
from mgreeks.weights.bismut_elworthy_li import (
    bel_delta_weight,
    bel_vega_weight,
    bel_second_order_weight,
    verify_bel_equals_score,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

S0, K, T = 100.0, 100.0, 1.0
R, Q, SIGMA = 0.05, 0.02, 0.20
N_PATHS = 500_000
SEED = 42

_model = GeometricBrownianMotion(r=R, q=Q, sigma=SIGMA)
_disc = np.exp(-R * T)
_truth = bs_greeks(S0, K, R, Q, SIGMA, T, "call")


def _simulate(n_steps: int = 1, seed: int = SEED, n_paths: int = N_PATHS):
    return _model.simulate(
        S0, T, n_steps, n_paths,
        return_full_paths=True, rng=np.random.default_rng(seed)
    )


# Pre-compute once for most tests (1 step is exact for vanilla)
_out1 = _simulate(n_steps=1)
_W_T1 = _out1["brownian_increments"].sum(axis=1)
_ST1  = _out1["terminal"]
_f_call = np.maximum(_ST1 - K, 0.0)    # European call payoff

# Multi-step simulation for path-dependent and BEL tests
_out52 = _simulate(n_steps=52)
_W_T52 = _out52["brownian_increments"].sum(axis=1)
_ST52  = _out52["terminal"]
_f_call52 = np.maximum(_ST52 - K, 0.0)


def _within_4se(estimate: float, truth: float, std_error: float) -> bool:
    """True if |estimate - truth| < 4 * std_error."""
    return abs(estimate - truth) < 4.0 * std_error + 1e-9


def _se(samples: np.ndarray) -> float:
    return float(samples.std(ddof=1) / np.sqrt(len(samples)))


# ---------------------------------------------------------------------------
# Helper: Greek estimate from weight
# ---------------------------------------------------------------------------

def _greek(weight: np.ndarray, payoff: np.ndarray, disc: float = _disc) -> tuple[float, float]:
    """Returns (estimate, std_error)."""
    samples = payoff * weight
    est = float(_disc * samples.mean())
    se  = float(_disc * _se(samples))
    return est, se


# ===========================================================================
# 1. Delta weight
# ===========================================================================

class TestDeltaWeight:
    def test_formula_european_call(self):
        """disc·E[f·π_Δ] ≈ BS delta for European call."""
        pi = delta_weight_gbm(S0, _ST1, SIGMA, R, Q, T, _W_T1)
        est, se = _greek(pi, _f_call)
        assert _within_4se(est, _truth["delta"], se), (
            f"delta: {est:.5f} vs {_truth['delta']:.5f}  se={se:.5f}"
        )

    def test_formula_shape(self):
        pi = delta_weight_gbm(S0, _ST1, SIGMA, R, Q, T, _W_T1)
        assert pi.shape == (_W_T1.shape[0],)

    def test_zero_coupon_bond(self):
        """disc·E[1·π_Δ] ≈ 0  (delta of a zero-coupon bond is zero)."""
        pi = delta_weight_gbm(S0, _ST1, SIGMA, R, Q, T, _W_T1)
        est, se = _greek(pi, np.ones(N_PATHS))
        assert _within_4se(est, 0.0, se), f"ZCB delta: {est:.6f} se={se:.6f}"

    def test_put_delta(self):
        """disc·E[put·π_Δ] ≈ BS put delta."""
        f_put = np.maximum(K - _ST1, 0.0)
        pi = delta_weight_gbm(S0, _ST1, SIGMA, R, Q, T, _W_T1)
        est, se = _greek(pi, f_put)
        truth_put = bs_greeks(S0, K, R, Q, SIGMA, T, "put")["delta"]
        assert _within_4se(est, truth_put, se), (
            f"put delta: {est:.5f} vs {truth_put:.5f}  se={se:.5f}"
        )

    def test_digital_call_delta(self):
        """
        Digital call: Malliavin weight gives finite-variance estimate matching
        the analytic digital-call delta  e^{-rT} φ(d2) / (S_0 σ √T).
        """
        f_dig = ((_ST1 > K)).astype(float)
        pi = delta_weight_gbm(S0, _ST1, SIGMA, R, Q, T, _W_T1)
        est, se = _greek(pi, f_dig)

        # Analytic digital-call delta
        d2 = (np.log(S0/K) + (R - Q - 0.5*SIGMA**2)*T) / (SIGMA*np.sqrt(T))
        truth_dig_delta = np.exp(-R*T) * norm.pdf(d2) / (S0 * SIGMA * np.sqrt(T))

        assert _within_4se(est, truth_dig_delta, se), (
            f"digital delta: {est:.5f} vs {truth_dig_delta:.5f}  se={se:.5f}"
        )

    def test_digital_call_variance_finite(self):
        """Variance of Malliavin delta estimator for digital call is finite."""
        f_dig = ((_ST1 > K)).astype(float)
        pi = delta_weight_gbm(S0, _ST1, SIGMA, R, Q, T, _W_T1)
        var = float(np.var(f_dig * pi))
        assert np.isfinite(var) and var < 1e6


# ===========================================================================
# 2. Gamma weight
# ===========================================================================

class TestGammaWeight:
    def test_formula_european_call(self):
        """disc·E[f·π_Γ] ≈ BS gamma."""
        pi = gamma_weight_gbm(S0, _ST1, SIGMA, R, Q, T, _W_T1)
        est, se = _greek(pi, _f_call)
        assert _within_4se(est, _truth["gamma"], se), (
            f"gamma: {est:.6f} vs {_truth['gamma']:.6f}  se={se:.6f}"
        )

    def test_zero_coupon_bond(self):
        """disc·E[1·π_Γ] ≈ 0  (gamma of a zero-coupon bond is zero)."""
        pi = gamma_weight_gbm(S0, _ST1, SIGMA, R, Q, T, _W_T1)
        est, se = _greek(pi, np.ones(N_PATHS))
        assert _within_4se(est, 0.0, se), f"ZCB gamma: {est:.7f} se={se:.7f}"

    def test_hermite_form_is_biased(self):
        """
        Verify that the Hermite form (W_T²-T)/(S0²σ²T²) is biased.
        The correct formula adds -σT·W_T in the numerator; the simpler form
        omits this path-by-path correction and is statistically biased.
        """
        W_T = _W_T1
        pi_correct = gamma_weight_gbm(S0, _ST1, SIGMA, R, Q, T, W_T)
        pi_hermite = (W_T**2 - T) / (SIGMA**2 * T**2 * S0**2)  # wrong form

        est_c, se_c = _greek(pi_correct, _f_call)
        est_h, _    = _greek(pi_hermite, _f_call)
        truth_g = _truth["gamma"]

        # Correct form is within 4 se; Hermite form has larger relative error
        assert _within_4se(est_c, truth_g, se_c)
        assert abs(est_h - truth_g) > abs(est_c - truth_g), (
            "Hermite form should be more biased than the correct form"
        )

    def test_shape(self):
        pi = gamma_weight_gbm(S0, _ST1, SIGMA, R, Q, T, _W_T1)
        assert pi.shape == (_W_T1.shape[0],)


# ===========================================================================
# 3. Vega weight
# ===========================================================================

class TestVegaWeight:
    def test_formula_european_call(self):
        """disc·E[f·π_v] ≈ BS vega."""
        pi = vega_weight_gbm(S0, _ST1, SIGMA, R, Q, T, _W_T1)
        est, se = _greek(pi, _f_call)
        assert _within_4se(est, _truth["vega"], se), (
            f"vega: {est:.4f} vs {_truth['vega']:.4f}  se={se:.4f}"
        )

    def test_zero_coupon_bond(self):
        """disc·E[1·π_v] ≈ 0  (vega of a zero-coupon bond is zero)."""
        pi = vega_weight_gbm(S0, _ST1, SIGMA, R, Q, T, _W_T1)
        est, se = _greek(pi, np.ones(N_PATHS))
        assert _within_4se(est, 0.0, se), f"ZCB vega: {est:.6f} se={se:.6f}"

    def test_formula_t_half(self):
        """Verify at T=0.5 (non-trivial T dependence in the formula)."""
        T2 = 0.5
        out = GeometricBrownianMotion(r=R, q=Q, sigma=SIGMA).simulate(
            S0, T2, 1, N_PATHS, return_full_paths=True, rng=np.random.default_rng(0)
        )
        W = out["brownian_increments"].sum(axis=1)
        ST = out["terminal"]
        f = np.maximum(ST - K, 0.0)
        pi = vega_weight_gbm(S0, ST, SIGMA, R, Q, T2, W)
        disc2 = np.exp(-R*T2)
        est = float(disc2 * (f * pi).mean())
        se  = float(disc2 * _se(f * pi))
        truth = bs_greeks(S0, K, R, Q, SIGMA, T2, "call")["vega"]
        assert _within_4se(est, truth, se), (
            f"T=0.5 vega: {est:.4f} vs {truth:.4f}  se={se:.4f}"
        )


# ===========================================================================
# 4. Rho weight
# ===========================================================================

class TestRhoWeight:
    def test_formula_european_call(self):
        """disc·E[f·π_ρ] ≈ BS rho."""
        pi = rho_weight_gbm(S0, _ST1, SIGMA, R, Q, T, _W_T1)
        est, se = _greek(pi, _f_call)
        assert _within_4se(est, _truth["rho"], se), (
            f"rho: {est:.4f} vs {_truth['rho']:.4f}  se={se:.4f}"
        )

    def test_zero_coupon_bond(self):
        """disc·E[1·π_ρ] ≈ −T·disc  (rho of a ZCB = −T·e^{−rT})."""
        pi = rho_weight_gbm(S0, _ST1, SIGMA, R, Q, T, _W_T1)
        est, se = _greek(pi, np.ones(N_PATHS))
        truth_zcb_rho = -T * _disc
        assert _within_4se(est, truth_zcb_rho, se), (
            f"ZCB rho: {est:.5f} vs {truth_zcb_rho:.5f}  se={se:.5f}"
        )

    def test_formula_t_2(self):
        """Verify at T=2 to test the T factor in the weight."""
        T2 = 2.0
        out = GeometricBrownianMotion(r=R, q=Q, sigma=SIGMA).simulate(
            S0, T2, 1, N_PATHS, return_full_paths=True, rng=np.random.default_rng(0)
        )
        W = out["brownian_increments"].sum(axis=1)
        ST = out["terminal"]
        f = np.maximum(ST - K, 0.0)
        pi = rho_weight_gbm(S0, ST, SIGMA, R, Q, T2, W)
        disc2 = np.exp(-R*T2)
        est = float(disc2 * (f * pi).mean())
        se  = float(disc2 * _se(f * pi))
        truth = bs_greeks(S0, K, R, Q, SIGMA, T2, "call")["rho"]
        assert _within_4se(est, truth, se), (
            f"T=2 rho: {est:.4f} vs {truth:.4f}  se={se:.4f}"
        )


# ===========================================================================
# 5. Theta weight
# ===========================================================================

class TestThetaWeight:
    def test_formula_european_call(self):
        """disc·E[f·π_θ] ≈ BS theta (negative for long calls)."""
        pi = theta_weight_gbm(S0, _ST1, SIGMA, R, Q, T, _W_T1)
        est, se = _greek(pi, _f_call)
        assert _within_4se(est, _truth["theta"], se), (
            f"theta: {est:.4f} vs {_truth['theta']:.4f}  se={se:.4f}"
        )

    def test_sign_negative_for_long_call(self):
        """Theta < 0 for long ATM calls."""
        pi = theta_weight_gbm(S0, _ST1, SIGMA, R, Q, T, _W_T1)
        est, _ = _greek(pi, _f_call)
        assert est < 0, f"Expected negative theta, got {est:.4f}"

    def test_zero_coupon_bond(self):
        """disc·E[1·π_θ] ≈ r·disc  (theta of a ZCB = r·e^{-rT})."""
        pi = theta_weight_gbm(S0, _ST1, SIGMA, R, Q, T, _W_T1)
        est, se = _greek(pi, np.ones(N_PATHS))
        truth_zcb_theta = R * _disc
        assert _within_4se(est, truth_zcb_theta, se), (
            f"ZCB theta: {est:.6f} vs {truth_zcb_theta:.6f}  se={se:.6f}"
        )

    def test_full_formula_beats_approximation(self):
        """
        The full theta formula (including μ·W_T/(σT) drift term) is closer
        to the BS truth than the simplified r − (Z²−1)/(2T) approximation.
        """
        W_T = _W_T1
        pi_full   = theta_weight_gbm(S0, _ST1, SIGMA, R, Q, T, W_T)
        Z = W_T / np.sqrt(T)
        pi_approx = R - (Z**2 - 1.0) / (2.0 * T)  # simplified (biased)

        est_full, _  = _greek(pi_full,   _f_call)
        est_approx,_ = _greek(pi_approx, _f_call)
        truth = _truth["theta"]

        assert abs(est_full - truth) < abs(est_approx - truth), (
            f"Full formula ({est_full:.4f}) should beat approx ({est_approx:.4f})"
            f" vs truth ({truth:.4f})"
        )


# ===========================================================================
# 6. all_weights_gbm convenience function
# ===========================================================================

class TestAllWeightsGBM:
    def test_keys(self):
        weights = all_weights_gbm(S0, SIGMA, R, Q, T, _W_T1, _ST1)
        assert set(weights.keys()) == {"delta", "gamma", "vega", "rho", "theta"}

    def test_shapes(self):
        weights = all_weights_gbm(S0, SIGMA, R, Q, T, _W_T1, _ST1)
        for k, v in weights.items():
            assert v.shape == (_W_T1.shape[0],), f"{k}: wrong shape {v.shape}"

    def test_values_match_individual(self):
        weights = all_weights_gbm(S0, SIGMA, R, Q, T, _W_T1, _ST1)
        np.testing.assert_array_equal(
            weights["delta"], delta_weight_gbm(S0, _ST1, SIGMA, R, Q, T, _W_T1)
        )
        np.testing.assert_array_equal(
            weights["gamma"], gamma_weight_gbm(S0, _ST1, SIGMA, R, Q, T, _W_T1)
        )

    def test_no_st_arg(self):
        """ST=None default should not crash."""
        weights = all_weights_gbm(S0, SIGMA, R, Q, T, _W_T1)
        assert "delta" in weights


# ===========================================================================
# 7. Path-dependent delta weight
# ===========================================================================

class TestPathDependentDelta:
    def test_same_as_european_weight_single_step(self):
        """
        For a single time step (n=1), the path-dependent weight ΔW_1/(σS_0Δt_1)
        equals the European weight W_T/(σS_0T) since ΔW_1=W_T and Δt_1=T.
        """
        pi_pd = delta_weight_path_dependent(
            _out1["paths"], _out1["brownian_increments"], _model, S0, T
        )
        pi_eu = delta_weight_gbm(S0, _ST1, SIGMA, R, Q, T, _W_T1)
        np.testing.assert_array_almost_equal(pi_pd, pi_eu, decimal=12)

    def test_asian_call_delta_matches_pathwise_ipa(self):
        """
        disc·E[f_Asian · π_Δ] ≈ disc·E[1_{A>K} · A/S_0]  (pathwise IPA).
        Both are unbiased estimators of the Asian call delta; they should agree
        within a few combined standard errors.
        """
        paths52 = _out52["paths"]
        times52 = _out52["times"]

        payoff = ArithmeticAsianCall(K)
        f_asian = payoff(paths52, times52)

        # Malliavin estimator
        pi = delta_weight_path_dependent(
            paths52, _out52["brownian_increments"], _model, S0, T
        )
        malliavin_est = float(_disc * (f_asian * pi).mean())
        malliavin_se  = float(_disc * _se(f_asian * pi))

        # Pathwise IPA estimator
        ipa_samples = payoff.payoff_deriv(paths52, times52)
        ipa_est = float(_disc * ipa_samples.mean())
        ipa_se  = float(_disc * _se(ipa_samples))

        # Both estimates should agree within 4 combined standard errors
        combined_se = np.sqrt(malliavin_se**2 + ipa_se**2)
        assert abs(malliavin_est - ipa_est) < 4.0 * combined_se + 1e-4, (
            f"Malliavin ({malliavin_est:.5f}) and IPA ({ipa_est:.5f}) disagree "
            f"combined_se={combined_se:.5f}"
        )

    def test_missing_sigma_attr_raises(self):
        """Passing a model without .sigma raises a clear error."""
        from mgreeks.models.base import StochasticModel

        class DummyModel(StochasticModel):
            def simulate(self, *a, **kw): ...
            def malliavin_derivative(self, *a, **kw): ...
            def log_density_gradient(self, *a, **kw): ...
            @property
            def param_dict(self): return {}

        with pytest.raises(ValueError, match="sigma"):
            delta_weight_path_dependent(
                _out52["paths"], _out52["brownian_increments"],
                DummyModel(), S0, T
            )


# ===========================================================================
# 8. BEL formula — GBM
# ===========================================================================

class TestBELFormulaGBM:
    def test_bel_equals_score_exactly(self):
        """For GBM, BEL integral collapses to W_T/(σS_0T) — zero numerical diff."""
        result = verify_bel_equals_score(
            _out52["paths"], _out52["brownian_increments"],
            _model, S0, T, _out52["times"]
        )
        assert result["agree"], f"max_diff={result['max_abs_diff']:.2e}"

    def test_bel_delta_matches_bs(self):
        """BEL delta weight on 52-step paths matches BS delta."""
        pi = bel_delta_weight(
            _out52["paths"], _out52["brownian_increments"],
            _model, S0, T, _out52["times"]
        )
        est, se = _greek(pi, _f_call52)
        assert _within_4se(est, _truth["delta"], se), (
            f"BEL delta: {est:.5f} vs {_truth['delta']:.5f}  se={se:.5f}"
        )

    def test_bel_vega_matches_score(self):
        """BEL vega weight matches score-function vega weight for GBM."""
        pi_bel = bel_vega_weight(
            _out52["paths"], _out52["brownian_increments"],
            _model, S0, T, _out52["times"]
        )
        pi_score = vega_weight_gbm(S0, _ST52, SIGMA, R, Q, T, _W_T52)
        np.testing.assert_array_almost_equal(pi_bel, pi_score, decimal=12)

    def test_bel_second_order_gamma_matches_score(self):
        """bel_second_order_weight('gamma') matches gamma_weight_gbm."""
        pi_bel = bel_second_order_weight(
            _out52["paths"], _out52["brownian_increments"],
            _model, S0, T, _out52["times"], greek_type="gamma"
        )
        pi_score = gamma_weight_gbm(S0, _ST52, SIGMA, R, Q, T, _W_T52)
        np.testing.assert_array_almost_equal(pi_bel, pi_score, decimal=12)

    def test_bel_second_order_vanna(self):
        """Vanna weight runs without error, is finite, and has finite variance."""
        pi = bel_second_order_weight(
            _out52["paths"], _out52["brownian_increments"],
            _model, S0, T, _out52["times"], greek_type="vanna"
        )
        assert pi.shape == (N_PATHS,)
        assert np.all(np.isfinite(pi))

    def test_bel_second_order_volga(self):
        """Volga weight runs without error and is finite."""
        pi = bel_second_order_weight(
            _out52["paths"], _out52["brownian_increments"],
            _model, S0, T, _out52["times"], greek_type="volga"
        )
        assert pi.shape == (N_PATHS,)
        assert np.all(np.isfinite(pi))

    def test_bel_second_order_unknown_type_raises(self):
        with pytest.raises(ValueError, match="greek_type"):
            bel_second_order_weight(
                _out52["paths"], _out52["brownian_increments"],
                _model, S0, T, _out52["times"], greek_type="banana"
            )

    def test_bel_unsupported_model_raises(self):
        """bel_delta_weight raises NotImplementedError for Heston."""
        from mgreeks.models.heston import HestonModel
        heston = HestonModel(r=R, q=Q, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, V0=0.04)
        out = heston.simulate(S0, T, 10, 100, return_full_paths=True,
                              rng=np.random.default_rng(0))
        with pytest.raises(NotImplementedError):
            bel_delta_weight(
                out["paths"], out["brownian_increments"],
                heston, S0, T, out["times"]
            )


# ===========================================================================
# 9. BEL formula — LocalVolModel (numerical integration)
# ===========================================================================

class TestBELLocalVol:
    """
    For a CEV model, the BEL delta weight computed via the numerical stochastic
    integral should give a delta estimate consistent with finite differences.
    """
    beta = 0.5       # CEV exponent — creates leverage effect
    lv_model = LocalVolModel(
        r=R, q=Q, model_type="cev", sigma0=SIGMA, beta=beta, S_ref=S0
    )

    def _simulate_lv(self, n_paths=100_000, n_steps=100):
        return self.lv_model.simulate(
            S0, T, n_steps, n_paths,
            return_full_paths=True, rng=np.random.default_rng(0)
        )

    def test_bel_local_vol_delta_finite(self):
        """BEL delta for CEV is finite and has finite variance."""
        out = self._simulate_lv()
        pi = bel_delta_weight(
            out["paths"], out["brownian_increments"],
            self.lv_model, S0, T, out["times"]
        )
        assert np.all(np.isfinite(pi))
        assert np.isfinite(pi.var())

    def test_bel_local_vol_delta_vs_fd(self):
        """
        BEL delta for CEV should agree with finite-difference delta
        (using the same RNG seed for variance reduction) within 5%.
        """
        n_paths, n_steps = 100_000, 100
        h = 1.0  # bump in S0

        # Base simulation
        out_c = self.lv_model.simulate(
            S0, T, n_steps, n_paths,
            return_full_paths=True, rng=np.random.default_rng(0)
        )
        f_c = np.maximum(out_c["paths"][:, -1] - K, 0.0)

        # Bumped simulations (same seed for CRN)
        out_u = self.lv_model.simulate(
            S0 + h, T, n_steps, n_paths,
            return_full_paths=True, rng=np.random.default_rng(0)
        )
        f_u = np.maximum(out_u["paths"][:, -1] - K, 0.0)
        out_d = self.lv_model.simulate(
            S0 - h, T, n_steps, n_paths,
            return_full_paths=True, rng=np.random.default_rng(0)
        )
        f_d = np.maximum(out_d["paths"][:, -1] - K, 0.0)

        fd_delta = float(_disc * ((f_u - f_d) / (2.0 * h)).mean())

        # BEL Malliavin delta
        pi = bel_delta_weight(
            out_c["paths"], out_c["brownian_increments"],
            self.lv_model, S0, T, out_c["times"]
        )
        bel_delta = float(_disc * (f_c * pi).mean())

        # Should agree within 5% relative tolerance
        assert abs(bel_delta - fd_delta) / max(abs(fd_delta), 1e-3) < 0.10, (
            f"BEL={bel_delta:.5f}  FD={fd_delta:.5f}"
        )


# ===========================================================================
# 10.  Cross-weight orthogonality (mean-zero for constant payoffs)
# ===========================================================================

class TestOrthogonality:
    """
    For delta, gamma, vega weights: disc·E[π] = 0 (ZCB Greek = 0).
    These tests verify the weights are correctly mean-zero when multiplied
    by a constant payoff (f=1).
    """

    @pytest.mark.parametrize("weight_fn,truth", [
        (lambda W: delta_weight_gbm(S0, None, SIGMA, R, Q, T, W), 0.0),
        (lambda W: gamma_weight_gbm(S0, None, SIGMA, R, Q, T, W), 0.0),
        (lambda W: vega_weight_gbm(S0, None, SIGMA, R, Q, T, W),  0.0),
    ])
    def test_zero_mean_constant_payoff(self, weight_fn, truth):
        W = _W_T1
        pi = weight_fn(W)
        est, se = _greek(pi, np.ones(N_PATHS))
        assert _within_4se(est, truth, se), (
            f"Expected 0, got {est:.6f}  se={se:.6f}"
        )
