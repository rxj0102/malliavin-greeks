"""
tests/test_variance_reduction.py — Variance reduction for Malliavin Greeks.

Techniques tested
-----------------
1. Antithetic variates  — odd weight functions benefit greatly
2. Control variate      — geometric Asian reduces arithmetic Asian variance
3. Localization         — truncated weight reduces std_error

Each test checks BOTH:
  a) Correctness: the reduced-variance estimator agrees with the truth.
  b) Efficiency:  the std_error (or variance) is measurably smaller.
"""

from __future__ import annotations

import numpy as np
import pytest

from mgreeks.greeks import analytical
from mgreeks.greeks.analytical import bs_delta, bs_gamma
from mgreeks.models.gbm import GeometricBrownianMotion
from mgreeks.payoffs.asian import ArithmeticAsianCall, GeometricAsianCall
from mgreeks.payoffs.european import EuropeanCall
from mgreeks.simulation import MonteCarloEngine
from mgreeks.variance_reduction import (
    antithetic_malliavin,
    geometric_asian_control_variate,
    delta_hedged_control_variate,
    localized_malliavin_weight,
)
from mgreeks.variance_reduction.antithetic import simulate_antithetic
from mgreeks.variance_reduction.localization import (
    auto_localization_radius,
    interval_localized_weight,
)
from mgreeks.weights.malliavin_weights import (
    delta_weight_gbm,
    gamma_weight_gbm,
)


# ---------------------------------------------------------------------------
# Shared parameters
# ---------------------------------------------------------------------------

S0, K, T = 100.0, 100.0, 1.0
R, Q, SIGMA = 0.05, 0.02, 0.20
N_PATHS = 400_000
N_STEPS = 52
SEED = 42

_disc = float(np.exp(-R * T))
_model = GeometricBrownianMotion(r=R, q=Q, sigma=SIGMA)
_engine = MonteCarloEngine(_model, n_paths=N_PATHS, n_steps=N_STEPS, rng_seed=SEED)

_Z95 = 1.959964


def _sim(seed=SEED, n_paths=N_PATHS):
    return _model.simulate(S0, T, N_STEPS, n_paths, return_full_paths=True,
                           rng=np.random.default_rng(seed))


# Pre-compute one shared simulation
_OUT = _sim()
_W_T = _OUT["brownian_increments"].sum(axis=1)
_DELTA_W = delta_weight_gbm(S0, _OUT["terminal"], SIGMA, R, Q, T, _W_T)
_GAMMA_W = gamma_weight_gbm(S0, _OUT["terminal"], SIGMA, R, Q, T, _W_T)


# ---------------------------------------------------------------------------
# 1. Antithetic variates
# ---------------------------------------------------------------------------

class TestAntitheticVariates:
    """Antithetic variates for Malliavin delta and gamma estimators."""

    def _anti_sims(self, seed=SEED, n_paths=N_PATHS):
        return simulate_antithetic(_model, S0, T, N_STEPS, n_paths,
                                   rng=np.random.default_rng(seed))

    # ---- delta: odd weight → large variance reduction ----

    def test_antithetic_delta_unbiased(self):
        """
        Antithetic delta estimator agrees with analytical BS delta.

        Antithetic sampling is unbiased by construction: both ΔW and −ΔW
        have the same marginal distribution under GBM.
        """
        truth = bs_delta(S0, K, T, R, Q, SIGMA, "call")
        payoff = EuropeanCall(K)
        sim_p, sim_n = self._anti_sims()

        def weight(paths, incr):
            W_T = incr.sum(axis=1)
            return delta_weight_gbm(S0, paths[:, -1], SIGMA, R, Q, T, W_T)

        res = antithetic_malliavin(payoff, weight,
                                   sim_p["paths"], sim_n["paths"],
                                   sim_p["brownian_increments"],
                                   sim_n["brownian_increments"], _disc)

        assert abs(res["value"] - truth) < 4.0 * res["std_error"] + 1e-4, (
            f"Antithetic delta={res['value']:.5f} vs truth={truth:.5f}"
        )

    def test_antithetic_delta_reduces_variance(self):
        """
        Antithetic variates reduce variance for the delta estimator.

        The delta weight W_T/(σ S_0 T) is ODD in W_T, and the call payoff
        is increasing in W_T, so their product is approximately odd →
        strong negative correlation with the antithetic → variance reduction.
        """
        payoff = EuropeanCall(K)
        sim_p, sim_n = self._anti_sims()

        def weight(paths, incr):
            W_T = incr.sum(axis=1)
            return delta_weight_gbm(S0, paths[:, -1], SIGMA, R, Q, T, W_T)

        res = antithetic_malliavin(payoff, weight,
                                   sim_p["paths"], sim_n["paths"],
                                   sim_p["brownian_increments"],
                                   sim_n["brownian_increments"], _disc)

        # Must achieve at least 2× variance reduction for delta (typical: 10-20×)
        assert res["variance_reduction_ratio"] >= 2.0, (
            f"Antithetic delta VRR={res['variance_reduction_ratio']:.2f}, expected ≥ 2"
        )

    def test_antithetic_delta_std_error_smaller(self):
        """std_error of antithetic delta < plain delta for same number of paths."""
        payoff = EuropeanCall(K)
        sim_p, sim_n = self._anti_sims()

        def weight(paths, incr):
            W_T = incr.sum(axis=1)
            return delta_weight_gbm(S0, paths[:, -1], SIGMA, R, Q, T, W_T)

        res_anti = antithetic_malliavin(payoff, weight,
                                        sim_p["paths"], sim_n["paths"],
                                        sim_p["brownian_increments"],
                                        sim_n["brownian_increments"], _disc)

        # Plain estimator using only the + paths
        f_pos = payoff(sim_p["paths"], sim_p["times"])
        w_pos = weight(sim_p["paths"], sim_p["brownian_increments"])
        plain_se = float((_disc * f_pos * w_pos).std(ddof=1) / np.sqrt(N_PATHS))

        assert res_anti["std_error"] < plain_se, (
            f"Antithetic SE={res_anti['std_error']:.5f} should be < plain SE={plain_se:.5f}"
        )

    # ---- gamma: even-dominated weight → smaller (but positive) reduction ----

    def test_antithetic_gamma_unbiased(self):
        """Antithetic gamma estimator is unbiased."""
        truth = bs_gamma(S0, K, T, R, Q, SIGMA)
        payoff = EuropeanCall(K)
        sim_p, sim_n = self._anti_sims()

        def weight(paths, incr):
            W_T = incr.sum(axis=1)
            return gamma_weight_gbm(S0, paths[:, -1], SIGMA, R, Q, T, W_T)

        res = antithetic_malliavin(payoff, weight,
                                   sim_p["paths"], sim_n["paths"],
                                   sim_p["brownian_increments"],
                                   sim_n["brownian_increments"], _disc)

        assert abs(res["value"] - truth) < 4.0 * res["std_error"] + 1e-4, (
            f"Antithetic gamma={res['value']:.5f} vs truth={truth:.5f}"
        )

    def test_antithetic_gamma_less_effective_than_delta(self):
        """
        Antithetic variance reduction is smaller for gamma than for delta.

        The gamma weight is approximately EVEN in W_T (dominated by W_T²),
        so the product f·π_Γ is not strongly odd, reducing the benefit.
        """
        payoff = EuropeanCall(K)
        sim_p, sim_n = self._anti_sims()

        def w_delta(paths, incr):
            W_T = incr.sum(axis=1)
            return delta_weight_gbm(S0, paths[:, -1], SIGMA, R, Q, T, W_T)

        def w_gamma(paths, incr):
            W_T = incr.sum(axis=1)
            return gamma_weight_gbm(S0, paths[:, -1], SIGMA, R, Q, T, W_T)

        res_d = antithetic_malliavin(payoff, w_delta,
                                     sim_p["paths"], sim_n["paths"],
                                     sim_p["brownian_increments"],
                                     sim_n["brownian_increments"], _disc)
        res_g = antithetic_malliavin(payoff, w_gamma,
                                     sim_p["paths"], sim_n["paths"],
                                     sim_p["brownian_increments"],
                                     sim_n["brownian_increments"], _disc)

        assert res_d["variance_reduction_ratio"] > res_g["variance_reduction_ratio"], (
            f"Delta VRR={res_d['variance_reduction_ratio']:.2f} should exceed "
            f"gamma VRR={res_g['variance_reduction_ratio']:.2f}"
        )


# ---------------------------------------------------------------------------
# 2. Control variate
# ---------------------------------------------------------------------------

class TestControlVariate:
    """Geometric Asian and delta-hedged control variates."""

    # ---- geometric Asian control variate ----

    def test_geometric_cv_reduces_variance_for_price(self):
        """
        Geometric Asian CV reduces variance for arithmetic Asian price estimation.

        Arithmetic and geometric Asian payoffs have correlation ≈ 0.99,
        giving near-optimal variance reduction.
        """
        arith_payoff = ArithmeticAsianCall(K)
        geo_payoff = GeometricAsianCall(K)

        f_arith = arith_payoff(_OUT["paths"], _OUT["times"])
        f_geo = geo_payoff(_OUT["paths"], _OUT["times"])

        analytic = analytical.geometric_asian_greeks(S0, K, T, R, Q, SIGMA, N_STEPS)
        geo_price = analytic["price"]

        res = geometric_asian_control_variate(f_arith, f_geo, geo_price, _disc)

        # Variance reduction ratio should be substantial (≥ 5×)
        assert res["variance_reduction_ratio"] >= 5.0, (
            f"Geometric CV VRR={res['variance_reduction_ratio']:.2f}, expected ≥ 5"
        )

    def test_geometric_cv_price_unbiased(self):
        """
        Arithmetic Asian price via geometric CV is unbiased.

        Since the CV estimator is constructed with the exact analytical value
        of the geometric Asian, the expectation is preserved.
        """
        arith_payoff = ArithmeticAsianCall(K)
        geo_payoff = GeometricAsianCall(K)

        f_arith = arith_payoff(_OUT["paths"], _OUT["times"])
        f_geo = geo_payoff(_OUT["paths"], _OUT["times"])

        analytic = analytical.geometric_asian_greeks(S0, K, T, R, Q, SIGMA, N_STEPS)
        geo_price = analytic["price"]

        res = geometric_asian_control_variate(f_arith, f_geo, geo_price, _disc)

        # Estimate should be close to FD benchmark (arithmetic Asian price)
        arith_price_plain = float(_disc * f_arith.mean())
        plain_se = float((_disc * f_arith).std(ddof=1) / np.sqrt(N_PATHS))

        # Both estimators should agree within 4 combined SE
        combined_se = np.sqrt(res["std_error"]**2 + plain_se**2)
        assert abs(res["value"] - arith_price_plain) < 4.0 * combined_se + 1e-3, (
            f"CV price={res['value']:.4f} vs plain={arith_price_plain:.4f}"
        )

    def test_geometric_cv_std_error_smaller(self):
        """CV std_error < plain estimator std_error."""
        arith_payoff = ArithmeticAsianCall(K)
        geo_payoff = GeometricAsianCall(K)

        f_arith = arith_payoff(_OUT["paths"], _OUT["times"])
        f_geo = geo_payoff(_OUT["paths"], _OUT["times"])

        analytic = analytical.geometric_asian_greeks(S0, K, T, R, Q, SIGMA, N_STEPS)
        geo_price = analytic["price"]

        res = geometric_asian_control_variate(f_arith, f_geo, geo_price, _disc)
        plain_se = float((_disc * f_arith).std(ddof=1) / np.sqrt(N_PATHS))

        assert res["std_error"] < plain_se, (
            f"CV SE={res['std_error']:.6f} should be < plain SE={plain_se:.6f}"
        )

    def test_geometric_cv_high_correlation(self):
        """Arithmetic and geometric Asian payoffs have high correlation (≥ 0.98)."""
        arith_payoff = ArithmeticAsianCall(K)
        geo_payoff = GeometricAsianCall(K)
        f_arith = arith_payoff(_OUT["paths"], _OUT["times"])
        f_geo = geo_payoff(_OUT["paths"], _OUT["times"])
        corr = float(np.corrcoef(f_arith, f_geo)[0, 1])
        assert corr >= 0.98, f"Correlation={corr:.4f} should be ≥ 0.98"

    def test_geometric_cv_delta_reduces_variance(self):
        """
        Geometric CV reduces variance for ARITHMETIC ASIAN DELTA estimation.

        The Greek estimator f_arith · π is controlled by f_geo · π using
        the analytical geometric Asian delta.
        """
        arith_payoff = ArithmeticAsianCall(K)
        geo_payoff = GeometricAsianCall(K)

        from mgreeks.weights.malliavin_weights import delta_weight_path_dependent
        pd_weight = delta_weight_path_dependent(
            _OUT["paths"], _OUT["brownian_increments"], _model, S0, T
        )

        f_arith_w = arith_payoff(_OUT["paths"], _OUT["times"]) * pd_weight
        f_geo_w = geo_payoff(_OUT["paths"], _OUT["times"]) * pd_weight

        analytic = analytical.geometric_asian_greeks(S0, K, T, R, Q, SIGMA, N_STEPS)
        geo_delta = analytic["delta"]

        res = geometric_asian_control_variate(
            f_arith_w, f_geo_w, geo_delta, _disc
        )

        # VRR ≥ 2 for delta (arithmetic/geometric delta estimators are correlated)
        assert res["variance_reduction_ratio"] >= 2.0, (
            f"Delta CV VRR={res['variance_reduction_ratio']:.2f}, expected ≥ 2"
        )

    # ---- delta-hedged control variate ----

    def test_delta_hedged_cv_reduces_variance(self):
        """
        Delta-hedged control variate reduces variance for European call price.

        f(S_T) and f(S_T) − Δ·S_T are highly correlated, so the CV
        captures most of the variance.
        """
        payoff = EuropeanCall(K)
        f = payoff(_OUT["paths"], _OUT["times"])
        delta_anal = bs_delta(S0, K, T, R, Q, SIGMA, "call")

        res = delta_hedged_control_variate(f, _OUT["paths"], delta_anal, S0, _disc)
        plain_se = float((_disc * f).std(ddof=1) / np.sqrt(N_PATHS))

        assert res["std_error"] < plain_se, (
            f"Delta-hedged CV SE={res['std_error']:.6f} should be < plain SE={plain_se:.6f}"
        )

    def test_delta_hedged_cv_unbiased(self):
        """Delta-hedged CV price estimate is unbiased."""
        from mgreeks.utils import bs_price
        payoff = EuropeanCall(K)
        f = payoff(_OUT["paths"], _OUT["times"])
        delta_anal = bs_delta(S0, K, T, R, Q, SIGMA, "call")

        res = delta_hedged_control_variate(f, _OUT["paths"], delta_anal, S0, _disc)
        truth = bs_price(S0, K, R, Q, SIGMA, T, "call")

        assert abs(res["value"] - truth) < 4.0 * res["std_error"] + 1e-4, (
            f"Delta-hedged CV price={res['value']:.5f} vs truth={truth:.5f}"
        )

    def test_delta_hedged_cv_high_correlation(self):
        """f(S_T) and S_T are highly correlated (≥ 0.9) for an ATM call."""
        payoff = EuropeanCall(K)
        f = payoff(_OUT["paths"], _OUT["times"])
        ST = _OUT["paths"][:, -1]
        # The control variate is z = S_T (zero-mean after subtracting E[S_T])
        # The correlation between call payoff and terminal spot is high for ATM
        corr = float(np.corrcoef(f, ST)[0, 1])
        assert corr >= 0.9, f"Corr(f, S_T)={corr:.4f} should be ≥ 0.9"


# ---------------------------------------------------------------------------
# 3. Localization
# ---------------------------------------------------------------------------

class TestLocalization:
    """Malliavin weight localization / truncation."""

    def test_localization_reduces_std_error(self):
        """
        Truncated weight has smaller std_error than untruncated weight.

        Clipping the weight at its 99.9th quantile removes heavy-tail
        spikes while barely affecting the estimator mean.
        """
        payoff = EuropeanCall(K)
        f = payoff(_OUT["paths"], _OUT["times"])

        # Raw (untruncated) weight
        w_raw = localized_malliavin_weight(
            _OUT["paths"], _OUT["brownian_increments"], _model, S0, T, _OUT["times"],
            localization_radius=None,
        )

        # Auto-choose radius at 99.9th quantile
        R_auto = auto_localization_radius(w_raw, quantile=0.999)

        w_loc = localized_malliavin_weight(
            _OUT["paths"], _OUT["brownian_increments"], _model, S0, T, _OUT["times"],
            localization_radius=R_auto,
        )

        se_raw = float((_disc * f * w_raw).std(ddof=1) / np.sqrt(N_PATHS))
        se_loc = float((_disc * f * w_loc).std(ddof=1) / np.sqrt(N_PATHS))

        assert se_loc < se_raw, (
            f"Localized SE={se_loc:.6f} should be < raw SE={se_raw:.6f}"
        )

    def test_localization_small_bias(self):
        """
        Truncation bias is negligible at the 99.9th-quantile radius.

        The truncated estimator should agree with the analytical delta
        within 5 standard errors.
        """
        truth = bs_delta(S0, K, T, R, Q, SIGMA, "call")
        payoff = EuropeanCall(K)
        f = payoff(_OUT["paths"], _OUT["times"])

        w_raw = localized_malliavin_weight(
            _OUT["paths"], _OUT["brownian_increments"], _model, S0, T, _OUT["times"],
        )
        R_auto = auto_localization_radius(w_raw, quantile=0.999)
        w_loc = localized_malliavin_weight(
            _OUT["paths"], _OUT["brownian_increments"], _model, S0, T, _OUT["times"],
            localization_radius=R_auto,
        )

        est = float(_disc * (f * w_loc).mean())
        se = float((_disc * f * w_loc).std(ddof=1) / np.sqrt(N_PATHS))

        assert abs(est - truth) < 5.0 * se + 1e-4, (
            f"Localized delta={est:.5f} vs truth={truth:.5f}, se={se:.5f}"
        )

    def test_localization_returns_correct_shape(self):
        """localized_malliavin_weight returns array of shape (n_paths,)."""
        w = localized_malliavin_weight(
            _OUT["paths"], _OUT["brownian_increments"], _model, S0, T, _OUT["times"],
        )
        assert w.shape == (N_PATHS,)

    def test_localization_clips_correctly(self):
        """Truncated weight values are all within [−R, R]."""
        w_raw = localized_malliavin_weight(
            _OUT["paths"], _OUT["brownian_increments"], _model, S0, T, _OUT["times"],
        )
        R = float(np.quantile(np.abs(w_raw), 0.95))
        w_loc = localized_malliavin_weight(
            _OUT["paths"], _OUT["brownian_increments"], _model, S0, T, _OUT["times"],
            localization_radius=R,
        )
        assert float(np.abs(w_loc).max()) <= R + 1e-12

    def test_localization_no_truncation_equals_raw(self):
        """Without truncation, localized weight equals the standard delta weight."""
        from mgreeks.weights.malliavin_weights import delta_weight_gbm
        W_T = _OUT["brownian_increments"].sum(axis=1)
        w_std = delta_weight_gbm(S0, _OUT["terminal"], SIGMA, R, Q, T, W_T)
        w_loc = localized_malliavin_weight(
            _OUT["paths"], _OUT["brownian_increments"], _model, S0, T, _OUT["times"],
            localization_radius=None,
        )
        np.testing.assert_array_almost_equal(w_loc, w_std, decimal=12)

    # ---- interval localization ----

    def test_interval_localization_unbiased(self):
        """
        Interval-localized weight (half of the time steps) gives unbiased delta.

        Any subset A ⊂ [0,T] gives an unbiased IBP weight; only the
        variance changes.
        """
        truth = bs_delta(S0, K, T, R, Q, SIGMA, "call")
        payoff = EuropeanCall(K)
        f = payoff(_OUT["paths"], _OUT["times"])

        # Use only the first half of the steps
        n_steps = _OUT["brownian_increments"].shape[1]
        active = np.zeros(n_steps, dtype=bool)
        active[: n_steps // 2] = True

        w = interval_localized_weight(
            _OUT["brownian_increments"], _model, S0, T, active
        )
        est = float(_disc * (f * w).mean())
        se = float((_disc * f * w).std(ddof=1) / np.sqrt(N_PATHS))

        assert abs(est - truth) < 5.0 * se + 1e-4, (
            f"Interval-localized delta={est:.5f} vs truth={truth:.5f}, se={se:.5f}"
        )

    def test_interval_localization_single_step_unbiased(self):
        """
        Using only a single step (the first ΔW) is equivalent to
        delta_weight_path_dependent and gives an unbiased delta for
        European (terminal) payoffs.
        """
        truth = bs_delta(S0, K, T, R, Q, SIGMA, "call")
        payoff = EuropeanCall(K)
        f = payoff(_OUT["paths"], _OUT["times"])

        active = np.zeros(N_STEPS, dtype=bool)
        active[0] = True

        w = interval_localized_weight(
            _OUT["brownian_increments"], _model, S0, T, active
        )
        est = float(_disc * (f * w).mean())
        se = float((_disc * f * w).std(ddof=1) / np.sqrt(N_PATHS))

        assert abs(est - truth) < 5.0 * se + 1e-4, (
            f"Single-step delta={est:.5f} vs truth={truth:.5f}, se={se:.5f}"
        )

    def test_interval_localization_full_steps_matches_standard(self):
        """Using ALL steps matches the standard W_T weight (up to rounding)."""
        from mgreeks.weights.malliavin_weights import delta_weight_gbm
        W_T = _OUT["brownian_increments"].sum(axis=1)
        w_std = delta_weight_gbm(S0, _OUT["terminal"], SIGMA, R, Q, T, W_T)

        active = np.ones(N_STEPS, dtype=bool)
        w_interval = interval_localized_weight(
            _OUT["brownian_increments"], _model, S0, T, active
        )
        np.testing.assert_array_almost_equal(w_interval, w_std, decimal=12)

    # ---- auto_localization_radius ----

    def test_auto_radius_is_positive(self):
        """auto_localization_radius returns a positive finite float."""
        w = localized_malliavin_weight(
            _OUT["paths"], _OUT["brownian_increments"], _model, S0, T, _OUT["times"],
        )
        R = auto_localization_radius(w)
        assert np.isfinite(R) and R > 0

    def test_auto_radius_quantile_monotone(self):
        """Higher quantile → larger radius."""
        w = localized_malliavin_weight(
            _OUT["paths"], _OUT["brownian_increments"], _model, S0, T, _OUT["times"],
        )
        R99 = auto_localization_radius(w, quantile=0.99)
        R999 = auto_localization_radius(w, quantile=0.999)
        assert R999 >= R99


# ---------------------------------------------------------------------------
# 4. Combined: antithetic + control variate
# ---------------------------------------------------------------------------

class TestCombinedReduction:
    """Combining antithetic variates with control variate gives additive gains."""

    def test_antithetic_then_cv_price_unbiased(self):
        """
        Antithetic + geometric-Asian CV for arithmetic Asian price is unbiased.

        Combine antithetic paths to get an unbiased arithmetic-Asian price
        estimate, then apply the geometric-Asian CV.
        """
        sim_p, sim_n = simulate_antithetic(_model, S0, T, N_STEPS, N_PATHS,
                                           rng=np.random.default_rng(SEED))

        arith_payoff = ArithmeticAsianCall(K)
        geo_payoff = GeometricAsianCall(K)

        # Antithetic average of the payoff (unbiased, lower variance)
        f_arith_p = arith_payoff(sim_p["paths"], sim_p["times"])
        f_arith_n = arith_payoff(sim_n["paths"], sim_n["times"])
        f_geo_p = geo_payoff(sim_p["paths"], sim_p["times"])
        f_geo_n = geo_payoff(sim_n["paths"], sim_n["times"])

        f_arith_anti = 0.5 * (f_arith_p + f_arith_n)
        f_geo_anti = 0.5 * (f_geo_p + f_geo_n)

        analytic = analytical.geometric_asian_greeks(S0, K, T, R, Q, SIGMA, N_STEPS)
        geo_price = analytic["price"]

        res = geometric_asian_control_variate(
            f_arith_anti, f_geo_anti, geo_price, _disc
        )

        # Compare against plain arithmetic Asian price (from pre-computed sim)
        f_arith_plain = arith_payoff(_OUT["paths"], _OUT["times"])
        arith_price_ref = float(_disc * f_arith_plain.mean())
        plain_se = float((_disc * f_arith_plain).std(ddof=1) / np.sqrt(N_PATHS))

        combined_se = np.sqrt(res["std_error"]**2 + plain_se**2)
        assert abs(res["value"] - arith_price_ref) < 4.0 * combined_se + 1e-3

    def test_combined_outperforms_plain(self):
        """
        Combined antithetic+CV std_error < plain std_error for arithmetic Asian.
        """
        sim_p, sim_n = simulate_antithetic(_model, S0, T, N_STEPS, N_PATHS,
                                           rng=np.random.default_rng(SEED + 1))

        arith_payoff = ArithmeticAsianCall(K)
        geo_payoff = GeometricAsianCall(K)

        f_arith_anti = 0.5 * (arith_payoff(sim_p["paths"], sim_p["times"])
                               + arith_payoff(sim_n["paths"], sim_n["times"]))
        f_geo_anti = 0.5 * (geo_payoff(sim_p["paths"], sim_p["times"])
                             + geo_payoff(sim_n["paths"], sim_n["times"]))

        analytic = analytical.geometric_asian_greeks(S0, K, T, R, Q, SIGMA, N_STEPS)
        res = geometric_asian_control_variate(
            f_arith_anti, f_geo_anti, analytic["price"], _disc
        )

        # Plain arithmetic Asian
        f_arith_plain = arith_payoff(_OUT["paths"], _OUT["times"])
        plain_se = float((_disc * f_arith_plain).std(ddof=1) / np.sqrt(N_PATHS))

        assert res["std_error"] < plain_se * 0.5, (
            f"Combined SE={res['std_error']:.6f} should be < 50% of plain SE={plain_se:.6f}"
        )
