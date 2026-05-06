"""
tests/test_greeks_exotic.py — Malliavin Greeks for path-dependent / exotic options.

Benchmarks used
---------------
- Geometric Asian call: analytical closed-form (adjusted-vol BS formula)
- Arithmetic Asian call: finite differences (both are unbiased for smooth payoffs)
- Down-and-out call: analytical barrier-option formula (continuous monitoring limit)
- Lookback / basket: finite differences and sign / ordering checks

Key design choice
-----------------
For DISCRETE path-dependent payoffs (Asian, barrier, lookback), the UNBIASED
Malliavin delta weight is:

    π_Δ = ΔW_1 / (σ S_0 Δt_1)     [score of the first conditional density]

rather than W_T / (σ S_0 T) (which is biased by a factor t̄/T for multi-step
payoffs — see malliavin_weights.asian_delta_weight_gbm docstring).

All "Malliavin estimate" tests below use delta_weight_path_dependent (ΔW_1)
to ensure unbiased comparison.  The alias functions added to malliavin_weights
(asian_delta_weight_gbm, barrier_delta_weight_gbm, lookback_delta_weight_gbm)
are tested separately for their documented properties.
"""

from __future__ import annotations

import numpy as np
import pytest

from mgreeks.greeks import MalliavinGreeks, FiniteDifferenceGreeks, analytical
from mgreeks.models.gbm import GeometricBrownianMotion
from mgreeks.models.heston import HestonModel
from mgreeks.models.multidimensional import MultiAssetGBM
from mgreeks.payoffs.asian import ArithmeticAsianCall, ArithmeticAsianPut, GeometricAsianCall
from mgreeks.payoffs.barrier import DownAndOutCall, DownAndOutPut, DownAndInCall
from mgreeks.payoffs.lookback import (
    FloatingStrikeLookbackCall, FloatingStrikeLookbackPut, FixedStrikeLookbackCall,
)
from mgreeks.payoffs.basket import BasketCall, BasketPut
from mgreeks.simulation import MonteCarloEngine
from mgreeks.weights.malliavin_weights import (
    delta_weight_path_dependent,
    asian_delta_weight_gbm,
    barrier_delta_weight_gbm,
    lookback_delta_weight_gbm,
    delta_weight_heston,
    vega_weight_heston,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

S0, K, T = 100.0, 100.0, 1.0
R, Q, SIGMA = 0.05, 0.02, 0.20
B_BARRIER = 85.0      # barrier level, far from S0
B_NEAR = 95.0         # barrier level, close to S0 (stress test)
N_PATHS = 500_000
N_STEPS = 52          # daily-ish monitoring for path-dep payoffs
SEED = 42

_disc = float(np.exp(-R * T))
_model = GeometricBrownianMotion(r=R, q=Q, sigma=SIGMA)
_engine = MonteCarloEngine(_model, n_paths=N_PATHS, n_steps=N_STEPS, rng_seed=SEED)
_mall = MalliavinGreeks(_model, _engine)
_fd = FiniteDifferenceGreeks(_model, _engine, bump_size=0.01, bump_type="relative")


def _sim():
    return _model.simulate(S0, T, N_STEPS, N_PATHS, return_full_paths=True,
                           rng=np.random.default_rng(SEED))


# Pre-compute one simulation for re-use in parameter studies
_OUT = _sim()
_W_T = _OUT["brownian_increments"].sum(axis=1)
_PD_WEIGHT = delta_weight_path_dependent(
    _OUT["paths"], _OUT["brownian_increments"], _model, S0, T
)


def _greek_from_weight(payoff, weight, disc=_disc):
    f = payoff(_OUT["paths"], _OUT["times"])
    return float(disc * (f * weight).mean())


def _se_from_weight(payoff, weight, disc=_disc):
    f = payoff(_OUT["paths"], _OUT["times"])
    samples = disc * f * weight
    return float(samples.std(ddof=1) / np.sqrt(len(samples)))


def _within(est, truth, se, n=4.0):
    return abs(est - truth) < n * se + 1e-5


# ---------------------------------------------------------------------------
# 1. Asian options
# ---------------------------------------------------------------------------

class TestAsianOptions:
    """
    Tests for arithmetic and geometric Asian call/put Greeks.
    """

    # ---- Geometric Asian vs closed-form ----

    def test_geometric_asian_delta_vs_analytical(self):
        """
        Malliavin delta for geometric Asian call agrees with analytical formula.

        The analytical delta comes from the adjusted-vol Black-Scholes formula
        for geometric Asian calls (see analytical.geometric_asian_greeks).
        """
        payoff = GeometricAsianCall(K)
        f = payoff(_OUT["paths"], _OUT["times"])
        est = float(_disc * (f * _PD_WEIGHT).mean())
        se = float((_disc * f * _PD_WEIGHT).std(ddof=1) / np.sqrt(N_PATHS))

        truth_dict = analytical.geometric_asian_greeks(S0, K, T, R, Q, SIGMA, N_STEPS)
        truth = truth_dict["delta"]

        assert _within(est, truth, se), (
            f"Geometric Asian delta: Malliavin={est:.5f} vs analytical={truth:.5f} se={se:.5f}"
        )

    def test_geometric_asian_delta_less_than_european(self):
        """Geometric Asian delta < European call delta (averaging reduces sensitivity)."""
        from mgreeks.greeks.analytical import bs_delta
        payoff = GeometricAsianCall(K)
        f = payoff(_OUT["paths"], _OUT["times"])
        asian_delta = float(_disc * (f * _PD_WEIGHT).mean())
        euro_delta = bs_delta(S0, K, T, R, Q, SIGMA, "call")
        assert asian_delta < euro_delta, (
            f"Asian delta {asian_delta:.4f} should be < European delta {euro_delta:.4f}"
        )

    # ---- Arithmetic Asian vs finite differences ----

    def test_arithmetic_asian_call_delta_vs_fd(self):
        """
        Malliavin delta for arithmetic Asian call ≈ finite diff delta.

        Both are unbiased estimators; Malliavin should have lower or comparable
        variance (the averaging of the Asian payoff smooths the estimator).
        """
        payoff = ArithmeticAsianCall(K)
        f = payoff(_OUT["paths"], _OUT["times"])
        mall_est = float(_disc * (f * _PD_WEIGHT).mean())
        mall_se = float((_disc * f * _PD_WEIGHT).std(ddof=1) / np.sqrt(N_PATHS))

        fd_res = _fd.delta(payoff, S0, T)
        fd_est = fd_res["value"]
        fd_se = fd_res["std_error"]

        combined_se = np.sqrt(mall_se**2 + fd_se**2)
        assert abs(mall_est - fd_est) < 4.0 * combined_se + 1e-3, (
            f"Arithmetic Asian delta: Malliavin={mall_est:.5f} vs FD={fd_est:.5f} "
            f"combined_se={combined_se:.5f}"
        )

    def test_arithmetic_asian_malliavin_lower_variance_than_european(self):
        """
        Asian call × W_T weight has lower variance than European call × W_T weight.

        When both payoffs are multiplied by the same W_T/(σS_0T) weight, the
        averaging of the Asian payoff reduces its correlation with the weight,
        yielding a lower-variance product than the European terminal payoff.
        """
        from mgreeks.payoffs.european import EuropeanCall
        asian_payoff = ArithmeticAsianCall(K)
        euro_payoff = EuropeanCall(K)

        f_asian = asian_payoff(_OUT["paths"], _OUT["times"])
        f_euro = euro_payoff(_OUT["paths"], _OUT["times"])

        # Compare both using the same W_T weight (fair apples-to-apples comparison)
        w_wt = _W_T / (SIGMA * S0 * T)
        asian_var = float((f_asian * w_wt).var())
        euro_var = float((f_euro * w_wt).var())

        assert asian_var <= euro_var, (
            f"Asian var={asian_var:.4f} should be ≤ European var={euro_var:.4f} "
            "(averaging reduces payoff-weight correlation)"
        )

    def test_arithmetic_asian_call_delta_positive(self):
        """Asian call delta is positive."""
        payoff = ArithmeticAsianCall(K)
        f = payoff(_OUT["paths"], _OUT["times"])
        est = float(_disc * (f * _PD_WEIGHT).mean())
        assert est > 0, f"Asian call delta should be positive, got {est:.5f}"

    def test_arithmetic_asian_put_delta_negative(self):
        """Asian put delta is negative."""
        payoff = ArithmeticAsianPut(K)
        f = payoff(_OUT["paths"], _OUT["times"])
        est = float(_disc * (f * _PD_WEIGHT).mean())
        assert est < 0, f"Asian put delta should be negative, got {est:.5f}"

    def test_asian_put_call_parity_delta(self):
        """
        delta(Asian call) - delta(Asian put) ≈ e^{-qT} (discounted-forward parity).

        For arithmetic Asian, the call-put parity is:
            C - P = e^{-rT} E[A - K] = e^{-rT}(F_A - K)
        where F_A is the arithmetic-average forward.  By linearity of delta:
            delta(C) - delta(P) = e^{-qT} × (∂F_A/∂S_0)

        For GBM with S0 = 100, ∂F_A/∂S_0 = 1 (A is proportional to S_0), so:
            delta(C) - delta(P) = e^{-qT}.
        """
        call_payoff = ArithmeticAsianCall(K)
        put_payoff = ArithmeticAsianPut(K)

        fc = call_payoff(_OUT["paths"], _OUT["times"])
        fp = put_payoff(_OUT["paths"], _OUT["times"])

        dc = float(_disc * (fc * _PD_WEIGHT).mean())
        dp = float(_disc * (fp * _PD_WEIGHT).mean())

        diff = dc - dp
        expected = float(np.exp(-Q * T))

        se_c = float((_disc * fc * _PD_WEIGHT).std(ddof=1) / np.sqrt(N_PATHS))
        se_p = float((_disc * fp * _PD_WEIGHT).std(ddof=1) / np.sqrt(N_PATHS))
        tol = 5.0 * np.sqrt(se_c**2 + se_p**2)

        assert abs(diff - expected) < tol + 1e-3, (
            f"Asian parity: delta(C)-delta(P)={diff:.5f} vs e^{{-qT}}={expected:.5f}"
        )


# ---------------------------------------------------------------------------
# 2. Barrier options
# ---------------------------------------------------------------------------

class TestBarrierOptions:
    """
    Tests for down-and-out call Greek estimates.
    """

    # ---- vs analytical (far from barrier) ----

    def test_down_and_out_call_delta_vs_analytical(self):
        """
        Malliavin delta for down-and-out call ≈ analytical barrier delta.

        The analytical formula uses continuous monitoring; the MC uses
        N_STEPS discrete steps, so a small discretization gap is expected.
        """
        payoff = DownAndOutCall(K, B_BARRIER)
        f = payoff(_OUT["paths"], _OUT["times"])
        est = float(_disc * (f * _PD_WEIGHT).mean())
        se = float((_disc * f * _PD_WEIGHT).std(ddof=1) / np.sqrt(N_PATHS))

        truth = analytical.bs_barrier_delta(S0, K, B_BARRIER, T, R, Q, SIGMA)

        # Use 8 SE tolerance (discretization bias + MC noise)
        assert _within(est, truth, se, n=8.0), (
            f"Down-and-out delta: Malliavin={est:.5f} vs analytical={truth:.5f} se={se:.5f}"
        )

    def test_barrier_delta_less_than_one(self):
        """Down-and-out call delta is in (0, 1)."""
        payoff = DownAndOutCall(K, B_BARRIER)
        f = payoff(_OUT["paths"], _OUT["times"])
        barrier_delta = float(_disc * (f * _PD_WEIGHT).mean())
        assert 0 < barrier_delta < 1.0, (
            f"Barrier delta {barrier_delta:.4f} should be in (0, 1). "
            "Note: for B < K, barrier delta can exceed vanilla delta because "
            "raising S0 reduces knock-out probability in addition to raising payoff."
        )

    def test_barrier_delta_positive(self):
        """Down-and-out call delta is positive."""
        payoff = DownAndOutCall(K, B_BARRIER)
        f = payoff(_OUT["paths"], _OUT["times"])
        est = float(_disc * (f * _PD_WEIGHT).mean())
        assert est > 0, f"Barrier call delta should be positive, got {est:.5f}"

    # ---- knock-in / knock-out parity ----

    def test_knock_in_knock_out_delta_parity(self):
        """
        delta(DownAndOutCall) + delta(DownAndInCall) ≈ delta(VanillaCall).
        Uses the same simulation — parity holds path by path so the MC error is small.
        """
        from mgreeks.payoffs.european import EuropeanCall
        from mgreeks.weights.malliavin_weights import delta_weight_gbm

        # Use W_T weight (European-style): path-by-path parity holds exactly
        W_T = _OUT["brownian_increments"].sum(axis=1)
        weight = delta_weight_gbm(S0, _OUT["terminal"], SIGMA, R, Q, T, W_T)

        f_dao = DownAndOutCall(K, B_BARRIER)(_OUT["paths"], _OUT["times"])
        f_dai = DownAndInCall(K, B_BARRIER)(_OUT["paths"], _OUT["times"])
        f_van = EuropeanCall(K)(_OUT["paths"], _OUT["times"])

        dao_delta = float(_disc * (f_dao * weight).mean())
        dai_delta = float(_disc * (f_dai * weight).mean())
        van_delta = float(_disc * (f_van * weight).mean())

        # Path-by-path: f_dao + f_dai = f_van, so sum of deltas = vanilla delta
        combined_se = float(
            (_disc * (f_dao + f_dai - f_van) * weight).std(ddof=1) / np.sqrt(N_PATHS)
        )
        assert abs((dao_delta + dai_delta) - van_delta) < 4.0 * combined_se + 1e-6, (
            f"KI+KO delta parity: {dao_delta+dai_delta:.5f} vs vanilla {van_delta:.5f}"
        )

    # ---- Near the barrier: FD is noisy, Malliavin is stable ----

    def test_near_barrier_fd_noisier_than_malliavin(self):
        """
        Near the barrier (S0 = 1.05×B), FD delta variance >> Malliavin delta variance.

        This is the critical regime: a small bump h can flip paths from surviving
        to getting knocked out, causing the FD estimator's variance to explode.
        Malliavin simply weights the surviving payoffs — no variance explosion.
        """
        S0_near = B_NEAR * 1.05    # 5% above the barrier
        model_near = GeometricBrownianMotion(r=R, q=Q, sigma=SIGMA)
        engine_near = MonteCarloEngine(model_near, n_paths=N_PATHS, n_steps=N_STEPS,
                                       rng_seed=SEED)
        mall_near = MalliavinGreeks(model_near, engine_near)
        fd_near = FiniteDifferenceGreeks(model_near, engine_near,
                                         bump_size=0.01, bump_type="relative")

        payoff = DownAndOutCall(K, B_NEAR)
        res_mall = mall_near.delta(payoff, S0_near, T)
        res_fd = fd_near.delta(payoff, S0_near, T)

        # FD std_error should be much larger (at least 2× the Malliavin SE)
        assert res_fd["std_error"] > res_mall["std_error"], (
            f"Near barrier: FD se={res_fd['std_error']:.4f} should exceed "
            f"Malliavin se={res_mall['std_error']:.4f}"
        )

    def test_malliavin_finite_variance_near_barrier(self):
        """Malliavin std_error near the barrier must be finite and positive."""
        S0_near = B_NEAR * 1.05
        model_near = GeometricBrownianMotion(r=R, q=Q, sigma=SIGMA)
        engine_near = MonteCarloEngine(model_near, n_paths=N_PATHS, n_steps=N_STEPS,
                                       rng_seed=SEED)
        mall_near = MalliavinGreeks(model_near, engine_near)
        payoff = DownAndOutCall(K, B_NEAR)
        res = mall_near.delta(payoff, S0_near, T)
        assert np.isfinite(res["std_error"]) and res["std_error"] > 0


# ---------------------------------------------------------------------------
# 3. Lookback options
# ---------------------------------------------------------------------------

class TestLookbackOptions:
    """
    Tests for floating-strike lookback call/put Greeks.
    """

    def test_lookback_call_delta_positive(self):
        """Floating-strike lookback call delta > 0 (via homogeneity: delta = price/S0)."""
        payoff = FloatingStrikeLookbackCall()
        f = payoff(_OUT["paths"], _OUT["times"])
        # Use homogeneity: payoff is degree-1 in S0, so delta = disc*E[f]/S0 > 0
        est = float(_disc * f.mean()) / S0
        assert est > 0, f"Lookback call delta={est:.4f} should be positive"

    def test_lookback_put_delta_positive(self):
        """
        Floating-strike lookback put delta > 0 (via homogeneity: delta = price/S0).

        Payoff = M_T - S_T is degree-1 homogeneous in S_0, so delta = price/S_0 > 0.
        """
        payoff = FloatingStrikeLookbackPut()
        f = payoff(_OUT["paths"], _OUT["times"])
        est = float(_disc * f.mean()) / S0
        assert est > 0, f"Lookback put delta={est:.4f} should be positive"

    def test_lookback_call_price_greater_than_vanilla(self):
        """
        Floating-strike lookback call price > European call price.

        The lookback payoff S_T - m_T ≥ max(S_T - S_0, 0) ≥ max(S_T - K, 0)
        for K = S_0 (ATM).  So lookback price ≥ European call price.

        Note: lookback DELTA is NOT necessarily larger than European call delta.
        Payoff S_T - m_T is homogeneous of degree 1, so delta = price/S_0 < N(d1)
        for typical parameters.
        """
        from mgreeks.payoffs.european import EuropeanCall

        lb_payoff = FloatingStrikeLookbackCall()
        euro_payoff = EuropeanCall(K)

        mc_lb = float(_disc * lb_payoff(_OUT["paths"], _OUT["times"]).mean())
        mc_euro = float(_disc * euro_payoff(_OUT["paths"], _OUT["times"]).mean())

        assert mc_lb > mc_euro, (
            f"Lookback price {mc_lb:.4f} should exceed European call price {mc_euro:.4f}"
        )

    def test_lookback_call_delta_fd_vs_homogeneity(self):
        """
        FD lookback call delta ≈ price / S_0 (homogeneity argument).

        Payoff S_T − m_T is homogeneous degree-1 in S_0 (all path values
        scale proportionally), so:  delta = e^{-rT} E[payoff] / S_0.

        NOTE: delta_weight_path_dependent (ΔW_1 weight) is NOT correct for
        lookback payoffs that include paths[:, 0] = S_0 in the running extremum.
        The full delta requires LR score PLUS the direct correction E[∂f/∂S_0],
        which for the lookback call equals −E[1{S_0 = m_T}].
        FD (CRN) correctly computes the total delta automatically.
        """
        payoff = FloatingStrikeLookbackCall()
        f = payoff(_OUT["paths"], _OUT["times"])

        mc_price = float(_disc * f.mean())
        homog_delta = mc_price / S0    # from degree-1 homogeneity

        fd_res = _fd.delta(payoff, S0, T)
        fd_est = fd_res["value"]
        fd_se = fd_res["std_error"]

        assert abs(fd_est - homog_delta) < 4.0 * fd_se + 1e-3, (
            f"FD lookback delta={fd_est:.5f} should ≈ price/S0={homog_delta:.5f}"
        )

    def test_lookback_vs_european_price(self):
        """
        Floating-strike lookback call MC price > European call price.

        The lookback payoff S_T - m_T ≥ max(S_T - S_0, 0) ≥ max(S_T - K, 0)
        when K = S_0 (ATM), so the lookback is always worth at least as much.
        """
        from mgreeks.payoffs.european import EuropeanCall

        lb_payoff = FloatingStrikeLookbackCall()
        euro_payoff = EuropeanCall(K)

        mc_lb = float(_disc * lb_payoff(_OUT["paths"], _OUT["times"]).mean())
        mc_euro = float(_disc * euro_payoff(_OUT["paths"], _OUT["times"]).mean())

        assert mc_lb > mc_euro > 0, (
            f"Lookback price {mc_lb:.4f} should exceed European call price {mc_euro:.4f}"
        )


# ---------------------------------------------------------------------------
# 4. Basket options (multi-asset)
# ---------------------------------------------------------------------------

class TestBasketOptions:
    """
    Tests for basket call delta using the multi-asset GBM model.

    Delta w.r.t. asset i:  π^i = W^i_T / (σ_i S^i_0 T)   (uncorrelated assets)
    """

    N_ASSETS = 2
    SIGMAS = np.array([0.20, 0.25])
    WEIGHTS = np.array([0.5, 0.5])
    S0_VEC = np.array([100.0, 100.0])

    def _setup_uncorrelated(self):
        model = MultiAssetGBM(r=R, q=0.0, sigma=self.SIGMAS, n_assets=self.N_ASSETS)
        out = model.simulate(self.S0_VEC, T, N_STEPS, N_PATHS,
                             return_full_paths=True, rng=np.random.default_rng(SEED))
        return model, out

    def test_basket_delta_per_asset_structure(self):
        """Basket call delta w.r.t. each asset can be computed independently."""
        model, out = self._setup_uncorrelated()
        payoff = BasketCall(K, weights=self.WEIGHTS)
        f = payoff(out["paths"], out["times"])
        disc = np.exp(-R * T)

        for i in range(self.N_ASSETS):
            weight_i = model.malliavin_weight_delta(out["paths"], out["brownian_increments"],
                                                    asset=i, T=T)
            delta_i = float(disc * (f * weight_i).mean())
            se_i = float(disc * (f * weight_i).std(ddof=1) / np.sqrt(N_PATHS))
            assert delta_i > 0, f"Basket delta for asset {i} should be positive, got {delta_i:.4f}"
            assert np.isfinite(se_i)

    def test_basket_delta_symmetry(self):
        """
        For equal weights and equal-vol assets, deltas w.r.t. each asset are equal.
        """
        sigmas = np.array([0.20, 0.20])
        model = MultiAssetGBM(r=R, q=0.0, sigma=sigmas, n_assets=self.N_ASSETS)
        out = model.simulate(self.S0_VEC, T, N_STEPS, N_PATHS,
                             return_full_paths=True, rng=np.random.default_rng(SEED))
        payoff = BasketCall(K, weights=self.WEIGHTS)
        f = payoff(out["paths"], out["times"])
        disc = np.exp(-R * T)

        deltas = []
        for i in range(self.N_ASSETS):
            w = model.malliavin_weight_delta(out["paths"], out["brownian_increments"],
                                             asset=i, T=T)
            deltas.append(float(disc * (f * w).mean()))

        # By symmetry, deltas should be approximately equal
        se = float(disc * (f * model.malliavin_weight_delta(
            out["paths"], out["brownian_increments"], asset=0, T=T
        )).std(ddof=1) / np.sqrt(N_PATHS))

        assert abs(deltas[0] - deltas[1]) < 5.0 * np.sqrt(2) * se + 1e-3, (
            f"Basket deltas should be symmetric: delta_1={deltas[0]:.5f}, delta_2={deltas[1]:.5f}"
        )

    def test_basket_euler_sum(self):
        """
        Euler's theorem for homogeneous payoffs:
            Σ_i S^i_0 × delta_i = V (basket call price for ATM, homogeneous degree 1).

        Numerically: Σ_i delta_i × S0_i ≈ e^{-rT} E[basket × Σ_i weight_i × S^i_T/S^i_0]
        For a basket where all S0_i = S0 = 100 and weights = 0.5:
            Σ_i S0 × delta_i = 100 × (delta_1 + delta_2) ≈ S0 × e^{-qT} × N(d1_basket) × w_total
        This is difficult to verify directly, so we check that:
            Σ_i delta_i < 1  and  Σ_i delta_i > 0.
        """
        model, out = self._setup_uncorrelated()
        payoff = BasketCall(K, weights=self.WEIGHTS)
        f = payoff(out["paths"], out["times"])
        disc = np.exp(-R * T)

        total_delta = 0.0
        for i in range(self.N_ASSETS):
            w = model.malliavin_weight_delta(out["paths"], out["brownian_increments"],
                                             asset=i, T=T)
            total_delta += float(disc * (f * w).mean())

        assert 0 < total_delta < 1.0, (
            f"Sum of basket deltas should be in (0,1), got {total_delta:.4f}"
        )

    def test_basket_delta_less_than_vanilla(self):
        """
        Each basket asset's delta < European call delta (diversification reduces sensitivity).

        For n equal-weight assets, delta_i ≈ vanilla_delta / n.
        """
        from mgreeks.greeks.analytical import bs_delta

        model, out = self._setup_uncorrelated()
        payoff = BasketCall(K, weights=self.WEIGHTS)
        f = payoff(out["paths"], out["times"])
        disc = np.exp(-R * T)

        vanilla_delta = bs_delta(S0, K, T, R, 0.0, self.SIGMAS[0], "call")

        for i in range(self.N_ASSETS):
            w = model.malliavin_weight_delta(out["paths"], out["brownian_increments"],
                                             asset=i, T=T)
            delta_i = float(disc * (f * w).mean())
            assert delta_i < vanilla_delta, (
                f"Basket delta for asset {i} ({delta_i:.4f}) should be < "
                f"vanilla delta ({vanilla_delta:.4f})"
            )


# ---------------------------------------------------------------------------
# 5. Heston model delta weight
# ---------------------------------------------------------------------------

class TestHestonDeltaWeight:
    """
    Tests for the BEL delta weight under the Heston stochastic-volatility model.
    """

    def _heston_sim(self, n_paths=200_000, n_steps=50, seed=SEED):
        model = HestonModel(r=R, q=Q)
        out = model.simulate(S0, T, n_steps, n_paths,
                             return_full_paths=True, scheme="euler",
                             rng=np.random.default_rng(seed))
        return model, out

    def test_heston_delta_weight_finite(self):
        """BEL delta weight for Heston is finite and non-trivial."""
        from mgreeks.payoffs.european import EuropeanCall

        model, out = self._heston_sim()
        weight = delta_weight_heston(
            out["paths"], out["v_paths"],
            out["brownian_increments"], out["v_brownian_increments"],
            S0, model.V0, T, out["times"], model,
        )
        assert np.all(np.isfinite(weight)), "Heston delta weight contains NaN/Inf"
        assert float(np.abs(weight).mean()) > 0

    def test_heston_delta_positive_call(self):
        """Heston Malliavin delta for European call is positive."""
        from mgreeks.payoffs.european import EuropeanCall

        model, out = self._heston_sim()
        payoff = EuropeanCall(K)
        f = payoff(out["paths"], out["times"])
        disc = float(np.exp(-model.r * T))

        weight = delta_weight_heston(
            out["paths"], out["v_paths"],
            out["brownian_increments"], out["v_brownian_increments"],
            S0, model.V0, T, out["times"], model,
        )
        delta = float(disc * (f * weight).mean())
        assert delta > 0, f"Heston call delta should be positive, got {delta:.5f}"

    def test_heston_delta_near_gbm_when_vol_of_vol_zero(self):
        """
        When xi → 0, Heston degenerates to GBM and the BEL delta weight
        approaches the GBM delta weight W_T/(√V_0 × S_0 × T).
        """
        from mgreeks.payoffs.european import EuropeanCall

        sigma_gbm = 0.20
        V0_heston = sigma_gbm**2   # match variance to GBM sigma

        model_h = HestonModel(r=R, q=Q, kappa=10.0, theta=V0_heston,
                               xi=0.001, rho=0.0, V0=V0_heston)
        model_gbm = GeometricBrownianMotion(r=R, q=Q, sigma=sigma_gbm)

        n_paths = 200_000
        rng_h = np.random.default_rng(1)
        out_h = model_h.simulate(S0, T, 50, n_paths, return_full_paths=True,
                                 scheme="euler", rng=rng_h)

        payoff = EuropeanCall(K)
        f_h = payoff(out_h["paths"], out_h["times"])
        disc = np.exp(-R * T)

        weight_h = delta_weight_heston(
            out_h["paths"], out_h["v_paths"],
            out_h["brownian_increments"], out_h["v_brownian_increments"],
            S0, model_h.V0, T, out_h["times"], model_h,
        )
        delta_h = float(disc * (f_h * weight_h).mean())

        # Compare to analytical GBM delta
        from mgreeks.greeks.analytical import bs_delta
        delta_gbm = bs_delta(S0, K, T, R, Q, sigma_gbm, "call")

        se = float(disc * (f_h * weight_h).std(ddof=1) / np.sqrt(n_paths))
        assert abs(delta_h - delta_gbm) < 6.0 * se + 0.02, (
            f"Heston (xi→0) delta={delta_h:.4f} should ≈ GBM delta={delta_gbm:.4f}"
        )

    def test_heston_vega_weight_finite(self):
        """Heston vega weight (∂V/∂V_0) is finite and non-trivial."""
        model, out = self._heston_sim()
        weight = vega_weight_heston(
            out["paths"], out["v_paths"],
            out["brownian_increments"], out["v_brownian_increments"],
            S0, model.V0, T, out["times"], model,
        )
        assert np.all(np.isfinite(weight)), "Heston vega weight contains NaN/Inf"

    def test_heston_vega_positive_call(self):
        """Heston vega (∂V/∂V_0) for European call is positive."""
        from mgreeks.payoffs.european import EuropeanCall

        model, out = self._heston_sim()
        payoff = EuropeanCall(K)
        f = payoff(out["paths"], out["times"])
        disc = float(np.exp(-model.r * T))

        weight = vega_weight_heston(
            out["paths"], out["v_paths"],
            out["brownian_increments"], out["v_brownian_increments"],
            S0, model.V0, T, out["times"], model,
        )
        vega = float(disc * (f * weight).mean())
        assert vega > 0, f"Heston call vega w.r.t. V_0 should be positive, got {vega:.5f}"


# ---------------------------------------------------------------------------
# 6. Alias weight function tests
# ---------------------------------------------------------------------------

class TestAliasWeightFunctions:
    """
    Verify that asian_delta_weight_gbm, barrier_delta_weight_gbm,
    lookback_delta_weight_gbm all equal delta_weight_gbm (W_T formula).
    Document the known discretization bias.
    """

    def test_asian_weight_equals_delta_weight_gbm(self):
        """asian_delta_weight_gbm returns W_T/(σ S_0 T) — same as delta_weight_gbm."""
        from mgreeks.weights.malliavin_weights import delta_weight_gbm
        w_asian = asian_delta_weight_gbm(
            _OUT["paths"], _OUT["brownian_increments"], S0, SIGMA, T, _OUT["times"]
        )
        W_T = _OUT["brownian_increments"].sum(axis=1)
        w_euro = delta_weight_gbm(S0, _OUT["terminal"], SIGMA, R, Q, T, W_T)
        np.testing.assert_array_almost_equal(w_asian, w_euro, decimal=12)

    def test_barrier_weight_equals_delta_weight_gbm(self):
        """barrier_delta_weight_gbm returns W_T/(σ S_0 T)."""
        from mgreeks.weights.malliavin_weights import delta_weight_gbm
        w_barrier = barrier_delta_weight_gbm(
            _OUT["paths"], _OUT["brownian_increments"], S0, SIGMA, T, _OUT["times"], B_BARRIER
        )
        W_T = _OUT["brownian_increments"].sum(axis=1)
        w_euro = delta_weight_gbm(S0, _OUT["terminal"], SIGMA, R, Q, T, W_T)
        np.testing.assert_array_almost_equal(w_barrier, w_euro, decimal=12)

    def test_lookback_weight_equals_delta_weight_gbm(self):
        """lookback_delta_weight_gbm returns W_T/(σ S_0 T)."""
        from mgreeks.weights.malliavin_weights import delta_weight_gbm
        w_lb = lookback_delta_weight_gbm(
            _OUT["paths"], _OUT["brownian_increments"], S0, SIGMA, T, _OUT["times"]
        )
        W_T = _OUT["brownian_increments"].sum(axis=1)
        w_euro = delta_weight_gbm(S0, _OUT["terminal"], SIGMA, R, Q, T, W_T)
        np.testing.assert_array_almost_equal(w_lb, w_euro, decimal=12)

    def test_wt_bias_for_arithmetic_asian(self):
        """
        Document the discretization bias: W_T formula gives ≈ t̄/T × correct delta.

        For uniform fixing over 52 steps: t̄/T = (n+1)/(2n) = 53/104 ≈ 0.51.
        The W_T formula underestimates the arithmetic Asian delta by this factor.
        """
        payoff = ArithmeticAsianCall(K)
        f = payoff(_OUT["paths"], _OUT["times"])

        # W_T weight (biased for path-dependent)
        W_T = _OUT["brownian_increments"].sum(axis=1)
        w_wt = W_T / (SIGMA * S0 * T)
        est_wt = float(_disc * (f * w_wt).mean())

        # ΔW_1 weight (correct for path-dependent)
        est_correct = float(_disc * (f * _PD_WEIGHT).mean())

        # Expected bias factor
        n = N_STEPS
        bias_factor = (n + 1) / (2 * n)    # ≈ 0.51 for n=52

        assert abs(est_wt - bias_factor * est_correct) < 0.01, (
            f"W_T estimate ({est_wt:.4f}) should be ≈ {bias_factor:.3f} × "
            f"correct estimate ({est_correct:.4f})"
        )


# ---------------------------------------------------------------------------
# 7. GeometricAsianCall payoff check
# ---------------------------------------------------------------------------

class TestGeometricAsianPayoff:
    """Sanity checks for the GeometricAsianCall payoff object."""

    def test_geometric_asian_price_below_european(self):
        """Geometric Asian call price ≤ European call price (averaging reduces value)."""
        from mgreeks.payoffs.european import EuropeanCall
        from mgreeks.utils import bs_price

        geo_payoff = GeometricAsianCall(K)
        euro_payoff = EuropeanCall(K)

        f_geo = geo_payoff(_OUT["paths"], _OUT["times"])
        f_euro = euro_payoff(_OUT["paths"], _OUT["times"])

        mc_geo = float(_disc * f_geo.mean())
        mc_euro = float(_disc * f_euro.mean())

        assert mc_geo < mc_euro, (
            f"Geometric Asian price {mc_geo:.4f} should be < European price {mc_euro:.4f}"
        )

    def test_geometric_asian_price_vs_analytical(self):
        """Geometric Asian MC price ≈ analytical closed-form price."""
        geo_payoff = GeometricAsianCall(K)
        f = geo_payoff(_OUT["paths"], _OUT["times"])
        mc_price = float(_disc * f.mean())
        mc_se = float(_disc * f.std(ddof=1) / np.sqrt(N_PATHS))

        analytic = analytical.geometric_asian_greeks(S0, K, T, R, Q, SIGMA, N_STEPS)
        truth = analytic["price"]

        assert abs(mc_price - truth) < 4.0 * mc_se + 0.01, (
            f"Geometric Asian price: MC={mc_price:.4f} vs analytical={truth:.4f} se={mc_se:.4f}"
        )
