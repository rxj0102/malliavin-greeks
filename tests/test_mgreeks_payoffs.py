"""
Tests for mgreeks payoff implementations.

Checks:
  European  : shapes, non-negativity, put-call parity
  Digital   : partition of unity, shapes
  Asian     : arithmetic/geometric average, payoff_deriv
  Barrier   : knock-in/knock-out parity, indicator correctness
  Lookback  : running max/min, non-negativity
  Basket    : weighted average, spread, rainbow options
"""

import numpy as np
import pytest

from mgreeks.payoffs.european import (
    EuropeanCall, EuropeanPut, DigitalCall, DigitalPut,
    AssetOrNothingCall, put_call_parity_check,
)
from mgreeks.payoffs.asian import (
    ArithmeticAsianCall, ArithmeticAsianPut, GeometricAsianCall,
)
from mgreeks.payoffs.barrier import (
    DownAndOutCall, DownAndInCall, DownAndOutPut, DownAndInPut,
    UpAndOutCall, UpAndInCall, barrier_parity_check,
)
from mgreeks.payoffs.lookback import (
    FixedStrikeLookbackCall, FixedStrikeLookbackPut,
    FloatingStrikeLookbackCall, FloatingStrikeLookbackPut,
    PartialLookbackCall,
)
from mgreeks.payoffs.basket import (
    BasketCall, BasketPut, SpreadCall, SpreadPut,
    BestOfCall, WorstOfCall,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def make_paths_1d(S0=100.0, n_paths=2000, n_steps=50, seed=0):
    """Simple GBM-like paths for 1-asset payoffs."""
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n_paths, n_steps))
    log_inc = 0.05 / n_steps + 0.20 * np.sqrt(1.0 / n_steps) * Z
    log_S = np.empty((n_paths, n_steps + 1))
    log_S[:, 0] = np.log(S0)
    np.cumsum(log_inc, axis=1, out=log_S[:, 1:])
    log_S[:, 1:] += np.log(S0)
    return np.exp(log_S)


def make_times_1d(T=1.0, n_steps=50):
    return np.linspace(0, T, n_steps + 1)


def make_paths_2d(S0=None, n_paths=2000, n_steps=50, seed=0):
    """2-asset paths, shape (n_paths, 2, n_steps+1)."""
    if S0 is None:
        S0 = np.array([100.0, 90.0])
    rng = np.random.default_rng(seed)
    n_assets = len(S0)
    Z = rng.standard_normal((n_paths, n_assets, n_steps))
    log_inc = 0.05 / n_steps + 0.20 * np.sqrt(1.0 / n_steps) * Z
    log_paths = np.empty((n_paths, n_assets, n_steps + 1))
    log_paths[:, :, 0] = np.log(S0)
    for i in range(n_steps):
        log_paths[:, :, i + 1] = log_paths[:, :, i] + log_inc[:, :, i]
    return np.exp(log_paths)


PATHS = make_paths_1d()
TIMES = make_times_1d()
K = 100.0


# ---------------------------------------------------------------------------
# European
# ---------------------------------------------------------------------------

class TestEuropean:
    def test_call_nonneg(self):
        payoff = EuropeanCall(K)
        assert np.all(payoff(PATHS, TIMES) >= 0)

    def test_put_nonneg(self):
        payoff = EuropeanPut(K)
        assert np.all(payoff(PATHS, TIMES) >= 0)

    def test_call_shape(self):
        assert EuropeanCall(K)(PATHS, TIMES).shape == (len(PATHS),)

    def test_known_itm_call(self):
        """Call payoff = S_T - K when S_T > K."""
        ST = np.array([[90.0, 110.0]])   # shape (1, 2) → terminal 110
        times = np.array([0.0, 1.0])
        val = EuropeanCall(100.0)(ST, times)
        np.testing.assert_allclose(val, [10.0])

    def test_known_otm_call(self):
        ST = np.array([[90.0, 80.0]])
        times = np.array([0.0, 1.0])
        val = EuropeanCall(100.0)(ST, times)
        np.testing.assert_allclose(val, [0.0])

    def test_put_call_parity(self):
        c = EuropeanCall(K)(PATHS, TIMES)
        p = EuropeanPut(K)(PATHS, TIMES)
        ST = PATHS[:, -1]
        assert put_call_parity_check(c, p, ST, K)

    def test_call_payoff_deriv(self):
        """∂/∂S_T call = 1_{S_T > K}."""
        payoff = EuropeanCall(K)
        d = payoff.payoff_deriv(PATHS, TIMES)
        expected = (PATHS[:, -1] > K).astype(float)
        np.testing.assert_array_equal(d, expected)

    def test_put_payoff_deriv(self):
        """∂/∂S_T put = -1_{S_T < K}."""
        payoff = EuropeanPut(K)
        d = payoff.payoff_deriv(PATHS, TIMES)
        expected = -(PATHS[:, -1] < K).astype(float)
        np.testing.assert_array_equal(d, expected)

    def test_repr(self):
        assert "100" in repr(EuropeanCall(100))


class TestDigital:
    def test_call_put_partition(self):
        """DigitalCall(K) + DigitalPut(K) = 1 everywhere (ignoring S_T = K)."""
        dc = DigitalCall(K)(PATHS, TIMES)
        dp = DigitalPut(K)(PATHS, TIMES)
        ST = PATHS[:, -1]
        mask = ST != K
        np.testing.assert_array_equal((dc + dp)[mask], 1.0)

    def test_call_binary(self):
        dc = DigitalCall(K)(PATHS, TIMES)
        assert set(dc).issubset({0.0, 1.0})

    def test_asset_or_nothing(self):
        payoff = AssetOrNothingCall(K)
        vals = payoff(PATHS, TIMES)
        ST = PATHS[:, -1]
        expected = ST * (ST > K)
        np.testing.assert_array_equal(vals, expected)

    def test_digital_no_payoff_deriv(self):
        """DigitalCall does NOT implement payoff_deriv."""
        with pytest.raises(NotImplementedError):
            DigitalCall(K).payoff_deriv(PATHS, TIMES)


# ---------------------------------------------------------------------------
# Asian
# ---------------------------------------------------------------------------

class TestAsian:
    def test_arithmetic_call_nonneg(self):
        payoff = ArithmeticAsianCall(K)
        assert np.all(payoff(PATHS, TIMES) >= 0)

    def test_arithmetic_put_nonneg(self):
        assert np.all(ArithmeticAsianPut(K)(PATHS, TIMES) >= 0)

    def test_arithmetic_parity(self):
        """A_call - A_put = A - K (put-call parity on the average)."""
        c = ArithmeticAsianCall(K)(PATHS, TIMES)
        p = ArithmeticAsianPut(K)(PATHS, TIMES)
        A = PATHS[:, 1:].mean(axis=1)   # exclude t_0 by default
        np.testing.assert_allclose(c - p, A - K, atol=1e-10)

    def test_geometric_call_nonneg(self):
        assert np.all(GeometricAsianCall(K)(PATHS, TIMES) >= 0)

    def test_arithmetic_payoff_deriv_call(self):
        """payoff_deriv = 1_{A>K} · A/S0."""
        payoff = ArithmeticAsianCall(K)
        d = payoff.payoff_deriv(PATHS, TIMES)
        A = PATHS[:, 1:].mean(axis=1)
        S0 = PATHS[:, 0]
        expected = (A > K).astype(float) * A / S0
        np.testing.assert_allclose(d, expected, rtol=1e-10)

    def test_arithmetic_payoff_deriv_put(self):
        """payoff_deriv = -1_{A<K} · A/S0."""
        payoff = ArithmeticAsianPut(K)
        d = payoff.payoff_deriv(PATHS, TIMES)
        A = PATHS[:, 1:].mean(axis=1)
        S0 = PATHS[:, 0]
        expected = -(A < K).astype(float) * A / S0
        np.testing.assert_allclose(d, expected, rtol=1e-10)

    def test_geometric_payoff_deriv(self):
        """payoff_deriv = 1_{G>K} · G/S0."""
        payoff = GeometricAsianCall(K)
        d = payoff.payoff_deriv(PATHS, TIMES)
        cols = PATHS[:, 1:]
        G = np.exp(np.log(np.maximum(cols, 1e-300)).mean(axis=1))
        S0 = PATHS[:, 0]
        expected = (G > K).astype(float) * G / S0
        np.testing.assert_allclose(d, expected, rtol=1e-10)

    def test_averaging_dates(self):
        """Custom averaging_dates restricts the average correctly."""
        dates = np.array([10, 20, 30])
        payoff = ArithmeticAsianCall(K, averaging_dates=dates)
        vals = payoff(PATHS, TIMES)
        A_manual = PATHS[:, dates].mean(axis=1)
        expected = np.maximum(A_manual - K, 0.0)
        np.testing.assert_allclose(vals, expected)

    def test_geometric_closed_form(self):
        """Kemna-Vorst closed form should be within 1% of MC for large n_paths."""
        from mgreeks.models.gbm import GeometricBrownianMotion
        model = GeometricBrownianMotion(r=0.05, q=0.0, sigma=0.20)
        n = 252
        out = model.simulate(100.0, 1.0, n, 200_000,
                             return_full_paths=True,
                             rng=np.random.default_rng(0))
        payoff = GeometricAsianCall(K=100.0)
        mc_price = np.exp(-0.05) * payoff(out["paths"], out["times"]).mean()
        cf_price = GeometricAsianCall.closed_form_price(
            S0=100.0, K=100.0, r=0.05, q=0.0, sigma=0.20, T=1.0, n=n
        )
        assert abs(mc_price - cf_price) / cf_price < 0.02, (
            f"MC={mc_price:.4f} CF={cf_price:.4f}"
        )


# ---------------------------------------------------------------------------
# Barrier
# ---------------------------------------------------------------------------

class TestBarrier:
    # Use barrier below S0 so most paths stay alive
    B_down = 85.0    # down barrier
    B_up = 120.0     # up barrier

    def test_down_parity_call(self):
        """DownAndIn + DownAndOut = Vanilla for calls."""
        dai = DownAndInCall(K, self.B_down)(PATHS, TIMES)
        dao = DownAndOutCall(K, self.B_down)(PATHS, TIMES)
        vanilla = EuropeanCall(K)(PATHS, TIMES)
        np.testing.assert_allclose(dai + dao, vanilla, atol=1e-12)

    def test_down_parity_put(self):
        dip = DownAndInPut(K, self.B_down)(PATHS, TIMES)
        dop = DownAndOutPut(K, self.B_down)(PATHS, TIMES)
        vanilla = EuropeanPut(K)(PATHS, TIMES)
        np.testing.assert_allclose(dip + dop, vanilla, atol=1e-12)

    def test_up_parity_call(self):
        uai = UpAndInCall(K, self.B_up)(PATHS, TIMES)
        uao = UpAndOutCall(K, self.B_up)(PATHS, TIMES)
        vanilla = EuropeanCall(K)(PATHS, TIMES)
        np.testing.assert_allclose(uai + uao, vanilla, atol=1e-12)

    def test_dao_nonneg(self):
        assert np.all(DownAndOutCall(K, self.B_down)(PATHS, TIMES) >= 0)

    def test_dip_nonneg(self):
        assert np.all(DownAndInPut(K, self.B_down)(PATHS, TIMES) >= 0)

    def test_knocked_out_paths_zero(self):
        """Knocked-out paths must have zero payoff."""
        dao = DownAndOutCall(K, self.B_down)
        payoffs = dao(PATHS, TIMES)
        # Paths that touch the barrier
        hit = PATHS.min(axis=1) <= self.B_down
        np.testing.assert_array_equal(payoffs[hit], 0.0)

    def test_knocked_in_complement(self):
        """Down-and-in: paths that never touch barrier have zero payoff."""
        dai = DownAndInCall(K, self.B_down)
        payoffs = dai(PATHS, TIMES)
        alive = PATHS.min(axis=1) > self.B_down
        np.testing.assert_array_equal(payoffs[alive], 0.0)

    def test_parity_check_helper(self):
        result = barrier_parity_check(PATHS, TIMES, K, self.B_down, direction="down")
        assert result["call_parity_error"] < 1e-12
        assert result["put_parity_error"] < 1e-12

    def test_barrier_dates_subset(self):
        """barrier_dates restricts monitoring to those indices."""
        dates = np.array([10, 25, 50])
        dao_full = DownAndOutCall(K, self.B_down)
        dao_sub = DownAndOutCall(K, self.B_down, barrier_dates=dates)
        # Subset cannot be knocked out more than full monitoring
        assert dao_sub(PATHS, TIMES).sum() >= dao_full(PATHS, TIMES).sum() - 1e-9


# ---------------------------------------------------------------------------
# Lookback
# ---------------------------------------------------------------------------

class TestLookback:
    def test_fixed_call_nonneg(self):
        assert np.all(FixedStrikeLookbackCall(K)(PATHS, TIMES) >= 0)

    def test_fixed_put_nonneg(self):
        assert np.all(FixedStrikeLookbackPut(K)(PATHS, TIMES) >= 0)

    def test_floating_call_nonneg(self):
        """Floating call: S_T - min_t S_t ≥ 0 always."""
        assert np.all(FloatingStrikeLookbackCall()(PATHS, TIMES) >= 0)

    def test_floating_put_nonneg(self):
        assert np.all(FloatingStrikeLookbackPut()(PATHS, TIMES) >= 0)

    def test_fixed_call_uses_max(self):
        """max(M_T - K, 0) where M_T = path max."""
        payoff = FixedStrikeLookbackCall(K)
        vals = payoff(PATHS, TIMES)
        M_T = PATHS.max(axis=1)
        expected = np.maximum(M_T - K, 0.0)
        np.testing.assert_allclose(vals, expected)

    def test_fixed_put_uses_min(self):
        payoff = FixedStrikeLookbackPut(K)
        vals = payoff(PATHS, TIMES)
        m_T = PATHS.min(axis=1)
        expected = np.maximum(K - m_T, 0.0)
        np.testing.assert_allclose(vals, expected)

    def test_floating_call_formula(self):
        """S_T - min_t S_t."""
        vals = FloatingStrikeLookbackCall()(PATHS, TIMES)
        expected = PATHS[:, -1] - PATHS.min(axis=1)
        np.testing.assert_allclose(vals, expected)

    def test_floating_put_formula(self):
        """max_t S_t - S_T."""
        vals = FloatingStrikeLookbackPut()(PATHS, TIMES)
        expected = PATHS.max(axis=1) - PATHS[:, -1]
        np.testing.assert_allclose(vals, expected)

    def test_running_max(self):
        call = FixedStrikeLookbackCall(K)
        M = call.running_max(PATHS)
        assert M.shape == PATHS.shape
        # Running max must be monotonically non-decreasing
        assert np.all(np.diff(M, axis=1) >= 0)

    def test_partial_lookback(self):
        dates = np.array([5, 15, 30, 50])
        payoff = PartialLookbackCall(K, monitor_dates=dates)
        vals = payoff(PATHS, TIMES)
        M = PATHS[:, dates].max(axis=1)
        expected = np.maximum(M - K, 0.0)
        np.testing.assert_allclose(vals, expected)

    def test_partial_le_full(self):
        """Partial lookback ≤ full lookback (fewer monitoring dates → smaller max)."""
        dates = np.arange(1, 26)
        partial = PartialLookbackCall(K, monitor_dates=dates)(PATHS, TIMES)
        full = FixedStrikeLookbackCall(K)(PATHS, TIMES)
        assert np.all(partial <= full + 1e-12)

    def test_floating_call_deriv(self):
        """payoff_deriv = 1 (constant)."""
        deriv = FloatingStrikeLookbackCall().payoff_deriv_ST(PATHS, TIMES)
        np.testing.assert_array_equal(deriv, np.ones(len(PATHS)))


# ---------------------------------------------------------------------------
# Basket / Spread / Rainbow
# ---------------------------------------------------------------------------

class TestBasket:
    paths2 = make_paths_2d()   # (n_paths, 2, n_steps+1)
    times = make_times_1d()
    K = 95.0

    def test_basket_call_nonneg(self):
        payoff = BasketCall(self.K)
        assert np.all(payoff(self.paths2, self.times) >= 0)

    def test_basket_put_nonneg(self):
        assert np.all(BasketPut(self.K)(self.paths2, self.times) >= 0)

    def test_basket_parity(self):
        """BasketCall - BasketPut = basket - K."""
        c = BasketCall(self.K)(self.paths2, self.times)
        p = BasketPut(self.K)(self.paths2, self.times)
        ST = self.paths2[:, :, -1]    # (n_paths, 2)
        basket = ST.mean(axis=1)      # equal weights
        np.testing.assert_allclose(c - p, basket - self.K, atol=1e-10)

    def test_basket_weights(self):
        """Custom weights produce correct weighted average."""
        w = np.array([0.7, 0.3])
        payoff = BasketCall(self.K, weights=w)
        vals = payoff(self.paths2, self.times)
        ST = self.paths2[:, :, -1]
        basket = ST @ w
        expected = np.maximum(basket - self.K, 0.0)
        np.testing.assert_allclose(vals, expected, atol=1e-12)

    def test_basket_payoff_deriv_i(self):
        """payoff_deriv_i = w_i * 1_{basket > K}."""
        w = np.array([0.6, 0.4])
        payoff = BasketCall(self.K, weights=w)
        ST = self.paths2[:, :, -1]
        basket = ST @ w
        d0 = payoff.payoff_deriv_i(self.paths2, self.times, asset=0)
        expected = w[0] * (basket > self.K).astype(float)
        np.testing.assert_allclose(d0, expected)

    def test_spread_call_nonneg(self):
        assert np.all(SpreadCall(0.0)(self.paths2, self.times) >= 0)

    def test_spread_put_nonneg(self):
        assert np.all(SpreadPut(0.0)(self.paths2, self.times) >= 0)

    def test_spread_formula(self):
        """max(S1 - S2 - K, 0)."""
        K_sp = 5.0
        payoff = SpreadCall(K_sp)
        vals = payoff(self.paths2, self.times)
        ST = self.paths2[:, :, -1]
        expected = np.maximum(ST[:, 0] - ST[:, 1] - K_sp, 0.0)
        np.testing.assert_allclose(vals, expected)

    def test_best_of_call(self):
        payoff = BestOfCall(self.K)
        vals = payoff(self.paths2, self.times)
        ST = self.paths2[:, :, -1]
        expected = np.maximum(ST.max(axis=1) - self.K, 0.0)
        np.testing.assert_allclose(vals, expected)

    def test_worst_of_call(self):
        payoff = WorstOfCall(self.K)
        vals = payoff(self.paths2, self.times)
        ST = self.paths2[:, :, -1]
        expected = np.maximum(ST.min(axis=1) - self.K, 0.0)
        np.testing.assert_allclose(vals, expected)

    def test_best_ge_worst(self):
        """Best-of payoff ≥ worst-of payoff on every path."""
        b = BestOfCall(self.K)(self.paths2, self.times)
        w = WorstOfCall(self.K)(self.paths2, self.times)
        assert np.all(b >= w - 1e-12)

    def test_2d_input(self):
        """Payoffs accept 2D input (n_paths, n_assets) for terminal-only."""
        ST_only = self.paths2[:, :, -1]   # (n_paths, 2)
        call_3d = BasketCall(self.K)(self.paths2, self.times)
        call_2d = BasketCall(self.K)(ST_only, self.times)
        np.testing.assert_allclose(call_2d, call_3d)

    def test_margrabe_price_nonneg(self):
        """Margrabe (K=0 spread) price should be positive."""
        price = SpreadCall.margrabe_price(
            S1=100.0, S2=90.0, sigma1=0.20, sigma2=0.25,
            rho=0.5, r=0.05, q1=0.0, q2=0.0, T=1.0
        )
        assert price > 0
