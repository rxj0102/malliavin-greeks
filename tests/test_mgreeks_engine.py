"""
Tests for MonteCarloEngine and mgreeks.utils.

Verifies that the engine produces Greek estimates consistent with
Black-Scholes analytics for vanilla European options.
"""

import numpy as np
import pytest

from mgreeks.models.gbm import GeometricBrownianMotion
from mgreeks.payoffs.european import EuropeanCall, EuropeanPut, DigitalCall
from mgreeks.simulation import MonteCarloEngine
from mgreeks.utils import bs_price, bs_greeks, relative_error, confidence_interval


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

S0, K, T = 100.0, 100.0, 1.0
R, Q, SIGMA = 0.05, 0.02, 0.20
N_PATHS = 200_000
N_STEPS = 1         # exact GBM: 1 step is enough for vanilla
SEED = 0

MODEL = GeometricBrownianMotion(r=R, q=Q, sigma=SIGMA)
ENGINE = MonteCarloEngine(MODEL, n_paths=N_PATHS, n_steps=N_STEPS, rng_seed=SEED)
CALL = EuropeanCall(K)
PUT = EuropeanPut(K)

TRUTH = bs_greeks(S0, K, R, Q, SIGMA, T, option="call")
TRUTH_P = bs_greeks(S0, K, R, Q, SIGMA, T, option="put")
BS_CALL = bs_price(S0, K, R, Q, SIGMA, T, "call")
BS_PUT = bs_price(S0, K, R, Q, SIGMA, T, "put")

TOL_4SIGMA = 4       # accept if estimate within 4 std errors of truth


def within_n_se(estimate, truth, std_error, n=TOL_4SIGMA):
    return abs(estimate - truth) < n * std_error + 1e-8


# ---------------------------------------------------------------------------
# Price tests
# ---------------------------------------------------------------------------

class TestPrice:
    def test_call_price(self):
        res = ENGINE.price(CALL, S0, T)
        assert "price" in res
        assert within_n_se(res["price"], BS_CALL, res["std_error"]), (
            f"price={res['price']:.4f} vs {BS_CALL:.4f} se={res['std_error']:.4f}"
        )

    def test_put_price(self):
        res = ENGINE.price(PUT, S0, T)
        assert within_n_se(res["price"], BS_PUT, res["std_error"])

    def test_result_keys(self):
        res = ENGINE.price(CALL, S0, T)
        for key in ("price", "std_error", "ci_lower", "ci_upper", "n_paths"):
            assert key in res

    def test_ci_contains_truth(self):
        res = ENGINE.price(CALL, S0, T)
        assert res["ci_lower"] <= BS_CALL <= res["ci_upper"] or (
            abs(res["price"] - BS_CALL) < 3 * res["std_error"]
        )

    def test_n_paths_stored(self):
        res = ENGINE.price(CALL, S0, T)
        assert res["n_paths"] == N_PATHS


# ---------------------------------------------------------------------------
# Malliavin Greek tests
# ---------------------------------------------------------------------------

class TestMalliavinGreeks:
    def _greek(self, greek_name):
        return ENGINE.greek(CALL, S0, T, greek_name, method="malliavin")

    def test_delta(self):
        res = self._greek("delta")
        assert within_n_se(res["greek"], TRUTH["delta"], res["std_error"]), (
            f"delta={res['greek']:.5f} vs {TRUTH['delta']:.5f} se={res['std_error']:.5f}"
        )

    def test_gamma(self):
        res = self._greek("gamma")
        assert within_n_se(res["greek"], TRUTH["gamma"], res["std_error"]), (
            f"gamma={res['greek']:.6f} vs {TRUTH['gamma']:.6f} se={res['std_error']:.6f}"
        )

    def test_vega(self):
        res = self._greek("vega")
        assert within_n_se(res["greek"], TRUTH["vega"], res["std_error"]), (
            f"vega={res['greek']:.4f} vs {TRUTH['vega']:.4f} se={res['std_error']:.4f}"
        )

    def test_rho(self):
        res = self._greek("rho")
        assert within_n_se(res["greek"], TRUTH["rho"], res["std_error"]), (
            f"rho={res['greek']:.4f} vs {TRUTH['rho']:.4f} se={res['std_error']:.4f}"
        )

    def test_result_has_method(self):
        res = self._greek("delta")
        assert res["method"] == "malliavin"

    def test_lr_alias(self):
        res1 = ENGINE.greek(CALL, S0, T, "delta", method="malliavin")
        res2 = ENGINE.greek(CALL, S0, T, "delta", method="lr")
        assert res2["method"] == "malliavin"

    def test_unknown_greek_raises(self):
        with pytest.raises(ValueError):
            ENGINE.greek(CALL, S0, T, "banana", method="malliavin")

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            ENGINE.greek(CALL, S0, T, "delta", method="unknown")


# ---------------------------------------------------------------------------
# Pathwise Greek tests
# ---------------------------------------------------------------------------

class TestPathwiseDelta:
    def test_delta_pathwise(self):
        res = ENGINE.greek(CALL, S0, T, "delta", method="pathwise")
        assert within_n_se(res["greek"], TRUTH["delta"], res["std_error"]), (
            f"IPA delta={res['greek']:.5f} vs {TRUTH['delta']:.5f} se={res['std_error']:.5f}"
        )

    def test_pathwise_non_delta_raises(self):
        with pytest.raises(ValueError):
            ENGINE.greek(CALL, S0, T, "vega", method="pathwise")

    def test_digital_pathwise_raises(self):
        with pytest.raises(NotImplementedError):
            ENGINE.greek(DigitalCall(K), S0, T, "delta", method="pathwise")


# ---------------------------------------------------------------------------
# Finite-difference Greek tests
# ---------------------------------------------------------------------------

class TestFiniteDiffGreeks:
    engine_fd = MonteCarloEngine(MODEL, n_paths=100_000, n_steps=N_STEPS, rng_seed=SEED)

    def test_fd_delta(self):
        res = self.engine_fd.greek(CALL, S0, T, "delta", method="finite_diff")
        assert within_n_se(res["greek"], TRUTH["delta"], max(res["std_error"], 1e-4))

    def test_fd_result_keys(self):
        res = self.engine_fd.greek(CALL, S0, T, "delta", method="finite_diff")
        assert "greek" in res


# ---------------------------------------------------------------------------
# all_greeks
# ---------------------------------------------------------------------------

class TestAllGreeks:
    def test_all_greeks_keys(self):
        results = ENGINE.all_greeks(CALL, S0, T, method="malliavin")
        assert set(results.keys()) == {"delta", "gamma", "vega", "rho", "theta"}

    def test_all_greeks_delta_close(self):
        results = ENGINE.all_greeks(CALL, S0, T, method="malliavin")
        d = results["delta"]
        assert within_n_se(d["greek"], TRUTH["delta"], d["std_error"])


# ---------------------------------------------------------------------------
# utils
# ---------------------------------------------------------------------------

class TestUtils:
    def test_bs_price_call(self):
        p = bs_price(S0, K, R, Q, SIGMA, T, "call")
        assert p > 0

    def test_bs_put_call_parity(self):
        c = bs_price(S0, K, R, Q, SIGMA, T, "call")
        p = bs_price(S0, K, R, Q, SIGMA, T, "put")
        disc = np.exp(-R * T)
        fwd = S0 * np.exp(-Q * T)
        parity = c - p - (fwd - K * disc)
        assert abs(parity) < 1e-10

    def test_bs_greeks_call_delta_range(self):
        g = bs_greeks(S0, K, R, Q, SIGMA, T, "call")
        assert 0 < g["delta"] < 1

    def test_bs_greeks_put_delta_range(self):
        g = bs_greeks(S0, K, R, Q, SIGMA, T, "put")
        assert -1 < g["delta"] < 0

    def test_bs_greeks_gamma_positive(self):
        g = bs_greeks(S0, K, R, Q, SIGMA, T, "call")
        assert g["gamma"] > 0

    def test_bs_greeks_vega_positive(self):
        g = bs_greeks(S0, K, R, Q, SIGMA, T, "call")
        assert g["vega"] > 0

    def test_relative_error(self):
        assert abs(relative_error(1.05, 1.0) - 5.0) < 1e-10

    def test_relative_error_zero_truth(self):
        assert np.isnan(relative_error(1.0, 0.0))

    def test_confidence_interval(self):
        samples = np.random.default_rng(0).standard_normal(10_000)
        ci = confidence_interval(samples)
        assert ci["ci_lower"] < ci["estimate"] < ci["ci_upper"]
        assert abs(ci["estimate"]) < 0.05  # mean ≈ 0 for standard normals
