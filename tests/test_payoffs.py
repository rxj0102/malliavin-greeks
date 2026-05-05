"""Tests for payoff functions."""

import numpy as np
import pytest
from malliavin_greeks.payoffs import (
    european_call, european_put,
    digital_call, digital_put,
    asian_call, asian_put,
    down_and_out_call, up_and_out_call,
    down_and_in_call,
    lookback_call_fixed, lookback_put_fixed,
    lookback_call_floating, lookback_put_floating,
)


def _make_paths(S0=100.0, n_paths=1000, n_steps=10, seed=0):
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n_paths, n_steps))
    log_S = np.cumsum(0.02 * Z, axis=1)
    log_S = np.concatenate([np.zeros((n_paths, 1)), log_S], axis=1)
    return S0 * np.exp(log_S)


S = _make_paths()


def test_european_call_nonneg():
    assert np.all(european_call(S, 100) >= 0)


def test_european_put_nonneg():
    assert np.all(european_put(S, 100) >= 0)


def test_put_call_parity():
    """C - P = S_T - K*e^{-rT} (undiscounted); we check C - P = S_T - K."""
    K = 100.0
    c = european_call(S, K)
    p = european_put(S, K)
    ST = S[:, -1]
    np.testing.assert_allclose(c - p, ST - K, atol=1e-10)


def test_digital_call_binary():
    d = digital_call(S, 100)
    assert set(d).issubset({0.0, 1.0})


def test_digital_put_binary():
    d = digital_put(S, 100)
    assert set(d).issubset({0.0, 1.0})


def test_digital_call_put_partition():
    c = digital_call(S, 100)
    p = digital_put(S, 100)
    # Ignoring ties (S_T == K has prob 0): c + p = 1 a.s.
    assert np.all(c + p == 1)


def test_asian_call_nonneg():
    assert np.all(asian_call(S, 100) >= 0)


def test_asian_put_nonneg():
    assert np.all(asian_put(S, 100) >= 0)


def test_down_and_out_call_knockout():
    """For paths that hit barrier, payoff must be zero."""
    # Set barrier above S0 so most paths are knocked out immediately
    H = S[:, 0].mean() * 0.99   # barrier just below S0; paths that dip get killed
    payoffs = down_and_out_call(S, 95, H)
    vanilla = european_call(S, 95)
    # Some paths should survive; the rest must have zero payoff
    alive = np.all(S > H, axis=1)
    assert alive.sum() > 0, "No paths survived — barrier too high"
    assert alive.sum() < len(S), "All paths survived — barrier too low"
    # Knocked-out paths have zero payoff regardless of vanilla payoff
    np.testing.assert_array_equal(payoffs[~alive], 0.0)


def test_barrier_in_plus_out_equals_vanilla():
    """Down-and-in + down-and-out = vanilla (knock-in/knock-out parity)."""
    K, H = 100.0, 85.0
    dao = down_and_out_call(S, K, H)
    dai = down_and_in_call(S, K, H)
    vanilla = european_call(S, K)
    np.testing.assert_allclose(dao + dai, vanilla, atol=1e-10)


def test_lookback_call_fixed_nonneg():
    assert np.all(lookback_call_fixed(S, 100) >= 0)


def test_lookback_floating_nonneg():
    assert np.all(lookback_call_floating(S) >= 0)
    assert np.all(lookback_put_floating(S) >= 0)
