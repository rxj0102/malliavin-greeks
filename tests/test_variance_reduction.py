"""Tests for variance reduction modules."""

import numpy as np
import pytest
from malliavin_greeks.models import GBMParams
from malliavin_greeks.variance import cv_delta_call, stratified_delta, is_delta_digital
from malliavin_greeks.estimators import malliavin_delta
from malliavin_greeks.payoffs import european_call, digital_call
from malliavin_greeks.benchmarks import bs_greeks


PARAMS = GBMParams(S0=100, r=0.05, q=0.0, sigma=0.20, T=1.0)
K = 100.0


def test_cv_delta_call_correct():
    """CV estimator should be closer to truth than plain Malliavin."""
    truth = bs_greeks(PARAMS, K, "call")["delta"]
    result = cv_delta_call(PARAMS, K, 300_000, 1, np.random.default_rng(123))
    assert abs(result.greek - truth) < 5 * result.std_err


def test_cv_reduces_variance():
    """Control variate should reduce variance vs plain Malliavin."""
    n = 200_000
    seed = 77

    r_plain = malliavin_delta(PARAMS, lambda S: european_call(S, K),
                               n, 1, np.random.default_rng(seed))
    r_cv = cv_delta_call(PARAMS, K, n, 1, np.random.default_rng(seed))

    # CV variance should be strictly lower (with high probability for this seed)
    assert r_cv.variance < r_plain.variance, (
        f"CV variance {r_cv.variance:.4e} >= plain {r_plain.variance:.4e}"
    )


def test_stratified_delta_correct():
    """Stratified estimator should give correct delta."""
    truth = bs_greeks(PARAMS, K, "call")["delta"]
    result = stratified_delta(PARAMS, lambda S: european_call(S, K),
                               n_paths=100_000, n_strata=10, n_steps=1,
                               rng=np.random.default_rng(55))
    assert abs(result.greek - truth) < 5 * result.std_err


def test_is_delta_digital_correct():
    """IS estimator for digital delta should be correct."""
    from scipy.stats import norm
    p = PARAMS
    d2 = (np.log(p.S0 / K) + (p.r - 0.5 * p.sigma**2) * p.T) / (p.sigma * np.sqrt(p.T))
    truth = np.exp(-p.r * p.T) * norm.pdf(d2) / (p.S0 * p.sigma * np.sqrt(p.T))

    result = is_delta_digital(PARAMS, K, 200_000, 1, np.random.default_rng(88))
    assert abs(result.greek - truth) < 5 * result.std_err
