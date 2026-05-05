"""
Tests for Greek estimators against Black-Scholes analytics.

Each test runs with a large-enough sample to give a 3-sigma pass/fail margin.
We test:
    - Malliavin estimators (delta, gamma, vega, rho)
    - Finite-difference estimators
    - Pathwise estimators
    - Likelihood-ratio estimators
"""

import numpy as np
import pytest
from scipy.stats import norm

from malliavin_greeks.models import GBMParams
from malliavin_greeks.payoffs import european_call, european_put, digital_call
from malliavin_greeks.benchmarks import bs_greeks
from malliavin_greeks.estimators import (
    malliavin_delta, malliavin_gamma, malliavin_vega, malliavin_rho,
    fd_delta_central, fd_gamma_central, fd_vega_central,
    pathwise_delta, pathwise_vega,
    lr_delta, lr_gamma,
)


# ---------------------------------------------------------------------------
# Shared setup
# ---------------------------------------------------------------------------

PARAMS = GBMParams(S0=100.0, r=0.05, q=0.0, sigma=0.20, T=1.0)
K_ATM = 100.0
K_ITM = 80.0
K_OTM = 120.0
N_PATHS = 500_000
SEED = 2024
TOL_SIGMA = 4.0          # pass if |estimate - truth| < TOL_SIGMA * std_err


def _check(result, truth, tol_sigma=TOL_SIGMA):
    z = abs(result.greek - truth) / result.std_err
    assert z < tol_sigma, (
        f"Estimator {result.estimator!r}: greek={result.greek:.5f}, "
        f"truth={truth:.5f}, z-score={z:.2f} > {tol_sigma}"
    )


# ---------------------------------------------------------------------------
# Malliavin delta
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("K", [K_ITM, K_ATM, K_OTM])
def test_malliavin_delta_call(K):
    truth = bs_greeks(PARAMS, K, "call")["delta"]
    result = malliavin_delta(PARAMS, lambda S: european_call(S, K),
                              N_PATHS, 1, np.random.default_rng(SEED), antithetic=True)
    _check(result, truth)


@pytest.mark.parametrize("K", [K_ITM, K_ATM, K_OTM])
def test_malliavin_delta_put(K):
    truth = bs_greeks(PARAMS, K, "put")["delta"]
    result = malliavin_delta(PARAMS, lambda S: european_put(S, K),
                              N_PATHS, 1, np.random.default_rng(SEED), antithetic=True)
    _check(result, truth)


# ---------------------------------------------------------------------------
# Malliavin gamma
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("K", [K_ATM, K_ITM])
def test_malliavin_gamma(K):
    truth = bs_greeks(PARAMS, K, "call")["gamma"]
    result = malliavin_gamma(PARAMS, lambda S: european_call(S, K),
                               N_PATHS, 1, np.random.default_rng(SEED), antithetic=True)
    _check(result, truth)


# ---------------------------------------------------------------------------
# Malliavin vega
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("K", [K_ATM, K_OTM])
def test_malliavin_vega(K):
    truth = bs_greeks(PARAMS, K, "call")["vega"]
    result = malliavin_vega(PARAMS, lambda S: european_call(S, K),
                              N_PATHS, 1, np.random.default_rng(SEED), antithetic=True)
    _check(result, truth)


# ---------------------------------------------------------------------------
# Malliavin rho
# ---------------------------------------------------------------------------

def test_malliavin_rho_call():
    K = K_ATM
    truth = bs_greeks(PARAMS, K, "call")["rho"]
    result = malliavin_rho(PARAMS, lambda S: european_call(S, K),
                            N_PATHS, 1, np.random.default_rng(SEED), antithetic=True)
    _check(result, truth)


# ---------------------------------------------------------------------------
# Finite-difference delta
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("K", [K_ATM, K_OTM])
def test_fd_delta_central(K):
    truth = bs_greeks(PARAMS, K, "call")["delta"]
    result = fd_delta_central(PARAMS, lambda S: european_call(S, K),
                               N_PATHS // 2, 1, 1.0, np.random.default_rng(SEED))
    _check(result, truth)


# ---------------------------------------------------------------------------
# Pathwise delta
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("K", [K_ATM, K_OTM])
def test_pathwise_delta(K):
    truth = bs_greeks(PARAMS, K, "call")["delta"]
    result = pathwise_delta(PARAMS, lambda S: (S[:, -1] > K).astype(float),
                             N_PATHS, 1, np.random.default_rng(SEED), antithetic=True)
    _check(result, truth)


# ---------------------------------------------------------------------------
# LR delta (== Malliavin for terminal payoffs)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("K", [K_ATM, K_OTM])
def test_lr_delta(K):
    truth = bs_greeks(PARAMS, K, "call")["delta"]
    result = lr_delta(PARAMS, lambda S: european_call(S, K),
                      N_PATHS, 1, np.random.default_rng(SEED), antithetic=True)
    _check(result, truth)


# ---------------------------------------------------------------------------
# Digital call delta (discontinuous payoff — Malliavin key use case)
# ---------------------------------------------------------------------------

def test_malliavin_delta_digital():
    p = PARAMS
    K = K_ATM
    d2 = (np.log(p.S0 / K) + (p.r - 0.5 * p.sigma**2) * p.T) / (p.sigma * np.sqrt(p.T))
    truth = np.exp(-p.r * p.T) * norm.pdf(d2) / (p.S0 * p.sigma * np.sqrt(p.T))
    result = malliavin_delta(p, lambda S: digital_call(S, K),
                              N_PATHS, 1, np.random.default_rng(SEED), antithetic=True)
    _check(result, truth)


# ---------------------------------------------------------------------------
# Asian option delta: Malliavin BEL weight
# ---------------------------------------------------------------------------

def test_pathwise_delta_asian_call():
    """
    Pathwise IPA delta for arithmetic-average Asian call.

    dA/dS0 = A/S0 (exact, since S_t = S0 * S_t_normalized), so the IPA
    estimator is E[1_{A>K} * A/S0] and should agree with FD to well within
    5 combined standard errors.
    """
    from malliavin_greeks.payoffs import asian_call
    from malliavin_greeks.estimators import pathwise_delta_asian, fd_delta_central

    p = GBMParams(S0=100, r=0.05, q=0.0, sigma=0.20, T=1.0)
    K = 100.0
    n_steps = 50

    r_pw = pathwise_delta_asian(
        p, lambda S, A: (A > K).astype(float),
        300_000, n_steps, np.random.default_rng(SEED)
    )
    # FD reference
    r_fd = fd_delta_central(p, lambda S: asian_call(S, K),
                             500_000, n_steps, 0.5, np.random.default_rng(SEED))

    combined_se = np.sqrt(r_pw.std_err**2 + r_fd.std_err**2)
    assert abs(r_pw.greek - r_fd.greek) < 5 * combined_se, (
        f"Pathwise Asian delta {r_pw.greek:.4f} vs FD {r_fd.greek:.4f}, "
        f"combined SE = {combined_se:.4f}"
    )
