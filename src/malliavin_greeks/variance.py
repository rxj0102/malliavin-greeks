"""
Variance-reduction techniques for Malliavin Greek estimators.

1. Control variates  — subtract a correlated estimator with known expectation
2. Antithetic variates — already supported in simulate_gbm (antithetic=True)
3. Optimal localization — choose u(t) in the BEL formula to minimize variance
4. Importance sampling — tilt the path measure towards the exercise region

All functions return a GreekResult (or equivalent dict) consistent with the
estimators module.
"""

from __future__ import annotations

import numpy as np
from typing import Callable
from malliavin_greeks.models import GBMParams, simulate_gbm
from malliavin_greeks import weights as W
from malliavin_greeks.estimators import GreekResult


# ---------------------------------------------------------------------------
# 1. Control variate for Malliavin delta using Black-Scholes delta
# ---------------------------------------------------------------------------

def _bs_price(S0, K, r, q, sigma, T, option="call"):
    """Black-Scholes closed-form price."""
    from scipy.stats import norm
    d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option == "call":
        return S0 * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S0 * np.exp(-q * T) * norm.cdf(-d1)


def _bs_delta(S0, K, r, q, sigma, T, option="call"):
    """Black-Scholes delta (analytic)."""
    from scipy.stats import norm
    d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    if option == "call":
        return np.exp(-q * T) * norm.cdf(d1)
    else:
        return -np.exp(-q * T) * norm.cdf(-d1)


def cv_delta_call(
    params: GBMParams,
    K: float,
    n_paths: int,
    n_steps: int = 1,
    rng=None,
    antithetic: bool = True,
) -> GreekResult:
    """
    Malliavin delta for a European call with control variate.

    Control: Y = payoff(S) * pi_delta  with known mean = BS delta.
    The control variate is the call payoff itself:
        CV = disc * (S_T - K)_+ - BS_price
    whose mean is zero.  We regress the Malliavin estimator on CV.

    This is particularly effective because the call price and its delta
    are both driven by the same path, giving high correlation.
    """
    from malliavin_greeks.payoffs import european_call
    S, Z = simulate_gbm(params, n_paths, n_steps, rng, antithetic)
    pi = W.weight_delta_gbm(params, Z)
    disc = np.exp(-params.r * params.T)

    raw = disc * european_call(S, K) * pi          # Malliavin delta samples

    # Control variate: discounted payoff, mean = BS price
    cv_samples = disc * european_call(S, K)
    cv_mean = _bs_price(params.S0, K, params.r, params.q, params.sigma, params.T, "call")

    # Optimal coefficient: beta = Cov(raw, cv) / Var(cv)
    cov = np.cov(raw, cv_samples)
    beta = cov[0, 1] / cov[1, 1] if cov[1, 1] > 0 else 0.0
    controlled = raw - beta * (cv_samples - cv_mean)
    return GreekResult(controlled, "cv_delta_call")


def cv_delta_put(
    params: GBMParams,
    K: float,
    n_paths: int,
    n_steps: int = 1,
    rng=None,
    antithetic: bool = True,
) -> GreekResult:
    """Malliavin delta for European put with control variate."""
    from malliavin_greeks.payoffs import european_put
    S, Z = simulate_gbm(params, n_paths, n_steps, rng, antithetic)
    pi = W.weight_delta_gbm(params, Z)
    disc = np.exp(-params.r * params.T)

    raw = disc * european_put(S, K) * pi
    cv_samples = disc * european_put(S, K)
    cv_mean = _bs_price(params.S0, K, params.r, params.q, params.sigma, params.T, "put")

    cov = np.cov(raw, cv_samples)
    beta = cov[0, 1] / cov[1, 1] if cov[1, 1] > 0 else 0.0
    controlled = raw - beta * (cv_samples - cv_mean)
    return GreekResult(controlled, "cv_delta_put")


# ---------------------------------------------------------------------------
# 2. Optimal localization function u(t) for the BEL weight
# ---------------------------------------------------------------------------

def weight_delta_optimal_localization(
    params: GBMParams,
    Z: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    BEL delta weight with power-law localization u(t) = (T - t)^alpha / C_alpha.

    The default u(t) = 1/T (uniform) is optimal for terminal payoffs.
    For path-dependent payoffs, u(t) = (T - t)^alpha concentrates weight
    near time 0, reducing variance for payoffs that are more sensitive to
    early path segments.

    C_alpha = int_0^T (T-t)^alpha dt = T^{alpha+1} / (alpha+1).
    """
    p = params
    n_steps = Z.shape[1]
    dt = p.T / n_steps
    sqrt_dt = np.sqrt(dt)

    t = np.arange(1, n_steps + 1) * dt - dt / 2   # midpoints
    u = (p.T - t) ** alpha
    C = p.T ** (alpha + 1) / (alpha + 1)
    u_normalized = u / C                            # (n_steps,)

    # BEL weight: pi = (1/(sigma * S0)) * int_0^T u(t)/J_t dW_t
    # Under GBM: J_t = S_t/S0, so 1/J_t = S0/S_t.
    # Without S path: approximate J_t = 1 (first variation at t=0 is 1).
    # Full form requires S path; here we use the leading-order approximation.
    weighted_Z = (u_normalized * Z).sum(axis=1) * sqrt_dt
    return weighted_Z / (p.sigma * p.S0)


# ---------------------------------------------------------------------------
# 3. Importance sampling for digital options (near-ATM)
# ---------------------------------------------------------------------------

def is_delta_digital(
    params: GBMParams,
    K: float,
    n_paths: int,
    n_steps: int = 1,
    rng=None,
) -> GreekResult:
    """
    Importance sampling + Malliavin delta for digital call options.

    For a digital, the standard Malliavin estimator has high variance near ATM
    because the payoff is a step function.  IS shifts the drift so paths are
    more likely to land near K, then reweights by the Radon-Nikodym derivative.

    IS drift: shift log S_T so that E*[S_T] = K  =>  log K - log S0 = (r-q-sig^2/2)T + mu_shift*T
    mu_shift = (log(K/S0) - (r-q-sig^2/2)*T) / T
    """
    from malliavin_greeks.payoffs import digital_call
    rng = np.random.default_rng(rng)
    p = params

    # Compute IS drift shift
    mu_0 = p.r - p.q - 0.5 * p.sigma**2
    mu_shift = (np.log(K / p.S0) / p.T) - mu_0   # shift so mean log = log K

    # Simulate under shifted measure
    n_steps_ = n_steps
    dt = p.T / n_steps_
    Z = rng.standard_normal((n_paths, n_steps_))
    sqrt_dt = np.sqrt(dt)

    log_increments = (mu_0 + mu_shift) * dt + p.sigma * sqrt_dt * Z
    log_S = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(log_increments, axis=1)], axis=1)
    S = p.S0 * np.exp(log_S)

    # Radon-Nikodym: dQ/dP = exp(-mu_shift/sigma * W_T - 0.5*(mu_shift/sigma)^2 * T)
    W_T = sqrt_dt * Z.sum(axis=1)
    theta_is = mu_shift / p.sigma
    RN = np.exp(-theta_is * W_T - 0.5 * theta_is**2 * p.T)

    # Under Q*, the Q-Brownian motion is W^Q = W^{Q*} + mu_shift*T/sigma.
    # The Malliavin delta weight is w.r.t. Q:
    #   pi^Q = W^Q_T / (sigma * S0 * T) = (W_T + mu_shift*T/sigma) / (sigma*S0*T)
    W_Q_T = W_T + theta_is * p.T          # Girsanov correction: W^Q = W^{Q*} + theta*T
    pi = W_Q_T / (p.sigma * p.S0 * p.T)

    disc = np.exp(-p.r * p.T)
    samples = disc * digital_call(S, K) * pi * RN
    return GreekResult(samples, "is_delta_digital")


# ---------------------------------------------------------------------------
# 4. Stratified sampling for variance reduction
# ---------------------------------------------------------------------------

def stratified_delta(
    params: GBMParams,
    payoff_fn: Callable[[np.ndarray], np.ndarray],
    n_paths: int,
    n_strata: int = 10,
    n_steps: int = 1,
    rng=None,
) -> GreekResult:
    """
    Stratified sampling on the terminal Brownian motion W_T.

    Divide [0,1] into n_strata equal-probability strata.  Draw one uniform
    per stratum, convert to normal via inverse CDF, simulate paths, and
    apply the Malliavin delta weight.

    This ensures uniform coverage of the W_T distribution and reduces variance
    of the estimator approximately by a factor of n_strata for smooth payoffs.
    """
    from scipy.stats import norm as sp_norm
    rng_ = np.random.default_rng(rng)

    paths_per_stratum = n_paths // n_strata
    all_samples = []

    for j in range(n_strata):
        lo = j / n_strata
        hi = (j + 1) / n_strata
        u = rng_.uniform(lo, hi, size=paths_per_stratum)
        z_terminal = sp_norm.ppf(u)   # stratified normal draws for W_T

        # For n_steps=1, Z is just z_terminal
        if n_steps == 1:
            Z = z_terminal.reshape(-1, 1)
        else:
            # Split: first Brownian increment is stratified; rest are free
            Z_rest = rng_.standard_normal((paths_per_stratum, n_steps - 1))
            Z = np.concatenate([z_terminal.reshape(-1, 1), Z_rest], axis=1)

        dt = params.T / n_steps
        log_inc = (params.r - params.q - 0.5 * params.sigma**2) * dt + params.sigma * np.sqrt(dt) * Z
        log_S = np.concatenate([np.zeros((paths_per_stratum, 1)), np.cumsum(log_inc, axis=1)], axis=1)
        S = params.S0 * np.exp(log_S)

        pi = W.weight_delta_gbm(params, Z)
        disc = np.exp(-params.r * params.T)
        all_samples.append(disc * payoff_fn(S) * pi)

    return GreekResult(np.concatenate(all_samples), "stratified_delta")
