"""
Analytical (closed-form) Greek formulas.

Functions
---------
bs_delta, bs_gamma, bs_vega, bs_theta, bs_rho
bs_digital_delta, bs_digital_gamma
bs_barrier_delta
geometric_asian_greeks
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _d1d2(S0: float, K: float, T: float, r: float, q: float, sigma: float):
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return d1, d2, sqrt_T


# ---------------------------------------------------------------------------
# Black-Scholes Greeks
# ---------------------------------------------------------------------------

def bs_delta(
    S0: float, K: float, T: float, r: float, q: float, sigma: float,
    option_type: str = "call",
) -> float:
    """Black-Scholes delta: e^{-qT} N(d1) for call, -e^{-qT} N(-d1) for put."""
    d1, _, _ = _d1d2(S0, K, T, r, q, sigma)
    dq = np.exp(-q * T)
    if option_type == "call":
        return float(dq * norm.cdf(d1))
    return float(-dq * norm.cdf(-d1))


def bs_gamma(
    S0: float, K: float, T: float, r: float, q: float, sigma: float,
) -> float:
    """Black-Scholes gamma: e^{-qT} phi(d1) / (S0 sigma sqrt(T))."""
    d1, _, sqrt_T = _d1d2(S0, K, T, r, q, sigma)
    return float(np.exp(-q * T) * norm.pdf(d1) / (S0 * sigma * sqrt_T))


def bs_vega(
    S0: float, K: float, T: float, r: float, q: float, sigma: float,
) -> float:
    """Black-Scholes vega: S0 e^{-qT} phi(d1) sqrt(T)."""
    d1, _, sqrt_T = _d1d2(S0, K, T, r, q, sigma)
    return float(S0 * np.exp(-q * T) * norm.pdf(d1) * sqrt_T)


def bs_theta(
    S0: float, K: float, T: float, r: float, q: float, sigma: float,
    option_type: str = "call",
) -> float:
    """Black-Scholes theta (negative for long options)."""
    d1, d2, sqrt_T = _d1d2(S0, K, T, r, q, sigma)
    disc = np.exp(-r * T)
    fwd = S0 * np.exp(-q * T)
    phi_d1 = norm.pdf(d1)
    decay = -fwd * phi_d1 * sigma / (2.0 * sqrt_T)
    if option_type == "call":
        return float(decay - r * K * disc * norm.cdf(d2) + q * fwd * norm.cdf(d1))
    return float(decay + r * K * disc * norm.cdf(-d2) - q * fwd * norm.cdf(-d1))


def bs_rho(
    S0: float, K: float, T: float, r: float, q: float, sigma: float,
    option_type: str = "call",
) -> float:
    """Black-Scholes rho: K T e^{-rT} N(d2) for call."""
    _, d2, _ = _d1d2(S0, K, T, r, q, sigma)
    disc = np.exp(-r * T)
    if option_type == "call":
        return float(K * T * disc * norm.cdf(d2))
    return float(-K * T * disc * norm.cdf(-d2))


# ---------------------------------------------------------------------------
# Digital call Greeks
# ---------------------------------------------------------------------------

def bs_digital_delta(
    S0: float, K: float, T: float, r: float, q: float, sigma: float,
) -> float:
    """
    Delta of a digital (cash-or-nothing) call.

    Price = e^{-rT} N(d2).
    Delta = e^{-rT} phi(d2) / (S0 sigma sqrt(T)).
    """
    _, d2, sqrt_T = _d1d2(S0, K, T, r, q, sigma)
    return float(np.exp(-r * T) * norm.pdf(d2) / (S0 * sigma * sqrt_T))


def bs_digital_gamma(
    S0: float, K: float, T: float, r: float, q: float, sigma: float,
) -> float:
    """
    Gamma of a digital (cash-or-nothing) call.

    Gamma = -e^{-rT} d1 phi(d2) / (S0^2 sigma^2 T).
    """
    d1, d2, sqrt_T = _d1d2(S0, K, T, r, q, sigma)
    return float(-np.exp(-r * T) * d1 * norm.pdf(d2) / (S0**2 * sigma**2 * T))


# ---------------------------------------------------------------------------
# Barrier option delta
# ---------------------------------------------------------------------------

def bs_barrier_delta(
    S0: float,
    K: float,
    B: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    barrier_type: str = "down_and_out_call",
) -> float:
    """
    Analytical delta for a barrier option (where available).

    Down-and-out call (B ≤ S0, B ≤ K):
        V_doc = BS_call(S0) - (B/S0)^{2λ} BS_call(B^2/S0)

    where λ = (r - q) / sigma^2 + 0.5.

    Delta is computed via ∂V/∂S0.
    """
    if barrier_type != "down_and_out_call":
        raise NotImplementedError(f"barrier_type={barrier_type!r} not implemented.")
    if S0 <= B:
        return 0.0

    lam = (r - q) / sigma**2 + 0.5
    ratio = B / S0

    def _bs_call_price(s: float) -> float:
        if s <= 0:
            return 0.0
        d1, d2, _ = _d1d2(s, K, T, r, q, sigma)
        disc = np.exp(-r * T)
        return float(s * np.exp(-q * T) * norm.cdf(d1) - K * disc * norm.cdf(d2))

    def _bs_call_delta(s: float) -> float:
        return bs_delta(s, K, T, r, q, sigma, "call")

    s_image = B**2 / S0

    # ∂/∂S0 [(B/S0)^{2λ} BS_call(B^2/S0)]
    # = (-2λ/S0) * (B/S0)^{2λ} * BS_call(B^2/S0)
    #   + (B/S0)^{2λ} * BS_call_delta(B^2/S0) * (-B^2/S0^2)
    mirror_factor = ratio ** (2 * lam)
    mirror_price = _bs_call_price(s_image)
    mirror_delta_image = _bs_call_delta(s_image)
    mirror_delta_wrt_S0 = (
        (-2 * lam / S0) * mirror_factor * mirror_price
        + mirror_factor * mirror_delta_image * (-B**2 / S0**2)
    )

    vanilla_delta = bs_delta(S0, K, T, r, q, sigma, "call")
    return float(vanilla_delta - mirror_delta_wrt_S0)


# ---------------------------------------------------------------------------
# Geometric Asian option Greeks
# ---------------------------------------------------------------------------

def geometric_asian_greeks(
    S0: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    n_averaging: int,
) -> dict:
    """
    Greeks of a geometric Asian call option under GBM (closed-form).

    For n discrete averaging dates t_i = i*T/n (i=1,...,n), the geometric
    mean G = (Π S_{t_i})^{1/n} is log-normal under GBM with:

        sigma_adj = sigma * sqrt((n+1)(2n+1) / (6n^2))
        r_adj     = 0.5*(r - q - sigma^2/2) + 0.5*sigma_adj^2 + q

    The geometric Asian call price equals BS(S0, K, T, r_eff, q_eff, sigma_adj)
    where the effective rates are chosen so E[disc * G] = forward of Asian.

    Returns
    -------
    dict with keys: price, delta, gamma, vega, theta, rho
    """
    n = n_averaging
    # Adjusted volatility (exact for discrete geometric average)
    sigma_adj = sigma * np.sqrt((n + 1) * (2 * n + 1) / (6 * n**2))

    # Drift of log G under risk-neutral measure (for n equally-spaced fixing dates)
    mu = r - q - 0.5 * sigma**2
    mu_adj = 0.5 * mu * (n + 1) / n + 0.5 * sigma_adj**2

    # Forward of G: S_0 * exp(mu_adj * T).  To use the BS formula with
    # discount rate r (not mu_adj+q), set q_adj so that r - q_adj = mu_adj.
    q_adj = r - mu_adj

    from mgreeks.utils import bs_price, bs_greeks

    price = bs_price(S0, K, r, q_adj, sigma_adj, T, "call")
    greeks = bs_greeks(S0, K, r, q_adj, sigma_adj, T, "call")

    return {"price": price, **greeks}
