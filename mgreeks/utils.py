"""
Numerical utilities for Monte Carlo Greek estimation.

Contents
--------
bs_price        : Black-Scholes call/put price
bs_greeks       : Black-Scholes analytic Greeks (delta, gamma, vega, rho, theta)
convergence_plot: plot std_error vs sqrt(n_paths) curve
relative_error  : percentage error table helper
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Black-Scholes analytics
# ---------------------------------------------------------------------------

def bs_price(
    S0: float, K: float, r: float, q: float, sigma: float, T: float,
    option: str = "call",
) -> float:
    """Black-Scholes price for a European call or put."""
    from scipy.stats import norm
    if T <= 0 or sigma <= 0:
        if option == "call":
            return max(S0 * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
        else:
            return max(K * np.exp(-r * T) - S0 * np.exp(-q * T), 0.0)
    d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    disc = np.exp(-r * T)
    fwd = S0 * np.exp(-q * T)
    if option == "call":
        return fwd * norm.cdf(d1) - K * disc * norm.cdf(d2)
    else:
        return K * disc * norm.cdf(-d2) - fwd * norm.cdf(-d1)


def bs_greeks(
    S0: float, K: float, r: float, q: float, sigma: float, T: float,
    option: str = "call",
) -> dict:
    """
    Analytic Black-Scholes Greeks.

    Returns
    -------
    dict with keys: delta, gamma, vega, rho, theta
    """
    from scipy.stats import norm
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    disc = np.exp(-r * T)
    fwd = S0 * np.exp(-q * T)
    phi_d1 = norm.pdf(d1)
    Nd1 = norm.cdf(d1)
    Nd2 = norm.cdf(d2)

    gamma = np.exp(-q * T) * phi_d1 / (S0 * sigma * sqrt_T)
    vega = fwd * phi_d1 * sqrt_T

    if option == "call":
        delta = np.exp(-q * T) * Nd1
        rho = K * T * disc * Nd2
        theta = (
            -np.exp(-q * T) * S0 * phi_d1 * sigma / (2 * sqrt_T)
            - r * K * disc * Nd2
            + q * fwd * Nd1
        )
    else:
        delta = -np.exp(-q * T) * norm.cdf(-d1)
        rho = -K * T * disc * norm.cdf(-d2)
        theta = (
            -np.exp(-q * T) * S0 * phi_d1 * sigma / (2 * sqrt_T)
            + r * K * disc * norm.cdf(-d2)
            - q * fwd * norm.cdf(-d1)
        )

    return {
        "delta": float(delta),
        "gamma": float(gamma),
        "vega": float(vega),
        "rho": float(rho),
        "theta": float(theta),
    }


# ---------------------------------------------------------------------------
# Confidence interval
# ---------------------------------------------------------------------------

def confidence_interval(
    samples: np.ndarray, disc: float = 1.0, alpha: float = 0.95
) -> dict:
    """95% CI from raw (undiscounted) samples."""
    n = len(samples)
    vals = disc * samples
    mean = float(vals.mean())
    stderr = float(vals.std(ddof=1) / np.sqrt(n))
    z = 1.959964
    return {
        "estimate": mean,
        "std_error": stderr,
        "ci_lower": mean - z * stderr,
        "ci_upper": mean + z * stderr,
    }


# ---------------------------------------------------------------------------
# Relative error table
# ---------------------------------------------------------------------------

def relative_error(estimate: float, truth: float) -> float:
    """Signed relative error in percent: (estimate - truth) / |truth| * 100."""
    if truth == 0:
        return float("nan")
    return (estimate - truth) / abs(truth) * 100.0


def print_greek_table(results: dict, truth: dict, title: str = "") -> None:
    """
    Print a comparison table of MC estimates vs analytic Greeks.

    Parameters
    ----------
    results : dict of greek_name → result dict (from MonteCarloEngine.all_greeks)
    truth   : dict of greek_name → float (from bs_greeks)
    title   : header string
    """
    if title:
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")
    header = f"{'Greek':<8} {'Estimate':>12} {'Std Err':>10} {'Truth':>12} {'Rel Err%':>10}"
    print(header)
    print("-" * len(header))
    for name in ("delta", "gamma", "vega", "rho", "theta"):
        if name not in results or name not in truth:
            continue
        est = results[name].get("greek", float("nan"))
        se = results[name].get("std_error", float("nan"))
        tr = truth[name]
        re = relative_error(est, tr)
        print(f"{name:<8} {est:>12.6f} {se:>10.6f} {tr:>12.6f} {re:>9.2f}%")


# ---------------------------------------------------------------------------
# Convergence diagnostics
# ---------------------------------------------------------------------------

def convergence_table(
    samples: np.ndarray,
    disc: float = 1.0,
    n_levels: int = 6,
) -> list[dict]:
    """
    Show how the estimate and std_error evolve as n increases.

    Parameters
    ----------
    samples  : raw (undiscounted) payoff samples
    disc     : discount factor
    n_levels : number of doubling levels (starts at len(samples) // 2^n_levels)

    Returns
    -------
    list of dicts with keys: n_paths, estimate, std_error
    """
    rows = []
    n_total = len(samples)
    sizes = [n_total // (2 ** (n_levels - i)) for i in range(n_levels + 1)]
    sizes = [s for s in sizes if s >= 100]
    for n in sizes:
        sub = disc * samples[:n]
        mean = float(sub.mean())
        se = float(sub.std(ddof=1) / np.sqrt(n))
        rows.append({"n_paths": n, "estimate": mean, "std_error": se})
    return rows
