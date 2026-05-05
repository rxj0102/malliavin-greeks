"""
Systematic variance comparison across estimators.

The key metric is the *variance of the Greek estimator* (proportional to
Monte Carlo cost for a given accuracy target).  We report:

    - Point estimate
    - Standard error
    - Variance (var of individual samples, not of the mean)
    - Relative efficiency = Var(reference) / Var(estimator)
    - Bias (where analytic value is available)

All tables are returned as plain dicts for easy downstream processing
(pandas, JSON, LaTeX).
"""

from __future__ import annotations

import time
import numpy as np
from typing import Callable, Optional
from malliavin_greeks.models import GBMParams
from malliavin_greeks.estimators import (
    GreekResult,
    malliavin_delta, malliavin_gamma, malliavin_vega,
    fd_delta_central, fd_gamma_central, fd_vega_central,
    pathwise_delta, pathwise_vega,
    lr_delta, lr_gamma,
)
from malliavin_greeks.variance import cv_delta_call, cv_delta_put


# ---------------------------------------------------------------------------
# Black-Scholes analytics (benchmark truth)
# ---------------------------------------------------------------------------

def bs_greeks(params: GBMParams, K: float, option: str = "call") -> dict:
    """Closed-form BS Greeks for European vanilla options."""
    from scipy.stats import norm
    p = params
    d1 = (np.log(p.S0 / K) + (p.r - p.q + 0.5 * p.sigma**2) * p.T) / (p.sigma * np.sqrt(p.T))
    d2 = d1 - p.sigma * np.sqrt(p.T)
    disc = np.exp(-p.r * p.T)
    df_q = np.exp(-p.q * p.T)

    if option == "call":
        price = p.S0 * df_q * norm.cdf(d1) - K * disc * norm.cdf(d2)
        delta = df_q * norm.cdf(d1)
        rho = K * p.T * disc * norm.cdf(d2)
    else:
        price = K * disc * norm.cdf(-d2) - p.S0 * df_q * norm.cdf(-d1)
        delta = -df_q * norm.cdf(-d1)
        rho = -K * p.T * disc * norm.cdf(-d2)

    gamma = df_q * norm.pdf(d1) / (p.S0 * p.sigma * np.sqrt(p.T))
    vega = p.S0 * df_q * norm.pdf(d1) * np.sqrt(p.T)
    theta = (
        -p.S0 * df_q * norm.pdf(d1) * p.sigma / (2 * np.sqrt(p.T))
        - p.r * K * disc * (norm.cdf(d2) if option == "call" else norm.cdf(-d2))
        + p.q * p.S0 * df_q * (norm.cdf(d1) if option == "call" else norm.cdf(-d1))
    )

    return dict(price=price, delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)


# ---------------------------------------------------------------------------
# Single-estimator benchmark runner
# ---------------------------------------------------------------------------

def run_estimator(
    name: str,
    estimator_fn: Callable[[], GreekResult],
    truth: Optional[float] = None,
) -> dict:
    t0 = time.perf_counter()
    result = estimator_fn()
    elapsed = time.perf_counter() - t0
    out = dict(
        name=name,
        greek=result.greek,
        std_err=result.std_err,
        variance=result.variance,
        n_paths=result.n_paths,
        wall_time_s=elapsed,
    )
    if truth is not None:
        out["bias"] = result.greek - truth
        out["rmse"] = np.sqrt(result.variance / result.n_paths + (result.greek - truth)**2)
    return out


# ---------------------------------------------------------------------------
# Delta benchmark: Malliavin vs FD vs pathwise vs LR
# ---------------------------------------------------------------------------

def benchmark_delta(
    params: GBMParams,
    K: float,
    n_paths: int = 100_000,
    n_steps: int = 1,
    h_fd: float = 1.0,
    option: str = "call",
    seed: int = 42,
) -> list[dict]:
    """
    Compare delta estimators for a European call/put.

    Returns a list of benchmark result dicts, one per estimator.
    """
    from malliavin_greeks.payoffs import european_call, european_put
    from malliavin_greeks.estimators import malliavin_rho

    payoff = european_call if option == "call" else european_put
    truth = bs_greeks(params, K, option)["delta"]

    rng_base = np.random.default_rng(seed)

    results = []

    results.append(run_estimator(
        "Malliavin",
        lambda: malliavin_delta(params, payoff, n_paths, n_steps,
                                 np.random.default_rng(seed), antithetic=True),
        truth,
    ))
    results.append(run_estimator(
        "Malliavin+CV",
        lambda: cv_delta_call(params, K, n_paths, n_steps,
                               np.random.default_rng(seed), antithetic=True)
               if option == "call"
               else cv_delta_put(params, K, n_paths, n_steps,
                                  np.random.default_rng(seed), antithetic=True),
        truth,
    ))
    results.append(run_estimator(
        "FD Central",
        lambda: fd_delta_central(params, payoff, n_paths, n_steps, h_fd,
                                   np.random.default_rng(seed)),
        truth,
    ))
    # Pathwise delta (derivative of payoff)
    if option == "call":
        deriv = lambda S: (S[:, -1] > K).astype(float)
    else:
        deriv = lambda S: -(S[:, -1] < K).astype(float)

    results.append(run_estimator(
        "Pathwise (IPA)",
        lambda: pathwise_delta(params, deriv, n_paths, n_steps,
                                np.random.default_rng(seed), antithetic=True),
        truth,
    ))
    results.append(run_estimator(
        "Likelihood Ratio",
        lambda: lr_delta(params, payoff, n_paths, n_steps,
                          np.random.default_rng(seed), antithetic=True),
        truth,
    ))

    # Compute relative efficiency vs Malliavin
    ref_var = results[0]["variance"]
    for r in results:
        r["rel_efficiency"] = ref_var / r["variance"] if r["variance"] > 0 else np.inf

    return results


def benchmark_gamma(
    params: GBMParams,
    K: float,
    n_paths: int = 100_000,
    n_steps: int = 1,
    h_fd: float = 1.0,
    option: str = "call",
    seed: int = 42,
) -> list[dict]:
    """Compare gamma estimators: Malliavin vs FD central vs LR."""
    from malliavin_greeks.payoffs import european_call, european_put
    payoff = european_call if option == "call" else european_put
    truth = bs_greeks(params, K, option)["gamma"]

    results = []
    results.append(run_estimator(
        "Malliavin",
        lambda: malliavin_gamma(params, payoff, n_paths, n_steps,
                                 np.random.default_rng(seed), antithetic=True),
        truth,
    ))
    results.append(run_estimator(
        "FD Central",
        lambda: fd_gamma_central(params, payoff, n_paths, n_steps, h_fd,
                                   np.random.default_rng(seed)),
        truth,
    ))
    results.append(run_estimator(
        "LR Gamma",
        lambda: lr_gamma(params, payoff, n_paths, n_steps,
                          np.random.default_rng(seed), antithetic=True),
        truth,
    ))

    ref_var = results[0]["variance"]
    for r in results:
        r["rel_efficiency"] = ref_var / r["variance"] if r["variance"] > 0 else np.inf
    return results


def benchmark_vega(
    params: GBMParams,
    K: float,
    n_paths: int = 100_000,
    n_steps: int = 1,
    h_fd: float = 0.001,
    option: str = "call",
    seed: int = 42,
) -> list[dict]:
    """Compare vega estimators: Malliavin vs FD vs pathwise."""
    from malliavin_greeks.payoffs import european_call, european_put
    payoff = european_call if option == "call" else european_put
    truth = bs_greeks(params, K, option)["vega"]

    if option == "call":
        deriv = lambda S: (S[:, -1] > K).astype(float) * S[:, -1]
    else:
        deriv = lambda S: -(S[:, -1] < K).astype(float) * S[:, -1]

    results = []
    results.append(run_estimator(
        "Malliavin",
        lambda: malliavin_vega(params, payoff, n_paths, n_steps,
                                np.random.default_rng(seed), antithetic=True),
        truth,
    ))
    results.append(run_estimator(
        "FD Central",
        lambda: fd_vega_central(params, payoff, n_paths, n_steps, h_fd,
                                  np.random.default_rng(seed)),
        truth,
    ))
    results.append(run_estimator(
        "Pathwise",
        lambda: pathwise_vega(params, deriv, n_paths, n_steps,
                               np.random.default_rng(seed), antithetic=True),
        truth,
    ))

    ref_var = results[0]["variance"]
    for r in results:
        r["rel_efficiency"] = ref_var / r["variance"] if r["variance"] > 0 else np.inf
    return results


# ---------------------------------------------------------------------------
# Digital option benchmark (discontinuous payoff — key use case)
# ---------------------------------------------------------------------------

def benchmark_digital_delta(
    params: GBMParams,
    K: float,
    n_paths: int = 100_000,
    n_steps: int = 1,
    h_fd: float = 1.0,
    seed: int = 42,
) -> list[dict]:
    """
    Delta of a digital call: key test for discontinuous payoffs.

    BS analytic digital delta: d/dS0 [e^{-rT} N(d2)] = e^{-rT} N'(d2) * d(d2)/dS0
    """
    from scipy.stats import norm
    from malliavin_greeks.payoffs import digital_call
    from malliavin_greeks.variance import is_delta_digital
    from malliavin_greeks.estimators import lr_delta

    p = params
    d1 = (np.log(p.S0 / K) + (p.r - p.q + 0.5 * p.sigma**2) * p.T) / (p.sigma * np.sqrt(p.T))
    d2 = d1 - p.sigma * np.sqrt(p.T)
    truth = np.exp(-p.r * p.T) * norm.pdf(d2) / (p.S0 * p.sigma * np.sqrt(p.T))

    results = []
    results.append(run_estimator(
        "Malliavin",
        lambda: malliavin_delta(params, digital_call, n_paths, n_steps,
                                 np.random.default_rng(seed), antithetic=True),
        truth,
    ))
    results.append(run_estimator(
        "IS+Malliavin",
        lambda: is_delta_digital(params, K, n_paths, n_steps, np.random.default_rng(seed)),
        truth,
    ))
    results.append(run_estimator(
        "FD Central",
        lambda: fd_delta_central(params, digital_call, n_paths, n_steps, h_fd,
                                   np.random.default_rng(seed)),
        truth,
    ))
    results.append(run_estimator(
        "LR (=Malliavin)",
        lambda: lr_delta(params, digital_call, n_paths, n_steps,
                          np.random.default_rng(seed), antithetic=True),
        truth,
    ))

    ref_var = results[0]["variance"]
    for r in results:
        r["rel_efficiency"] = ref_var / r["variance"] if r["variance"] > 0 else np.inf
    return results


# ---------------------------------------------------------------------------
# Moneyness sweep: variance vs moneyness for delta
# ---------------------------------------------------------------------------

def moneyness_sweep(
    params: GBMParams,
    strikes: np.ndarray,
    n_paths: int = 50_000,
    n_steps: int = 1,
    seed: int = 42,
) -> dict:
    """
    For each strike in `strikes`, compute delta variance for Malliavin, FD, pathwise.

    Returns dict of {estimator: array of variances}, shape (len(strikes),).
    """
    from malliavin_greeks.payoffs import european_call

    mall_vars, fd_vars, pw_vars, lr_vars = [], [], [], []
    deriv = lambda S, K=None: (S[:, -1] > K).astype(float)

    for K in strikes:
        truth = bs_greeks(params, K)["delta"]
        r_m = run_estimator(
            "Malliavin",
            lambda K=K: malliavin_delta(params, lambda S: european_call(S, K),
                                        n_paths, n_steps, np.random.default_rng(seed)),
            truth,
        )
        r_fd = run_estimator(
            "FD",
            lambda K=K: fd_delta_central(params, lambda S: european_call(S, K),
                                          n_paths, n_steps, 1.0, np.random.default_rng(seed)),
            truth,
        )
        r_pw = run_estimator(
            "Pathwise",
            lambda K=K: pathwise_delta(params, lambda S: (S[:, -1] > K).astype(float),
                                        n_paths, n_steps, np.random.default_rng(seed)),
            truth,
        )
        r_lr = run_estimator(
            "LR",
            lambda K=K: lr_delta(params, lambda S: european_call(S, K),
                                   n_paths, n_steps, np.random.default_rng(seed)),
            truth,
        )
        mall_vars.append(r_m["variance"])
        fd_vars.append(r_fd["variance"])
        pw_vars.append(r_pw["variance"])
        lr_vars.append(r_lr["variance"])

    return dict(
        strikes=strikes,
        malliavin=np.array(mall_vars),
        fd_central=np.array(fd_vars),
        pathwise=np.array(pw_vars),
        lr=np.array(lr_vars),
    )


# ---------------------------------------------------------------------------
# Pretty-print utilities
# ---------------------------------------------------------------------------

def print_benchmark_table(results: list[dict], title: str = "") -> None:
    if title:
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")
    header = f"{'Estimator':<20} {'Greek':>10} {'Std Err':>10} {'Variance':>12} {'Rel Eff':>10}"
    if "bias" in results[0]:
        header += f" {'Bias':>12}"
    print(header)
    print("-" * len(header))
    for r in results:
        row = (
            f"{r['name']:<20} {r['greek']:>10.5f} {r['std_err']:>10.5f} "
            f"{r['variance']:>12.4e} {r['rel_efficiency']:>10.3f}"
        )
        if "bias" in r:
            row += f" {r['bias']:>12.5f}"
        print(row)
