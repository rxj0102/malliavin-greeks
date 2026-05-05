"""
Stochastic process simulators.

All simulators return paths of shape (n_paths, n_steps+1) for a scalar asset,
or additional arrays as documented.  The first column is always S_0.

Conventions
-----------
- T       : maturity in years
- n_steps : number of Euler-Maruyama steps (n_steps=1 gives exact GBM)
- rng     : numpy Generator; pass np.random.default_rng(seed) for reproducibility
"""

from __future__ import annotations

import numpy as np
from numpy.random import Generator
from dataclasses import dataclass, field
from typing import Optional


def _default_rng(rng: Optional[Generator]) -> Generator:
    return rng if rng is not None else np.random.default_rng()


# ---------------------------------------------------------------------------
# Geometric Brownian Motion
# ---------------------------------------------------------------------------

@dataclass
class GBMParams:
    S0: float = 100.0
    r: float = 0.05
    q: float = 0.0
    sigma: float = 0.20
    T: float = 1.0


def simulate_gbm(
    params: GBMParams,
    n_paths: int,
    n_steps: int = 1,
    rng: Optional[Generator] = None,
    antithetic: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Exact log-normal paths for GBM.

    Returns
    -------
    S : (n_paths, n_steps+1) asset price paths
    Z : (n_paths, n_steps) standard normals driving the paths
        (antithetic: first half paths use Z, second half use -Z)
    """
    rng = _default_rng(rng)
    p = params
    dt = p.T / n_steps
    mu_dt = (p.r - p.q - 0.5 * p.sigma**2) * dt
    sig_sqrt_dt = p.sigma * np.sqrt(dt)

    if antithetic:
        half = n_paths // 2
        Z_half = rng.standard_normal((half, n_steps))
        Z = np.concatenate([Z_half, -Z_half], axis=0)
    else:
        Z = rng.standard_normal((n_paths, n_steps))

    log_increments = mu_dt + sig_sqrt_dt * Z          # (n_paths, n_steps)
    log_S = np.cumsum(log_increments, axis=1)          # cumulative log-returns
    log_S = np.concatenate([np.zeros((n_paths, 1)), log_S], axis=1)
    S = p.S0 * np.exp(log_S)
    return S, Z


# ---------------------------------------------------------------------------
# Local Volatility (Dupire) — Euler-Maruyama
# ---------------------------------------------------------------------------

@dataclass
class LocalVolParams:
    S0: float = 100.0
    r: float = 0.05
    q: float = 0.0
    T: float = 1.0
    # sigma(t, S) callable; defaults to flat 20%
    sigma_fn: object = field(default_factory=lambda: lambda t, S: 0.20 * np.ones_like(S))


def simulate_local_vol(
    params: LocalVolParams,
    n_paths: int,
    n_steps: int = 100,
    rng: Optional[Generator] = None,
    antithetic: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Euler-Maruyama for dS = (r-q)S dt + sigma(t,S) S dW.

    Returns
    -------
    S    : (n_paths, n_steps+1) paths
    Z    : (n_paths, n_steps) driving normals
    sigmas : (n_paths, n_steps) sigma(t_i, S_i) evaluated on the path
    """
    rng = _default_rng(rng)
    p = params
    dt = p.T / n_steps
    sqrt_dt = np.sqrt(dt)

    if antithetic:
        half = n_paths // 2
        Z_half = rng.standard_normal((half, n_steps))
        Z = np.concatenate([Z_half, -Z_half], axis=0)
    else:
        Z = rng.standard_normal((n_paths, n_steps))

    S = np.empty((n_paths, n_steps + 1))
    sigmas = np.empty((n_paths, n_steps))
    S[:, 0] = p.S0

    for i in range(n_steps):
        t = i * dt
        sig = p.sigma_fn(t, S[:, i])
        sigmas[:, i] = sig
        S[:, i + 1] = S[:, i] * np.exp(
            (p.r - p.q - 0.5 * sig**2) * dt + sig * sqrt_dt * Z[:, i]
        )

    return S, Z, sigmas


# ---------------------------------------------------------------------------
# Heston Stochastic Volatility — Euler-Maruyama (full truncation scheme)
# ---------------------------------------------------------------------------

@dataclass
class HestonParams:
    S0: float = 100.0
    V0: float = 0.04       # initial variance (sigma^2 = 0.04 -> sigma = 0.20)
    r: float = 0.05
    q: float = 0.0
    kappa: float = 2.0     # mean-reversion speed
    theta: float = 0.04    # long-run variance
    xi: float = 0.30       # vol-of-vol
    rho: float = -0.70     # correlation between W^S and W^V
    T: float = 1.0


def simulate_heston(
    params: HestonParams,
    n_paths: int,
    n_steps: int = 200,
    rng: Optional[Generator] = None,
    antithetic: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Full-truncation Euler-Maruyama for the Heston model.

    Returns
    -------
    S  : (n_paths, n_steps+1)
    V  : (n_paths, n_steps+1) variance process (truncated >= 0)
    Z1 : (n_paths, n_steps) normals driving log S
    Z2 : (n_paths, n_steps) independent normals (W^V component)
    """
    rng = _default_rng(rng)
    p = params
    dt = p.T / n_steps
    sqrt_dt = np.sqrt(dt)
    rho_bar = np.sqrt(1.0 - p.rho**2)

    if antithetic:
        half = n_paths // 2
        Z1h = rng.standard_normal((half, n_steps))
        Z2h = rng.standard_normal((half, n_steps))
        Z1 = np.concatenate([Z1h, -Z1h], axis=0)
        Z2 = np.concatenate([Z2h, -Z2h], axis=0)
    else:
        Z1 = rng.standard_normal((n_paths, n_steps))
        Z2 = rng.standard_normal((n_paths, n_steps))

    # Correlated Brownians: dW^S = Z1, dW^V = rho*Z1 + rho_bar*Z2
    W_V = p.rho * Z1 + rho_bar * Z2

    S = np.empty((n_paths, n_steps + 1))
    V = np.empty((n_paths, n_steps + 1))
    S[:, 0] = p.S0
    V[:, 0] = p.V0

    for i in range(n_steps):
        V_pos = np.maximum(V[:, i], 0.0)   # full truncation
        sqrt_V = np.sqrt(V_pos)
        S[:, i + 1] = S[:, i] * np.exp(
            (p.r - p.q - 0.5 * V_pos) * dt + sqrt_V * sqrt_dt * Z1[:, i]
        )
        V[:, i + 1] = (
            V[:, i]
            + p.kappa * (p.theta - V_pos) * dt
            + p.xi * sqrt_V * sqrt_dt * W_V[:, i]
        )

    return S, V, Z1, Z2
