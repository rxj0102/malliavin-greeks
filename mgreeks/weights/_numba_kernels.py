"""
Numba-accelerated inner loops for Malliavin weight computation.

Falls back to equivalent pure-NumPy implementations when numba is not
installed — call sites need not check availability.
"""

from __future__ import annotations

import numpy as np

try:
    from numba import njit as _njit
    _NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover
    _NUMBA_AVAILABLE = False

    def _njit(*args, **kwargs):  # noqa: E303
        def _decorator(fn):
            return fn
        return _decorator


# ---------------------------------------------------------------------------
# BEL stochastic integral for CEV / local-vol models
# ---------------------------------------------------------------------------
# The inner loop iterates over n_steps time steps for ALL paths in parallel.
# Vectorised over paths (axis 0), sequential over time steps (axis 1).
#
# Inputs (all pre-computed by the caller from model._sigma_func):
#   paths               : (n_paths, n_steps+1)
#   brownian_increments : (n_paths, n_steps)
#   sigma_loc           : (n_paths, n_steps)   — σ_loc(t_i, S_{t_i})
#   sigma_loc_deriv     : (n_paths, n_steps)   — ∂σ_loc/∂S at (t_i, S_{t_i})
#   dt                  : scalar time-step size
#   T                   : scalar maturity
#
# Output: weight array (n_paths,)


@_njit(cache=True)
def _bel_integral_kernel(
    paths: np.ndarray,
    brownian_increments: np.ndarray,
    sigma_loc: np.ndarray,
    sigma_loc_deriv: np.ndarray,
    dt: float,
    T: float,
) -> np.ndarray:
    """
    Euler approximation of the BEL stochastic integral for local-vol models.

    Computes (1/T) Σ_{i=0}^{n-1} [Y_{t_i} / σ_diff(S_{t_i})] · ΔW_i

    where σ_diff(S) = σ_loc(t,S)·S and Y_{t_i} = ∂S_{t_i}/∂S_0.

    Y is propagated via the log-Euler variational update:
        Y_{i+1} = Y_i · (S_{i+1}/S_i) · exp(S_i·∂σ_loc/∂S·ΔW_i
                                              − 0.5·(S_i·∂σ_loc/∂S)²·dt)
    """
    n_paths = paths.shape[0]
    n_steps = brownian_increments.shape[1]

    weight = np.zeros(n_paths)
    Y = np.ones(n_paths)

    for i in range(n_steps):
        for p in range(n_paths):
            S_i = paths[p, i]
            sig = sigma_loc[p, i]
            dsig = sigma_loc_deriv[p, i]
            dW = brownian_increments[p, i]

            sigma_diff = sig * S_i
            integrand = Y[p] / sigma_diff
            weight[p] += integrand * dW

            # First-variation update
            ratio = paths[p, i + 1] / S_i
            corr = S_i * dsig
            Y[p] *= ratio * np.exp(corr * dW - 0.5 * corr * corr * dt)

    for p in range(n_paths):
        weight[p] /= T

    return weight


def bel_integral_local_vol(
    paths: np.ndarray,
    brownian_increments: np.ndarray,
    sigma_loc: np.ndarray,
    sigma_loc_deriv: np.ndarray,
    dt: float,
    T: float,
) -> np.ndarray:
    """
    BEL stochastic integral for local-vol models, JIT-compiled when numba is
    available and falling back to vectorised NumPy otherwise.

    Parameters
    ----------
    paths               : (n_paths, n_steps+1)
    brownian_increments : (n_paths, n_steps)
    sigma_loc           : (n_paths, n_steps)  — pre-computed σ_loc(t_i, S_i)
    sigma_loc_deriv     : (n_paths, n_steps)  — pre-computed ∂σ_loc/∂S
    dt                  : uniform time step
    T                   : maturity

    Returns
    -------
    weight : (n_paths,)
    """
    if _NUMBA_AVAILABLE:
        return _bel_integral_kernel(
            paths, brownian_increments,
            sigma_loc, sigma_loc_deriv,
            dt, T,
        )
    return _bel_integral_numpy(
        paths, brownian_increments,
        sigma_loc, sigma_loc_deriv,
        dt, T,
    )


def _bel_integral_numpy(
    paths: np.ndarray,
    brownian_increments: np.ndarray,
    sigma_loc: np.ndarray,
    sigma_loc_deriv: np.ndarray,
    dt: float,
    T: float,
) -> np.ndarray:
    """Pure-NumPy vectorised fallback (no per-step Python loop)."""
    n_steps = brownian_increments.shape[1]
    weight = np.zeros(paths.shape[0])
    Y = np.ones(paths.shape[0])

    for i in range(n_steps):
        S_i = paths[:, i]
        sig = sigma_loc[:, i]
        dsig = sigma_loc_deriv[:, i]
        dW = brownian_increments[:, i]

        sigma_diff = sig * S_i
        weight += (Y / sigma_diff) * dW

        ratio = paths[:, i + 1] / S_i
        corr = S_i * dsig
        Y *= ratio * np.exp(corr * dW - 0.5 * corr**2 * dt)

    return weight / T


# ---------------------------------------------------------------------------
# Heston weight inner loop
# ---------------------------------------------------------------------------

@_njit(cache=True)
def _heston_delta_kernel(
    paths_V: np.ndarray,
    brownian_increments_S: np.ndarray,
    S0: float,
    T: float,
) -> np.ndarray:
    """
    π_Δ^{Heston} = (1/(S_0 T)) Σ_i (1/√V_{t_i}) · ΔW^S_i

    Clamps V_{t_i} ≥ 1e-8 to avoid division by zero near the zero boundary.
    """
    n_paths = paths_V.shape[0]
    n_steps = brownian_increments_S.shape[1]
    weight = np.zeros(n_paths)

    for p in range(n_paths):
        acc = 0.0
        for i in range(n_steps):
            v = paths_V[p, i]
            if v < 1e-8:
                v = 1e-8
            acc += brownian_increments_S[p, i] / np.sqrt(v)
        weight[p] = acc / (S0 * T)

    return weight


def heston_delta_weight_jit(
    paths_V: np.ndarray,
    brownian_increments_S: np.ndarray,
    S0: float,
    T: float,
) -> np.ndarray:
    """
    JIT-compiled Heston delta weight (falls back to vectorised NumPy).

    Parameters
    ----------
    paths_V             : variance paths, shape (n_paths, n_steps+1)
    brownian_increments_S : spot BM increments, shape (n_paths, n_steps)
    S0                  : initial spot
    T                   : maturity

    Returns
    -------
    weight : (n_paths,)
    """
    if _NUMBA_AVAILABLE:
        return _heston_delta_kernel(paths_V[:, :-1], brownian_increments_S, S0, T)
    V_mid = np.maximum(paths_V[:, :-1], 1e-8)
    return np.sum(brownian_increments_S / np.sqrt(V_mid), axis=1) / (S0 * T)
