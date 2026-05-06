"""
Antithetic variates for Malliavin Greek estimators.

For each path driven by increments ΔW, we also evaluate with −ΔW.
The antithetic path has the same marginal distribution as the original
(GBM log-returns are symmetric around their mean), so the combined
estimator is unbiased.

Variance analysis
-----------------
The combined estimator is:

    Ŷ = (1/2) [f(S⁺) π(S⁺) + f(S⁻) π(S⁻)]

where S⁺ is driven by ΔW and S⁻ by −ΔW.

Var(Ŷ) = (1/4) [Var(X⁺) + Var(X⁻) + 2 Cov(X⁺, X⁻)]
        = (1/2) Var(X) + (1/2) Cov(X⁺, X⁻)           (by symmetry)

Antithetic reduces variance iff Cov(X⁺, X⁻) < 0, i.e., iff
the mapping ΔW ↦ f(S) π(S) is MONOTONE (negatively correlated
with its antithetic counterpart).

Delta weight (odd in W_T):
    π_Δ(W_T) = W_T / (σ S_0 T) is ODD.
    f(S_T(W_T)) is increasing in W_T (call payoff).
    → f·π_Δ ≈ odd × increasing → negatively correlated with antithetic.
    LARGE variance reduction for delta estimator.

Gamma weight (even in W_T up to the −1 correction):
    π_Γ = (W_T(W_T − σT) − T) / (S_0² σ² T²) ≈ W_T² / (S_0² σ² T²).
    f(S_T) is increasing in W_T.
    → f·π_Γ ≈ even × increasing → positively correlated with antithetic.
    LITTLE OR NO variance reduction for gamma.
"""

from __future__ import annotations

import numpy as np


_Z95 = 1.959964


def antithetic_malliavin(
    payoff,
    weight_func,
    paths_pos: np.ndarray,
    paths_neg: np.ndarray,
    increments_pos: np.ndarray,
    increments_neg: np.ndarray,
    discount: float,
) -> dict:
    """
    Antithetic variates for Malliavin Greek estimators.

    Evaluates the payoff and Malliavin weight on both the original paths
    (driven by +ΔW) and their antithetic counterparts (driven by −ΔW),
    then averages.  The estimator is unbiased for any payoff.

    Parameters
    ----------
    payoff       : callable(paths, times) → (n_paths,) payoff array
    weight_func  : callable(paths, increments) → (n_paths,) weight array
    paths_pos    : full paths for +ΔW, shape (n_paths, n_steps+1)
    paths_neg    : full paths for −ΔW, shape (n_paths, n_steps+1)
    increments_pos: Brownian increments for +ΔW, shape (n_paths, n_steps)
    increments_neg: Brownian increments for −ΔW, shape (n_paths, n_steps)
    discount     : e^{-rT}

    Returns
    -------
    dict with keys: value, std_error, ci_lower, ci_upper, n_paths,
                    variance_reduction_ratio (Var_plain / Var_antithetic)
    """
    times = np.linspace(0, 1, paths_pos.shape[1])  # placeholder; payoff uses times

    f_pos = payoff(paths_pos, times)
    f_neg = payoff(paths_neg, times)
    w_pos = weight_func(paths_pos, increments_pos)
    w_neg = weight_func(paths_neg, increments_neg)

    samples_pos = f_pos * w_pos
    samples_neg = f_neg * w_neg
    samples_anti = 0.5 * (samples_pos + samples_neg)

    vals = discount * samples_anti
    mean = float(vals.mean())
    se = float(vals.std(ddof=1) / np.sqrt(len(vals)))

    # Variance-reduction ratio vs plain estimator (using only + paths)
    se_plain = float((discount * samples_pos).std(ddof=1) / np.sqrt(len(samples_pos)))
    vr_ratio = (se_plain / se) ** 2 if se > 0 else float("nan")

    return {
        "value": mean,
        "std_error": se,
        "ci_lower": mean - _Z95 * se,
        "ci_upper": mean + _Z95 * se,
        "n_paths": len(vals),
        "variance_reduction_ratio": vr_ratio,
    }


def simulate_antithetic(model, S0: float, T: float, n_steps: int, n_paths: int,
                         rng=None) -> tuple[dict, dict]:
    """
    Simulate a pair of antithetic path sets under a GBM model.

    Returns (sim_pos, sim_neg) where sim_neg uses −ΔW increments.
    Both dicts have keys: paths, brownian_increments, terminal, times, dt.
    """
    from mgreeks.models.gbm import GeometricBrownianMotion
    if not isinstance(model, GeometricBrownianMotion):
        raise NotImplementedError(
            "simulate_antithetic only supports GeometricBrownianMotion."
        )

    if rng is None:
        rng = np.random.default_rng()

    times, dt = model._make_time_grid(T, n_steps)
    sqrt_dt = np.sqrt(dt)
    drift = (model.r - model.q - 0.5 * model.sigma**2) * dt

    Z = rng.standard_normal((n_paths, n_steps))

    def _build(z):
        dW = sqrt_dt * z
        log_inc = drift + model.sigma * dW
        log_S = np.empty((n_paths, n_steps + 1))
        log_S[:, 0] = np.log(S0)
        np.cumsum(log_inc, axis=1, out=log_S[:, 1:])
        log_S[:, 1:] += np.log(S0)
        paths = np.exp(log_S)
        return {
            "paths": paths,
            "brownian_increments": dW,
            "terminal": paths[:, -1],
            "times": times,
            "dt": dt,
        }

    return _build(Z), _build(-Z)
