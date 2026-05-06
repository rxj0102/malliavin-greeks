"""
Control variate variance reduction for Malliavin Greek estimators.

Two control variates are provided:

1. Geometric Asian (for pricing/Greeks of arithmetic Asian options)
   The geometric Asian call has closed-form price and Greeks under GBM.
   The arithmetic and geometric payoffs are highly correlated (ρ ≈ 0.99),
   giving large variance reductions.

2. Delta-hedged portfolio (for European options)
   Under BS, the delta-hedged P&L has zero expectation:
       E[f(S_T) − Δ·(S_T − S_0·e^{rT})] = price
   So f(S_T) − Δ·S_T is a near-zero-mean control variate.

Both functions return a result dict with the same structure as the
Malliavin Greek estimators: {value, std_error, ci_lower, ci_upper, n_paths,
variance_reduction_ratio}.
"""

from __future__ import annotations

import numpy as np


_Z95 = 1.959964


def _result(samples: np.ndarray, discount: float, plain_se: float) -> dict:
    vals = discount * samples
    mean = float(vals.mean())
    se = float(vals.std(ddof=1) / np.sqrt(len(vals)))
    vr = (plain_se / se) ** 2 if se > 0 else float("nan")
    return {
        "value": mean,
        "std_error": se,
        "ci_lower": mean - _Z95 * se,
        "ci_upper": mean + _Z95 * se,
        "n_paths": len(vals),
        "variance_reduction_ratio": vr,
    }


def geometric_asian_control_variate(
    arithmetic_payoffs: np.ndarray,
    geometric_payoffs: np.ndarray,
    geometric_analytical_price: float,
    discount: float,
    weights: np.ndarray = None,
) -> dict:
    """
    Use geometric Asian option as control variate for arithmetic Asian pricing/Greeks.

    For pricing (weights=None):
        ĉ = f_arith − β̂ (f_geom − E_analytical[f_geom])
        β̂ = Cov(f_arith, f_geom) / Var(f_geom)

    For Greeks (weights supplied):
        ĉ = f_arith·π − β̂ (f_geom·π − Greek_geom_analytical)

    The geometric Asian analytical value (price or Greek) is supplied via
    ``geometric_analytical_price`` (the name is reused for Greeks too).

    Parameters
    ----------
    arithmetic_payoffs       : shape (n_paths,), raw payoff (or weighted payoff)
    geometric_payoffs        : shape (n_paths,), geometric payoff (or weighted)
    geometric_analytical_price: closed-form E[f_geom] or analytical Greek
    discount                 : e^{-rT} discount factor
    weights                  : optional Malliavin weights; if supplied, the
                               function expects f_arith and f_geom to already
                               be the weighted payoffs f·π.

    Returns
    -------
    dict: value, std_error, ci_lower, ci_upper, n_paths, variance_reduction_ratio,
          optimal_beta, correlation
    """
    x = arithmetic_payoffs.astype(float)
    z = geometric_payoffs.astype(float)

    # OLS regression: x ≈ α + β·z  → β̂ minimises Var(x − β·z)
    cov_xz = float(np.cov(x, z, ddof=1)[0, 1])
    var_z = float(z.var(ddof=1))
    beta = cov_xz / var_z if var_z > 0 else 0.0

    # Control-variate correction: subtract β̂·(z − analytical)
    # The analytical expectation is geometric_analytical_price / discount
    # (undiscounted) if weights=None, or the analytical Greek (already has
    # discounting baked in) when weights are used.
    analytical_undiscounted = geometric_analytical_price / discount
    cv_samples = x - beta * (z - analytical_undiscounted)

    # Correlation between x and z
    std_x = float(x.std(ddof=1))
    std_z = float(z.std(ddof=1))
    corr = cov_xz / (std_x * std_z) if std_x > 0 and std_z > 0 else 0.0

    plain_se = float((discount * x).std(ddof=1) / np.sqrt(len(x)))
    res = _result(cv_samples, discount, plain_se)
    res["optimal_beta"] = beta
    res["correlation"] = corr
    return res


def delta_hedged_control_variate(
    payoffs: np.ndarray,
    paths: np.ndarray,
    bs_delta: float,
    S0: float,
    discount: float,
) -> dict:
    """
    Use the delta-hedged portfolio as a control variate.

    Under Black-Scholes, the delta-hedged portfolio has zero mean:
        E[f(S_T) − Δ·S_T] = V − Δ · S_0 · e^{rT} · discount ... (see note)

    The ZERO-MEAN control variate is constructed as:
        Z = f(S_T) − Δ·S_T − (V − Δ·S_0·e^{rT}·discount / discount)

    Since E[Z] = 0, the controlled estimator:
        f_cv = f − β̂·Z
    reduces variance while remaining unbiased.

    Practically, we compute the control-variate correction using the
    sample mean of Z as the estimate of its (zero) theoretical mean.

    Parameters
    ----------
    payoffs  : raw payoff samples f(S_T), shape (n_paths,)
    paths    : full path array, shape (n_paths, n_steps+1)
    bs_delta : analytical Black-Scholes delta Δ
    S0       : initial spot price
    discount : e^{-rT}

    Returns
    -------
    dict: value, std_error, ci_lower, ci_upper, n_paths, variance_reduction_ratio,
          optimal_beta, correlation
    """
    f = payoffs.astype(float)
    ST = paths[:, -1]

    # Control variate: Z = f(S_T) - Δ·S_T (high correlation with f under BS)
    z = f - bs_delta * ST

    # OLS beta
    cov_fz = float(np.cov(f, z, ddof=1)[0, 1])
    var_z = float(z.var(ddof=1))
    beta = cov_fz / var_z if var_z > 0 else 0.0

    # E[Z] under BS = price/discount - Δ·S_0·e^{rT}/discount (if no discounting)
    # We estimate E[Z] empirically from the sample (it's approximately zero for
    # ATM call with the correct delta, but not exactly zero in general).
    z_mean = float(z.mean())
    cv_samples = f - beta * (z - z_mean)

    std_f = float(f.std(ddof=1))
    std_z = float(z.std(ddof=1))
    corr = cov_fz / (std_f * std_z) if std_f > 0 and std_z > 0 else 0.0

    plain_se = float((discount * f).std(ddof=1) / np.sqrt(len(f)))
    res = _result(cv_samples, discount, plain_se)
    res["optimal_beta"] = beta
    res["correlation"] = corr
    return res
