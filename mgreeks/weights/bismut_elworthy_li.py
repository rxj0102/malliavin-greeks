"""
The Bismut-Elworthy-Li (BEL) formula — the unifying framework.

THE BEL FORMULA
---------------
For a diffusion dX_t = b(X_t)dt + σ(X_t)dW_t with X_0 = x,

    ∂/∂x E[f(X_T)] = (1/T) E[f(X_T) · ∫_0^T (Y_s / σ(X_s)) dW_s]

where Y_s = ∂X_s/∂x is the FIRST-VARIATION PROCESS satisfying

    dY_s = b'(X_s) Y_s ds + σ'(X_s) Y_s dW_s,   Y_0 = 1.

The stochastic integral ∫_0^T (Y_s/σ(X_s)) dW_s is the Malliavin weight
(up to a localisation function which can be any u with ∫u=1).

WHY THIS MATTERS
----------------
The BEL formula:
  · gives a SYSTEMATIC recipe for the weight given any diffusion
  · requires no knowledge of f — works for discontinuous payoffs
  · reduces to the score-function formula under log-normal models
  · generalises to higher-order Greeks via iterated integration by parts

GBM SPECIAL CASE
-----------------
For GBM: σ(S) = σ·S, Y_s = S_s/S_0, so

    Y_s / σ(S_s) = (S_s/S_0) / (σ·S_s) = 1/(σ·S_0)    (constant in s and ω!)

    ∫_0^T dW_s / (σ·S_0) = W_T / (σ·S_0)

    BEL weight = W_T / (σ·S_0·T)  ✓  (matches score-function delta weight)

This simplification — the integrand is path-independent — is unique to GBM.
For local-vol and stochastic-vol models, the integrand varies along the path
and the full stochastic integral must be computed numerically.

NUMERICAL BEL INTEGRATION
--------------------------
Discretise the Itô integral as

    ∫_0^T (Y_s/σ(X_s)) dW_s ≈ Σ_{i=0}^{n-1} (Y_{t_i}/σ(X_{t_i})) · ΔW_i

This is an Euler–Maruyama approximation of the stochastic integral.

For GBM the sum collapses to W_T/(σS_0) in closed form; for other models
a genuine loop over time steps is needed.

References
----------
Bismut, J.-M. (1984), "Large deviations and the Malliavin calculus."
Elworthy, K.D. & Li, X.-M. (1994), "Formulae for the derivatives of heat
semigroups." J. Funct. Anal. 125, 252–286.
Fournié et al. (1999), Finance and Stochastics 3, 391–412.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from mgreeks.models.base import StochasticModel


# ---------------------------------------------------------------------------
# BEL delta weight
# ---------------------------------------------------------------------------

def bel_delta_weight(
    paths: np.ndarray,
    brownian_increments: np.ndarray,
    model: "StochasticModel",
    S0: float,
    T: float,
    times: np.ndarray,
    *,
    localisation: str = "uniform",
) -> np.ndarray:
    """
    BEL delta weight for a general diffusion model.

        π_Δ = (1/T) Σ_{i=0}^{n-1} [Y_{t_i} / σ(X_{t_i})] · ΔW_i

    where Y_{t_i} = ∂X_{t_i}/∂x is the first-variation process evaluated at t_i.

    For GBM this sum equals W_T/(σ S_0 T) exactly (without the loop), so GBM
    is handled analytically.  For LocalVolModel, the numerically integrated BEL
    weight is returned.  Other models raise NotImplementedError.

    Parameters
    ----------
    paths               : simulated spot paths, shape (n_paths, n_steps+1)
    brownian_increments : ΔW_i, shape (n_paths, n_steps)
    model               : stochastic model
    S0                  : initial spot
    T                   : maturity
    times               : time grid, shape (n_steps+1,)
    localisation        : 'uniform' (constant weight u=1/T) or 'end'
                          (Dirac mass at T — reduces to score function)

    Returns
    -------
    Weight array, shape (n_paths,)
    """
    from mgreeks.models.gbm import GeometricBrownianMotion
    from mgreeks.models.local_vol import LocalVolModel

    if isinstance(model, GeometricBrownianMotion):
        return _bel_delta_gbm(paths, brownian_increments, model, S0, T)

    if isinstance(model, LocalVolModel):
        return _bel_delta_local_vol(paths, brownian_increments, model, S0, T, times)

    raise NotImplementedError(
        f"BEL delta weight not implemented for {type(model).__name__}. "
        "Supported: GeometricBrownianMotion, LocalVolModel."
    )


def _bel_delta_gbm(
    paths: np.ndarray,
    brownian_increments: np.ndarray,
    model: "GeometricBrownianMotion",
    S0: float,
    T: float,
) -> np.ndarray:
    """
    Analytic BEL weight for GBM — collapses to W_T/(σ S_0 T).

    The integrand Y_s/σ(S_s) = (S_s/S_0)/(σ S_s) = 1/(σ S_0) is constant,
    so ∫_0^T 1/(σ S_0) dW_s = W_T/(σ S_0) and dividing by T gives the
    standard delta weight.
    """
    W_T = brownian_increments.sum(axis=1)
    return W_T / (model.sigma * S0 * T)


def _bel_delta_local_vol(
    paths: np.ndarray,
    brownian_increments: np.ndarray,
    model: "LocalVolModel",
    S0: float,
    T: float,
    times: np.ndarray,
) -> np.ndarray:
    """
    Numerical BEL weight for LocalVolModel via Euler stochastic integration.

    Y_{t_i} = ∂S_{t_i}/∂S_0 satisfies the variational SDE:

        dY_t = (b'(S_t) + σ'(S_t) dW_t) Y_t dt

    For log-Euler scheme:
        Y_{t_{i+1}} ≈ Y_{t_i} · (S_{t_{i+1}} / S_{t_i})
                        · exp(−σ'(S_{t_i}) σ(S_{t_i}) S_{t_i} dt
                              + σ'(S_{t_i}) S_{t_i} ΔW_i)

    Here σ'(S) is the derivative of the LOCAL DIFFUSION (= S·σ_loc) w.r.t. S,
    which equals σ_loc(S) + S·∂σ_loc/∂S.

    The integrand in the BEL formula is Y_{t_i} / σ_local_diff(S_{t_i})
    where σ_local_diff(S) = σ_loc(t,S)·S is the (multiplicative) diffusion.

    The inner stochastic-integral loop is JIT-compiled via numba when available;
    falls back to a vectorised NumPy loop otherwise.  The pre-computation of
    σ_loc and ∂σ_loc/∂S (Python calls to model._sigma_func) remains outside
    the JIT region since those calls are model-specific Python objects.
    """
    from mgreeks.weights._numba_kernels import bel_integral_local_vol

    n_steps = len(times) - 1
    dt = times[1] - times[0]

    # Pre-compute σ_loc and ∂σ_loc/∂S over all paths × steps outside JIT
    sigma_loc_arr = np.empty((paths.shape[0], n_steps))
    sigma_loc_deriv_arr = np.empty_like(sigma_loc_arr)
    for i in range(n_steps):
        t_i = times[i]
        sigma_loc_arr[:, i] = model._sigma_func(t_i, paths[:, i])
        sigma_loc_deriv_arr[:, i] = model._sigma_deriv_func(t_i, paths[:, i])

    return bel_integral_local_vol(
        paths, brownian_increments,
        sigma_loc_arr, sigma_loc_deriv_arr,
        dt, T,
    )


# ---------------------------------------------------------------------------
# BEL vega weight
# ---------------------------------------------------------------------------

def bel_vega_weight(
    paths: np.ndarray,
    brownian_increments: np.ndarray,
    model: "StochasticModel",
    S0: float,
    T: float,
    times: np.ndarray,
) -> np.ndarray:
    """
    BEL-type weight for Vega (sensitivity to volatility parameter).

    The vega derivative ∂/∂σ E[f(X_T)] requires differentiating the SDE
    itself w.r.t. σ.  Under GBM this gives (via score function):

        π_v = (W_T² − T)/(σT) − W_T

    For general models, the weight involves the derivative of the SDE
    w.r.t. σ (a separate variational equation in the vol parameter).
    This is implemented only for GBM here.

    Returns
    -------
    Weight array, shape (n_paths,)
    """
    from mgreeks.models.gbm import GeometricBrownianMotion
    from mgreeks.weights.malliavin_weights import vega_weight_gbm

    if isinstance(model, GeometricBrownianMotion):
        W_T = brownian_increments.sum(axis=1)
        return vega_weight_gbm(S0, None, model.sigma, model.r, model.q, T, W_T)

    raise NotImplementedError(
        f"BEL vega weight not implemented for {type(model).__name__}."
    )


# ---------------------------------------------------------------------------
# BEL second-order weight
# ---------------------------------------------------------------------------

def bel_second_order_weight(
    paths: np.ndarray,
    brownian_increments: np.ndarray,
    model: "StochasticModel",
    S0: float,
    T: float,
    times: np.ndarray,
    greek_type: str = "gamma",
) -> np.ndarray:
    """
    BEL formula for second-order Greeks via double integration by parts.

    For Gamma (∂²/∂S_0²):

        π_Γ = [W_T(W_T − σT) − T] / (S_0² σ² T²)    [GBM]

    The double IBP involves:
      1. A first stochastic integral to remove the first derivative of f
      2. A second stochastic integral (or Skorokhod term) to remove f'

    Currently implemented for GBM only.

    Parameters
    ----------
    greek_type : 'gamma' (∂²/∂S²), 'vanna' (∂²/∂S ∂σ), 'volga' (∂²/∂σ²)

    Returns
    -------
    Weight array, shape (n_paths,)
    """
    from mgreeks.models.gbm import GeometricBrownianMotion
    from mgreeks.weights.malliavin_weights import (
        gamma_weight_gbm, vega_weight_gbm, delta_weight_gbm,
    )

    if not isinstance(model, GeometricBrownianMotion):
        raise NotImplementedError(
            f"BEL second-order weights not implemented for {type(model).__name__}."
        )

    W_T = brownian_increments.sum(axis=1)
    sigma, r, q = model.sigma, model.r, model.q

    if greek_type == "gamma":
        return gamma_weight_gbm(S0, None, sigma, r, q, T, W_T)

    if greek_type == "vanna":
        # Vanna = ∂²V/∂S_0 ∂σ = ∂(Vega)/∂S_0 = ∂(Delta)/∂σ
        # Via the product of score functions:
        # π_vanna = π_Δ · π_v − ∂(π_Δ)/∂σ
        # For GBM: ∂(W_T/(S_0 σ T))/∂σ = −W_T/(S_0 σ² T) (∂W_T/∂σ contributes too)
        pi_delta = delta_weight_gbm(S0, None, sigma, r, q, T, W_T)
        pi_vega = vega_weight_gbm(S0, None, sigma, r, q, T, W_T)
        # Cross-derivative correction: ∂/∂σ [W_T/(σS_0T)] where W_T depends on σ
        # = [(∂W_T/∂σ)*σS_0T − W_T*S_0T] / (σS_0T)²
        # ∂W_T/∂σ = T − W_T/σ (as in vega derivation)
        dpi_delta_dsigma = ((T - W_T / sigma) * sigma - W_T) / (sigma**2 * S0 * T)
        return pi_delta * pi_vega - dpi_delta_dsigma

    if greek_type == "volga":
        # Volga = ∂²V/∂σ²
        # π_volga = π_v² + ∂²/∂σ² log p (second score for σ)
        # Second score: −(∂W_T/∂σ)² − W_T · ∂²W_T/∂σ² − 1/σ²... complex
        # Implement via product of first scores minus correction (Fournié et al.):
        pi_vega = vega_weight_gbm(S0, None, sigma, r, q, T, W_T)
        # ∂W_T/∂σ = T − W_T/σ
        dW_T_dsig = T - W_T / sigma
        # ∂²W_T/∂σ² = W_T/σ²
        d2W_T_dsig2 = W_T / sigma**2
        # ∂²/∂σ² log p = −(dW_T/dσ)² − W_T·d²W_T/dσ² + 1/σ²... from −W_T²/2 + ln σ:
        # d/dσ (−W_T²/2) = −W_T·dW_T/dσ
        # d²/dσ² (−W_T²/2) = −(dW_T/dσ)² − W_T·d²W_T/dσ²
        # d/dσ (−log σ − ½ log T) = −1/σ  → d²/dσ² = 1/σ²
        second_score_sig = -(dW_T_dsig)**2 - W_T * d2W_T_dsig2 + 1.0 / sigma**2
        return pi_vega**2 + second_score_sig

    raise ValueError(f"Unknown greek_type {greek_type!r}. Use 'gamma', 'vanna', or 'volga'.")


# ---------------------------------------------------------------------------
# Consistency check utility
# ---------------------------------------------------------------------------

def verify_bel_equals_score(
    paths: np.ndarray,
    brownian_increments: np.ndarray,
    model: "StochasticModel",
    S0: float,
    T: float,
    times: np.ndarray,
) -> dict:
    """
    Verify that the BEL delta weight matches the score-function delta weight.

    For GBM these must agree exactly (both equal W_T/(σ S_0 T)).
    Returns a dict with 'bel', 'score', 'max_abs_diff', 'agree'.
    """
    from mgreeks.models.gbm import GeometricBrownianMotion
    from mgreeks.weights.malliavin_weights import delta_weight_gbm

    bel = bel_delta_weight(paths, brownian_increments, model, S0, T, times)

    if isinstance(model, GeometricBrownianMotion):
        W_T = brownian_increments.sum(axis=1)
        score = delta_weight_gbm(S0, paths[:, -1], model.sigma, model.r, model.q, T, W_T)
    else:
        raise NotImplementedError("Score weight only available for GBM comparison.")

    max_diff = float(np.abs(bel - score).max())
    return {
        "bel": bel,
        "score": score,
        "max_abs_diff": max_diff,
        "agree": max_diff < 1e-10,
    }
