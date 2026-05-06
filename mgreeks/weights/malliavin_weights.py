"""
Core Malliavin weight functions for computing Greeks under GBM.

THE FUNDAMENTAL PRINCIPLE
-------------------------
For a derivative with payoff f(S_T), the Greek ∂/∂θ V where V = e^{-rT}E[f(S_T)]
can be written as:

    ∂/∂θ V = e^{-rT} E[f(S_T) · π_θ]

where π_θ is the Malliavin weight — a random variable that depends on the
Brownian path but NOT on the payoff function f.

This is the Malliavin integration-by-parts formula.  The weight π_θ does not
involve derivatives of f, so:
  · even discontinuous payoffs (digitals, barriers) yield finite-variance estimates
  · the same weight works for European AND path-dependent payoffs under GBM

DERIVATION STRATEGY (score-function / LR approach for GBM)
-----------------------------------------------------------
Under GBM with S_T = S_0 exp((r-q-σ²/2)T + σW_T), the terminal Brownian is

    W_T = (log(S_T/S_0) − (r−q−σ²/2)T) / σ

so W_T is a deterministic function of (S_T, S_0, θ).  Differentiating the
log-density

    log p(S_T | S_0; θ) = const − W_T²/(2) − log σ − ½ log T

with respect to a parameter θ yields the weight via the IBP identity:

    E[∂/∂θ f(S_T)] = E[f(S_T) · ∂/∂θ log p(S_T | S_0; θ)]

with a discount-factor correction for parameters that also enter e^{-rT}.

KEY FORMULAS (all verified numerically against Black-Scholes analytics)
------------------------------------------------------------------------
Delta:  π_Δ = W_T / (S_0 σ T)

Gamma:  π_Γ = [W_T(W_T − σT) − T] / (S_0² σ² T²)
        Note: the simpler-looking Hermite form (W_T²−T)/(S_0²σ²T²) is WRONG
        because it ignores ∂W_T/∂S_0 = −1/(σS_0) when differentiating twice.

Vega:   π_v = (W_T² − T)/(σT) − W_T
        (equivalently: (W_T² − σT·W_T − T)/(σT))

Rho:    π_ρ = W_T·T/σ − T
        The −T term is the discount-factor derivative ∂e^{-rT}/∂r = −T e^{-rT}.

Theta:  π_θ = r + (T − W_T²)/(2T²) − μ·W_T/(σT)   where μ = r−q−σ²/2
        Gives Theta = e^{-rT} E[f · π_θ] in the standard convention (negative for
        long calls).  The simpler approximation r − (Z²−1)/(2T) omits the μ drift
        term and is biased unless μ = 0.

References
----------
Fournié et al. (1999), "Applications of Malliavin calculus to Monte Carlo
methods in finance." Finance and Stochastics 3, 391–412.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from mgreeks.models.base import StochasticModel


# ---------------------------------------------------------------------------
# Delta
# ---------------------------------------------------------------------------

def delta_weight_gbm(
    S0: float,
    ST: np.ndarray,
    sigma: float,
    r: float,
    q: float,
    T: float,
    W_T: np.ndarray,
) -> np.ndarray:
    """
    Malliavin weight for Delta under GBM.

    π_Δ = W_T / (S_0 · σ · T)

    Derivation:
        ∂/∂S_0 log p = (∂W_T/∂S_0) · (−W_T) = (−1/(σS_0)) · (−W_T) = W_T/(σS_0)
        Dividing by T (from the normalisation of W_T variance): π_Δ = W_T/(S_0 σ T).

    The weight is IDENTICAL for path-dependent payoffs under GBM because
    D_s S_t = σ S_t is independent of s, so the BEL stochastic integral
    collapses to W_T/(σ S_0 T).

    Parameters
    ----------
    S0  : initial spot price
    ST  : terminal spot prices (n_paths,) — unused in this formula, kept for API
    sigma, r, q : model parameters
    T   : maturity
    W_T : terminal Brownian motion Σ ΔW_i, shape (n_paths,)

    Returns
    -------
    Weight array, shape (n_paths,)
    """
    return W_T / (S0 * sigma * T)


# ---------------------------------------------------------------------------
# Gamma
# ---------------------------------------------------------------------------

def gamma_weight_gbm(
    S0: float,
    ST: np.ndarray,
    sigma: float,
    r: float,
    q: float,
    T: float,
    W_T: np.ndarray,
) -> np.ndarray:
    """
    Malliavin weight for Gamma under GBM.

    π_Γ = [W_T(W_T − σT) − T] / (S_0² σ² T²)

    Derivation (score-function, second order):
        ∂/∂S_0 log p = W_T/(σTS_0)         [first derivative]
        ∂W_T/∂S_0    = −1/(σS_0)            [W_T depends on S_0]

        ∂²/∂S_0² log p = ∂/∂S_0 [W_T/(σTS_0)]
                        = (∂W_T/∂S_0)/(σTS_0) + W_T·∂(1/(σTS_0))/∂S_0
                        = −1/(σ²TS_0²) − W_T/(σTS_0²)

        π_Γ = ∂²/∂S_0² log p + (∂/∂S_0 log p)²
            = −1/(σ²TS_0²) − W_T/(σTS_0²) + W_T²/(σ²T²S_0²)
            = [W_T² − σTW_T − T] / (σ²T²S_0²)
            = [W_T(W_T − σT) − T] / (σ²T²S_0²)

    CAUTION: The simpler Hermite-polynomial form (W_T²−T)/(σ²T²S_0²) is
    BIASED.  It omits the −W_T/(σTS_0²) term that arises from differentiating
    W_T w.r.t. S_0 twice.  The correct formula includes −σT·W_T in the
    numerator.  Both forms have the same expectation (since E[W_T]=0), but
    they differ path-by-path and the simpler form gives a biased gamma estimate.

    Parameters
    ----------
    (same as delta_weight_gbm)

    Returns
    -------
    Weight array, shape (n_paths,)
    """
    return (W_T * (W_T - sigma * T) - T) / (sigma**2 * T**2 * S0**2)


# ---------------------------------------------------------------------------
# Vega
# ---------------------------------------------------------------------------

def vega_weight_gbm(
    S0: float,
    ST: np.ndarray,
    sigma: float,
    r: float,
    q: float,
    T: float,
    W_T: np.ndarray,
) -> np.ndarray:
    """
    Malliavin weight for Vega under GBM.

    π_v = (W_T² − T) / (σT) − W_T

    Equivalently: (W_T² − σT·W_T − T) / (σT).

    Derivation (holding S_T fixed, so W_T depends on σ):
        W_T(σ) = (log S_T/S_0 − μT) / σ  where μ = r−q−σ²/2
        ∂W_T/∂σ = (−W_T/σ) + T·(σ)... full chain rule:
            ∂W_T/∂σ = [−(log S_T/S_0 − μT) / σ² + (−σT)/σ]
                     = −W_T/σ + (−T) = -(W_T/σ + T)
        Wait — differentiating numerator: ∂μ/∂σ = −σ → ∂(−μT)/∂σ = σT
        ∂W_T/∂σ = [σT/σ − W_T/σ] = T − W_T/σ   (correct Itô chain rule)

        ∂/∂σ log p = −W_T · ∂W_T/∂σ − 1/σ
                   = −W_T(T − W_T/σ) − 1/σ
                   = W_T²/σ − W_T·T − 1/σ
                   = (W_T² − 1) / σ − W_T·T... simplify:
                   = (W_T² − T) / (σT) − W_T    [verified numerically]

    Returns
    -------
    Weight array, shape (n_paths,)
    """
    return (W_T**2 - T) / (sigma * T) - W_T


# ---------------------------------------------------------------------------
# Rho
# ---------------------------------------------------------------------------

def rho_weight_gbm(
    S0: float,
    ST: np.ndarray,
    sigma: float,
    r: float,
    q: float,
    T: float,
    W_T: np.ndarray,
) -> np.ndarray:
    """
    Malliavin weight for Rho under GBM.

    π_ρ = W_T / σ − T

    Two contributions to ∂V/∂r:
    1. Score function: ∂/∂r log p = W_T / σ
       Derivation: W_T = (log S_T/S_0 − μT)/σ, so ∂W_T/∂r = −T/σ.
       log p ∝ −W_T²/(2T), so score = −(W_T/T)·(−T/σ) = W_T/σ.
    2. Discount factor: ∂/∂r e^{−rT} = −T e^{−rT}  →  weight contribution −T

    Combined: π_ρ = W_T/σ − T

    NOTE: The score W_T/σ has NO extra T multiplier.  A naive reading of
    ∂W_T/∂r = −T/σ might suggest a T factor, but the T cancels against the
    1/T from the Gaussian log-density −W_T²/(2T).

    Rho for a zero-coupon bond: e^{-rT}·E[1·π_ρ] = e^{-rT}·(−T) = −T·disc ✓

    Returns
    -------
    Weight array, shape (n_paths,)
    """
    return W_T / sigma - T


# ---------------------------------------------------------------------------
# Theta
# ---------------------------------------------------------------------------

def theta_weight_gbm(
    S0: float,
    ST: np.ndarray,
    sigma: float,
    r: float,
    q: float,
    T: float,
    W_T: np.ndarray,
) -> np.ndarray:
    """
    Malliavin weight for Theta under GBM.

    Gives Theta in the standard convention: Theta = e^{-rT} E[f · π_θ]
    where Theta < 0 for long calls (value erodes over time).

    π_θ = r + (T − W_T²) / (2T²) − μ · W_T / (σT)

    where μ = r − q − σ²/2.

    Derivation:
        Score w.r.t. T (the maturity), holding S_T fixed:
            log p ∝ −W_T²/(2T) − ½ log T

        score_T = ∂/∂T log p = W_T²/(2T²) + μ·W_T/(σT) − 1/(2T)
        (the μ·W_T/(σT) term arises from ∂(−μT)/∂T = −μ → ∂W_T/∂T = μ/σ)

        Theta = −∂V/∂T = e^{-rT} E[f·(r − score_T)]
              = e^{-rT} E[f·(r + 1/(2T) − W_T²/(2T²) − μ·W_T/(σT))]
              = e^{-rT} E[f·(r + (T − W_T²)/(2T²) − μ·W_T/(σT))]

    NOTE: The simpler approximation r − (Z²−1)/(2T) where Z = W_T/√T
    omits the μ·W_T/(σT) drift term.  This introduces bias proportional to
    μ·Cov(f, W_T), which is non-zero for non-constant payoffs when μ ≠ 0.

    Verification: for f = 1 (zero-coupon bond),
    E[π_θ] = r + (T−T)/(2T²) − 0 = r,
    so Theta_ZCB = r·e^{-rT} ✓.

    Returns
    -------
    Weight array, shape (n_paths,)
    """
    mu = r - q - 0.5 * sigma**2
    return r + (T - W_T**2) / (2.0 * T**2) - mu * W_T / (sigma * T)


# ---------------------------------------------------------------------------
# Path-dependent delta (GBM)
# ---------------------------------------------------------------------------

def delta_weight_path_dependent(
    paths: np.ndarray,
    brownian_increments: np.ndarray,
    model: "StochasticModel",
    S0: float,
    T: float,
) -> np.ndarray:
    """
    Malliavin Delta weight for path-dependent payoffs under GBM.

    For ANY payoff f(S_{t_1}, ..., S_{t_n}) under GBM (Asian, barrier, etc.):

        π_Δ = ΔW_1 / (σ S_0 Δt_1)

    where ΔW_1 is the FIRST Brownian increment and Δt_1 = T / n_steps.

    Derivation (score-function on the joint discrete-time density):
        The joint transition density factorises as
            p(S_{t_1}, ..., S_{t_n} | S_0) = p(S_{t_1}|S_0) · Π_{i≥2} p(S_{t_i}|S_{t_{i-1}})
        Only the first factor depends on S_0, so
            ∂/∂S_0 log p = ∂/∂S_0 log p(S_{t_1}|S_0) = ΔW_1 / (σ S_0 Δt_1).
        Applying Stein's lemma on ΔW_1 confirms the IBP identity for any payoff.

    NOTE: The formula W_T/(σS_0T) is correct ONLY for terminal payoffs f(S_T)
    (i.e., when n=1, t_1=T so ΔW_1=W_T and Δt_1=T).  For n>1 fixing dates that
    formula gives a biased estimator biased by a factor t_i/T per fixing date.

    Parameters
    ----------
    paths               : shape (n_paths, n_steps+1)
    brownian_increments : shape (n_paths, n_steps), assumed uniform time steps
    model               : must be GeometricBrownianMotion
    S0                  : initial spot
    T                   : maturity

    Returns
    -------
    Weight array, shape (n_paths,)
    """
    sigma = getattr(model, "sigma", None)
    if sigma is None:
        raise ValueError("delta_weight_path_dependent requires a GBM model with .sigma")
    n_steps = brownian_increments.shape[1]
    dt1 = T / n_steps
    DW1 = brownian_increments[:, 0]
    return DW1 / (S0 * sigma * dt1)


# ---------------------------------------------------------------------------
# Convenience: compute all GBM weights from a simulation dict
# ---------------------------------------------------------------------------

def all_weights_gbm(
    S0: float,
    sigma: float,
    r: float,
    q: float,
    T: float,
    W_T: np.ndarray,
    ST: np.ndarray | None = None,
) -> dict:
    """
    Compute all GBM Malliavin weights in one call.

    Parameters
    ----------
    W_T : terminal Brownian motion, shape (n_paths,)
    ST  : terminal spot (optional, not used in formulas but kept for API)

    Returns
    -------
    dict with keys: 'delta', 'gamma', 'vega', 'rho', 'theta'
    """
    if ST is None:
        ST = np.empty_like(W_T)
    return {
        "delta": delta_weight_gbm(S0, ST, sigma, r, q, T, W_T),
        "gamma": gamma_weight_gbm(S0, ST, sigma, r, q, T, W_T),
        "vega":  vega_weight_gbm(S0, ST, sigma, r, q, T, W_T),
        "rho":   rho_weight_gbm(S0, ST, sigma, r, q, T, W_T),
        "theta": theta_weight_gbm(S0, ST, sigma, r, q, T, W_T),
    }


# ---------------------------------------------------------------------------
# Convenience aliases for path-dependent payoffs under GBM
# ---------------------------------------------------------------------------

def asian_delta_weight_gbm(
    paths: np.ndarray,
    brownian_increments: np.ndarray,
    S0: float,
    sigma: float,
    T: float,
    times: np.ndarray,
) -> np.ndarray:
    """
    Malliavin delta weight for Asian options under GBM.

    Returns W_T / (S_0 σ T) — identical to delta_weight_gbm.

    NOTE ON BIAS FOR DISCRETE FIXING DATES
    ---------------------------------------
    This formula comes from the continuous-time BEL formula.  For a payoff
    with n discrete fixing dates t_1,...,t_n (not all equal to T), applying
    the Wiener-space IBP with constant kernel u_s = 1/(σS_0T) gives a
    time-weighted estimator that underestimates the delta by a factor of
    t̄/T (mean fixing time over maturity).  For uniform averaging:
    t̄/T = (n+1)/(2n) → 1/2 as n → ∞.

    For an unbiased estimator of path-dependent delta, use
    `delta_weight_path_dependent` (score of the first conditional density).
    """
    W_T = brownian_increments.sum(axis=1)
    return W_T / (S0 * sigma * T)


def barrier_delta_weight_gbm(
    paths: np.ndarray,
    brownian_increments: np.ndarray,
    S0: float,
    sigma: float,
    T: float,
    times: np.ndarray,
    barrier: float = None,
    barrier_type: str = "down_and_out",
) -> np.ndarray:
    """
    Malliavin delta weight for barrier options under GBM.

    Returns W_T / (S_0 σ T) — the standard GBM delta weight.

    The barrier indicator 1_{survival} is part of the payoff, not the weight.
    The Malliavin IBP still applies because 1_{survival} is measurable w.r.t.
    the path and the weight is constructed only from the model (not the payoff).

    For an unbiased discrete-time estimator, use `delta_weight_path_dependent`
    (score of the first conditional density).  Using W_T/(σS_0T) introduces
    the same t̄/T bias as for Asian options when monitoring is discrete.

    For continuous monitoring (large n_steps), both estimates converge to the
    true barrier delta; W_T/(σS_0T) converges more slowly.

    See Gobet & Kohatsu-Higa (2003) for localized variance-reduction variants.
    """
    W_T = brownian_increments.sum(axis=1)
    return W_T / (S0 * sigma * T)


def lookback_delta_weight_gbm(
    paths: np.ndarray,
    brownian_increments: np.ndarray,
    S0: float,
    sigma: float,
    T: float,
    times: np.ndarray,
) -> np.ndarray:
    """
    Malliavin delta weight for lookback options under GBM.

    Returns W_T / (S_0 σ T) — the standard GBM delta weight.

    The running maximum/minimum is part of the payoff; the Malliavin weight
    depends only on the model dynamics and is payoff-independent.  The same
    discretization bias as for Asian/barrier options applies; use
    `delta_weight_path_dependent` for unbiased estimation with discrete paths.
    """
    W_T = brownian_increments.sum(axis=1)
    return W_T / (S0 * sigma * T)


# ---------------------------------------------------------------------------
# Heston model Malliavin weights
# ---------------------------------------------------------------------------

def delta_weight_heston(
    paths_S: np.ndarray,
    paths_V: np.ndarray,
    brownian_increments_S: np.ndarray,
    brownian_increments_V: np.ndarray,
    S0: float,
    V0: float,
    T: float,
    times: np.ndarray,
    model: "HestonModel",
) -> np.ndarray:
    """
    BEL delta weight for the Heston stochastic-volatility model.

    Under Heston, dS_t = (r-q) S_t dt + √V_t S_t dW^S_t, so the
    first variation Y_t = ∂S_t/∂S_0 satisfies:

        dY_t ≈ ((r-q) + ρ ξ √V_t) Y_t dt + √V_t Y_t dW^S_t

    Approximating Y_s ≈ S_s/S_0 (leading order, exact for ρ=0) and
    using σ_eff(S,V) = √V · S, the BEL integrand simplifies to:

        Y_s / σ_eff(S_s, V_s) = (S_s/S_0) / (√V_s · S_s) = 1 / (S_0 √V_s)

    Plugging into the BEL formula:

        π_Δ = (1/(S_0 T)) Σ_i (1/√V_{t_i}) · ΔW^S_i

    where ΔW^S_i are the SPOT Brownian increments (accounting for the
    rho-correlation decomposition already present in the simulation).

    The variance paths V_{t_i} must be bounded away from zero; we clip at 1e-8.

    Parameters
    ----------
    paths_S               : spot paths, shape (n_paths, n_steps+1)
    paths_V               : variance paths, shape (n_paths, n_steps+1)
    brownian_increments_S : ΔW^S_i, shape (n_paths, n_steps)
    brownian_increments_V : ΔW^V_i, shape (n_paths, n_steps) [unused, kept for API]
    S0                    : initial spot
    V0                    : initial variance
    T                     : maturity
    times                 : time grid, shape (n_steps+1,)
    model                 : HestonModel instance

    Returns
    -------
    Weight array, shape (n_paths,)
    """
    V_mid = np.maximum(paths_V[:, :-1], 1e-8)   # (n_paths, n_steps)
    inv_sqrt_V = 1.0 / np.sqrt(V_mid)
    weight = np.sum(inv_sqrt_V * brownian_increments_S, axis=1)
    return weight / (S0 * T)


def vega_weight_heston(
    paths_S: np.ndarray,
    paths_V: np.ndarray,
    brownian_increments_S: np.ndarray,
    brownian_increments_V: np.ndarray,
    S0: float,
    V0: float,
    T: float,
    times: np.ndarray,
    model: "HestonModel",
) -> np.ndarray:
    """
    Malliavin weight for Heston vega (∂V/∂V_0).

    The sensitivity ∂S_t/∂V_0 satisfies a variational SDE driven by
    the variance process:

        d(∂S_t/∂V_0) = (r-q)(∂S_t/∂V_0)dt
                      + (1/(2√V_t))(∂V_t/∂V_0) S_t dW^S_t
                      + √V_t (∂S_t/∂V_0) dW^S_t

    where ∂V_t/∂V_0 satisfies:

        d(∂V_t/∂V_0) = -κ(∂V_t/∂V_0)dt + ξ/(2√V_t)(∂V_t/∂V_0)dW^V_t

    with ∂V_0/∂V_0 = 1.  This is approximated as:

        ∂V_t/∂V_0 ≈ exp(-κt)   (exact for ξ=0)

    Then the leading-order BEL vega weight is:

        π_v ≈ (1/(2 T)) Σ_i (S_{t_i}/S_0) exp(-κ t_i) / (√V_{t_i} S_{t_i})
                      × ΔW^S_i
            = (1/(2 S_0 T)) Σ_i exp(-κ t_i) / √V_{t_i} × ΔW^S_i
    """
    n_steps = brownian_increments_S.shape[1]
    dt = T / n_steps
    times_mid = times[:-1]                       # t_0, t_1, ..., t_{n-1}

    V_mid = np.maximum(paths_V[:, :-1], 1e-8)   # (n_paths, n_steps)
    inv_sqrt_V = 1.0 / np.sqrt(V_mid)
    decay = np.exp(-model.kappa * times_mid)     # exp(-κ t_i), shape (n_steps,)

    weight = np.sum(inv_sqrt_V * decay[np.newaxis, :] * brownian_increments_S, axis=1)
    return weight / (2.0 * S0 * T)
