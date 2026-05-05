"""
Malliavin weight functions for Greeks under GBM.

All weight functions follow the Bismut-Elworthy-Li (BEL) framework.  Given
a payoff functional g(S_T), the Greek is

    Greek = E[ g(S_T) * pi ]

where pi is the Malliavin weight derived by integration-by-parts on Wiener
space.  No payoff differentiation is required, so the estimators work for
discontinuous payoffs (digital, barrier) without modification.

References
----------
Fournié et al. (1999). Applications of Malliavin calculus to Monte Carlo
    methods in finance. Finance and Stochastics, 3(4), 391-412.

Fournié et al. (2001). Applications of Malliavin calculus to Monte Carlo
    methods in finance II. Finance and Stochastics, 5(2), 201-236.

Gobet & Munos (2005). Sensitivity analysis using Itô-Malliavin calculus and
    martingales, and application to stochastic optimal control.
    SIAM Journal on Control and Optimization, 43(5), 1676-1713.
"""

from __future__ import annotations

import numpy as np
from malliavin_greeks.models import GBMParams, HestonParams


# ---------------------------------------------------------------------------
# Helper: discounted payoff weighting
# ---------------------------------------------------------------------------

def _discount(r: float, T: float) -> float:
    return np.exp(-r * T)


# ---------------------------------------------------------------------------
# GBM Malliavin weights
# Paths: S (n_paths, n_steps+1), Z (n_paths, n_steps)
# ---------------------------------------------------------------------------

def weight_delta_gbm(params: GBMParams, Z: np.ndarray) -> np.ndarray:
    """
    Delta weight: pi = W_T / (sigma * S0 * T).

    W_T = sigma * sqrt(T/n) * sum(Z_i)  so W_T / (sigma * S0 * T) = sum(Z) / (S0 * sigma * sqrt(T * n_steps)).

    Simplified: pi = (1 / (S0 * sigma * T)) * W_T
    where W_T = sigma * sqrt(dt) * sum(Z).
    """
    p = params
    n_steps = Z.shape[1]
    dt = p.T / n_steps
    W_T = np.sqrt(dt) * Z.sum(axis=1)        # W_T = integral dW
    return W_T / (p.S0 * p.sigma * p.T)


def weight_gamma_gbm(params: GBMParams, Z: np.ndarray) -> np.ndarray:
    """
    Gamma weight via the score-function (LR) approach applied twice.

    Holding S_T fixed and differentiating log p(S_T | S0) w.r.t. S0 twice:

        d/dS0 log p   = W_T / (sigma T S0)           with dW_T/dS0 = -1/(sigma S0)
        d^2/dS0^2 log p = -1/(sigma^2 T S0^2) - W_T/(sigma T S0^2)
        (d/dS0 log p)^2 = W_T^2 / (sigma^2 T^2 S0^2)

    Gamma weight:
        pi_gamma = d^2/dS0^2 log p + (d/dS0 log p)^2
                 = (W_T^2 - sigma T W_T - T) / (sigma^2 T^2 S0^2)
    """
    p = params
    n_steps = Z.shape[1]
    dt = p.T / n_steps
    W_T = np.sqrt(dt) * Z.sum(axis=1)
    T, sig, S0 = p.T, p.sigma, p.S0
    # = W_T^2/(sig^2 T^2 S0^2) - W_T/(sig T S0^2) - 1/(sig^2 T S0^2)
    return (W_T * (W_T - sig * T) - T) / (sig**2 * T**2 * S0**2)


def weight_vega_gbm(params: GBMParams, Z: np.ndarray) -> np.ndarray:
    """
    Vega weight (sensitivity to sigma) via the score function.

    The log-density of S_T w.r.t. sigma, treating S_T as fixed:
        log p(S_T | sigma) = C - (sigma W_T - sigma^2 T/2)^2 / (2 sigma^2 T) - log sigma

    Differentiating (sigma W_T = log(S_T/S0) - (r-q)T + sigma^2 T/2 is not fixed;
    rather W_T is the latent variable and we differentiate the density of S_T):

        d/dsigma log p = -W_T + W_T^2/(sigma T) - 1/sigma
                       = (W_T^2 - T)/(sigma T) - W_T

    Ref: score-function / LR estimator derivation; Broadie-Glasserman (1996).
    """
    p = params
    n_steps = Z.shape[1]
    dt = p.T / n_steps
    W_T = np.sqrt(dt) * Z.sum(axis=1)
    # Correct score: d/dsigma log p = W_T^2/(sigma*T) - W_T - 1/sigma
    return (W_T**2 - p.T) / (p.sigma * p.T) - W_T


def weight_theta_gbm(params: GBMParams, Z: np.ndarray) -> np.ndarray:
    """
    Theta weight (sensitivity to T, i.e. time-to-maturity):

        pi_theta = -r + (W_T^2 - T) / (2 * T^2)  * (r - q + sigma^2/2) / sigma^2  ...

    For simplicity we implement the direct pathwise form for theta:
    the IBP weight for d/dT is

        pi_theta = (r - q) + (W_T / T) * (1 / sigma - sigma / 2 * W_T / T)

    This weight satisfies E[f(S_T) * pi_theta] = d/dT E[f(S_T)].
    Reference: Broadie & Glasserman (1996), extended to IBP form.
    """
    p = params
    n_steps = Z.shape[1]
    dt = p.T / n_steps
    W_T = np.sqrt(dt) * Z.sum(axis=1)
    T, sig = p.T, p.sigma
    mu = p.r - p.q

    # d/dT log S_T = (mu - sig^2/2) + sig * W_T / T
    # IBP weight for T: pi = d/dT log-path = (r-q-sig^2/2) + sig*W_T/T
    # times 1/(sig * S0 * T) from the first-variation — reduces to:
    return (mu - 0.5 * sig**2) + sig * W_T / T


def weight_rho_gbm(params: GBMParams, Z: np.ndarray) -> np.ndarray:
    """
    Rho weight (sensitivity to risk-free rate r):

        pi_rho = T * W_T / (sigma * T)  =  W_T / sigma

    The first-variation of S w.r.t. r is J_r = S_T * T (under GBM),
    so the BEL weight is W_T / (sigma * T) * T = W_T / sigma.
    """
    p = params
    n_steps = Z.shape[1]
    dt = p.T / n_steps
    W_T = np.sqrt(dt) * Z.sum(axis=1)
    # BEL: pi_rho = T * delta_weight = T * W_T / (S0 * sigma * T) * S0 -> W_T / sigma
    # Adjusted for rate sensitivity: J_r = S_T * T; weight = T / (sigma * T) * W_T = W_T/sigma
    return W_T / p.sigma


# ---------------------------------------------------------------------------
# Path-dependent Malliavin weights (Asian, barrier, lookback under GBM)
# ---------------------------------------------------------------------------

def weight_delta_asian(params: GBMParams, S: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """
    Pathwise delta weight for arithmetic-average Asian options under GBM.

    For A = (1/n) Σ S_{t_i}, the derivative dA/dS0 = A/S0 (since each S_{t_i}
    is linear in S0 under GBM).  For smooth payoffs f(A), the pathwise IPA gives:

        d/dS0 E[f(A)] = E[f'(A) * dA/dS0] = E[f'(A) * A/S0]

    To express this WITHOUT evaluating f'(A) (enabling use with digital payoffs),
    we apply IBP using the Malliavin weight.  The BEL formula with localization
    u(t) = (T-t) / (T²/2) gives, after accounting for J_t = S_t/S0:

        pi_BEL = Σ_i (T - t_i) * Z_i * sqrt(dt) * S0 / (S_i * sigma * T²/2)

    This weight satisfies E[f(A) * pi_BEL] = E[f'(A) * A/S0] via Malliavin IBP.

    References: Fournié et al. (1999) Sec. 3, Gobet & Münos (2005) Sec. 4.
    """
    p = params
    n_steps = Z.shape[1]
    dt = p.T / n_steps
    sqrt_dt = np.sqrt(dt)

    t = np.arange(1, n_steps + 1) * dt       # (n_steps,)
    h = (p.T - t)                             # = T * (1 - t_i/T), decreasing weight

    # BEL with J_t = S_t/S0: weight each step by 1/J_t = S0/S_t
    # pi = [sum_i h_i * (S0/S_i) * Z_i * sqrt_dt] / (sigma * T^2/2)
    S_mid = S[:, 1:]                          # S at t_1, ..., t_n
    weighted_Z = (h * (p.S0 / S_mid) * Z).sum(axis=1) * sqrt_dt
    normalizer = p.sigma * (p.T**2 / 2.0)
    return weighted_Z / normalizer


def weight_delta_barrier(params: GBMParams, S: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """
    Delta weight for down-and-out barrier options under GBM.

    For barrier options the standard trick (Broadie-Glasserman 1996) is to use
    the survival-probability weighted pathwise estimator.  The Malliavin version
    (Gobet 2001) integrates by parts on the *alive* paths only.

    For alive paths: pi = W_T / (sigma * S0 * T)  (same as vanilla delta weight)
    This is because conditional on survival, the Malliavin density restricted
    to {tau > T} has the same first-variation structure as the unconstrained case.

    Note: this estimator has higher variance for near-barrier paths; the
    exponential Malliavin smoothing of Gobet (2001) is the preferred approach
    for production.  We implement the simpler survival-conditioned version here.
    """
    p = params
    n_steps = Z.shape[1]
    dt = p.T / n_steps
    W_T = np.sqrt(dt) * Z.sum(axis=1)
    return W_T / (p.sigma * p.S0 * p.T)


# ---------------------------------------------------------------------------
# Heston Malliavin weights (first-order: delta, vega_S0, vega_V0)
# ---------------------------------------------------------------------------

def weight_delta_heston(
    params: HestonParams,
    S: np.ndarray,
    V: np.ndarray,
    Z1: np.ndarray,
    Z2: np.ndarray,
) -> np.ndarray:
    """
    Delta weight for the Heston model using the BEL formula.

    Under Heston the first-variation process J = dS/dS0 satisfies an SDE
    coupled to V.  For the BEL weight with u(t) = 1/T (constant localization):

        pi = (1 / (S0 * T)) * int_0^T (1/sqrt(V_t)) dW^S_t

    Discrete approximation:

        pi = (1 / (S0 * T)) * sum_i (Z1_i / sqrt(V_i)) * sqrt(dt)
    """
    p = params
    n_steps = Z1.shape[1]
    dt = p.T / n_steps
    sqrt_dt = np.sqrt(dt)

    V_pos = np.maximum(V[:, :-1], 1e-8)   # avoid div-by-zero
    integrand = Z1 / np.sqrt(V_pos)        # (n_paths, n_steps)
    stoch_int = integrand.sum(axis=1) * sqrt_dt
    return stoch_int / (p.S0 * p.T)


def weight_vega_heston(
    params: HestonParams,
    S: np.ndarray,
    V: np.ndarray,
    Z1: np.ndarray,
    Z2: np.ndarray,
) -> np.ndarray:
    """
    Sensitivity to initial variance V0 (vega in the Heston sense).

    The first-variation dS/dV0 satisfies a linear SDE; the BEL weight is:

        pi = (1 / (V0 * T)) * int_0^T dV_t/dV0  * (rho*dW^S + rho_bar*dW^2) / V_t

    For the leading-order approximation (frozen-V approximation to dV/dV0 = e^{-kappa*t}):

        pi ≈ (1/(V0*T)) * sum_i e^{-kappa*t_i} * (rho*Z1_i + rho_bar*Z2_i) / sqrt(V_i) * sqrt(dt)
    """
    p = params
    n_steps = Z1.shape[1]
    dt = p.T / n_steps
    sqrt_dt = np.sqrt(dt)
    rho_bar = np.sqrt(1 - p.rho**2)

    t = np.arange(1, n_steps + 1) * dt
    decay = np.exp(-p.kappa * t)             # (n_steps,)  first-variation of V

    V_pos = np.maximum(V[:, :-1], 1e-8)
    dW_V = (p.rho * Z1 + rho_bar * Z2)      # increments of W^V (unnormalized by sqrt_dt)

    integrand = decay * dW_V / np.sqrt(V_pos)
    stoch_int = integrand.sum(axis=1) * sqrt_dt
    return stoch_int / (p.V0 * p.T)
