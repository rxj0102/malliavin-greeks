"""
Greek estimators: Malliavin, finite-difference, pathwise, and likelihood-ratio.

All estimators return a dict with keys:
    greek       : point estimate
    std_err     : Monte Carlo standard error
    variance    : sample variance of the estimator (proportional to MC cost)
    n_paths     : number of paths used

Finite-difference estimators require two (or three) simulations; all others
use a single simulation.  This difference is fundamental to the efficiency
argument for the Malliavin approach.

References
----------
Glasserman (2003). Monte Carlo Methods in Financial Engineering. Springer.

Broadie & Glasserman (1996). Estimating security price derivatives using
    simulation. Management Science, 42(2), 269-285.

Fournié et al. (1999). Applications of Malliavin calculus to Monte Carlo
    methods in finance. Finance and Stochastics, 3(4), 391-412.
"""

from __future__ import annotations

import numpy as np
from typing import Callable
from malliavin_greeks.models import GBMParams, simulate_gbm
from malliavin_greeks import weights as W


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

class GreekResult:
    __slots__ = ("greek", "std_err", "variance", "n_paths", "estimator")

    def __init__(self, samples: np.ndarray, estimator: str = "unknown"):
        self.greek = float(samples.mean())
        self.variance = float(samples.var(ddof=1))
        self.std_err = float(np.sqrt(self.variance / len(samples)))
        self.n_paths = len(samples)
        self.estimator = estimator

    def __repr__(self):
        return (
            f"GreekResult(estimator={self.estimator!r}, greek={self.greek:.6f}, "
            f"std_err={self.std_err:.6f}, variance={self.variance:.6f})"
        )


# ---------------------------------------------------------------------------
# Malliavin estimators (GBM)
# ---------------------------------------------------------------------------

def malliavin_delta(
    params: GBMParams,
    payoff_fn: Callable[[np.ndarray], np.ndarray],
    n_paths: int,
    n_steps: int = 1,
    rng=None,
    antithetic: bool = True,
) -> GreekResult:
    """Delta via Malliavin IBP: E[payoff(S) * pi_delta]."""
    S, Z = simulate_gbm(params, n_paths, n_steps, rng, antithetic)
    pi = W.weight_delta_gbm(params, Z)
    disc = np.exp(-params.r * params.T)
    samples = disc * payoff_fn(S) * pi
    return GreekResult(samples, "malliavin_delta")


def malliavin_gamma(
    params: GBMParams,
    payoff_fn: Callable[[np.ndarray], np.ndarray],
    n_paths: int,
    n_steps: int = 1,
    rng=None,
    antithetic: bool = True,
) -> GreekResult:
    """Gamma via Malliavin second IBP: E[payoff(S) * pi_gamma]."""
    S, Z = simulate_gbm(params, n_paths, n_steps, rng, antithetic)
    pi = W.weight_gamma_gbm(params, Z)
    disc = np.exp(-params.r * params.T)
    samples = disc * payoff_fn(S) * pi
    return GreekResult(samples, "malliavin_gamma")


def malliavin_vega(
    params: GBMParams,
    payoff_fn: Callable[[np.ndarray], np.ndarray],
    n_paths: int,
    n_steps: int = 1,
    rng=None,
    antithetic: bool = True,
) -> GreekResult:
    """Vega via Malliavin IBP w.r.t. sigma."""
    S, Z = simulate_gbm(params, n_paths, n_steps, rng, antithetic)
    pi = W.weight_vega_gbm(params, Z)
    disc = np.exp(-params.r * params.T)
    samples = disc * payoff_fn(S) * pi
    return GreekResult(samples, "malliavin_vega")


def malliavin_rho(
    params: GBMParams,
    payoff_fn: Callable[[np.ndarray], np.ndarray],
    n_paths: int,
    n_steps: int = 1,
    rng=None,
    antithetic: bool = True,
) -> GreekResult:
    """Rho via Malliavin IBP w.r.t. r."""
    S, Z = simulate_gbm(params, n_paths, n_steps, rng, antithetic)
    pi = W.weight_rho_gbm(params, Z)
    disc = np.exp(-params.r * params.T)
    # Rho also gets a -T * price contribution from discounting: d/dr [e^{-rT} E[f]] = -T * price + e^{-rT} E[f * pi]
    price = disc * payoff_fn(S)
    samples = -params.T * price + disc * payoff_fn(S) * pi
    return GreekResult(samples, "malliavin_rho")


def malliavin_theta(
    params: GBMParams,
    payoff_fn: Callable[[np.ndarray], np.ndarray],
    n_paths: int,
    n_steps: int = 1,
    rng=None,
    antithetic: bool = True,
) -> GreekResult:
    """Theta (dV/dT) via Malliavin IBP w.r.t. T."""
    S, Z = simulate_gbm(params, n_paths, n_steps, rng, antithetic)
    pi = W.weight_theta_gbm(params, Z)
    disc = np.exp(-params.r * params.T)
    price = payoff_fn(S)
    # d/dT [e^{-rT} E[f(S_T)]] = -r * e^{-rT} E[f] + e^{-rT} E[f * pi_theta]
    samples = disc * price * (pi - params.r)
    return GreekResult(samples, "malliavin_theta")


def pathwise_delta_asian(
    params: GBMParams,
    payoff_deriv_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    n_paths: int,
    n_steps: int = 50,
    rng=None,
    antithetic: bool = True,
) -> GreekResult:
    """
    Pathwise (IPA) delta for arithmetic-average Asian options.

    Since A = (1/n) Σ S_{t_i} is smooth in S0 (A = S0 * A_normalized),
    the pathwise formula applies:

        Delta = e^{-rT} E[f'(A) * dA/dS0] = e^{-rT} E[f'(A) * A/S0]

    payoff_deriv_fn : callable(S, A) -> array of f'(A) values
        For a call: lambda S, A: (A > K).astype(float)
        For a digital: NOT valid (f'(A) is a Dirac delta)
    """
    S, Z = simulate_gbm(params, n_paths, n_steps, rng, antithetic)
    A = S[:, 1:].mean(axis=1)
    dphi = payoff_deriv_fn(S, A)
    disc = np.exp(-params.r * params.T)
    samples = disc * dphi * A / params.S0
    return GreekResult(samples, "pathwise_delta_asian")


def malliavin_delta_asian_digital(
    params: GBMParams,
    K: float,
    n_paths: int,
    n_steps: int = 50,
    rng=None,
    antithetic: bool = True,
) -> GreekResult:
    """
    Malliavin delta for a DIGITAL Asian option via the BEL weight.

    For payoffs f(A) = 1_{A>K} (digital), pathwise IPA fails because f'(A)
    is a Dirac delta.  The BEL weight with stochastic u_t provides an
    unbiased estimator that avoids differentiating the indicator.

    The weight follows from the Clark-Ocone representation of G = dA/dS0:

        G = A/S0 = E[A/S0] + integral_0^T sigma*(T-t)/T * (S_t/S0) dW_t  (r=q=0)

    Combined with the Malliavin IBP (Fournié et al. 2001, Sec. 3), this gives:

        pi = integral_0^T (T-t_i)/(T²/2) * (S0/S_{t_i}) * Z_i * sqrt(dt)

    which satisfies E[f(A) * pi] = E[f'(A) * A/S0] for all f ∈ L².
    """
    S, Z = simulate_gbm(params, n_paths, n_steps, rng, antithetic)
    pi = W.weight_delta_asian(params, S, Z)
    disc = np.exp(-params.r * params.T)
    from malliavin_greeks.payoffs import digital_call
    samples = disc * digital_call(S, K) * pi
    return GreekResult(samples, "malliavin_delta_asian_digital")


# ---------------------------------------------------------------------------
# Finite-difference estimators
# ---------------------------------------------------------------------------

def fd_delta_forward(
    params: GBMParams,
    payoff_fn: Callable[[np.ndarray], np.ndarray],
    n_paths: int,
    n_steps: int = 1,
    h: float = 1.0,
    rng=None,
) -> GreekResult:
    """Forward finite-difference delta: [V(S0+h) - V(S0)] / h."""
    rng = np.random.default_rng(rng)
    seed1, seed2 = rng.integers(0, 2**31, size=2)

    p_up = GBMParams(
        S0=params.S0 + h, r=params.r, q=params.q, sigma=params.sigma, T=params.T
    )
    S_base, _ = simulate_gbm(params, n_paths, n_steps, np.random.default_rng(seed1))
    S_up, _ = simulate_gbm(p_up, n_paths, n_steps, np.random.default_rng(seed1))
    disc = np.exp(-params.r * params.T)
    samples = disc * (payoff_fn(S_up) - payoff_fn(S_base)) / h
    return GreekResult(samples, "fd_delta_forward")


def fd_delta_central(
    params: GBMParams,
    payoff_fn: Callable[[np.ndarray], np.ndarray],
    n_paths: int,
    n_steps: int = 1,
    h: float = 1.0,
    rng=None,
) -> GreekResult:
    """Central finite-difference delta: [V(S0+h) - V(S0-h)] / (2h)."""
    rng = np.random.default_rng(rng)
    seed = int(rng.integers(0, 2**31))

    p_up = GBMParams(S0=params.S0 + h, r=params.r, q=params.q, sigma=params.sigma, T=params.T)
    p_dn = GBMParams(S0=params.S0 - h, r=params.r, q=params.q, sigma=params.sigma, T=params.T)
    S_up, _ = simulate_gbm(p_up, n_paths, n_steps, np.random.default_rng(seed))
    S_dn, _ = simulate_gbm(p_dn, n_paths, n_steps, np.random.default_rng(seed))
    disc = np.exp(-params.r * params.T)
    samples = disc * (payoff_fn(S_up) - payoff_fn(S_dn)) / (2 * h)
    return GreekResult(samples, "fd_delta_central")


def fd_gamma_central(
    params: GBMParams,
    payoff_fn: Callable[[np.ndarray], np.ndarray],
    n_paths: int,
    n_steps: int = 1,
    h: float = 1.0,
    rng=None,
) -> GreekResult:
    """Central FD gamma: [V(S0+h) - 2*V(S0) + V(S0-h)] / h^2."""
    rng = np.random.default_rng(rng)
    seed = int(rng.integers(0, 2**31))

    p_up = GBMParams(S0=params.S0 + h, r=params.r, q=params.q, sigma=params.sigma, T=params.T)
    p_dn = GBMParams(S0=params.S0 - h, r=params.r, q=params.q, sigma=params.sigma, T=params.T)
    S_base, _ = simulate_gbm(params, n_paths, n_steps, np.random.default_rng(seed))
    S_up, _ = simulate_gbm(p_up, n_paths, n_steps, np.random.default_rng(seed))
    S_dn, _ = simulate_gbm(p_dn, n_paths, n_steps, np.random.default_rng(seed))
    disc = np.exp(-params.r * params.T)
    samples = disc * (payoff_fn(S_up) - 2 * payoff_fn(S_base) + payoff_fn(S_dn)) / h**2
    return GreekResult(samples, "fd_gamma_central")


def fd_vega_central(
    params: GBMParams,
    payoff_fn: Callable[[np.ndarray], np.ndarray],
    n_paths: int,
    n_steps: int = 1,
    h: float = 0.001,
    rng=None,
) -> GreekResult:
    """Central FD vega w.r.t. sigma."""
    rng = np.random.default_rng(rng)
    seed = int(rng.integers(0, 2**31))

    p_up = GBMParams(S0=params.S0, r=params.r, q=params.q, sigma=params.sigma + h, T=params.T)
    p_dn = GBMParams(S0=params.S0, r=params.r, q=params.q, sigma=params.sigma - h, T=params.T)
    S_up, _ = simulate_gbm(p_up, n_paths, n_steps, np.random.default_rng(seed))
    S_dn, _ = simulate_gbm(p_dn, n_paths, n_steps, np.random.default_rng(seed))
    disc = np.exp(-params.r * params.T)
    samples = disc * (payoff_fn(S_up) - payoff_fn(S_dn)) / (2 * h)
    return GreekResult(samples, "fd_vega_central")


# ---------------------------------------------------------------------------
# Pathwise (Broadie-Glasserman) estimators — require differentiable payoffs
# ---------------------------------------------------------------------------

def pathwise_delta(
    params: GBMParams,
    payoff_deriv_fn: Callable[[np.ndarray], np.ndarray],
    n_paths: int,
    n_steps: int = 1,
    rng=None,
    antithetic: bool = True,
) -> GreekResult:
    """
    Pathwise (IPA) delta: E[payoff'(S_T) * S_T / S0].

    payoff_deriv_fn : derivative of payoff w.r.t. S_T (e.g. indicator for call).
    Only valid for differentiable (a.e.) payoffs.
    """
    S, Z = simulate_gbm(params, n_paths, n_steps, rng, antithetic)
    disc = np.exp(-params.r * params.T)
    dphi_dST = payoff_deriv_fn(S)
    samples = disc * dphi_dST * S[:, -1] / params.S0
    return GreekResult(samples, "pathwise_delta")


def pathwise_vega(
    params: GBMParams,
    payoff_deriv_fn: Callable[[np.ndarray], np.ndarray],
    n_paths: int,
    n_steps: int = 1,
    rng=None,
    antithetic: bool = True,
) -> GreekResult:
    """
    Pathwise vega: E[payoff'(S_T) * S_T * (W_T / T - sigma)].

    d/dsigma log S_T = W_T - sigma*T, so d S_T/dsigma = S_T*(W_T/T - sigma).
    """
    S, Z = simulate_gbm(params, n_paths, n_steps, rng, antithetic)
    n_steps_ = Z.shape[1]
    dt = params.T / n_steps_
    W_T = np.sqrt(dt) * Z.sum(axis=1)
    disc = np.exp(-params.r * params.T)
    dphi = payoff_deriv_fn(S)
    dST_dsig = S[:, -1] * (W_T - params.sigma * params.T)
    samples = disc * dphi * dST_dsig
    return GreekResult(samples, "pathwise_vega")


# ---------------------------------------------------------------------------
# Likelihood-ratio (score function) estimators
# ---------------------------------------------------------------------------

def lr_delta(
    params: GBMParams,
    payoff_fn: Callable[[np.ndarray], np.ndarray],
    n_paths: int,
    n_steps: int = 1,
    rng=None,
    antithetic: bool = True,
) -> GreekResult:
    """
    Likelihood-ratio (LR) delta.

    Score function: d/dS0 log p(Z; S0) = W_T / (sigma * S0 * T).
    This is identical to the Malliavin delta weight — the LR method coincides
    with the Malliavin IBP method for terminal-value payoffs under GBM.
    """
    S, Z = simulate_gbm(params, n_paths, n_steps, rng, antithetic)
    n_steps_ = Z.shape[1]
    dt = params.T / n_steps_
    W_T = np.sqrt(dt) * Z.sum(axis=1)
    score = W_T / (params.sigma * params.S0 * params.T)
    disc = np.exp(-params.r * params.T)
    samples = disc * payoff_fn(S) * score
    return GreekResult(samples, "lr_delta")


def lr_gamma(
    params: GBMParams,
    payoff_fn: Callable[[np.ndarray], np.ndarray],
    n_paths: int,
    n_steps: int = 1,
    rng=None,
    antithetic: bool = True,
) -> GreekResult:
    """
    LR gamma: second derivative of log-density w.r.t. S0.

    score_2 = d^2/dS0^2 log p = [(W_T/(sigma*T))^2 - 1/T] / S0^2 - score/S0
    """
    S, Z = simulate_gbm(params, n_paths, n_steps, rng, antithetic)
    n_steps_ = Z.shape[1]
    dt = params.T / n_steps_
    W_T = np.sqrt(dt) * Z.sum(axis=1)
    T, sig, S0 = params.T, params.sigma, params.S0

    score1 = W_T / (sig * S0 * T)
    score2 = (W_T**2 - T) / (sig**2 * S0**2 * T**2) - 1.0 / (sig * S0**2 * T)
    disc = np.exp(-params.r * params.T)
    samples = disc * payoff_fn(S) * score2
    return GreekResult(samples, "lr_gamma")
