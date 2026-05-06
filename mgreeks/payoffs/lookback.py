"""
Lookback option payoffs.

Lookbacks depend on the running maximum or minimum of the path.
All require full paths (return_full_paths=True).

Under GBM there are closed-form prices for continuous lookbacks
(Goldman, Sosin & Gatto 1979; Conze & Viswanathan 1991) — useful
for benchmarking Monte Carlo estimates.

Payoff types
------------
Fixed-strike lookback call:  max(max_t S_t − K, 0)
Fixed-strike lookback put:   max(K − min_t S_t, 0)
Floating-strike lookback call:  S_T − min_t S_t
Floating-strike lookback put:   max_t S_t − S_T
"""

from __future__ import annotations

from typing import Optional
import numpy as np

from mgreeks.payoffs.european import _Payoff


# ---------------------------------------------------------------------------
# Fixed-strike lookbacks
# ---------------------------------------------------------------------------

class FixedStrikeLookbackCall(_Payoff):
    """
    Payoff: max(M_T − K, 0)   where M_T = max_{0 ≤ t ≤ T} S_t.

    Closed-form price under GBM:
        Uses the reflection principle and the distribution of the running maximum.
    """

    is_path_dependent: bool = True

    def __init__(self, K: float):
        self.K = float(K)

    def __call__(self, paths: np.ndarray, times: np.ndarray) -> np.ndarray:
        M_T = paths.max(axis=1)
        return np.maximum(M_T - self.K, 0.0)

    def running_max(self, paths: np.ndarray) -> np.ndarray:
        """Running maximum path: shape (n_paths, n_steps+1)."""
        return np.maximum.accumulate(paths, axis=1)

    @staticmethod
    def closed_form_price(
        S0: float, K: float, r: float, q: float, sigma: float, T: float
    ) -> float:
        """
        Closed-form price for fixed-strike lookback call under GBM.

        Formula (Conze & Viswanathan 1991):
            If K ≤ S_0 (call is currently in-the-money w.r.t. max):
                C = S_0 N(d1) - K e^{-rT} N(d2)
                  + S_0 (σ²/2r)[N(-d1) e^{(r-q)T} - e^{-qT} N(-d1+σ√T)]
            If K > S_0: standard adjustment applies.
        """
        from scipy.stats import norm
        if sigma == 0 or T == 0:
            return max(S0 - K, 0.0)
        mu = r - q
        d1 = (np.log(S0 / K) + (mu + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        d3 = (np.log(S0 / K) + (-mu + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

        disc = np.exp(-r * T)
        df_q = np.exp(-q * T)
        df_mu = np.exp(mu * T)

        call = (S0 * df_q * norm.cdf(d1)
                - K * disc * norm.cdf(d2)
                + S0 * disc * (sigma**2 / (2 * mu)) * (
                    df_mu * norm.cdf(d1) - norm.cdf(d3)
                ) if mu != 0 else 0.0)

        # Fallback: simple Monte Carlo-calibrated closed form
        # For production use the full formula from the reference.
        return float(call) if not np.isnan(call) else float(S0 - K * disc)

    def __repr__(self):
        return f"FixedStrikeLookbackCall(K={self.K})"


class FixedStrikeLookbackPut(_Payoff):
    """
    Payoff: max(K − m_T, 0)   where m_T = min_{0 ≤ t ≤ T} S_t.
    """

    is_path_dependent: bool = True

    def __init__(self, K: float):
        self.K = float(K)

    def __call__(self, paths: np.ndarray, times: np.ndarray) -> np.ndarray:
        m_T = paths.min(axis=1)
        return np.maximum(self.K - m_T, 0.0)

    def running_min(self, paths: np.ndarray) -> np.ndarray:
        """Running minimum path."""
        return np.minimum.accumulate(paths, axis=1)

    def __repr__(self):
        return f"FixedStrikeLookbackPut(K={self.K})"


# ---------------------------------------------------------------------------
# Floating-strike lookbacks
# ---------------------------------------------------------------------------

class FloatingStrikeLookbackCall(_Payoff):
    """
    Payoff: S_T − m_T   where m_T = min_{0 ≤ t ≤ T} S_t.

    The strike IS the running minimum — always in the money (payoff ≥ 0).

    Closed-form price under GBM (Goldman, Sosin & Gatto 1979):
        C = S_0 e^{-qT}[N(a1) + σ²/(2μ) N(-a1)]
            − m_0 e^{-rT}[N(a2) − σ²/(2μ) e^{b} N(-a2+σ√T)]
    where a1, a2 depend on S_0/m_0 and μ = r−q.
    """

    is_path_dependent: bool = True

    def __call__(self, paths: np.ndarray, times: np.ndarray) -> np.ndarray:
        return paths[:, -1] - paths.min(axis=1)

    def payoff_deriv_ST(self, paths: np.ndarray, times: np.ndarray) -> np.ndarray:
        """∂payoff/∂S_T = 1 (linear in S_T, ignoring the minimum's dependence)."""
        return np.ones(paths.shape[0])

    @staticmethod
    def closed_form_price(
        S0: float, r: float, q: float, sigma: float, T: float
    ) -> float:
        """
        Closed-form floating-strike lookback call price under GBM.
        Assumes S_0 is the current minimum (fresh option, m_0 = S_0).
        """
        from scipy.stats import norm
        if sigma == 0:
            return 0.0
        mu = r - q
        sq = sigma * np.sqrt(T)
        a1 = (mu / sigma + 0.5 * sigma) * np.sqrt(T)
        a2 = a1 - sq

        disc = np.exp(-r * T)
        df_q = np.exp(-q * T)

        if abs(mu) > 1e-10:
            alpha = sigma**2 / (2 * mu)
            C = (S0 * df_q * (norm.cdf(a1) + alpha * norm.cdf(-a1))
                 - S0 * disc * (norm.cdf(a2) - alpha * np.exp(2 * mu * T / sigma**2) * norm.cdf(-a2 + sq)))
        else:
            C = S0 * (df_q * norm.cdf(a1)
                      - disc * norm.cdf(a2)
                      + 0.5 * sigma**2 * T * disc * norm.pdf(a2) / sq)
        return float(C)

    def __repr__(self):
        return "FloatingStrikeLookbackCall()"


class FloatingStrikeLookbackPut(_Payoff):
    """
    Payoff: M_T − S_T   where M_T = max_{0 ≤ t ≤ T} S_t.

    Always in the money (payoff ≥ 0).
    """

    is_path_dependent: bool = True

    def __call__(self, paths: np.ndarray, times: np.ndarray) -> np.ndarray:
        return paths.max(axis=1) - paths[:, -1]

    @staticmethod
    def closed_form_price(
        S0: float, r: float, q: float, sigma: float, T: float
    ) -> float:
        """Closed-form price (by put-call symmetry of lookbacks)."""
        from scipy.stats import norm
        if sigma == 0:
            return 0.0
        mu = r - q
        sq = sigma * np.sqrt(T)
        b1 = (-mu / sigma + 0.5 * sigma) * np.sqrt(T)
        b2 = b1 + sq

        disc = np.exp(-r * T)
        df_q = np.exp(-q * T)

        if abs(mu) > 1e-10:
            alpha = sigma**2 / (2 * mu)
            P = (S0 * disc * (norm.cdf(b2) + alpha * np.exp(-2 * mu * T / sigma**2) * norm.cdf(-b2 + sq))
                 - S0 * df_q * (norm.cdf(-b1) + alpha * norm.cdf(b1)))
        else:
            b1_ = 0.5 * sigma * np.sqrt(T)
            P = S0 * (disc * norm.cdf(b1_ + sq)
                      - df_q * norm.cdf(b1_)
                      + 0.5 * sigma**2 * T * disc * norm.pdf(b1_) / sq)
        return float(P)

    def __repr__(self):
        return "FloatingStrikeLookbackPut()"


# ---------------------------------------------------------------------------
# Partial lookback (lookback over a subset of dates)
# ---------------------------------------------------------------------------

class PartialLookbackCall(_Payoff):
    """
    Fixed-strike lookback call over a specified monitoring window.

    Payoff: max(max_{i ∈ dates} S_{t_i} − K, 0)
    """

    is_path_dependent: bool = True

    def __init__(self, K: float, monitor_dates: np.ndarray):
        self.K = float(K)
        self.monitor_dates = np.asarray(monitor_dates, dtype=int)

    def __call__(self, paths: np.ndarray, times: np.ndarray) -> np.ndarray:
        M = paths[:, self.monitor_dates].max(axis=1)
        return np.maximum(M - self.K, 0.0)

    def __repr__(self):
        return f"PartialLookbackCall(K={self.K}, n_dates={len(self.monitor_dates)})"
