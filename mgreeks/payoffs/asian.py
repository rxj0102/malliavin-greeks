"""
Asian (average-rate) payoffs.

Two averaging conventions:
  - Arithmetic: A = (1/n) Σ S_{t_i}
  - Geometric:  G = (Π S_{t_i})^{1/n}  — has closed-form price under GBM

The geometric Asian is useful as a control variate for arithmetic Asian pricing.

Path-dependence
---------------
Both require the full path (return_full_paths=True in simulate()).
The averaging dates can be a subset of the simulation time grid.

Malliavin delta note
--------------------
For arithmetic Asian calls, A is smooth in S_0 (A = S_0 · Ā where Ā depends
only on the normalised path), so the pathwise (IPA) estimator applies:

    Δ_Asian = e^{-rT} E[1_{A>K} · dA/dS_0] = e^{-rT} E[1_{A>K} · A/S_0]

For geometric Asian calls, the closed-form delta can be used for benchmarking.
"""

from __future__ import annotations

from typing import Optional
import numpy as np

from mgreeks.payoffs.european import _Payoff


class ArithmeticAsianCall(_Payoff):
    """
    Payoff: max(A − K, 0)   where A = (1/n) Σ_{i∈dates} S_{t_i}

    Parameters
    ----------
    K              : strike
    averaging_dates: indices into the time grid to include in average.
                     None → use all steps (including t_0=0 if include_S0=True).
    include_S0     : whether to include S_0 in the average (default False).
    """

    is_path_dependent: bool = True

    def __init__(
        self,
        K: float,
        averaging_dates: Optional[np.ndarray] = None,
        include_S0: bool = False,
    ):
        self.K = float(K)
        self.averaging_dates = (
            np.asarray(averaging_dates, dtype=int) if averaging_dates is not None
            else None
        )
        self.include_S0 = include_S0

    def _average(self, paths: np.ndarray) -> np.ndarray:
        if self.averaging_dates is not None:
            cols = self.averaging_dates
        elif self.include_S0:
            cols = slice(None)          # all columns including t_0
        else:
            cols = slice(1, None)       # exclude t_0
        return paths[:, cols].mean(axis=1)

    def __call__(self, paths: np.ndarray, times: np.ndarray) -> np.ndarray:
        A = self._average(paths)
        return np.maximum(A - self.K, 0.0)

    def payoff_deriv(self, paths: np.ndarray, times: np.ndarray) -> np.ndarray:
        """
        Pathwise IPA: ∂/∂S_T max(A−K, 0) is not well-defined for path-dependent A.
        The correct pathwise Greek for delta is dA/dS_0, not dA/dS_T.

        Returns the indicator 1_{A>K} · dA/dS_0 = 1_{A>K} · A/S_0,
        so that the estimator is  disc * payoff_deriv * S_0  gives delta.

        Usage:  delta_samples = disc * payoff.payoff_deriv(paths, times)
        """
        A = self._average(paths)
        S0 = paths[:, 0]
        return (A > self.K).astype(float) * A / S0

    def __repr__(self):
        return f"ArithmeticAsianCall(K={self.K}, include_S0={self.include_S0})"


class ArithmeticAsianPut(_Payoff):
    """Payoff: max(K − A, 0)."""

    is_path_dependent: bool = True

    def __init__(
        self,
        K: float,
        averaging_dates: Optional[np.ndarray] = None,
        include_S0: bool = False,
    ):
        self.K = float(K)
        self.averaging_dates = (
            np.asarray(averaging_dates, dtype=int) if averaging_dates is not None
            else None
        )
        self.include_S0 = include_S0

    def _average(self, paths: np.ndarray) -> np.ndarray:
        if self.averaging_dates is not None:
            cols = self.averaging_dates
        elif self.include_S0:
            cols = slice(None)
        else:
            cols = slice(1, None)
        return paths[:, cols].mean(axis=1)

    def __call__(self, paths: np.ndarray, times: np.ndarray) -> np.ndarray:
        A = self._average(paths)
        return np.maximum(self.K - A, 0.0)

    def payoff_deriv(self, paths: np.ndarray, times: np.ndarray) -> np.ndarray:
        """Pathwise: −1_{A<K} · A/S_0."""
        A = self._average(paths)
        S0 = paths[:, 0]
        return -(A < self.K).astype(float) * A / S0

    def __repr__(self):
        return f"ArithmeticAsianPut(K={self.K})"


class GeometricAsianCall(_Payoff):
    """
    Payoff: max(G − K, 0)   where G = (Π_{i∈dates} S_{t_i})^{1/n}

    Geometric average options have a closed-form price under GBM
    (because log G is Gaussian) — useful as a control variate.

    Closed-form price under GBM (Kemna & Vorst 1990):
        G ~ log-normal with:
            μ_G = log S_0 + (r − q − σ²/2)(T̄ + h/2) − σ² h / 12
            σ_G² = σ² h (2n+1) / (6n)
        where h = T/n and T̄ = T(n−1)/(2n).

    Parameters
    ----------
    K              : strike
    averaging_dates: None → use all steps (excluding t_0)
    """

    is_path_dependent: bool = True

    def __init__(self, K: float, averaging_dates: Optional[np.ndarray] = None):
        self.K = float(K)
        self.averaging_dates = (
            np.asarray(averaging_dates, dtype=int) if averaging_dates is not None
            else None
        )

    def _geo_average(self, paths: np.ndarray) -> np.ndarray:
        if self.averaging_dates is not None:
            cols = paths[:, self.averaging_dates]
        else:
            cols = paths[:, 1:]             # exclude t_0
        n = cols.shape[1]
        return np.exp(np.log(np.maximum(cols, 1e-300)).mean(axis=1))

    def __call__(self, paths: np.ndarray, times: np.ndarray) -> np.ndarray:
        G = self._geo_average(paths)
        return np.maximum(G - self.K, 0.0)

    def payoff_deriv(self, paths: np.ndarray, times: np.ndarray) -> np.ndarray:
        """
        Pathwise: ∂/∂S_0 max(G−K, 0) = 1_{G>K} · G/S_0
        (same structure as arithmetic because G is also linear-homogeneous in S_0).
        """
        G = self._geo_average(paths)
        S0 = paths[:, 0]
        return (G > self.K).astype(float) * G / S0

    @staticmethod
    def closed_form_price(
        S0: float, K: float, r: float, q: float, sigma: float, T: float, n: int
    ) -> float:
        """
        Kemna–Vorst closed-form price for geometric Asian call under GBM.

        Parameters
        ----------
        n : number of averaging dates (equally spaced in [0, T])
        """
        from scipy.stats import norm
        # Exact: log G ~ N(mu_G, sigma_G²) under Q for discrete GBM
        # Var[log G] = σ² T (n+1)(2n+1) / (6n²)
        sigma_G = sigma * np.sqrt(T * (n + 1) * (2 * n + 1) / (6 * n * n))
        mu_adj = (r - q - 0.5 * sigma**2) * (n + 1) / (2 * n)
        mu_G = np.log(S0) + mu_adj * T
        # E_Q[G] = exp(mu_G + sigma_G² / 2)
        E_G = np.exp(mu_G + 0.5 * sigma_G**2)
        d1 = (np.log(E_G / K) + 0.5 * sigma_G**2) / sigma_G
        d2 = d1 - sigma_G
        return float(np.exp(-r * T) * (E_G * norm.cdf(d1) - K * norm.cdf(d2)))

    def __repr__(self):
        return f"GeometricAsianCall(K={self.K})"
