"""
Multi-asset basket and spread option payoffs.

These payoffs operate on multi-asset paths of shape (n_paths, n_assets, n_steps+1)
produced by MultiAssetGBM.simulate().

Included types
--------------
BasketCall / BasketPut      : weighted average of n_assets, vanilla call/put
SpreadCall / SpreadPut      : S^1 − S^2 spread option (n_assets=2)
BestOfCall                  : call on max(S^1,...,S^n)
WorstOfCall                 : call on min(S^1,...,S^n)
RainbowCall                 : max of n_assets (exchange option special case)
"""

from __future__ import annotations

from typing import Optional
import numpy as np

from mgreeks.payoffs.european import _Payoff


# ---------------------------------------------------------------------------
# Basket call / put
# ---------------------------------------------------------------------------

class BasketCall(_Payoff):
    """
    Payoff: max(Σ_i w_i S^i_T − K, 0)

    Parameters
    ----------
    K       : strike
    weights : asset weights w_i, shape (n_assets,); default uniform
    """

    is_path_dependent: bool = False

    def __init__(self, K: float, weights: Optional[np.ndarray] = None):
        self.K = float(K)
        self.weights = np.asarray(weights, dtype=float) if weights is not None else None

    def _basket_value(self, paths: np.ndarray) -> np.ndarray:
        """
        Weighted basket at maturity.

        Parameters
        ----------
        paths : shape (n_paths, n_assets, n_steps+1)  or  (n_paths, n_assets)
                if only terminal values are provided.
        """
        if paths.ndim == 3:
            ST = paths[:, :, -1]           # (n_paths, n_assets)
        elif paths.ndim == 2:
            ST = paths
        else:
            raise ValueError(f"Expected paths.ndim in {{2, 3}}, got {paths.ndim}")

        n_assets = ST.shape[1]
        w = self.weights if self.weights is not None else np.ones(n_assets) / n_assets
        return ST @ w                       # (n_paths,)

    def __call__(self, paths: np.ndarray, times: np.ndarray) -> np.ndarray:
        basket = self._basket_value(paths)
        return np.maximum(basket - self.K, 0.0)

    def payoff_deriv_i(
        self, paths: np.ndarray, times: np.ndarray, asset: int
    ) -> np.ndarray:
        """
        ∂payoff/∂S^i_T = w_i · 1_{basket > K}
        (for pathwise delta of asset i).
        """
        basket = self._basket_value(paths)
        n_assets = paths.shape[1] if paths.ndim == 3 else paths.shape[1]
        w = self.weights if self.weights is not None else np.ones(n_assets) / n_assets
        return w[asset] * (basket > self.K).astype(float)

    def __repr__(self):
        return f"BasketCall(K={self.K}, weights={self.weights})"


class BasketPut(_Payoff):
    """Payoff: max(K − Σ_i w_i S^i_T, 0)."""

    is_path_dependent: bool = False

    def __init__(self, K: float, weights: Optional[np.ndarray] = None):
        self.K = float(K)
        self.weights = np.asarray(weights, dtype=float) if weights is not None else None

    def _basket_value(self, paths: np.ndarray) -> np.ndarray:
        ST = paths[:, :, -1] if paths.ndim == 3 else paths
        n_assets = ST.shape[1]
        w = self.weights if self.weights is not None else np.ones(n_assets) / n_assets
        return ST @ w

    def __call__(self, paths: np.ndarray, times: np.ndarray) -> np.ndarray:
        basket = self._basket_value(paths)
        return np.maximum(self.K - basket, 0.0)

    def __repr__(self):
        return f"BasketPut(K={self.K}, weights={self.weights})"


# ---------------------------------------------------------------------------
# Spread options
# ---------------------------------------------------------------------------

class SpreadCall(_Payoff):
    """
    Payoff: max(S^1_T − S^2_T − K, 0)  (Margrabe / spread call)

    For K=0: reduces to the Margrabe exchange option with closed-form price.
    """

    is_path_dependent: bool = False

    def __init__(self, K: float = 0.0):
        self.K = float(K)

    def __call__(self, paths: np.ndarray, times: np.ndarray) -> np.ndarray:
        ST = paths[:, :, -1] if paths.ndim == 3 else paths
        return np.maximum(ST[:, 0] - ST[:, 1] - self.K, 0.0)

    @staticmethod
    def margrabe_price(
        S1: float, S2: float, sigma1: float, sigma2: float,
        rho: float, r: float, q1: float, q2: float, T: float
    ) -> float:
        """Margrabe (1978) exchange option price (K=0 case)."""
        from scipy.stats import norm
        sigma_spread = np.sqrt(sigma1**2 + sigma2**2 - 2 * rho * sigma1 * sigma2)
        if sigma_spread < 1e-10 or T < 1e-10:
            return max(S1 * np.exp(-q1 * T) - S2 * np.exp(-q2 * T), 0.0)
        d1 = (np.log(S1 / S2) + (q2 - q1 + 0.5 * sigma_spread**2) * T) \
             / (sigma_spread * np.sqrt(T))
        d2 = d1 - sigma_spread * np.sqrt(T)
        return (S1 * np.exp(-q1 * T) * norm.cdf(d1)
                - S2 * np.exp(-q2 * T) * norm.cdf(d2))

    def __repr__(self):
        return f"SpreadCall(K={self.K})"


class SpreadPut(_Payoff):
    """Payoff: max(K − (S^1_T − S^2_T), 0)."""

    is_path_dependent: bool = False

    def __init__(self, K: float = 0.0):
        self.K = float(K)

    def __call__(self, paths: np.ndarray, times: np.ndarray) -> np.ndarray:
        ST = paths[:, :, -1] if paths.ndim == 3 else paths
        return np.maximum(self.K - (ST[:, 0] - ST[:, 1]), 0.0)

    def __repr__(self):
        return f"SpreadPut(K={self.K})"


# ---------------------------------------------------------------------------
# Rainbow options
# ---------------------------------------------------------------------------

class BestOfCall(_Payoff):
    """Payoff: max(max_i S^i_T − K, 0)."""

    is_path_dependent: bool = False

    def __init__(self, K: float):
        self.K = float(K)

    def __call__(self, paths: np.ndarray, times: np.ndarray) -> np.ndarray:
        ST = paths[:, :, -1] if paths.ndim == 3 else paths
        return np.maximum(ST.max(axis=1) - self.K, 0.0)

    def __repr__(self):
        return f"BestOfCall(K={self.K})"


class WorstOfCall(_Payoff):
    """Payoff: max(min_i S^i_T − K, 0)."""

    is_path_dependent: bool = False

    def __init__(self, K: float):
        self.K = float(K)

    def __call__(self, paths: np.ndarray, times: np.ndarray) -> np.ndarray:
        ST = paths[:, :, -1] if paths.ndim == 3 else paths
        return np.maximum(ST.min(axis=1) - self.K, 0.0)

    def __repr__(self):
        return f"WorstOfCall(K={self.K})"
