"""
European and digital payoffs.

All payoff classes are callable:
    payoff(paths, times) -> np.ndarray of shape (n_paths,)

Paths shape: (n_paths, n_steps+1).
Times shape: (n_steps+1,).

Payoff objects also expose:
    .is_path_dependent  : bool
    .payoff_deriv(paths, times) -> ∂payoff/∂S_T  (for pathwise Greeks)
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Base mixin
# ---------------------------------------------------------------------------

class _Payoff:
    is_path_dependent: bool = False

    def payoff_deriv(self, paths: np.ndarray, times: np.ndarray) -> np.ndarray:
        """
        ∂payoff/∂S_T — used by the pathwise (IPA) Greek estimator.
        Must be overridden for differentiable payoffs.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement payoff_deriv. "
            "Use the Malliavin or LR method for non-smooth payoffs."
        )


# ---------------------------------------------------------------------------
# European call
# ---------------------------------------------------------------------------

class EuropeanCall(_Payoff):
    """
    Payoff: max(S_T − K, 0)

    Pathwise derivative: 1_{S_T > K}  (a.e. well-defined).
    """

    def __init__(self, K: float):
        self.K = float(K)

    def __call__(self, paths: np.ndarray, times: np.ndarray) -> np.ndarray:
        return np.maximum(paths[:, -1] - self.K, 0.0)

    def payoff_deriv(self, paths: np.ndarray, times: np.ndarray) -> np.ndarray:
        """∂/∂S_T max(S_T − K, 0) = 1_{S_T > K}."""
        return (paths[:, -1] > self.K).astype(float)

    def __repr__(self):
        return f"EuropeanCall(K={self.K})"


# ---------------------------------------------------------------------------
# European put
# ---------------------------------------------------------------------------

class EuropeanPut(_Payoff):
    """
    Payoff: max(K − S_T, 0)

    Pathwise derivative: −1_{S_T < K}.
    """

    def __init__(self, K: float):
        self.K = float(K)

    def __call__(self, paths: np.ndarray, times: np.ndarray) -> np.ndarray:
        return np.maximum(self.K - paths[:, -1], 0.0)

    def payoff_deriv(self, paths: np.ndarray, times: np.ndarray) -> np.ndarray:
        return -(paths[:, -1] < self.K).astype(float)

    def __repr__(self):
        return f"EuropeanPut(K={self.K})"


# ---------------------------------------------------------------------------
# Digital (binary) options
# ---------------------------------------------------------------------------

class DigitalCall(_Payoff):
    """
    Cash-or-nothing digital call: pays 1 if S_T > K, else 0.

    DISCONTINUOUS payoff — this is where finite differences FAIL and
    Malliavin / LR methods SHINE.

    The pathwise derivative ∂/∂S_T 1_{S_T > K} = δ(S_T − K) is a Dirac
    delta — the pathwise (IPA) estimator has infinite variance.

    The Malliavin/LR estimator uses the weight π = W_T/(σ S_0 T) which
    has finite variance regardless of moneyness.
    """

    def __init__(self, K: float):
        self.K = float(K)

    def __call__(self, paths: np.ndarray, times: np.ndarray) -> np.ndarray:
        return (paths[:, -1] > self.K).astype(float)

    # No payoff_deriv — pathwise not applicable; leave as NotImplementedError.

    def __repr__(self):
        return f"DigitalCall(K={self.K})"


class DigitalPut(_Payoff):
    """Pays 1 if S_T < K, else 0."""

    def __init__(self, K: float):
        self.K = float(K)

    def __call__(self, paths: np.ndarray, times: np.ndarray) -> np.ndarray:
        return (paths[:, -1] < self.K).astype(float)

    def __repr__(self):
        return f"DigitalPut(K={self.K})"


class AssetOrNothingCall(_Payoff):
    """Pays S_T if S_T > K, else 0."""

    def __init__(self, K: float):
        self.K = float(K)

    def __call__(self, paths: np.ndarray, times: np.ndarray) -> np.ndarray:
        ST = paths[:, -1]
        return ST * (ST > self.K)

    def payoff_deriv(self, paths: np.ndarray, times: np.ndarray) -> np.ndarray:
        """∂/∂S_T [S_T 1_{S_T>K}] = 1_{S_T>K}  (ignoring Dirac term)."""
        return (paths[:, -1] > self.K).astype(float)

    def __repr__(self):
        return f"AssetOrNothingCall(K={self.K})"


# ---------------------------------------------------------------------------
# Put-call parity helper
# ---------------------------------------------------------------------------

def put_call_parity_check(
    call_payoff: np.ndarray,
    put_payoff: np.ndarray,
    ST: np.ndarray,
    K: float,
) -> bool:
    """
    Verify C − P = S_T − K on each path (undiscounted parity).
    Returns True if max absolute error < 1e-10.
    """
    diff = call_payoff - put_payoff - (ST - K)
    return float(np.abs(diff).max()) < 1e-10
