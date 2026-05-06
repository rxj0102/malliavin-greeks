"""
Barrier option payoffs.

Barrier options are knocked in or out when the path crosses a barrier level B.
All barrier payoffs require the full path (return_full_paths=True).

Monitoring conventions
----------------------
Continuous monitoring is approximated by checking every simulated path point.
For accurate continuous-barrier pricing, use many time steps (n_steps ≥ 252).
For discrete barriers, pass the relevant column indices via barrier_dates.

Why Malliavin excels here
-------------------------
Finite differences for barrier option delta suffer severely near the barrier:
  - Small bump h → the bump path crosses the barrier when the unbumped path
    does not, creating a discontinuous payoff difference.
  - The FD estimator variance → ∞ as h → 0.

The Malliavin weight (same as vanilla delta) works without modification:
    delta = e^{-rT} E[payoff(S) · W_T / (σ S_0 T)]
Because the weight does not depend on which side of the barrier the path lands.

Knock-in / Knock-out parity
----------------------------
For each barrier type: Knock-In + Knock-Out = Vanilla (same K, no barrier).
This is a useful numerical check.
"""

from __future__ import annotations

from typing import Optional
import numpy as np

from mgreeks.payoffs.european import _Payoff, EuropeanCall, EuropeanPut


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _alive_down(paths: np.ndarray, B: float,
                barrier_dates: Optional[np.ndarray]) -> np.ndarray:
    """True where path never touches or crosses below barrier B."""
    cols = paths[:, barrier_dates] if barrier_dates is not None else paths
    return np.all(cols > B, axis=1)


def _alive_up(paths: np.ndarray, B: float,
              barrier_dates: Optional[np.ndarray]) -> np.ndarray:
    """True where path never touches or crosses above barrier B."""
    cols = paths[:, barrier_dates] if barrier_dates is not None else paths
    return np.all(cols < B, axis=1)


def _hit_down(paths: np.ndarray, B: float,
              barrier_dates: Optional[np.ndarray]) -> np.ndarray:
    """True where path touched or crossed below barrier B."""
    return ~_alive_down(paths, B, barrier_dates)


def _hit_up(paths: np.ndarray, B: float,
            barrier_dates: Optional[np.ndarray]) -> np.ndarray:
    return ~_alive_up(paths, B, barrier_dates)


# ---------------------------------------------------------------------------
# Down-and-out
# ---------------------------------------------------------------------------

class DownAndOutCall(_Payoff):
    """
    Payoff: max(S_T − K, 0) · 1_{min_t S_t > B}

    Knocked out (zero payoff) if the path ever goes below barrier B.
    Typically B < S_0 < K (or B < K < S_0 for deep ITM case).
    """

    is_path_dependent: bool = True

    def __init__(self, K: float, B: float,
                 barrier_dates: Optional[np.ndarray] = None):
        self.K = float(K)
        self.B = float(B)
        self.barrier_dates = (
            np.asarray(barrier_dates, dtype=int) if barrier_dates is not None
            else None
        )

    def __call__(self, paths: np.ndarray, times: np.ndarray) -> np.ndarray:
        alive = _alive_down(paths, self.B, self.barrier_dates).astype(float)
        return np.maximum(paths[:, -1] - self.K, 0.0) * alive

    def __repr__(self):
        return f"DownAndOutCall(K={self.K}, B={self.B})"


class DownAndOutPut(_Payoff):
    """Payoff: max(K − S_T, 0) · 1_{min_t S_t > B}."""

    is_path_dependent: bool = True

    def __init__(self, K: float, B: float,
                 barrier_dates: Optional[np.ndarray] = None):
        self.K, self.B = float(K), float(B)
        self.barrier_dates = (
            np.asarray(barrier_dates, dtype=int) if barrier_dates is not None
            else None
        )

    def __call__(self, paths: np.ndarray, times: np.ndarray) -> np.ndarray:
        alive = _alive_down(paths, self.B, self.barrier_dates).astype(float)
        return np.maximum(self.K - paths[:, -1], 0.0) * alive

    def __repr__(self):
        return f"DownAndOutPut(K={self.K}, B={self.B})"


# ---------------------------------------------------------------------------
# Down-and-in
# ---------------------------------------------------------------------------

class DownAndInCall(_Payoff):
    """
    Payoff: max(S_T − K, 0) · 1_{min_t S_t ≤ B}

    Knocked IN (gains the vanilla payoff) only if the path touches B.
    Note: DownAndInCall + DownAndOutCall = VanillaCall (knock-in/out parity).
    """

    is_path_dependent: bool = True

    def __init__(self, K: float, B: float,
                 barrier_dates: Optional[np.ndarray] = None):
        self.K, self.B = float(K), float(B)
        self.barrier_dates = (
            np.asarray(barrier_dates, dtype=int) if barrier_dates is not None
            else None
        )

    def __call__(self, paths: np.ndarray, times: np.ndarray) -> np.ndarray:
        knocked_in = _hit_down(paths, self.B, self.barrier_dates).astype(float)
        return np.maximum(paths[:, -1] - self.K, 0.0) * knocked_in

    def __repr__(self):
        return f"DownAndInCall(K={self.K}, B={self.B})"


class DownAndInPut(_Payoff):
    """Payoff: max(K − S_T, 0) · 1_{min_t S_t ≤ B}."""

    is_path_dependent: bool = True

    def __init__(self, K: float, B: float,
                 barrier_dates: Optional[np.ndarray] = None):
        self.K, self.B = float(K), float(B)
        self.barrier_dates = (
            np.asarray(barrier_dates, dtype=int) if barrier_dates is not None
            else None
        )

    def __call__(self, paths: np.ndarray, times: np.ndarray) -> np.ndarray:
        knocked_in = _hit_down(paths, self.B, self.barrier_dates).astype(float)
        return np.maximum(self.K - paths[:, -1], 0.0) * knocked_in

    def __repr__(self):
        return f"DownAndInPut(K={self.K}, B={self.B})"


# ---------------------------------------------------------------------------
# Up-and-out
# ---------------------------------------------------------------------------

class UpAndOutCall(_Payoff):
    """Payoff: max(S_T − K, 0) · 1_{max_t S_t < B}."""

    is_path_dependent: bool = True

    def __init__(self, K: float, B: float,
                 barrier_dates: Optional[np.ndarray] = None):
        self.K, self.B = float(K), float(B)
        self.barrier_dates = (
            np.asarray(barrier_dates, dtype=int) if barrier_dates is not None
            else None
        )

    def __call__(self, paths: np.ndarray, times: np.ndarray) -> np.ndarray:
        alive = _alive_up(paths, self.B, self.barrier_dates).astype(float)
        return np.maximum(paths[:, -1] - self.K, 0.0) * alive

    def __repr__(self):
        return f"UpAndOutCall(K={self.K}, B={self.B})"


class UpAndOutPut(_Payoff):
    """Payoff: max(K − S_T, 0) · 1_{max_t S_t < B}."""

    is_path_dependent: bool = True

    def __init__(self, K: float, B: float,
                 barrier_dates: Optional[np.ndarray] = None):
        self.K, self.B = float(K), float(B)
        self.barrier_dates = (
            np.asarray(barrier_dates, dtype=int) if barrier_dates is not None
            else None
        )

    def __call__(self, paths: np.ndarray, times: np.ndarray) -> np.ndarray:
        alive = _alive_up(paths, self.B, self.barrier_dates).astype(float)
        return np.maximum(self.K - paths[:, -1], 0.0) * alive

    def __repr__(self):
        return f"UpAndOutPut(K={self.K}, B={self.B})"


# ---------------------------------------------------------------------------
# Up-and-in
# ---------------------------------------------------------------------------

class UpAndInCall(_Payoff):
    """Payoff: max(S_T − K, 0) · 1_{max_t S_t ≥ B}."""

    is_path_dependent: bool = True

    def __init__(self, K: float, B: float,
                 barrier_dates: Optional[np.ndarray] = None):
        self.K, self.B = float(K), float(B)
        self.barrier_dates = (
            np.asarray(barrier_dates, dtype=int) if barrier_dates is not None
            else None
        )

    def __call__(self, paths: np.ndarray, times: np.ndarray) -> np.ndarray:
        knocked_in = _hit_up(paths, self.B, self.barrier_dates).astype(float)
        return np.maximum(paths[:, -1] - self.K, 0.0) * knocked_in

    def __repr__(self):
        return f"UpAndInCall(K={self.K}, B={self.B})"


class UpAndInPut(_Payoff):
    """Payoff: max(K − S_T, 0) · 1_{max_t S_t ≥ B}."""

    is_path_dependent: bool = True

    def __init__(self, K: float, B: float,
                 barrier_dates: Optional[np.ndarray] = None):
        self.K, self.B = float(K), float(B)
        self.barrier_dates = (
            np.asarray(barrier_dates, dtype=int) if barrier_dates is not None
            else None
        )

    def __call__(self, paths: np.ndarray, times: np.ndarray) -> np.ndarray:
        knocked_in = _hit_up(paths, self.B, self.barrier_dates).astype(float)
        return np.maximum(self.K - paths[:, -1], 0.0) * knocked_in

    def __repr__(self):
        return f"UpAndInPut(K={self.K}, B={self.B})"


# ---------------------------------------------------------------------------
# Parity checker
# ---------------------------------------------------------------------------

def barrier_parity_check(
    paths: np.ndarray,
    times: np.ndarray,
    K: float,
    B: float,
    direction: str = "down",
) -> dict:
    """
    Verify knock-in + knock-out = vanilla on each path.

    Returns dict with max_abs_error for call and put.
    """
    from mgreeks.payoffs.european import EuropeanCall, EuropeanPut

    vanilla_call = EuropeanCall(K)(paths, times)
    vanilla_put  = EuropeanPut(K)(paths, times)

    if direction == "down":
        dao = DownAndOutCall(K, B)(paths, times)
        dai = DownAndInCall(K, B)(paths, times)
        dop = DownAndOutPut(K, B)(paths, times)
        dip = DownAndInPut(K, B)(paths, times)
        call_err = float(np.abs(dao + dai - vanilla_call).max())
        put_err  = float(np.abs(dop + dip - vanilla_put).max())
    else:
        uao = UpAndOutCall(K, B)(paths, times)
        uai = UpAndInCall(K, B)(paths, times)
        uop = UpAndOutPut(K, B)(paths, times)
        uip = UpAndInPut(K, B)(paths, times)
        call_err = float(np.abs(uao + uai - vanilla_call).max())
        put_err  = float(np.abs(uop + uip - vanilla_put).max())

    return {"call_parity_error": call_err, "put_parity_error": put_err}
