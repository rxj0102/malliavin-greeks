"""
Payoff functions operating on path arrays.

All functions accept path arrays S of shape (n_paths, n_steps+1) and return
a 1-D array of shape (n_paths,).  Discounting is NOT applied here.

Notation
--------
K  : strike
H  : barrier level
r  : risk-free rate (needed for Asian put-call)
T  : maturity
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Vanilla payoffs (terminal value only)
# ---------------------------------------------------------------------------

def european_call(S: np.ndarray, K: float) -> np.ndarray:
    return np.maximum(S[:, -1] - K, 0.0)


def european_put(S: np.ndarray, K: float) -> np.ndarray:
    return np.maximum(K - S[:, -1], 0.0)


def digital_call(S: np.ndarray, K: float) -> np.ndarray:
    """Cash-or-nothing digital: pays 1 if S_T > K."""
    return (S[:, -1] > K).astype(float)


def digital_put(S: np.ndarray, K: float) -> np.ndarray:
    return (S[:, -1] < K).astype(float)


def binary_call(S: np.ndarray, K: float) -> np.ndarray:
    """Asset-or-nothing: pays S_T if S_T > K."""
    ST = S[:, -1]
    return ST * (ST > K)


# ---------------------------------------------------------------------------
# Asian (arithmetic average) payoffs
# ---------------------------------------------------------------------------

def asian_call(S: np.ndarray, K: float, include_S0: bool = False) -> np.ndarray:
    """Arithmetic-average-rate call: (A - K)+."""
    cols = slice(None) if include_S0 else slice(1, None)
    A = S[:, cols].mean(axis=1)
    return np.maximum(A - K, 0.0)


def asian_put(S: np.ndarray, K: float, include_S0: bool = False) -> np.ndarray:
    cols = slice(None) if include_S0 else slice(1, None)
    A = S[:, cols].mean(axis=1)
    return np.maximum(K - A, 0.0)


# ---------------------------------------------------------------------------
# Barrier payoffs  (down-and-out, up-and-out, knock-in variants)
# ---------------------------------------------------------------------------

def down_and_out_call(S: np.ndarray, K: float, H: float) -> np.ndarray:
    """Pays (S_T - K)+ if S never touched barrier H < S_0."""
    alive = np.all(S > H, axis=1)
    return alive * european_call(S, K)


def up_and_out_call(S: np.ndarray, K: float, H: float) -> np.ndarray:
    alive = np.all(S < H, axis=1)
    return alive * european_call(S, K)


def down_and_in_call(S: np.ndarray, K: float, H: float) -> np.ndarray:
    """Knock-in: pays (S_T - K)+ only if barrier H was hit."""
    knocked_in = np.any(S <= H, axis=1)
    return knocked_in * european_call(S, K)


def up_and_in_call(S: np.ndarray, K: float, H: float) -> np.ndarray:
    knocked_in = np.any(S >= H, axis=1)
    return knocked_in * european_call(S, K)


# ---------------------------------------------------------------------------
# Lookback payoffs
# ---------------------------------------------------------------------------

def lookback_call_fixed(S: np.ndarray, K: float) -> np.ndarray:
    """Fixed-strike lookback call: (max S - K)+."""
    return np.maximum(S.max(axis=1) - K, 0.0)


def lookback_put_fixed(S: np.ndarray, K: float) -> np.ndarray:
    return np.maximum(K - S.min(axis=1), 0.0)


def lookback_call_floating(S: np.ndarray) -> np.ndarray:
    """Floating-strike lookback call: S_T - min S."""
    return S[:, -1] - S.min(axis=1)


def lookback_put_floating(S: np.ndarray) -> np.ndarray:
    return S.max(axis=1) - S[:, -1]
