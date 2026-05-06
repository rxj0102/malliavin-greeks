"""
Geometric Brownian Motion model.

SDE (risk-neutral measure):
    dS_t = (r - q) S_t dt + σ S_t dW_t

Exact solution:
    S_T = S_0 · exp((r - q - σ²/2) T + σ W_T)

Key Malliavin objects
---------------------
Malliavin derivative of S_t (any s ≤ t):
    D_s S_t = σ S_t

Remarkably, this is INDEPENDENT of s — a perturbation δW_s at any time s ≤ t
scales all future values by the same multiplicative factor.  This follows from
differentiating the exact solution w.r.t. the Brownian path:

    D_s [S_0 exp(α t + σ W_t)] = σ S_t · D_s W_t = σ S_t · 1_{s≤t}

Score functions (∂/∂θ log p(S_T | S_0; θ))
--------------------------------------------
Holding S_T fixed, the log-density of the log-normal distribution gives:

    ∂/∂S_0 log p = W_T / (σ T S_0)

    ∂/∂σ log p   = (W_T² - T) / (σ T) - W_T

    ∂/∂r log p   = W_T / σ      [from r entering through the drift μ = r−q−σ²/2]

where W_T = (log(S_T/S_0) − (r−q−σ²/2)T) / σ  is the terminal Brownian motion
expressed as a function of the OBSERVED S_T (holding S_T fixed, W_T depends on θ).

Derivation note
~~~~~~~~~~~~~~~
The score w.r.t. σ (holding S_T fixed):
  log p = C − W_T²/(2) − log σ − log √T  (in terms of W_T = σ-standardised terminal BM)
  But W_T(σ) = (log S_T/S_0 − (r−q)T + σ²T/2) / σ, so dW_T/dσ = T − W_T/σ.
  Differentiating gives  ∂/∂σ log p = (W_T² − T)/(σT) − W_T.

This is the Malliavin / score-function (LR) vega weight.
"""

from __future__ import annotations

from typing import Optional
import numpy as np

from mgreeks.models.base import StochasticModel


class GeometricBrownianMotion(StochasticModel):
    """
    GBM: dS = (r − q) S dt + σ S dW  (risk-neutral measure).

    Simulation uses the EXACT log-normal solution — no Euler discretisation
    error.  A single time-step per monitoring date suffices for vanilla payoffs;
    more steps are needed for barrier/lookback monitoring.

    Parameters
    ----------
    r     : risk-free rate
    q     : continuous dividend yield
    sigma : volatility (σ > 0)
    """

    def __init__(self, r: float = 0.05, q: float = 0.0, sigma: float = 0.20):
        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")
        self.r = float(r)
        self.q = float(q)
        self.sigma = float(sigma)

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def simulate(
        self,
        S0: float,
        T: float,
        n_steps: int,
        n_paths: int,
        *,
        return_full_paths: bool = True,
        rng: Optional[np.random.Generator] = None,
    ) -> dict:
        """
        Exact GBM simulation.

        S_{t_{i+1}} = S_{t_i} · exp((r − q − σ²/2) Δt + σ √Δt · Z_i)

        Brownian increments:  ΔW_i = √Δt · Z_i  (stored for weight computations).
        """
        rng = self._default_rng(rng)
        times, dt = self._make_time_grid(T, n_steps)
        sqrt_dt = np.sqrt(dt)
        drift = (self.r - self.q - 0.5 * self.sigma**2) * dt

        # Draw n_paths × n_steps standard normals
        Z = rng.standard_normal((n_paths, n_steps))
        dW = sqrt_dt * Z                            # Brownian increments, (n_paths, n_steps)
        log_increments = drift + self.sigma * dW    # log-return per step

        if return_full_paths:
            log_S = np.empty((n_paths, n_steps + 1))
            log_S[:, 0] = np.log(S0)
            np.cumsum(log_increments, axis=1, out=log_S[:, 1:])
            log_S[:, 1:] += np.log(S0)
            paths = np.exp(log_S)
            terminal = paths[:, -1]
        else:
            log_ST = np.log(S0) + log_increments.sum(axis=1)
            terminal = np.exp(log_ST)
            paths = None

        return {
            "terminal": terminal,
            "paths": paths,
            "brownian_increments": dW,
            "times": times,
            "dt": dt,
        }

    # ------------------------------------------------------------------
    # Malliavin derivatives
    # ------------------------------------------------------------------

    def malliavin_derivative(
        self,
        paths: np.ndarray,
        brownian_increments: np.ndarray,
        s_index: int,
        t_index: int,
    ) -> np.ndarray:
        """
        D_{t_s} S_{t_t} = σ · S_{t_t}   for all s ≤ t.

        Independent of s — a consequence of GBM's multiplicative structure.
        """
        if s_index > t_index:
            raise ValueError(f"s_index ({s_index}) must be ≤ t_index ({t_index})")
        return self.sigma * paths[:, t_index]

    def malliavin_derivative_log(
        self,
        paths: np.ndarray,
        s_index: int,
        t_index: int,
    ) -> np.ndarray:
        """
        D_{t_s} log S_{t_t} = σ   for all s ≤ t.

        Constant — does not depend on the path.
        """
        return np.full(paths.shape[0], self.sigma)

    def first_variation(
        self,
        paths: np.ndarray,
        s_index: int = 0,
    ) -> np.ndarray:
        """
        First-variation process J_t = ∂S_t/∂S_0 = S_t / S_0.

        Returns the full path of J from s_index onward:
            J_{t_i} = S_{t_i} / S_{t_{s_index}}   (relative to the base at s_index)
        Shape: (n_paths, n_steps+1 - s_index).
        """
        S_s = paths[:, s_index : s_index + 1]     # (n_paths, 1) for broadcasting
        return paths[:, s_index:] / S_s

    # ------------------------------------------------------------------
    # Score functions
    # ------------------------------------------------------------------

    def log_density_gradient(
        self,
        S0: float,
        ST: np.ndarray,
        T: float,
        param_name: str,
        brownian_increments: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Score ∂/∂θ log p(S_T | S_0; θ), holding S_T fixed.

        The terminal Brownian W_T is inferred from S_T:
            W_T = (log(S_T/S_0) − (r−q−σ²/2)T) / σ

        Supported param_name values:
            'S0'    →  W_T / (σ T S_0)
            'sigma' →  (W_T² − T) / (σ T) − W_T
            'r'     →  W_T / σ
            'q'     →  −W_T / σ    (opposite sign to r)
            'T'     →  (r−q−σ²/2) + σ W_T / T  (Ito chain rule)
        """
        mu = self.r - self.q - 0.5 * self.sigma**2
        W_T = (np.log(ST / S0) - mu * T) / self.sigma

        if param_name == "S0":
            return W_T / (self.sigma * T * S0)

        elif param_name == "sigma":
            # d/dσ log p = (W_T² − T)/(σT) − W_T
            return (W_T**2 - T) / (self.sigma * T) - W_T

        elif param_name == "r":
            return W_T / self.sigma

        elif param_name == "q":
            return -W_T / self.sigma

        elif param_name == "T":
            # d/dT [discounted log-density]: used for theta weight
            return mu + self.sigma * W_T / T

        else:
            raise ValueError(f"Unknown param_name {param_name!r}")

    # ------------------------------------------------------------------
    # Terminal Brownian from simulated paths (convenience)
    # ------------------------------------------------------------------

    def W_T_from_increments(self, brownian_increments: np.ndarray) -> np.ndarray:
        """W_T = Σ ΔW_i, shape (n_paths,)."""
        return brownian_increments.sum(axis=1)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def param_dict(self) -> dict:
        return {"r": self.r, "q": self.q, "sigma": self.sigma}
