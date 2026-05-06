"""
Abstract base class for stochastic models.

A model specifies the SDE

    dX_t = μ(t, X_t) dt + σ(t, X_t) dW_t

and provides:
  - simulate()              : generate Monte Carlo paths
  - malliavin_derivative()  : D_s X_t — sensitivity of X_t to δW_s
  - log_density_gradient()  : score function ∂/∂θ log p(X_T | X_0; θ)

The Malliavin derivative D_s X_t is the fundamental object connecting
stochastic calculus to Greek computation.  It satisfies the VARIATIONAL
(first-variation) equation:

    d(D_s X_t) = ∂μ/∂x(t, X_t) · D_s X_t dt
               + ∂σ/∂x(t, X_t) · D_s X_t dW_t,   t > s

with initial condition D_s X_s = σ(s, X_s).

For GBM (dS = μS dt + σS dW) this simplifies to D_s S_t = σ S_t, constant in s.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class StochasticModel(ABC):
    """
    Abstract base for stochastic models used in Malliavin Greek computation.
    """

    # ------------------------------------------------------------------
    # Core abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def simulate(
        self,
        S0: float,
        T: float,
        n_steps: int,
        n_paths: int,
        *,
        return_full_paths: bool = False,
        rng: Optional[np.random.Generator] = None,
    ) -> dict:
        """
        Simulate paths of the model.

        Parameters
        ----------
        S0            : initial spot price
        T             : maturity in years
        n_steps       : number of time steps
        n_paths       : number of Monte Carlo paths
        return_full_paths : if False, only terminal values are returned
        rng           : numpy Generator for reproducibility

        Returns
        -------
        dict with keys:
            'terminal'            : S_T,  shape (n_paths,)
            'paths'               : S paths, shape (n_paths, n_steps+1)
                                    (None if return_full_paths=False)
            'brownian_increments' : ΔW_i = W_{t_{i+1}} - W_{t_i},
                                    shape (n_paths, n_steps)
            'times'               : time grid t_0 … t_n, shape (n_steps+1,)
            'dt'                  : step size T / n_steps
        """

    @abstractmethod
    def malliavin_derivative(
        self,
        paths: np.ndarray,
        brownian_increments: np.ndarray,
        s_index: int,
        t_index: int,
    ) -> np.ndarray:
        """
        Compute D_{t_s} X_{t_t} — the Malliavin derivative of X at t_t
        with respect to a Brownian perturbation at t_s (s_index ≤ t_index).

        Parameters
        ----------
        paths                : shape (n_paths, n_steps+1)
        brownian_increments  : shape (n_paths, n_steps)
        s_index              : column index of the perturbation time
        t_index              : column index of the evaluation time

        Returns
        -------
        shape (n_paths,)
        """

    @abstractmethod
    def log_density_gradient(
        self,
        S0: float,
        ST: np.ndarray,
        T: float,
        param_name: str,
        brownian_increments: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Score function: ∂/∂θ log p(S_T | S_0; θ).

        Used for likelihood-ratio (LR) Greek estimation.

        Parameters
        ----------
        S0            : initial spot
        ST            : terminal values, shape (n_paths,)
        T             : maturity
        param_name    : 'S0', 'sigma', 'r', 'T', etc.
        brownian_increments : may be needed for some parameterisations

        Returns
        -------
        score values, shape (n_paths,)
        """

    @property
    @abstractmethod
    def param_dict(self) -> dict:
        """Return model parameters as a plain dict."""

    # ------------------------------------------------------------------
    # Convenience helpers (non-abstract)
    # ------------------------------------------------------------------

    def _default_rng(self, rng: Optional[np.random.Generator]) -> np.random.Generator:
        return rng if rng is not None else np.random.default_rng()

    def _make_time_grid(self, T: float, n_steps: int) -> tuple[np.ndarray, float]:
        """Return (times, dt)."""
        dt = T / n_steps
        times = np.linspace(0.0, T, n_steps + 1)
        return times, dt

    def terminal_brownian(self, brownian_increments: np.ndarray) -> np.ndarray:
        """W_T = sum of increments, shape (n_paths,)."""
        return brownian_increments.sum(axis=1)

    def __repr__(self) -> str:
        pstr = ", ".join(f"{k}={v}" for k, v in self.param_dict.items())
        return f"{self.__class__.__name__}({pstr})"
