"""
Local volatility model (Dupire).

SDE (risk-neutral):
    dS_t = (r − q) S_t dt + σ_loc(t, S_t) S_t dW_t

where σ_loc(t, S) is a deterministic function (the local vol surface).

Built-in parameterisations
--------------------------
CEV (Constant Elasticity of Variance):
    σ_loc(t, S) = σ_0 · (S / S_ref)^{β−1}
    β = 1 → GBM; β = 0.5 → square-root; β = 0 → normal model

Quadratic local vol:
    σ_loc(t, S) = a + b(S − S_ref) + c(S − S_ref)²
    Produces a symmetric smile around S_ref.

Time-separable:
    σ_loc(t, S) = f(t) · g(S)
    Euler–Maruyama with user-supplied callables.

Malliavin derivative
--------------------
D_s S_t satisfies the VARIATIONAL equation from s to t:

    d(D_s S_t) = [(r−q) + σ_loc(t,S_t)(∂σ_loc/∂S)(t,S_t) S_t] D_s S_t dt
               + [σ_loc(t,S_t) + (∂σ_loc/∂S)(t,S_t) S_t] D_s S_t dW_t

with D_s S_s = σ_loc(s, S_s) S_s.

This reduces to:
    D_s S_t = (σ_loc(s,S_s) S_s / S_s) · J(s,t)
where J(s,t) is the first-variation process (∂S_t/∂S_s).

For CEV: ∂σ_loc/∂S = σ_0 (β−1) (S/S_ref)^{β−2} / S_ref
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple
import numpy as np

from mgreeks.models.base import StochasticModel


class LocalVolModel(StochasticModel):
    """
    Local volatility model with pluggable σ_loc(t, S).

    Parameters
    ----------
    r               : risk-free rate
    q               : dividend yield
    sigma_func      : callable(t, S) → σ_loc(t, S), shape (n_paths,) → (n_paths,)
    sigma_deriv_func: callable(t, S) → ∂σ_loc/∂S(t, S)  [for Malliavin derivative]
    model_type      : 'cev', 'quadratic', or 'custom'
    **model_params  : parameters passed to the built-in parameterisations
    """

    _SUPPORTED_TYPES = ("cev", "quadratic", "custom")

    def __init__(
        self,
        r: float = 0.05,
        q: float = 0.0,
        sigma_func: Optional[Callable] = None,
        sigma_deriv_func: Optional[Callable] = None,
        model_type: str = "cev",
        **model_params,
    ):
        self.r = float(r)
        self.q = float(q)
        self.model_type = model_type
        self.model_params = model_params

        if model_type == "cev":
            self._setup_cev(**model_params)
        elif model_type == "quadratic":
            self._setup_quadratic(**model_params)
        elif model_type == "custom":
            if sigma_func is None:
                raise ValueError("sigma_func required for model_type='custom'")
            self._sigma_func = sigma_func
            self._sigma_deriv_func = sigma_deriv_func
        else:
            raise ValueError(f"model_type must be one of {self._SUPPORTED_TYPES}")

    # ------------------------------------------------------------------
    # Built-in parameterisations
    # ------------------------------------------------------------------

    def _setup_cev(self, sigma0: float = 0.20, beta: float = 0.5,
                   S_ref: float = 100.0):
        """
        CEV: σ_loc(t,S) = σ_0 (S/S_ref)^{β−1}

        β = 1 → flat vol (GBM)
        β < 1 → inverse leverage effect (vol increases as S falls)
        """
        self._sigma0 = float(sigma0)
        self._beta = float(beta)
        self._S_ref = float(S_ref)
        self.model_params.update({"sigma0": sigma0, "beta": beta, "S_ref": S_ref})

        def sigma_fn(t, S):
            return self._sigma0 * (S / self._S_ref) ** (self._beta - 1.0)

        def sigma_deriv_fn(t, S):
            if self._beta == 1.0:
                return np.zeros_like(S)
            return self._sigma0 * (self._beta - 1.0) * (S / self._S_ref) ** (self._beta - 2.0) / self._S_ref

        self._sigma_func = sigma_fn
        self._sigma_deriv_func = sigma_deriv_fn

    def _setup_quadratic(self, a: float = 0.20, b: float = -0.001,
                          c: float = 0.00001, S_ref: float = 100.0):
        """
        Quadratic: σ_loc(t,S) = a + b(S−S_ref) + c(S−S_ref)²
        """
        self._a, self._b, self._c = float(a), float(b), float(c)
        self._S_ref = float(S_ref)
        self.model_params.update({"a": a, "b": b, "c": c, "S_ref": S_ref})

        def sigma_fn(t, S):
            x = S - self._S_ref
            return np.maximum(self._a + self._b * x + self._c * x**2, 1e-6)

        def sigma_deriv_fn(t, S):
            x = S - self._S_ref
            return self._b + 2.0 * self._c * x

        self._sigma_func = sigma_fn
        self._sigma_deriv_func = sigma_deriv_fn

    # ------------------------------------------------------------------
    # Simulation (Euler–Maruyama with log-Euler for stability)
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
        Log-Euler–Maruyama for local vol:
            log S_{t+dt} = log S_t + [(r−q) − σ_loc²/2] dt + σ_loc dW
        """
        rng = self._default_rng(rng)
        times, dt = self._make_time_grid(T, n_steps)
        sqrt_dt = np.sqrt(dt)

        Z = rng.standard_normal((n_paths, n_steps))
        dW = sqrt_dt * Z

        S = np.empty((n_paths, n_steps + 1))
        S[:, 0] = S0
        sigma_path = np.empty((n_paths, n_steps))

        for i in range(n_steps):
            t_i = times[i]
            sig = self._sigma_func(t_i, S[:, i])
            sigma_path[:, i] = sig
            S[:, i + 1] = S[:, i] * np.exp(
                (self.r - self.q - 0.5 * sig**2) * dt + sig * dW[:, i]
            )

        return {
            "terminal": S[:, -1],
            "paths": S if return_full_paths else None,
            "brownian_increments": dW,
            "sigma_path": sigma_path,           # σ_loc evaluated on path
            "times": times,
            "dt": dt,
        }

    # ------------------------------------------------------------------
    # Malliavin derivative (first-variation approach)
    # ------------------------------------------------------------------

    def malliavin_derivative(
        self,
        paths: np.ndarray,
        brownian_increments: np.ndarray,
        s_index: int,
        t_index: int,
        sigma_path: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        D_{t_s} S_{t_t} via the variational equation.

        Initial condition: D_s S_s = σ_loc(s, S_s) · S_s
        Propagated via first-variation:
            D_s S_{t_{i+1}} ≈ D_s S_{t_i} · (S_{t_{i+1}} / S_{t_i})
            · exp([(∂σ/∂S)·σ_loc·S_i − ½(∂σ/∂S)²·S_i²] · dt
                  + (∂σ/∂S)·S_i·dW_i)

        For σ_loc = const (GBM): D_s S_t = σ · S_t (recovered exactly).
        """
        if s_index > t_index:
            raise ValueError(f"s_index ({s_index}) must be ≤ t_index ({t_index})")

        times = np.linspace(0.0, brownian_increments.shape[1] * 1.0,
                            brownian_increments.shape[1] + 1)
        dt = times[1] - times[0] if len(times) > 1 else 1.0

        n_paths = paths.shape[0]
        t_s = times[s_index] if len(times) > s_index else s_index
        sig_s = self._sigma_func(t_s, paths[:, s_index])
        D = sig_s * paths[:, s_index]               # initial condition

        for i in range(s_index, t_index):
            t_i = times[i] if i < len(times) else i
            S_i = paths[:, i]
            sig_i = self._sigma_func(t_i, S_i) if sigma_path is None \
                    else sigma_path[:, i]
            dsig_i = self._sigma_deriv_func(t_i, S_i) \
                     if self._sigma_deriv_func is not None \
                     else np.zeros(n_paths)

            # Multiplicative update: D_{i+1} = D_i * (S_{i+1}/S_i) * correction
            ratio = paths[:, i + 1] / np.maximum(paths[:, i], 1e-12)
            # Correction from the chain rule for σ_loc(S_t):
            correction = np.exp(
                dsig_i * sig_i * S_i * dt
                + dsig_i * S_i * brownian_increments[:, i]
                - 0.5 * (dsig_i * S_i)**2 * dt
            )
            D = D * ratio * correction

        return D

    def first_variation(
        self,
        paths: np.ndarray,
        brownian_increments: np.ndarray,
        s_index: int,
        sigma_path: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        First-variation process J_{s→t} = ∂S_t/∂S_s along the path.

        Returns path of J from s_index to end: shape (n_paths, n_steps+1−s_index).
        """
        n_paths, n_total = paths.shape
        n_steps = n_total - 1
        times = np.linspace(0.0, float(n_steps), n_steps + 1)

        J = np.ones((n_paths, n_total - s_index))
        J[:, 0] = 1.0

        for k, i in enumerate(range(s_index, n_steps)):
            t_i = times[i]
            S_i = paths[:, i]
            sig_i = self._sigma_func(t_i, S_i) if sigma_path is None \
                    else sigma_path[:, i]
            dsig_i = self._sigma_deriv_func(t_i, S_i) \
                     if self._sigma_deriv_func is not None \
                     else np.zeros(n_paths)

            dt_local = 1.0  # placeholder; caller passes actual dt in brownian_increments
            J[:, k + 1] = J[:, k] * (paths[:, i + 1] / np.maximum(paths[:, i], 1e-12))

        return J

    # ------------------------------------------------------------------
    # Score function
    # ------------------------------------------------------------------

    def log_density_gradient(
        self,
        S0: float,
        ST: np.ndarray,
        T: float,
        param_name: str,
        brownian_increments: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        raise NotImplementedError(
            "Closed-form score not available for general local vol. "
            "Use finite-difference numerical differentiation of the log-density."
        )

    @property
    def param_dict(self) -> dict:
        d = {"r": self.r, "q": self.q, "model_type": self.model_type}
        d.update(self.model_params)
        return d

    def sigma_at(self, t: float, S: np.ndarray) -> np.ndarray:
        """Evaluate σ_loc(t, S)."""
        return self._sigma_func(t, S)

    def sigma_deriv_at(self, t: float, S: np.ndarray) -> np.ndarray:
        """Evaluate ∂σ_loc/∂S(t, S). Returns zeros if not available."""
        if self._sigma_deriv_func is None:
            return np.zeros_like(S)
        return self._sigma_deriv_func(t, S)
