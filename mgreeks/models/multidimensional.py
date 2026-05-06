"""
Correlated multi-asset GBM model.

SDE for asset i (risk-neutral):
    dS^i_t = (r − q_i) S^i_t dt + σ_i S^i_t dW^i_t

with corr(dW^i, dW^j) = ρ_{ij}.

Simulation via Cholesky decomposition:
    dW = L · dZ,  L = chol(Ρ),  dZ ~ N(0, I_n dt)

so W^i_t = Σ_j L_{ij} Z^j_t.

Malliavin derivative of S^i_t w.r.t. W^j:
    D^j_s S^i_t = σ_i L_{ij} S^i_t   for all s ≤ t.

This follows from differentiating the exact solution:
    S^i_t = S^i_0 exp((r−q_i−σ_i²/2)t + σ_i Σ_j L_{ij} Z^j_t)

    D^j_s [S^i_t] = σ_i L_{ij} S^i_t · D^j_s Z^j_t = σ_i L_{ij} S^i_t.

Applications:
    - Basket option Greeks (sensitivity to each asset's initial price)
    - Cross-asset Greeks (correlation sensitivities)
    - Multi-asset Malliavin weights via BEL formula
"""

from __future__ import annotations

from typing import Optional
import numpy as np

from mgreeks.models.base import StochasticModel


class MultiAssetGBM(StochasticModel):
    """
    Correlated multi-asset GBM.

    Parameters
    ----------
    r           : risk-free rate (scalar, common)
    q           : dividend yields, shape (n_assets,) or scalar
    sigma       : volatilities, shape (n_assets,)
    correlation : correlation matrix, shape (n_assets, n_assets); default = I
    n_assets    : number of assets (inferred from sigma if provided)
    """

    def __init__(
        self,
        r: float = 0.05,
        q: Optional[np.ndarray] = None,
        sigma: Optional[np.ndarray] = None,
        correlation: Optional[np.ndarray] = None,
        n_assets: int = 2,
    ):
        self.r = float(r)
        self.n_assets = n_assets

        # Dividend yields
        if q is None:
            q = np.zeros(n_assets)
        self.q = np.asarray(q, dtype=float)
        if self.q.ndim == 0:
            self.q = np.full(n_assets, float(q))

        # Volatilities
        if sigma is None:
            sigma = np.full(n_assets, 0.20)
        self.sigma = np.asarray(sigma, dtype=float)
        if self.sigma.shape != (n_assets,):
            raise ValueError(f"sigma must have shape ({n_assets},)")

        # Correlation matrix
        if correlation is None:
            self.correlation = np.eye(n_assets)
        else:
            self.correlation = np.asarray(correlation, dtype=float)
            if self.correlation.shape != (n_assets, n_assets):
                raise ValueError(f"correlation must have shape ({n_assets}, {n_assets})")

        # Cholesky decomposition: W = L @ Z  (lower triangular)
        try:
            self.chol = np.linalg.cholesky(self.correlation)
        except np.linalg.LinAlgError:
            raise ValueError("Correlation matrix is not positive definite.")

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def simulate(
        self,
        S0: float | np.ndarray,
        T: float,
        n_steps: int,
        n_paths: int,
        *,
        return_full_paths: bool = True,
        rng: Optional[np.random.Generator] = None,
    ) -> dict:
        """
        Simulate n_assets correlated GBM paths simultaneously.

        Parameters
        ----------
        S0 : initial prices, shape (n_assets,) or scalar (replicated)

        Returns
        -------
        dict with:
            'terminal'            : shape (n_paths, n_assets)
            'paths'               : shape (n_paths, n_assets, n_steps+1)
            'brownian_increments' : shape (n_paths, n_assets, n_steps)
                                    correlated ΔW (after Cholesky)
            'indep_increments'    : shape (n_paths, n_assets, n_steps)
                                    independent ΔZ (before Cholesky)
            'times'               : shape (n_steps+1,)
        """
        rng = self._default_rng(rng)
        times, dt = self._make_time_grid(T, n_steps)
        sqrt_dt = np.sqrt(dt)

        # Initial prices
        S0_arr = np.asarray(S0, dtype=float)
        if S0_arr.ndim == 0:
            S0_arr = np.full(self.n_assets, float(S0))
        if S0_arr.shape != (self.n_assets,):
            raise ValueError(f"S0 must have shape ({self.n_assets},)")

        # Independent standard normals: (n_paths, n_assets, n_steps)
        Z = rng.standard_normal((n_paths, self.n_assets, n_steps))

        # Correlated increments: dW[path, asset, step] = sqrt_dt * (chol @ Z[:,asset,step])
        # chol has shape (n_assets, n_assets); Z has shape (n_paths, n_assets, n_steps)
        # dW[p, :, i] = sqrt_dt * chol @ Z[p, :, i]
        dW = sqrt_dt * np.einsum("ij,pjk->pik", self.chol, Z)   # (n_paths, n_assets, n_steps)

        # Log-returns per step: (r−q−σ²/2)dt + σ dW  for each asset
        drift = (self.r - self.q - 0.5 * self.sigma**2) * dt   # (n_assets,)

        # Simulate paths
        if return_full_paths:
            paths = np.empty((n_paths, self.n_assets, n_steps + 1))
            paths[:, :, 0] = S0_arr[np.newaxis, :]
            for i in range(n_steps):
                log_inc = drift[np.newaxis, :] + self.sigma[np.newaxis, :] * dW[:, :, i]
                paths[:, :, i + 1] = paths[:, :, i] * np.exp(log_inc)
            terminal = paths[:, :, -1]
        else:
            log_returns = drift[np.newaxis, :, np.newaxis] + self.sigma[np.newaxis, :, np.newaxis] * dW
            log_ST = np.log(S0_arr)[np.newaxis, :] + log_returns.sum(axis=2)
            terminal = np.exp(log_ST)
            paths = None

        return {
            "terminal": terminal,
            "paths": paths,
            "brownian_increments": dW,
            "indep_increments": Z,
            "times": times,
            "dt": dt,
            "S0": S0_arr,
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
        D^{W^1}_{t_s} S^1_{t_t} for asset 0.
        Use malliavin_derivative_ij for the full (i,j) cross-derivative.
        """
        return self.malliavin_derivative_ij(paths, s_index, t_index, asset=0, brownian=0)

    def malliavin_derivative_ij(
        self,
        paths: np.ndarray,
        s_index: int,
        t_index: int,
        asset: int,
        brownian: int,
    ) -> np.ndarray:
        """
        D^{W^j}_{t_s} S^i_{t_t} = σ_i L_{ij} S^i_{t_t}.

        Parameters
        ----------
        asset     : index i (which asset's price is differentiated)
        brownian  : index j (which Brownian drives the perturbation)

        Returns
        -------
        shape (n_paths,)
        """
        if s_index > t_index:
            raise ValueError(f"s_index ({s_index}) must be ≤ t_index ({t_index})")
        return self.sigma[asset] * self.chol[asset, brownian] * paths[:, asset, t_index]

    def malliavin_weight_delta(
        self,
        paths: np.ndarray,
        brownian_increments: np.ndarray,
        asset: int,
        T: float,
    ) -> np.ndarray:
        """
        BEL delta weight for asset `asset`:
            π_i = W^i_T / (σ_i L_{ii} S^i_0 T)

        where W^i_T = Σ_k ΔW^i_k are the correlated Brownian sums for asset i.
        """
        W_T_i = brownian_increments[:, asset, :].sum(axis=1)
        S0_i = paths[:, asset, 0]
        return W_T_i / (self.sigma[asset] * self.chol[asset, asset] * S0_i * T)

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
            "Multi-asset score not implemented. Use single-asset GBM for each asset."
        )

    # ------------------------------------------------------------------
    # Correlation sensitivity (vanna-volga style)
    # ------------------------------------------------------------------

    def correlation_sensitivity(
        self,
        payoff_fn,
        paths: np.ndarray,
        brownian_increments: np.ndarray,
        i: int,
        j: int,
        T: float,
    ) -> np.ndarray:
        """
        Sensitivity to ρ_{ij} via likelihood-ratio method.

        ∂/∂ρ_{ij} log p = d/dρ_{ij} [−½ Z^T Σ^{-1} Z]  evaluated on path.
        Returns per-path weight (not yet averaged).
        """
        raise NotImplementedError("Correlation Greek not yet implemented.")

    @property
    def param_dict(self) -> dict:
        return {
            "r": self.r,
            "n_assets": self.n_assets,
            "q": self.q.tolist(),
            "sigma": self.sigma.tolist(),
            "correlation": self.correlation.tolist(),
        }
