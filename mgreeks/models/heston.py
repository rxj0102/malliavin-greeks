"""
Heston stochastic volatility model.

SDE (risk-neutral):
    dS_t = (r − q) S_t dt + √V_t S_t dW^S_t
    dV_t = κ(θ − V_t) dt + ξ √V_t dW^V_t
    corr(dW^S, dW^V) = ρ

Two simulation schemes are provided:

1. Euler-Maruyama (full truncation)
   - Simple, fast, but biased for large dt or when Feller condition fails
   - V_{t+dt} = max(V_t + κ(θ−max(V_t,0))dt + ξ√max(V_t,0)·ΔW^V, 0)

2. Quadratic-Exponential (QE) scheme  [Andersen 2008]
   - Accurately matches the conditional distribution of V_{t+dt}|V_t
   - The conditional distribution is noncentral chi-squared; QE fits it
     by matching the first two moments with either a quadratic or
     exponential form depending on ψ = Var/Mean²

   QE algorithm for V_{t+dt}|V_t ~ noncentral χ²:
     m  = θ + (V_t − θ) e^{−κ dt}                     (conditional mean)
     s² = V_t ξ² e^{−κdt}(1−e^{−κdt})/κ
          + θ ξ²(1−e^{−κdt})²/(2κ)                     (conditional variance)
     ψ  = s²/m²

     If ψ ≤ ψ_c (≈ 1.5):  quadratic form
       b² = 2ψ^{-1} − 1 + √(2ψ^{-1}) √(2ψ^{-1}−1)
       a  = m / (1 + b²)
       V_{t+dt} = a(b + Z)²,  Z ~ N(0,1)

     If ψ > ψ_c:  exponential form
       p  = (ψ−1)/(ψ+1),  β = (1−p)/m
       V_{t+dt} = Ψ^{-1}(U),  U ~ Uniform(0,1)
                = { 0              if U ≤ p
                  { ln((1−p)/(1−U)) / β  if U > p

   The spot increment given V is then simulated using the log-Euler scheme
   for S with the QE variance realisation and correlated Brownian:
     log S_{t+dt} = log S_t + (r−q−V_t/2) dt + √V_t · ΔW^S
   where ΔW^S ~ N(ρ·ΔW^V / √dt, √(1−ρ²)·√dt · Z_S).

Malliavin Derivative (variational equations)
--------------------------------------------
D^{W^S}_s S_t satisfies (using Ito's product rule on S_t = S_0 exp(...)):

    d(D^{W^S}_s S_t) = (r−q) D^S S_t dt + D(√V_t S_t) dW^S_t

In practice, for the BEL delta weight we use the approximation:

    D^{W^S}_s S_t ≈ √V_s · S_t / S_s   (leading-order in dt)

which follows from the first-variation equation solved at the initial
state.  The full path-dependent version integrates the variational SDE
forward; we provide both options.
"""

from __future__ import annotations

from typing import Optional
import numpy as np

from mgreeks.models.base import StochasticModel


class HestonModel(StochasticModel):
    """
    Heston stochastic volatility model.

    Parameters
    ----------
    r     : risk-free rate
    q     : dividend yield
    kappa : mean-reversion speed of variance (κ > 0)
    theta : long-run variance (θ > 0)
    xi    : vol-of-vol (ξ > 0)
    rho   : correlation W^S and W^V (|ρ| < 1)
    V0    : initial variance
    """

    def __init__(
        self,
        r: float = 0.05,
        q: float = 0.0,
        kappa: float = 2.0,
        theta: float = 0.04,
        xi: float = 0.30,
        rho: float = -0.70,
        V0: float = 0.04,
    ):
        self.r = float(r)
        self.q = float(q)
        self.kappa = float(kappa)
        self.theta = float(theta)
        self.xi = float(xi)
        self.rho = float(rho)
        self.V0 = float(V0)

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
        scheme: str = "qe",
        rng: Optional[np.random.Generator] = None,
    ) -> dict:
        """
        Simulate Heston paths.

        Parameters
        ----------
        scheme : 'euler' or 'qe' (Andersen Quadratic-Exponential, default)

        Returns
        -------
        dict with keys:
            'terminal', 'paths', 'v_paths', 'brownian_increments',
            'v_brownian_increments', 'times', 'dt'

        brownian_increments[i] = ΔW^S_i (the spot Brownian increments).
        v_brownian_increments[i] = ΔW^V_i (the variance Brownian increments).
        """
        rng = self._default_rng(rng)
        times, dt = self._make_time_grid(T, n_steps)

        if scheme == "euler":
            return self._simulate_euler(S0, T, n_steps, n_paths, dt, times, rng,
                                        return_full_paths)
        elif scheme == "qe":
            return self._simulate_qe(S0, T, n_steps, n_paths, dt, times, rng,
                                     return_full_paths)
        else:
            raise ValueError(f"Unknown scheme {scheme!r}. Use 'euler' or 'qe'.")

    def _simulate_euler(self, S0, T, n_steps, n_paths, dt, times, rng,
                        return_full_paths):
        sqrt_dt = np.sqrt(dt)
        rho_bar = np.sqrt(max(1.0 - self.rho**2, 0.0))

        # Independent standard normals
        Z_S = rng.standard_normal((n_paths, n_steps))
        Z_V = rng.standard_normal((n_paths, n_steps))

        # Correlated increments
        dW_S = sqrt_dt * (self.rho * Z_V + rho_bar * Z_S)
        dW_V = sqrt_dt * Z_V

        S = np.empty((n_paths, n_steps + 1))
        V = np.empty((n_paths, n_steps + 1))
        S[:, 0] = S0
        V[:, 0] = self.V0

        for i in range(n_steps):
            V_pos = np.maximum(V[:, i], 0.0)
            sqrt_V = np.sqrt(V_pos)
            S[:, i + 1] = S[:, i] * np.exp(
                (self.r - self.q - 0.5 * V_pos) * dt + sqrt_V * dW_S[:, i]
            )
            V[:, i + 1] = (
                V[:, i]
                + self.kappa * (self.theta - V_pos) * dt
                + self.xi * sqrt_V * dW_V[:, i]
            )
            # Full truncation: set used V to 0 if negative (already done above)

        return self._pack_result(S, V, dW_S, dW_V, times, dt, return_full_paths)

    def _simulate_qe(self, S0, T, n_steps, n_paths, dt, times, rng,
                     return_full_paths):
        """
        Quadratic-Exponential scheme for V; log-Euler for S.
        Following Andersen (2008), with ψ_c = 1.5.
        """
        psi_c = 1.5
        sqrt_dt = np.sqrt(dt)
        rho_bar = np.sqrt(max(1.0 - self.rho**2, 0.0))
        exp_kdt = np.exp(-self.kappa * dt)
        xi2 = self.xi**2

        # Precompute QE parameters for V transition
        c1 = xi2 * exp_kdt * (1.0 - exp_kdt) / self.kappa
        c2 = self.theta * xi2 * (1.0 - exp_kdt)**2 / (2.0 * self.kappa)

        # Random draws
        Z_S = rng.standard_normal((n_paths, n_steps))
        Z_QE = rng.standard_normal((n_paths, n_steps))   # for quadratic branch
        U_QE = rng.uniform(0.0, 1.0, (n_paths, n_steps)) # for exponential branch

        S = np.empty((n_paths, n_steps + 1))
        V = np.empty((n_paths, n_steps + 1))
        dW_S = np.empty((n_paths, n_steps))
        dW_V = np.empty((n_paths, n_steps))   # approximate variance Brownian

        S[:, 0] = S0
        V[:, 0] = self.V0

        for i in range(n_steps):
            V_cur = V[:, i]

            # --- Step 1: simulate V_{t+dt} via QE ---
            m  = self.theta + (V_cur - self.theta) * exp_kdt      # conditional mean
            s2 = V_cur * c1 + c2                                   # conditional variance
            psi = s2 / np.maximum(m**2, 1e-12)

            V_next = np.empty(n_paths)

            # Quadratic branch: ψ ≤ ψ_c
            mask_q = psi <= psi_c
            if mask_q.any():
                b2 = 2.0 / psi[mask_q] - 1.0 + np.sqrt(2.0 / psi[mask_q]) * np.sqrt(2.0 / psi[mask_q] - 1.0)
                b2 = np.maximum(b2, 0.0)
                a  = m[mask_q] / (1.0 + b2)
                V_next[mask_q] = a * (np.sqrt(b2) + Z_QE[mask_q, i])**2

            # Exponential branch: ψ > ψ_c
            mask_e = ~mask_q
            if mask_e.any():
                p  = (psi[mask_e] - 1.0) / (psi[mask_e] + 1.0)
                beta = (1.0 - p) / np.maximum(m[mask_e], 1e-12)
                U = U_QE[mask_e, i]
                V_exp = np.where(U <= p, 0.0, np.log((1.0 - p) / np.maximum(1.0 - U, 1e-15)) / beta)
                V_next[mask_e] = V_exp

            V[:, i + 1] = V_next

            # --- Step 2: log-Euler for S given V ---
            V_mid = 0.5 * (V_cur + V_next)               # trapezoidal V average
            sqrt_V_cur = np.sqrt(np.maximum(V_cur, 0.0))

            # Recover approximate ΔW^V from V transition for correlation
            # ΔW^V ≈ (V_next - V_cur - κ(θ - V_cur)dt) / (ξ √V_cur)
            dV = V_next - V_cur
            dW_V[:, i] = np.where(
                sqrt_V_cur > 1e-8,
                (dV - self.kappa * (self.theta - V_cur) * dt) / (self.xi * sqrt_V_cur),
                0.0,
            )

            # Spot Brownian increment: ρ · ΔW^V + √(1-ρ²) · Z_S · √dt
            dW_S[:, i] = self.rho * dW_V[:, i] + rho_bar * sqrt_dt * Z_S[:, i]

            S[:, i + 1] = S[:, i] * np.exp(
                (self.r - self.q - 0.5 * np.maximum(V_mid, 0.0)) * dt
                + np.sqrt(np.maximum(V_cur, 0.0)) * dW_S[:, i]
            )

        return self._pack_result(S, V, dW_S, dW_V, times, dt, return_full_paths)

    def _pack_result(self, S, V, dW_S, dW_V, times, dt, return_full_paths):
        result = {
            "terminal": S[:, -1],
            "v_terminal": V[:, -1],
            "paths": S if return_full_paths else None,
            "v_paths": V if return_full_paths else None,
            "brownian_increments": dW_S,
            "v_brownian_increments": dW_V,
            "times": times,
            "dt": dt,
        }
        return result

    # ------------------------------------------------------------------
    # Malliavin derivative
    # ------------------------------------------------------------------

    def malliavin_derivative(
        self,
        paths: np.ndarray,
        brownian_increments: np.ndarray,
        s_index: int,
        t_index: int,
    ) -> np.ndarray:
        """
        Approximate D^{W^S}_{t_s} S_{t_t} for the Heston model.

        Leading-order approximation (valid when ξ is small or dt is fine):
            D^{W^S}_s S_t ≈ √V_s · S_t / S_s

        For the full variational equation approach, use
        malliavin_derivative_full() which integrates the 2D variational SDE.
        """
        raise NotImplementedError(
            "Call malliavin_derivative_spot() for the spot perturbation "
            "or malliavin_derivative_full() for the complete 2D system."
        )

    def malliavin_derivative_spot(
        self,
        paths: np.ndarray,
        v_paths: np.ndarray,
        s_index: int,
        t_index: int,
    ) -> np.ndarray:
        """
        Leading-order Malliavin derivative D^{W^S}_{t_s} S_{t_t}.

        Approximation:  D^{W^S}_s S_t ≈ √V_s · (S_t / S_s)

        Exact for GBM (ξ=0).  Increasingly accurate for fine grids.
        """
        if s_index > t_index:
            raise ValueError(f"s_index ({s_index}) must be ≤ t_index ({t_index})")
        sqrt_Vs = np.sqrt(np.maximum(v_paths[:, s_index], 0.0))
        ratio = paths[:, t_index] / np.maximum(paths[:, s_index], 1e-12)
        return sqrt_Vs * ratio

    def malliavin_derivative_full(
        self,
        paths: np.ndarray,
        v_paths: np.ndarray,
        dW_S: np.ndarray,
        dW_V: np.ndarray,
        s_index: int,
        t_index: int,
        dt: float,
    ) -> np.ndarray:
        """
        Full variational SDE for D^{W^S}_s (S_t, V_t).

        Integrates the linearised system forward from s_index to t_index:

            d(DS) = (r−q) DS dt + ½ DV · S / √V dW^S dt + √V DS dW^S
            d(DV) = −κ DV dt + ½ ξ DV / √V dW^V dt

        with (DS, DV)|_{t_s} = (√V_s · S_s, 0).

        Returns DS at t_index, shape (n_paths,).
        """
        n_paths = paths.shape[0]
        DS = np.sqrt(np.maximum(v_paths[:, s_index], 0.0)) * paths[:, s_index]
        DV = np.zeros(n_paths)

        for i in range(s_index, t_index):
            V_i = np.maximum(v_paths[:, i], 1e-10)
            sqrt_V = np.sqrt(V_i)
            S_i = paths[:, i]

            # Variational SDE for DS (Euler step)
            DS_new = DS + (
                (self.r - self.q) * DS * dt
                + 0.5 * DV * S_i / sqrt_V * dW_S[:, i]
                + sqrt_V * DS * dW_S[:, i]
            )
            # Variational SDE for DV
            DV_new = DV + (
                -self.kappa * DV * dt
                + 0.5 * self.xi * DV / sqrt_V * dW_V[:, i]
            )
            DS, DV = DS_new, DV_new

        return DS

    # ------------------------------------------------------------------
    # Score function (Euler approximation)
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
        Score for the Heston model (approximate, using integrated variance).

        Under Heston, the conditional distribution of log S_T | {V_t}
        is Gaussian with mean (r−q) T − ½∫V dt and variance ∫V dt.

        For delta (∂/∂S0): W_T / (√V̄ T S0) where V̄ = (∫V dt) / T.
        """
        if brownian_increments is None:
            raise ValueError(
                "Heston score requires brownian_increments (ΔW^S path)."
            )
        W_T = brownian_increments.sum(axis=1)

        if param_name == "S0":
            # Use the spot Brownian for the score approximation
            dt = T / brownian_increments.shape[1]
            avg_sqrt_V = np.sqrt(np.maximum((W_T**2) / T, 0.01))
            return W_T / (avg_sqrt_V * T * S0)
        else:
            raise NotImplementedError(
                f"Score for {param_name!r} not implemented for Heston; "
                "use numerical finite differences."
            )

    # ------------------------------------------------------------------
    # Feller condition check
    # ------------------------------------------------------------------

    def feller_satisfied(self) -> bool:
        """Return True if 2κθ > ξ² (V stays strictly positive a.s.)."""
        return 2.0 * self.kappa * self.theta > self.xi**2

    @property
    def param_dict(self) -> dict:
        return {
            "r": self.r,
            "q": self.q,
            "kappa": self.kappa,
            "theta": self.theta,
            "xi": self.xi,
            "rho": self.rho,
            "V0": self.V0,
        }
