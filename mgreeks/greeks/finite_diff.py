"""
Finite-difference (bump-and-revalue) Greek estimators.

Each Greek requires 2–3 full Monte Carlo simulations:
  Delta : V(S+h), V(S-h)          → 2 sims
  Gamma : V(S+h), V(S), V(S-h)   → 3 sims
  Vega  : V(σ+h), V(σ-h)         → 2 sims
  Rho   : V(r+h), V(r-h)         → 2 sims
  Theta : V(T+h), V(T-h)         → 2 sims

Common Random Numbers (CRN) is used by default: both bumped simulations
share the same seed, so path-level differences cancel noise.

CRITICAL LIMITATION — discontinuous payoffs
-------------------------------------------
For digital/barrier options, the per-path difference f(S+h) - f(S-h) is
non-zero only for paths near the strike/barrier, and Var[(f_up - f_dn)/h]
→ ∞ as h → 0.  The optimal bump h balances bias (O(h²)) and variance
(O(1/(n h²))):  h_opt ~ n^{-1/4} for smooth payoffs; for discontinuous
payoffs no finite h gives a consistent estimator.
"""

from __future__ import annotations

import numpy as np

from mgreeks.models.base import StochasticModel
from mgreeks.simulation import MonteCarloEngine


_Z95 = 1.959964


class FiniteDifferenceGreeks:
    """
    Bump-and-revalue Greek estimator using central finite differences.

    Parameters
    ----------
    model     : stochastic model
    engine    : MonteCarloEngine (provides n_paths, n_steps, rng_seed)
    bump_size : size of the parameter bump
    bump_type : 'relative' (h = bump_size × |param|) or 'absolute' (h = bump_size)
    """

    def __init__(
        self,
        model: StochasticModel,
        engine: MonteCarloEngine,
        bump_size: float = 0.01,
        bump_type: str = "relative",
    ):
        self.model = model
        self.engine = engine
        self.bump_size = float(bump_size)
        self.bump_type = bump_type

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _bump(self, param_value: float) -> float:
        if self.bump_type == "relative":
            return self.bump_size * max(abs(param_value), 1e-8)
        return self.bump_size

    def _simulate(self, S0: float, T: float, seed=None) -> dict:
        rng = np.random.default_rng(seed if seed is not None else self.engine.rng_seed)
        return self.model.simulate(
            S0, T, self.engine.n_steps, self.engine.n_paths,
            return_full_paths=True, rng=rng,
        )

    @staticmethod
    def _result(samples: np.ndarray, disc: float) -> dict:
        vals = disc * samples
        mean = float(vals.mean())
        se = float(vals.std(ddof=1) / np.sqrt(len(vals)))
        return {
            "value": mean,
            "std_error": se,
            "ci_lower": mean - _Z95 * se,
            "ci_upper": mean + _Z95 * se,
            "n_paths": len(vals),
        }

    @staticmethod
    def _scalar_result(value: float, n_paths: int) -> dict:
        return {
            "value": float(value),
            "std_error": float("nan"),
            "ci_lower": float("nan"),
            "ci_upper": float("nan"),
            "n_paths": n_paths,
        }

    def _disc(self, T: float, r: float = None) -> float:
        if r is None:
            r = getattr(self.model, "r", 0.0)
        return float(np.exp(-r * T))

    # ------------------------------------------------------------------
    # Greeks
    # ------------------------------------------------------------------

    def delta(
        self,
        payoff,
        S0: float,
        T: float,
        use_common_randoms: bool = True,
    ) -> dict:
        """
        Central-difference delta: (V(S+h) - V(S-h)) / (2h).

        With use_common_randoms=True, both sims share the same seed, so
        the per-path difference f_up - f_dn cancels most of the variance.
        """
        h = self._bump(S0)
        seed = self.engine.rng_seed if use_common_randoms else None

        sim_up = self._simulate(S0 + h, T, seed=seed)
        sim_dn = self._simulate(S0 - h, T, seed=seed)

        f_up = payoff(sim_up["paths"], sim_up["times"])
        f_dn = payoff(sim_dn["paths"], sim_dn["times"])

        samples = (f_up - f_dn) / (2.0 * h)
        return self._result(samples, self._disc(T))

    def gamma(
        self,
        payoff,
        S0: float,
        T: float,
        use_common_randoms: bool = True,
    ) -> dict:
        """
        Second-difference gamma: (V(S+h) - 2V(S) + V(S-h)) / h².
        """
        h = self._bump(S0)
        seed = self.engine.rng_seed if use_common_randoms else None

        sim_c = self._simulate(S0, T, seed=seed)
        sim_u = self._simulate(S0 + h, T, seed=seed)
        sim_d = self._simulate(S0 - h, T, seed=seed)

        f_c = payoff(sim_c["paths"], sim_c["times"])
        f_u = payoff(sim_u["paths"], sim_u["times"])
        f_d = payoff(sim_d["paths"], sim_d["times"])

        samples = (f_u - 2.0 * f_c + f_d) / (h**2)
        return self._result(samples, self._disc(T))

    def vega(
        self,
        payoff,
        S0: float,
        T: float,
        use_common_randoms: bool = True,
    ) -> dict:
        """Central-difference vega: (V(σ+h) - V(σ-h)) / (2h)."""
        sigma0 = self.model.sigma
        h = self._bump(sigma0)
        seed = self.engine.rng_seed if use_common_randoms else None

        try:
            self.model.sigma = sigma0 + h
            sim_u = self._simulate(S0, T, seed=seed)
            f_u = payoff(sim_u["paths"], sim_u["times"])

            self.model.sigma = sigma0 - h
            sim_d = self._simulate(S0, T, seed=seed)
            f_d = payoff(sim_d["paths"], sim_d["times"])
        finally:
            self.model.sigma = sigma0

        samples = (f_u - f_d) / (2.0 * h)
        return self._result(samples, self._disc(T))

    def rho(self, payoff, S0: float, T: float) -> dict:
        """
        Central-difference rho: (V(r+h) - V(r-h)) / (2h).

        V includes the discount factor e^{-rT}, so both the drift and
        the discount change with r.
        """
        r0 = getattr(self.model, "r", 0.0)
        h = self._bump(max(abs(r0), 1e-4))
        seed = self.engine.rng_seed

        try:
            self.model.r = r0 + h
            sim_u = self._simulate(S0, T, seed=seed)
            f_u = payoff(sim_u["paths"], sim_u["times"])
            price_u = self._disc(T, r0 + h) * f_u

            self.model.r = r0 - h
            sim_d = self._simulate(S0, T, seed=seed)
            f_d = payoff(sim_d["paths"], sim_d["times"])
            price_d = self._disc(T, r0 - h) * f_d
        finally:
            self.model.r = r0

        samples = (price_u - price_d) / (2.0 * h)
        mean = float(samples.mean())
        se = float(samples.std(ddof=1) / np.sqrt(len(samples)))
        return {
            "value": mean,
            "std_error": se,
            "ci_lower": mean - _Z95 * se,
            "ci_upper": mean + _Z95 * se,
            "n_paths": len(samples),
        }

    def theta(self, payoff, S0: float, T: float) -> dict:
        """
        Central-difference theta: -(V(T+h) - V(T-h)) / (2h).

        Negative sign: Theta = -∂V/∂T (erodes with time).
        Both the maturity and the discount factor change with T.
        """
        r = getattr(self.model, "r", 0.0)
        h = self._bump(T)
        h = max(h, 1e-4)
        seed = self.engine.rng_seed

        sim_u = self._simulate(S0, T + h, seed=seed)
        f_u = payoff(sim_u["paths"], sim_u["times"])
        price_u = self._disc(T + h, r) * f_u

        sim_d = self._simulate(S0, T - h, seed=seed)
        f_d = payoff(sim_d["paths"], sim_d["times"])
        price_d = self._disc(T - h, r) * f_d

        # Theta = -dV/dT
        samples = -(price_u - price_d) / (2.0 * h)
        mean = float(samples.mean())
        se = float(samples.std(ddof=1) / np.sqrt(len(samples)))
        return {
            "value": mean,
            "std_error": se,
            "ci_lower": mean - _Z95 * se,
            "ci_upper": mean + _Z95 * se,
            "n_paths": len(samples),
        }
