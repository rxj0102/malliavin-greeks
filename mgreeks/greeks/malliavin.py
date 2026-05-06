"""
Malliavin calculus Greek estimators.

All Greeks share a SINGLE simulation — the key efficiency advantage over
finite differences (which need 2+ extra simulations per Greek).

For each Greek ∂/∂θ V where V = e^{-rT} E[f(S)]:

    Greek ≈ disc · (1/N) Σ f_i · π_{θ,i}

where π_θ is the Malliavin weight (path-dependent, payoff-independent).
"""

from __future__ import annotations

import numpy as np

from mgreeks.models.base import StochasticModel
from mgreeks.simulation import MonteCarloEngine


_Z95 = 1.959964  # 95% two-sided z


class MalliavinGreeks:
    """
    Compute Greeks using Malliavin calculus weights.

    All Greeks can be estimated from a single Monte Carlo simulation —
    no re-simulation is needed between Greeks.
    """

    def __init__(self, model: StochasticModel, engine: MonteCarloEngine):
        self.model = model
        self.engine = engine

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _simulate(self, S0: float, T: float) -> dict:
        return self.model.simulate(
            S0, T, self.engine.n_steps, self.engine.n_paths,
            return_full_paths=True, rng=self.engine._rng(),
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

    def _require_gbm(self) -> "GeometricBrownianMotion":
        from mgreeks.models.gbm import GeometricBrownianMotion
        if not isinstance(self.model, GeometricBrownianMotion):
            raise NotImplementedError(
                f"Malliavin weights not implemented for {type(self.model).__name__}. "
                "Only GeometricBrownianMotion is supported."
            )
        return self.model

    def _disc(self, T: float) -> float:
        return float(np.exp(-getattr(self.model, "r", 0.0) * T))

    # ------------------------------------------------------------------
    # Individual Greeks
    # ------------------------------------------------------------------

    def delta(
        self, payoff, S0: float, T: float, sim_result: dict = None,
    ) -> dict:
        """Delta = ∂V/∂S_0.  Weight: W_T / (S_0 σ T)."""
        from mgreeks.weights.malliavin_weights import delta_weight_gbm

        gbm = self._require_gbm()
        sim = sim_result if sim_result is not None else self._simulate(S0, T)
        dW = sim["brownian_increments"]
        W_T = dW.sum(axis=1)
        f = payoff(sim["paths"], sim["times"])
        weight = delta_weight_gbm(S0, sim["terminal"], gbm.sigma, gbm.r, gbm.q, T, W_T)
        return self._result(f * weight, self._disc(T))

    def gamma(
        self, payoff, S0: float, T: float, sim_result: dict = None,
    ) -> dict:
        """Gamma = ∂²V/∂S_0².  Weight: [W_T(W_T-σT)-T] / (S_0² σ² T²)."""
        from mgreeks.weights.malliavin_weights import gamma_weight_gbm

        gbm = self._require_gbm()
        sim = sim_result if sim_result is not None else self._simulate(S0, T)
        W_T = sim["brownian_increments"].sum(axis=1)
        f = payoff(sim["paths"], sim["times"])
        weight = gamma_weight_gbm(S0, sim["terminal"], gbm.sigma, gbm.r, gbm.q, T, W_T)
        return self._result(f * weight, self._disc(T))

    def vega(
        self, payoff, S0: float, T: float, sim_result: dict = None,
    ) -> dict:
        """Vega = ∂V/∂σ.  Weight: (W_T²-T)/(σT) - W_T."""
        from mgreeks.weights.malliavin_weights import vega_weight_gbm

        gbm = self._require_gbm()
        sim = sim_result if sim_result is not None else self._simulate(S0, T)
        W_T = sim["brownian_increments"].sum(axis=1)
        f = payoff(sim["paths"], sim["times"])
        weight = vega_weight_gbm(S0, sim["terminal"], gbm.sigma, gbm.r, gbm.q, T, W_T)
        return self._result(f * weight, self._disc(T))

    def rho(
        self, payoff, S0: float, T: float, sim_result: dict = None,
    ) -> dict:
        """Rho = ∂V/∂r.  Weight: W_T/σ - T."""
        from mgreeks.weights.malliavin_weights import rho_weight_gbm

        gbm = self._require_gbm()
        sim = sim_result if sim_result is not None else self._simulate(S0, T)
        W_T = sim["brownian_increments"].sum(axis=1)
        f = payoff(sim["paths"], sim["times"])
        weight = rho_weight_gbm(S0, sim["terminal"], gbm.sigma, gbm.r, gbm.q, T, W_T)
        return self._result(f * weight, self._disc(T))

    def theta(
        self, payoff, S0: float, T: float, sim_result: dict = None,
    ) -> dict:
        """Theta = -∂V/∂T.  Weight: r + (T-W_T²)/(2T²) - μW_T/(σT)."""
        from mgreeks.weights.malliavin_weights import theta_weight_gbm

        gbm = self._require_gbm()
        sim = sim_result if sim_result is not None else self._simulate(S0, T)
        W_T = sim["brownian_increments"].sum(axis=1)
        f = payoff(sim["paths"], sim["times"])
        weight = theta_weight_gbm(S0, sim["terminal"], gbm.sigma, gbm.r, gbm.q, T, W_T)
        return self._result(f * weight, self._disc(T))

    # ------------------------------------------------------------------
    # All Greeks from a SINGLE simulation
    # ------------------------------------------------------------------

    def all_greeks(self, payoff, S0: float, T: float) -> dict:
        """
        Compute delta, gamma, vega, rho, theta from ONE simulation.

        This is the key efficiency advantage of Malliavin calculus:
        all Greeks share the same paths and payoff evaluations.

        Returns
        -------
        dict mapping Greek name → result dict (each with 'value', 'std_error', ...)
        """
        from mgreeks.weights.malliavin_weights import all_weights_gbm

        self._require_gbm()
        gbm = self.model
        sim = self._simulate(S0, T)
        W_T = sim["brownian_increments"].sum(axis=1)
        f = payoff(sim["paths"], sim["times"])
        disc = self._disc(T)

        weights = all_weights_gbm(S0, gbm.sigma, gbm.r, gbm.q, T, W_T, sim["terminal"])
        return {name: self._result(f * w, disc) for name, w in weights.items()}

    # ------------------------------------------------------------------
    # Second-order Greeks
    # ------------------------------------------------------------------

    def higher_order(
        self,
        payoff,
        S0: float,
        T: float,
        greek_name: str = "gamma",
    ) -> dict:
        """
        Compute a second-order Greek via double Malliavin integration by parts.

        Parameters
        ----------
        greek_name : 'gamma' (∂²V/∂S²), 'vanna' (∂²V/∂S∂σ), 'volga' (∂²V/∂σ²)
        """
        from mgreeks.weights.bismut_elworthy_li import bel_second_order_weight

        self._require_gbm()
        sim = self._simulate(S0, T)
        f = payoff(sim["paths"], sim["times"])
        weight = bel_second_order_weight(
            sim["paths"], sim["brownian_increments"],
            self.model, S0, T, sim["times"],
            greek_type=greek_name,
        )
        return self._result(f * weight, self._disc(T))
