"""
Likelihood-ratio (score-function) Greek estimators.

    Greek = e^{-rT} E[f(S_T) · ∂/∂θ log p(S_T | S_0; θ)]

where ∂/∂θ log p is the SCORE FUNCTION — the derivative of the
log-transition-density with respect to the parameter θ.

The score does not involve f at all, so this method works for
discontinuous payoffs (digitals, barriers, etc.).

GBM note
--------
For GBM terminal payoffs, the LR and Malliavin weights are IDENTICAL:

    ∂/∂S_0 log p = W_T / (σ T S_0)   ← same as Malliavin delta weight

They differ conceptually:
  - LR differentiates the DENSITY
  - Malliavin uses integration-by-parts in Wiener space

They also differ for path-dependent payoffs with n > 1 steps.
For the JOINT density p(S_{t_1},...,S_{t_n}|S_0), only the first
conditional p(S_{t_1}|S_0) depends on S_0, so the LR score is:

    ∂/∂S_0 log p(joint) = ΔW_1 / (σ S_0 Δt_1)

This is the delta_weight_path_dependent formula.  The "naive" W_T/(σS_0T)
gives biased estimates for non-terminal payoffs.

Variance note
-------------
For multi-step path-dependent payoffs, the LR weight can have variance
that GROWS with the number of steps (the product of per-step scores).
Malliavin's representation avoids this by using a single stochastic
integral rather than a product of scores.  For terminal (one-step) payoffs
there is no difference.
"""

from __future__ import annotations

import numpy as np

from mgreeks.models.base import StochasticModel
from mgreeks.simulation import MonteCarloEngine


_Z95 = 1.959964


class LikelihoodRatioGreeks:
    """
    Greek estimator via the likelihood-ratio (score-function) method.

    For GBM this is numerically equivalent to MalliavinGreeks for terminal
    payoffs.  The class delegates to the same weight functions, emphasising
    the score-function derivation.
    """

    def __init__(self, model: StochasticModel, engine: MonteCarloEngine):
        self.model = model
        self.engine = engine

    # ------------------------------------------------------------------
    # Helpers
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

    def _require_gbm(self):
        from mgreeks.models.gbm import GeometricBrownianMotion
        if not isinstance(self.model, GeometricBrownianMotion):
            raise NotImplementedError(
                f"LR weights not implemented for {type(self.model).__name__}. "
                "Only GeometricBrownianMotion is supported."
            )
        return self.model

    def _disc(self, T: float) -> float:
        return float(np.exp(-getattr(self.model, "r", 0.0) * T))

    # ------------------------------------------------------------------
    # Greeks (all use GBM score functions = Malliavin weights)
    # ------------------------------------------------------------------

    def delta(
        self, payoff, S0: float, T: float, sim_result: dict = None,
    ) -> dict:
        """
        LR delta.  Score: ∂/∂S_0 log p = W_T / (σ T S_0).

        For terminal payoffs this is identical to the Malliavin weight.
        For path-dependent payoffs, use delta_path_dependent() instead.
        """
        from mgreeks.weights.malliavin_weights import delta_weight_gbm

        gbm = self._require_gbm()
        sim = sim_result if sim_result is not None else self._simulate(S0, T)
        W_T = sim["brownian_increments"].sum(axis=1)
        f = payoff(sim["paths"], sim["times"])
        score = delta_weight_gbm(S0, sim["terminal"], gbm.sigma, gbm.r, gbm.q, T, W_T)
        return self._result(f * score, self._disc(T))

    def delta_path_dependent(
        self, payoff, S0: float, T: float, sim_result: dict = None,
    ) -> dict:
        """
        LR delta for path-dependent payoffs.

        Uses the JOINT score ΔW_1/(σ S_0 Δt_1) (score of the first
        conditional density only).  This is the correct LR weight for
        any payoff f(S_{t_1},...,S_{t_n}) under GBM.
        """
        from mgreeks.weights.malliavin_weights import delta_weight_path_dependent

        gbm = self._require_gbm()
        sim = sim_result if sim_result is not None else self._simulate(S0, T)
        f = payoff(sim["paths"], sim["times"])
        score = delta_weight_path_dependent(
            sim["paths"], sim["brownian_increments"], gbm, S0, T
        )
        return self._result(f * score, self._disc(T))

    def gamma(
        self, payoff, S0: float, T: float, sim_result: dict = None,
    ) -> dict:
        """LR gamma.  Score: [W_T(W_T-σT)-T] / (S_0² σ² T²)."""
        from mgreeks.weights.malliavin_weights import gamma_weight_gbm

        gbm = self._require_gbm()
        sim = sim_result if sim_result is not None else self._simulate(S0, T)
        W_T = sim["brownian_increments"].sum(axis=1)
        f = payoff(sim["paths"], sim["times"])
        score = gamma_weight_gbm(S0, sim["terminal"], gbm.sigma, gbm.r, gbm.q, T, W_T)
        return self._result(f * score, self._disc(T))

    def vega(
        self, payoff, S0: float, T: float, sim_result: dict = None,
    ) -> dict:
        """LR vega.  Score: (W_T²-T)/(σT) - W_T."""
        from mgreeks.weights.malliavin_weights import vega_weight_gbm

        gbm = self._require_gbm()
        sim = sim_result if sim_result is not None else self._simulate(S0, T)
        W_T = sim["brownian_increments"].sum(axis=1)
        f = payoff(sim["paths"], sim["times"])
        score = vega_weight_gbm(S0, sim["terminal"], gbm.sigma, gbm.r, gbm.q, T, W_T)
        return self._result(f * score, self._disc(T))

    def rho(
        self, payoff, S0: float, T: float, sim_result: dict = None,
    ) -> dict:
        """LR rho.  Score: W_T/σ - T  (score of r + discount correction)."""
        from mgreeks.weights.malliavin_weights import rho_weight_gbm

        gbm = self._require_gbm()
        sim = sim_result if sim_result is not None else self._simulate(S0, T)
        W_T = sim["brownian_increments"].sum(axis=1)
        f = payoff(sim["paths"], sim["times"])
        score = rho_weight_gbm(S0, sim["terminal"], gbm.sigma, gbm.r, gbm.q, T, W_T)
        return self._result(f * score, self._disc(T))

    def theta(
        self, payoff, S0: float, T: float, sim_result: dict = None,
    ) -> dict:
        """LR theta.  Score: r + (T-W_T²)/(2T²) - μW_T/(σT)."""
        from mgreeks.weights.malliavin_weights import theta_weight_gbm

        gbm = self._require_gbm()
        sim = sim_result if sim_result is not None else self._simulate(S0, T)
        W_T = sim["brownian_increments"].sum(axis=1)
        f = payoff(sim["paths"], sim["times"])
        score = theta_weight_gbm(S0, sim["terminal"], gbm.sigma, gbm.r, gbm.q, T, W_T)
        return self._result(f * score, self._disc(T))
