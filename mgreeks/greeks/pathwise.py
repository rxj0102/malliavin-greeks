"""
Pathwise (Infinitesimal Perturbation Analysis, IPA) Greek estimators.

Requires payoffs to be (almost everywhere) differentiable with respect to
the asset price.  Works for:
  - European calls/puts: ∂f/∂S_T = ±1_{S_T ≷ K}
  - Asian calls/puts:    ∂f/∂S_0 = 1_{A>K} · A/S_0

FAILS for discontinuous payoffs:
  - Digitals:  ∂f/∂S_T = δ(S_T - K)  — undefined (infinite variance)
  - Barriers:  ∂f/∂S_T involves Dirac mass on the barrier

Under GBM, ∂S_T/∂S_0 = S_T/S_0, so:
  Delta = e^{-rT} E[f'(S_T) · S_T / S_0]

For path-dependent payoffs, ∂f/∂S_0 is computed via payoff.payoff_deriv()
which knows the payoff structure.
"""

from __future__ import annotations

import numpy as np

from mgreeks.models.base import StochasticModel
from mgreeks.simulation import MonteCarloEngine


_Z95 = 1.959964


class PathwiseGreeks:
    """
    Greek estimator via pathwise differentiation (IPA).

    Requires payoff.payoff_deriv() for delta (and payoff.payoff_deriv2()
    for gamma, if available).  Raises NotImplementedError for
    discontinuous payoffs.
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

    def _disc(self, T: float) -> float:
        return float(np.exp(-getattr(self.model, "r", 0.0) * T))

    # ------------------------------------------------------------------
    # Delta
    # ------------------------------------------------------------------

    def delta(
        self, payoff, S0: float, T: float, sim_result: dict = None,
    ) -> dict:
        """
        Pathwise delta for European payoffs under GBM.

            Delta = e^{-rT} E[f'(S_T) · S_T / S_0]

        payoff must implement .payoff_deriv(paths, times) returning ∂f/∂S_T.
        Raises NotImplementedError if the payoff is not differentiable.
        """
        sim = sim_result if sim_result is not None else self._simulate(S0, T)
        paths, times = sim["paths"], sim["times"]

        try:
            deriv = payoff.payoff_deriv(paths, times)
        except NotImplementedError:
            raise NotImplementedError(
                f"{type(payoff).__name__} does not support pathwise delta. "
                "Use MalliavinGreeks or LikelihoodRatioGreeks for discontinuous payoffs."
            )

        ST = paths[:, -1]
        samples = deriv * ST / S0
        return self._result(samples, self._disc(T))

    def delta_path_dependent(
        self, payoff, S0: float, T: float, sim_result: dict = None,
    ) -> dict:
        """
        Pathwise delta for path-dependent payoffs.

        payoff.payoff_deriv() should return ∂f/∂S_0 directly (not ∂f/∂S_T).
        For arithmetic Asian: returns 1_{A>K} · A/S_0, so the estimator is
        disc · E[payoff_deriv_samples] without the S_T/S_0 multiplication.
        """
        sim = sim_result if sim_result is not None else self._simulate(S0, T)
        paths, times = sim["paths"], sim["times"]

        try:
            samples = payoff.payoff_deriv(paths, times)
        except NotImplementedError:
            raise NotImplementedError(
                f"{type(payoff).__name__} does not support pathwise delta. "
                "Use MalliavinGreeks for discontinuous payoffs."
            )

        return self._result(samples, self._disc(T))

    # ------------------------------------------------------------------
    # Gamma
    # ------------------------------------------------------------------

    def gamma(
        self, payoff, S0: float, T: float, sim_result: dict = None,
    ) -> dict:
        """
        Pathwise gamma (requires second-order payoff derivative).

            Gamma = e^{-rT} E[f''(S_T) · (S_T/S_0)²]

        payoff must implement .payoff_deriv2(paths, times) returning ∂²f/∂S_T².
        """
        sim = sim_result if sim_result is not None else self._simulate(S0, T)
        paths, times = sim["paths"], sim["times"]

        if not hasattr(payoff, "payoff_deriv2"):
            raise NotImplementedError(
                f"{type(payoff).__name__} does not implement payoff_deriv2. "
                "Gamma via pathwise IPA requires a twice-differentiable payoff. "
                "Use MalliavinGreeks for discontinuous payoffs."
            )

        deriv2 = payoff.payoff_deriv2(paths, times)
        ST = paths[:, -1]
        samples = deriv2 * (ST / S0) ** 2
        return self._result(samples, self._disc(T))

    # ------------------------------------------------------------------
    # Vega
    # ------------------------------------------------------------------

    def vega(
        self, payoff, S0: float, T: float, sim_result: dict = None,
    ) -> dict:
        """
        Pathwise vega under GBM.

        Under exact GBM solution S_T = S_0 exp(μT + σW_T):
            ∂S_T/∂σ = S_T · (W_T - σT)   [holding the Brownian path fixed]

        So vega = e^{-rT} E[f'(S_T) · S_T · (W_T - σT)].
        """
        from mgreeks.models.gbm import GeometricBrownianMotion
        if not isinstance(self.model, GeometricBrownianMotion):
            raise NotImplementedError(
                "Pathwise vega is only implemented for GeometricBrownianMotion."
            )

        sigma = self.model.sigma
        sim = sim_result if sim_result is not None else self._simulate(S0, T)
        paths, times = sim["paths"], sim["times"]

        try:
            deriv = payoff.payoff_deriv(paths, times)
        except NotImplementedError:
            raise NotImplementedError(
                f"{type(payoff).__name__} does not support pathwise vega. "
                "Use MalliavinGreeks for discontinuous payoffs."
            )

        W_T = sim["brownian_increments"].sum(axis=1)
        ST = paths[:, -1]
        samples = deriv * ST * (W_T - sigma * T)
        return self._result(samples, self._disc(T))
