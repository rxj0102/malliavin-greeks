"""
malliavin-greeks
================
Malliavin calculus methods for computing Greeks of derivative securities
via Monte Carlo simulation.

Quick start
-----------
>>> from mgreeks.models.gbm import GeometricBrownianMotion
>>> from mgreeks.payoffs.european import EuropeanCall
>>> from mgreeks.simulation import MonteCarloEngine
>>> model = GeometricBrownianMotion(r=0.05, sigma=0.20)
>>> engine = MonteCarloEngine(model, n_paths=100_000, n_steps=252, rng_seed=42)
>>> result = engine.price(EuropeanCall(K=100), S0=100, T=1.0)
>>> print(result)
"""

from mgreeks.simulation import MonteCarloEngine

__version__ = "0.2.0"
__all__ = ["MonteCarloEngine"]
