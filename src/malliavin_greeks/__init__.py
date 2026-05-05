"""
malliavin-greeks: Malliavin calculus Greeks for Monte Carlo derivative pricing.

The library implements the Bismut-Elworthy-Li (BEL) framework for computing
sensitivities (Greeks) of derivative securities without finite-difference
perturbations. A single set of simulation paths yields unbiased estimates of
all first- and second-order Greeks, even for discontinuous payoffs.

Key modules
-----------
models      : GBM, local-volatility, and Heston stochastic-volatility simulators
weights     : Malliavin weight functions (delta, gamma, vega, theta, rho)
payoffs     : Vanilla and exotic payoff functions (European, Asian, barrier, lookback)
estimators  : Malliavin, finite-difference, pathwise, and likelihood-ratio Greeks
variance    : Variance-reduction techniques (control variates, importance sampling)
benchmarks  : Systematic variance comparison utilities
"""

from malliavin_greeks import models, weights, payoffs, estimators, variance, benchmarks

__version__ = "0.1.0"
__all__ = ["models", "weights", "payoffs", "estimators", "variance", "benchmarks"]
