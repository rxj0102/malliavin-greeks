# malliavin-greeks

A research-grade Python library implementing **Malliavin calculus methods** for
computing sensitivities (Greeks) of derivative securities via Monte Carlo simulation.

Instead of perturbing inputs and re-simulating (finite differences), Malliavin
calculus derives weight functions $\pi$ such that

$$\text{Greek} = e^{-rT}\,\mathbb{E}\bigl[f(S_T)\,\pi\bigr]$$

from a **single** set of simulation paths.  The weights are derived by
integration by parts on Wiener space (the Bismut–Elworthy–Li formula) and work
for **discontinuous payoffs** (digital options, barriers) without modification.

## Why Malliavin calculus for Greeks?

| Problem | FD / Bump-and-Revalue | Malliavin |
|---------|-----------------------|-----------|
| Discontinuous payoffs (digital, barrier) | Severe variance for small h, bias for large h | Exact, no bump needed |
| Path-dependent payoffs (Asian, lookback) | Propagation of bump through path is complex | Single simulation, clean weight |
| Higher-order Greeks (gamma) | Requires 3 simulations + instability | Single simulation, closed-form weight |
| Cost | >= 2x price simulation per Greek | 1x price simulation for all Greeks |

## Features

- **Models**: GBM (exact log-normal), local volatility (Dupire, Euler-Maruyama), Heston stochastic vol (full-truncation EM)
- **Malliavin weights** (Bismut-Elworthy-Li formula): delta, gamma, vega, theta, rho under GBM; delta and vega under Heston
- **Comparison estimators**: finite differences (forward, central), pathwise/IPA, likelihood-ratio (score function)
- **Exotic payoffs**: European, digital (cash-or-nothing, asset-or-nothing), Asian (arithmetic average), barrier (knock-in/out), lookback (fixed/floating strike)
- **Variance reduction**: antithetic variates, control variates (exact BS price as control), importance sampling, stratified sampling
- **Benchmark suite**: systematic variance and efficiency comparisons across all estimators

## Weight Formulas

Under GBM (score-function / LR derivation, coincides with Malliavin IBP for terminal payoffs):

| Greek | Weight pi |
|-------|-----------|
| Delta | W_T / (sigma S0 T) |
| Gamma | (W_T(W_T - sigma T) - T) / (sigma^2 T^2 S0^2) |
| Vega  | (W_T^2 - T)/(sigma T) - W_T |
| Rho   | W_T / sigma |

where W_T = sigma * sqrt(dt) * sum(Z_i) is the terminal Brownian motion, derived from the simulated path increments.

## Installation

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
import numpy as np
from malliavin_greeks.models import GBMParams
from malliavin_greeks.payoffs import european_call, digital_call
from malliavin_greeks.estimators import malliavin_delta, malliavin_gamma, malliavin_vega
from malliavin_greeks.benchmarks import benchmark_delta, print_benchmark_table

params = GBMParams(S0=100.0, r=0.05, q=0.0, sigma=0.20, T=1.0)
K = 100.0

# Delta for a European call — single simulation
delta = malliavin_delta(params, lambda S: european_call(S, K),
                        n_paths=100_000, rng=np.random.default_rng(42))
print(f"Delta: {delta.greek:.4f} +/- {delta.std_err:.4f}")

# Works for discontinuous payoffs without any modification
delta_dig = malliavin_delta(params, lambda S: digital_call(S, K),
                             n_paths=100_000, rng=np.random.default_rng(42))
print(f"Digital Delta: {delta_dig.greek:.4f} +/- {delta_dig.std_err:.4f}")

# Systematic variance comparison across estimators
results = benchmark_delta(params, K, n_paths=200_000)
print_benchmark_table(results, "Delta Comparison")
```

## Running Tests

```bash
pytest tests/ -v          # 42 tests in ~15 seconds
```

## Notebook Demo

```bash
jupyter notebook notebooks/malliavin_greeks_demo.ipynb
```

## References

Fournié, E., Lasry, J.-M., Lebuchoux, J., Lions, P.-L., & Touzi, N. (1999).
Applications of Malliavin calculus to Monte Carlo methods in finance.
*Finance and Stochastics*, 3(4), 391-412.

Fournié, E., Lasry, J.-M., Lebuchoux, J., & Lions, P.-L. (2001).
Applications of Malliavin calculus to Monte Carlo methods in finance II.
*Finance and Stochastics*, 5(2), 201-236.

Broadie, M., & Glasserman, P. (1996). Estimating security price derivatives
using simulation. *Management Science*, 42(2), 269-285.

Glasserman, P. (2003). *Monte Carlo Methods in Financial Engineering*. Springer.

Gobet, E., & Munos, R. (2005). Sensitivity analysis using Ito-Malliavin calculus
and martingales. *SIAM Journal on Control and Optimization*, 43(5), 1676-1713.
