# malliavin-greeks: Sensitivity Computation via Malliavin Calculus

A research-grade Python library for computing option Greeks (sensitivities) using **Malliavin calculus integration by parts** — a technique that avoids differentiating the payoff function and thereby works for **discontinuous payoffs** such as digital options and barrier options.

## Mathematical Foundation

For a derivative with payoff f(S_T) and option value V = e^{−rT} E[f(S_T)], the Malliavin integration by parts formula gives:

```
∂V/∂θ = e^{−rT} E[f(S_T) · π_θ]
```

where **π_θ** is the *Malliavin weight* — a random variable that depends on the Brownian path and the model parameters, but **not on the payoff function f**. This is the Bismut–Elworthy–Li (BEL) formula applied to option Greeks.

Key consequence: f is never differentiated, so the formula works equally well for smooth European calls and discontinuous digital or barrier payoffs.

## Key Results

Comparison of delta estimation for a **digital call** at-the-money (S₀=K=100, T=1, σ=20%):

| Method | SE ratio vs Malliavin | Note |
|--------|----------------------|------|
| Malliavin | 1.0 (baseline) | Finite variance, SE ∝ 1/√n |
| FD (h=1%) | 8.3× worse | SE ∝ 1/(h√n) → ∞ as h → 0 |
| FD (h=0.1%) | 83× worse | Variance explodes |
| Pathwise/IPA | N/A | Infinite variance (Dirac delta) |

For a **European call**, FD with common random numbers achieves lower variance than Malliavin for delta (smooth payoff), but Malliavin computes all Greeks from one simulation while FD needs 2+ extra simulations per Greek.

## Weight Formulas

Under GBM with W_T = Σ ΔWᵢ (terminal Brownian motion):

| Greek | Malliavin weight π |
|-------|-------------------|
| Delta | W_T / (σ S₀ T) |
| Gamma | [W_T(W_T − σT) − T] / (S₀² σ² T²) |
| Vega  | (W_T² − T)/(σT) − W_T |
| Rho   | W_T/σ − T |
| Theta | r + (T − W_T²)/(2T²) − μ W_T/(σT), μ = r−q−σ²/2 |

These weights are derived from the log-density score function (equivalent to BEL for GBM) and verified against Black–Scholes analytics.

## Quick Start

```python
import numpy as np
from mgreeks.models.gbm import GeometricBrownianMotion
from mgreeks.payoffs.european import EuropeanCall, DigitalCall
from mgreeks.simulation import MonteCarloEngine
from mgreeks.greeks import MalliavinGreeks, FiniteDifferenceGreeks
from mgreeks.greeks.analytical import bs_delta

# Model and payoff
model = GeometricBrownianMotion(r=0.05, q=0.02, sigma=0.20)
payoff = EuropeanCall(K=100.0)

# Monte Carlo engine (single shared simulation)
engine = MonteCarloEngine(model, n_paths=100_000, n_steps=1, rng_seed=42)

# Compute ALL Greeks from one simulation
mall = MalliavinGreeks(model, engine)
S0, T = 100.0, 1.0

delta = mall.delta(payoff, S0, T)
gamma = mall.gamma(payoff, S0, T)
vega  = mall.vega(payoff, S0, T)

print(f"Delta: {delta['value']:.4f} ± {delta['std_error']:.4f}")
print(f"Gamma: {gamma['value']:.4f} ± {gamma['std_error']:.4f}")
print(f"Vega:  {vega['value']:.4f} ± {vega['std_error']:.4f}")

# Compare with Black–Scholes
print(f"BS Delta: {bs_delta(S0, 100, T, 0.05, 0.02, 0.20, 'call'):.4f}")
```

Digital call — same weight, discontinuous payoff:

```python
digital = DigitalCall(K=100.0)
delta_d = mall.delta(digital, S0, T)
print(f"Digital Delta: {delta_d['value']:.4f} ± {delta_d['std_error']:.4f}")
# Finite variance regardless of proximity to strike
```

## Features

### Models

| Model | Class | Simulation |
|-------|-------|-----------|
| Geometric Brownian Motion | `GeometricBrownianMotion` | Exact log-normal |
| Constant Elasticity of Variance | `LocalVolModel(model_type='cev')` | Euler–Maruyama |
| Local volatility (Dupire surface) | `LocalVolModel(model_type='quadratic')` | Euler–Maruyama |
| Heston stochastic volatility | `HestonModel` | Full-truncation Euler (Andersen QE scheme) |

### Payoffs

| Payoff | Class | Path-dependent |
|--------|-------|---------------|
| European call/put | `EuropeanCall`, `EuropeanPut` | No |
| Digital (cash-or-nothing) | `DigitalCall` | No |
| Arithmetic Asian call | `ArithmeticAsianCall` | Yes |
| Down-and-out call | `DownAndOutCall` | Yes |
| Floating-strike lookback | `FloatingStrikeLookbackCall` | Yes |
| Basket call (multi-asset) | `BasketCall` | No |

### Greeks

| Greek | Malliavin | FD | Pathwise | LR |
|-------|-----------|----|---------|----|
| Delta | ✓ (all models) | ✓ | ✓ (smooth only) | ✓ |
| Gamma | ✓ (GBM, closed form) | ✓ | — | ✓ |
| Vega | ✓ (GBM, Heston) | ✓ | ✓ (smooth only) | ✓ |
| Rho | ✓ (GBM) | ✓ | — | — |
| Theta | ✓ (GBM) | ✓ | — | — |
| Vanna (∂²/∂S∂σ) | ✓ (GBM) | ✓ (4 sims) | — | — |
| Volga (∂²/∂σ²) | ✓ (GBM) | ✓ (3 sims) | — | — |

### Variance Reduction

- **Antithetic variates** (`mgreeks.variance_reduction.antithetic`): pairs each path with its Brownian reflection; VRR ≥ 10× for digital call delta
- **Control variates** (`mgreeks.variance_reduction.control_variate`): geometric Asian control (closed-form price); delta-hedged portfolio control
- **Localization** (`mgreeks.variance_reduction.localization`): weight truncation at 99.9th percentile; interval localization for barrier options

## Performance

The library is structured for efficiency:

- **Single simulation:** Malliavin computes delta, gamma, vega, rho, theta from one set of paths
- **FD comparison:** finite differences require 2–4× extra simulations per Greek
- **Numba JIT:** inner loops of the BEL stochastic integral (local vol, Heston) are compiled with `@numba.njit` when numba ≥ 0.57 is installed; pure NumPy fallback otherwise
- **Caching:** pass `sim_result=` to reuse paths across multiple Greeks

```python
# Cache simulation — one RNG draw, many estimates
sim = model.simulate(S0, T, n_steps=52, n_paths=200_000,
                     return_full_paths=True, rng=np.random.default_rng(42))
delta = mall.delta(payoff, S0, T, sim_result=sim)
gamma = mall.gamma(payoff, S0, T, sim_result=sim)
vega  = mall.vega(payoff, S0, T, sim_result=sim)
```

## Installation

```bash
pip install -e ".[dev]"
```

Requirements: Python ≥ 3.9, numpy, scipy, matplotlib, pyyaml.  
Optional: `numba` for JIT-compiled BEL inner loops (significant speedup for local vol / Heston).

## Running Experiments

```bash
# Variance comparison: 4 methods × 6 option types × 3 Greeks
python experiments/run_variance_comparison.py --n_paths 100000

# Discontinuous payoffs: digital and barrier delta sweep + convergence
python experiments/run_discontinuous_payoffs.py

# Higher-order Greeks: gamma surface, vanna/volga, digital gamma
python experiments/run_higher_order.py

# Model comparison: GBM vs CEV vs Heston
python experiments/run_model_comparison.py
```

All experiments read from `experiments/config.yaml` and accept `--n_paths`, `--n_steps`, `--seed` overrides. Results are saved to `experiments/results/`.

## Running Tests

```bash
pytest tests/ -v
```

Test coverage:
- `test_malliavin_weights.py` — GBM weight formulas vs analytical Greeks
- `test_greeks.py` — MalliavinGreeks, FiniteDifferenceGreeks, PathwiseGreeks, LikelihoodRatioGreeks
- `test_greeks_exotic.py` — Asian, barrier, lookback, Heston weights
- `test_variance_reduction.py` — antithetic VRR, control variate correlation, localization bias

## Documentation

- `docs/math_background.md` — Malliavin derivative, IBP formula, Clark–Ocone, Hermite polynomials, BEL formula
- `docs/weight_derivations.md` — Full derivations of all weight formulas (delta, gamma, vega, rho, theta, Heston, path-dependent)
- `docs/variance_analysis.md` — When FD fails, variance bounds, practical method-selection guide

## Notebooks

| Notebook | Topic |
|----------|-------|
| `01_malliavin_primer.ipynb` | What is the Malliavin derivative? Compute delta by hand. Payoff independence demonstrated. |
| `02_greeks_comparison.ipynb` | All four methods for European call. Convergence and bump-size dilemma. |
| `03_exotic_greeks.ipynb` | Asian, barrier, lookback Greeks. Barrier problem: FD fails near B, Malliavin stays flat. |
| `04_variance_analysis.ipynb` | Variance ratios, antithetic VRR, timing comparison, practical decision guide. |

## References

1. **Fournié, E., Lasry, J.-M., Lebuchoux, J., Lions, P.-L. & Touzi, N. (1999)** —  
   "Applications of Malliavin calculus to Monte Carlo methods in finance."  
   *Finance and Stochastics* **3**, 391–412.

2. **Fournié, E., Lasry, J.-M., Lebuchoux, J. & Lions, P.-L. (2001)** —  
   "Applications of Malliavin calculus to Monte Carlo methods in finance II."  
   *Finance and Stochastics* **5**, 201–236.

3. **Nualart, D. (2006)** — *The Malliavin Calculus and Related Topics*, 2nd ed. Springer.

4. **Bismut, J.-M. (1984)** — *Large Deviations and the Malliavin Calculus*. Birkhäuser.

5. **Elworthy, K.D. & Li, X.-M. (1994)** — "Formulae for the derivatives of heat semigroups."  
   *J. Functional Analysis* **125**, 252–286.

6. **Gobet, E. & Kohatsu-Higa, A. (2003)** — "Computation of Greeks for barrier and  
   look-back options using Malliavin calculus."  
   *Electronic Communications in Probability* **8**, 51–62.

7. **Andersen, L. (2008)** — "Simple and efficient simulation of the Heston stochastic  
   volatility model." *Journal of Computational Finance* **11(3)**, 1–42.

8. **Glasserman, P. (2004)** — *Monte Carlo Methods in Financial Engineering*. Springer.
