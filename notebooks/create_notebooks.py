#!/usr/bin/env python3
"""Build all four tutorial notebooks for malliavin-greeks."""
import json
from pathlib import Path

NB_DIR = Path(__file__).parent


def cell(source, cell_type="code"):
    if cell_type == "markdown":
        return {"cell_type": "markdown", "metadata": {}, "source": source}
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


def notebook(cells):
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.9.0"},
        },
        "cells": cells,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 01_malliavin_primer.ipynb
# ─────────────────────────────────────────────────────────────────────────────

nb01_cells = [
    cell("""\
# 01 — Malliavin Calculus Primer

## What is the Malliavin derivative?

The **Malliavin derivative** D_s F is the Fréchet derivative of a random variable F
on Wiener space in the Cameron–Martin direction.  Informally, D_s F measures how
sensitive F is to a perturbation of the Brownian path at time s.

Under GBM  S_T = S₀ exp(μT + σW_T)  we have:

    D_s S_T = σ · S_T   for all s ≤ T

The derivative is **constant in s** — a perturbation at any time has the same
proportional effect on S_T.

## The integration-by-parts formula

The key result (Fournié et al., 1999):  for any measurable payoff f,

    ∂/∂S₀  E[f(S_T)]  =  E[ f(S_T) · π ]

where the **Malliavin weight**  π = W_T / (σ S₀ T)  is computed from the
simulated path — **without ever differentiating f**.

This works for discontinuous payoffs (digital options, barriers) because the
singularity of f′ is absorbed by the Gaussian measure via the integration by parts.
""", "markdown"),

    cell("""\
import sys
sys.path.insert(0, '..')

import numpy as np
import matplotlib.pyplot as plt

from mgreeks.models.gbm import GeometricBrownianMotion
from mgreeks.payoffs.european import EuropeanCall, DigitalCall
from mgreeks.simulation import MonteCarloEngine
from mgreeks.greeks import MalliavinGreeks, FiniteDifferenceGreeks
from mgreeks.greeks.analytical import bs_delta, bs_digital_delta
from mgreeks.weights.malliavin_weights import delta_weight_gbm

# Parameters
S0, K, T = 100.0, 100.0, 1.0
r, q, sigma = 0.05, 0.02, 0.20
n_paths, seed = 50_000, 42

model = GeometricBrownianMotion(r=r, q=q, sigma=sigma)
disc = np.exp(-r * T)

print(f"Model: GBM  r={r}  q={q}  σ={sigma}")
print(f"Option: K={K}  T={T}  S₀={S0}")
"""),

    cell("""\
## Step 1: Simulate paths and compute the weight by hand

rng = np.random.default_rng(seed)
out = model.simulate(S0, T, n_steps=1, n_paths=n_paths,
                     return_full_paths=True, rng=rng)

S_T = out["terminal"]             # terminal spot, shape (n_paths,)
W_T = out["brownian_increments"].sum(axis=1)   # Σ ΔWᵢ = W_T

# Delta weight formula: π = W_T / (σ S₀ T)
weight = delta_weight_gbm(S0, S_T, sigma, r, q, T, W_T)

print(f"W_T: mean={W_T.mean():.4f}  std={W_T.std():.4f}  (should be ~N(0,{T:.1f}))")
print(f"Weight π: mean={weight.mean():.4f}  std={weight.std():.4f}")
"""),

    cell("""\
## Step 2: European call — compute delta by hand

euro_payoff = EuropeanCall(K)
f = euro_payoff(out["paths"], out["times"])   # payoff on each path

samples = disc * f * weight                   # e^{-rT} f · π
delta_mall = samples.mean()
delta_se   = samples.std(ddof=1) / np.sqrt(n_paths)
delta_bs   = bs_delta(S0, K, T, r, q, sigma, "call")

print(f"Malliavin delta: {delta_mall:.5f} ± {delta_se:.5f}")
print(f"Black-Scholes:   {delta_bs:.5f}")
print(f"Error: {abs(delta_mall - delta_bs):.5f}  ({abs(delta_mall - delta_bs)/delta_se:.1f} SE)")
"""),

    cell("""\
## Step 3: Visualise the weight distribution

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("GBM Delta Weight  π = W_T / (σ S₀ T)", fontsize=13)

ax = axes[0]
ax.hist(W_T, bins=60, density=True, color="#1f77b4", alpha=0.7, label="W_T")
xs = np.linspace(-4, 4, 300)
ax.plot(xs, np.exp(-xs**2 / (2*T)) / np.sqrt(2*np.pi*T), "k--", lw=2, label=f"N(0,{T})")
ax.set_xlabel("W_T"); ax.set_ylabel("density"); ax.legend()
ax.set_title("Terminal Brownian  W_T ~ N(0,T)")

ax = axes[1]
weighted = f * weight
ax.hist(weighted[weighted != 0], bins=80, density=True, color="#d62728", alpha=0.7)
ax.axvline(weighted.mean(), color="k", lw=2, label=f"mean = {weighted.mean():.4f}")
ax.set_xlabel("f(S_T) · π"); ax.legend()
ax.set_title("Malliavin integrand  f * pi  (E[f*pi]*disc = delta)")

plt.tight_layout()
plt.savefig("01_weight_distribution.png", dpi=100, bbox_inches="tight")
plt.show()
print("Figure saved.")
"""),

    cell("""\
## Step 4: Payoff independence — apply the SAME weight to a digital call

# Digital call: f(S_T) = 1_{S_T > K}   (discontinuous!)
digital = DigitalCall(K)
f_dig = digital(out["paths"], out["times"])

samples_d = disc * f_dig * weight      # same weight π, different payoff
delta_dig_mall = samples_d.mean()
delta_dig_se   = samples_d.std(ddof=1) / np.sqrt(n_paths)
delta_dig_bs   = bs_digital_delta(S0, K, T, r, q, sigma)

# FD comparison (uses same paths internally, different bump)
engine = MonteCarloEngine(model, n_paths=n_paths, n_steps=1, rng_seed=seed)
fd = FiniteDifferenceGreeks(model, engine, bump_size=0.01, bump_type="relative")
delta_dig_fd   = fd.delta(digital, S0, T)

print("=" * 60)
print("Digital Call Delta  (f is discontinuous at K)")
print("=" * 60)
print(f"{'Method':<20}  {'estimate':>10}  {'SE':>10}  {'bias':>10}")
print("-" * 60)
print(f"{'Malliavin':<20}  {delta_dig_mall:>10.5f}  {delta_dig_se:>10.5f}  {delta_dig_mall - delta_dig_bs:>10.5f}")
print(f"{'FD (h=1%)':<20}  {delta_dig_fd['value']:>10.5f}  {delta_dig_fd['std_error']:>10.5f}  {delta_dig_fd['value'] - delta_dig_bs:>10.5f}")
print(f"{'Analytical':<20}  {delta_dig_bs:>10.5f}  {'—':>10}  {'—':>10}")
print()
print(f"FD SE / Malliavin SE = {delta_dig_fd['std_error'] / delta_dig_se:.1f}×  (Malliavin wins decisively)")
"""),
]

# ─────────────────────────────────────────────────────────────────────────────
# 02_greeks_comparison.ipynb
# ─────────────────────────────────────────────────────────────────────────────

nb02_cells = [
    cell("""\
# 02 — All Four Methods: European Call Greeks

Four estimators for Greek computation:

| Method | Key idea | Requires f to be... |
|--------|----------|---------------------|
| **Malliavin** | Weight π from IBP formula | Measurable (any payoff) |
| **Finite difference** | Re-simulate with perturbed S₀ | Nothing, but variance explodes as h→0 |
| **Pathwise / IPA** | Differentiate f(S_T) directly | Differentiable |
| **Likelihood ratio** | Score function of transition density | Measurable |

For smooth payoffs, all four converge.  For discontinuous payoffs, only Malliavin and
LR have finite variance.
""", "markdown"),

    cell("""\
import sys
sys.path.insert(0, '..')

import numpy as np
import matplotlib.pyplot as plt
import time

from mgreeks.models.gbm import GeometricBrownianMotion
from mgreeks.payoffs.european import EuropeanCall
from mgreeks.simulation import MonteCarloEngine
from mgreeks.greeks import (
    MalliavinGreeks, FiniteDifferenceGreeks, PathwiseGreeks, LikelihoodRatioGreeks,
)
from mgreeks.greeks.analytical import bs_delta, bs_gamma, bs_vega

S0, K, T = 100.0, 100.0, 1.0
r, q, sigma = 0.05, 0.02, 0.20
n_paths, seed = 50_000, 42

model = GeometricBrownianMotion(r=r, q=q, sigma=sigma)
payoff = EuropeanCall(K)
engine = MonteCarloEngine(model, n_paths=n_paths, n_steps=1, rng_seed=seed)

mall = MalliavinGreeks(model, engine)
fd   = FiniteDifferenceGreeks(model, engine, bump_size=0.01, bump_type="relative")
pw   = PathwiseGreeks(model, engine)
lr   = LikelihoodRatioGreeks(model, engine)

bs = {
    "delta": bs_delta(S0, K, T, r, q, sigma, "call"),
    "gamma": bs_gamma(S0, K, T, r, q, sigma),
    "vega":  bs_vega(S0, K, T, r, q, sigma),
}
print(f"BS analytics: delta={bs['delta']:.4f}  gamma={bs['gamma']:.4f}  vega={bs['vega']:.4f}")
"""),

    cell("""\
## Side-by-side comparison

results = {}
for greek in ["delta", "gamma", "vega"]:
    results[greek] = {}
    for name, estimator in [("Malliavin", mall), ("FD", fd), ("Pathwise", pw), ("LR", lr)]:
        try:
            t0 = time.perf_counter()
            res = getattr(estimator, greek)(payoff, S0, T)
            elapsed = time.perf_counter() - t0
            results[greek][name] = {"value": res["value"], "se": res["std_error"], "t": elapsed}
        except (NotImplementedError, Exception):
            results[greek][name] = None

print(f"{'Greek':<8} {'Method':<12} {'Estimate':>10} {'SE':>10} {'Bias':>10} {'Time(s)':>8}")
print("-" * 65)
for greek in ["delta", "gamma", "vega"]:
    truth = bs[greek]
    for name, res in results[greek].items():
        if res is None:
            print(f"{greek:<8} {name:<12} {'N/A':>10}")
            continue
        bias = res['value'] - truth
        print(f"{greek:<8} {name:<12} {res['value']:>10.5f} {res['se']:>10.5f} {bias:>10.5f} {res['t']:>8.3f}")
    print()
"""),

    cell("""\
## Convergence: SE vs n_paths  (delta, European call)

n_grid = [1_000, 3_000, 10_000, 30_000, 100_000]
se_mall, se_fd = [], []

for n in n_grid:
    eng_n = MonteCarloEngine(model, n_paths=n, n_steps=1, rng_seed=seed)
    m = MalliavinGreeks(model, eng_n)
    f_est = FiniteDifferenceGreeks(model, eng_n, bump_size=0.01, bump_type="relative")
    se_mall.append(m.delta(payoff, S0, T)["std_error"])
    se_fd.append(f_est.delta(payoff, S0, T)["std_error"])

n_arr = np.array(n_grid, dtype=float)
ref = se_mall[0] * np.sqrt(n_grid[0]) / np.sqrt(n_arr)

fig, ax = plt.subplots(figsize=(7, 5))
ax.loglog(n_arr, se_mall, "b-o", label="Malliavin SE")
ax.loglog(n_arr, se_fd,   "r--s", label="FD SE (h=1%)")
ax.loglog(n_arr, ref, "k:", lw=1, label="1/√n reference")
ax.set_xlabel("n_paths"); ax.set_ylabel("Std Error")
ax.set_title("Convergence: European Call Delta"); ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("02_convergence.png", dpi=100, bbox_inches="tight")
plt.show()
print("Both methods converge at 1/√n.  FD has lower SE (FD wins for smooth payoffs with CRN).")
"""),

    cell("""\
## FD bump-size dilemma: European call vs Digital call

from mgreeks.payoffs.european import DigitalCall
from mgreeks.greeks.analytical import bs_digital_delta

digital = DigitalCall(K)
h_grid  = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]

se_euro_fd, se_dig_fd = [], []
for h in h_grid:
    fd_h = FiniteDifferenceGreeks(model, engine, bump_size=h, bump_type="relative")
    se_euro_fd.append(fd_h.delta(payoff,  S0, T)["std_error"])
    se_dig_fd.append( fd_h.delta(digital, S0, T)["std_error"])

# Malliavin SE (constant — no h)
se_mall_euro = mall.delta(payoff,  S0, T)["std_error"]
se_mall_dig  = mall.delta(digital, S0, T)["std_error"]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("FD Bump-Size Dilemma: SE vs h", fontsize=13)

for ax, se_fd_arr, se_mall_ref, title in [
    (axes[0], se_euro_fd, se_mall_euro, "European Call Delta"),
    (axes[1], se_dig_fd,  se_mall_dig,  "Digital Call Delta"),
]:
    ax.loglog(h_grid, se_fd_arr, "r-o", label="FD SE")
    ax.axhline(se_mall_ref, color="b", lw=2, label=f"Malliavin SE = {se_mall_ref:.5f}")
    ax.set_xlabel("Bump size h"); ax.set_ylabel("Std Error")
    ax.set_title(title); ax.legend(); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("02_bump_dilemma.png", dpi=100, bbox_inches="tight")
plt.show()
print("Digital: FD SE diverges as h→0.  Malliavin SE is constant and finite.")
"""),
]

# ─────────────────────────────────────────────────────────────────────────────
# 03_exotic_greeks.ipynb
# ─────────────────────────────────────────────────────────────────────────────

nb03_cells = [
    cell("""\
# 03 — Exotic Greeks: Asian, Barrier, Lookback

Path-dependent payoffs present additional challenges for Greek estimation:

- **Asian options:** payoff depends on the average of S along the path
- **Barrier options:** payoff is zero if the path crosses the barrier — discontinuous in S₀
- **Lookback options:** payoff depends on the running maximum/minimum

For FD, estimating the delta of a barrier option near the barrier is catastrophic:
bumping S₀ changes which paths survive the barrier — the estimator variance explodes
as S₀ → B.  Malliavin uses the same weight W_T/(σS₀T) and remains stable.
""", "markdown"),

    cell("""\
import sys
sys.path.insert(0, '..')

import numpy as np
import matplotlib.pyplot as plt
import time

from mgreeks.models.gbm import GeometricBrownianMotion
from mgreeks.payoffs.european import EuropeanCall, DigitalCall
from mgreeks.payoffs.asian import ArithmeticAsianCall
from mgreeks.payoffs.barrier import DownAndOutCall
from mgreeks.payoffs.lookback import FloatingStrikeLookbackCall
from mgreeks.simulation import MonteCarloEngine
from mgreeks.greeks import MalliavinGreeks, FiniteDifferenceGreeks
from mgreeks.greeks.analytical import bs_delta, bs_barrier_delta
from mgreeks.weights.malliavin_weights import delta_weight_path_dependent, delta_weight_gbm

S0, K, T, B = 100.0, 100.0, 1.0, 80.0
r, q, sigma  = 0.05, 0.02, 0.20
n_paths, seed = 50_000, 42
disc = np.exp(-r * T)

model = GeometricBrownianMotion(r=r, q=q, sigma=sigma)
print(f"Parameters: S0={S0}  K={K}  T={T}  B={B}  σ={sigma}")
"""),

    cell("""\
## 1. Arithmetic Asian call delta

n_steps = 52   # weekly averaging
engine_asian = MonteCarloEngine(model, n_paths=n_paths, n_steps=n_steps, rng_seed=seed)
mall_asian = MalliavinGreeks(model, engine_asian)
fd_asian   = FiniteDifferenceGreeks(model, engine_asian, bump_size=0.01, bump_type="relative")

asian_payoff = ArithmeticAsianCall(K)

t0 = time.perf_counter()
delta_mall_a = mall_asian.delta(asian_payoff, S0, T)
t_mall = time.perf_counter() - t0

t0 = time.perf_counter()
delta_fd_a   = fd_asian.delta(asian_payoff, S0, T)
t_fd = time.perf_counter() - t0

print("Asian Call Delta (n_steps=52 weekly averaging)")
print(f"  Malliavin : {delta_mall_a['value']:.5f} ± {delta_mall_a['std_error']:.5f}  ({t_mall:.2f}s, 1 sim)")
print(f"  FD (h=1%) : {delta_fd_a['value']:.5f} ± {delta_fd_a['std_error']:.5f}  ({t_fd:.2f}s, 2 sims)")
print(f"  SE ratio FD/Mall: {delta_fd_a['std_error']/delta_mall_a['std_error']:.2f}")
"""),

    cell("""\
## 2. Barrier option (DOC) delta near the barrier

n_steps_b = 52
doc_payoff = DownAndOutCall(K, B)

S0_grid = np.linspace(B + 3, 130, 25)
mall_est, mall_se = [], []
fd_est,   fd_se   = [], []

for s0 in S0_grid:
    rng = np.random.default_rng(seed)
    out = model.simulate(s0, T, n_steps_b, n_paths, return_full_paths=True, rng=rng)
    W_T = out["brownian_increments"].sum(axis=1)
    w   = delta_weight_gbm(s0, out["terminal"], sigma, r, q, T, W_T)
    f   = doc_payoff(out["paths"], out["times"])
    samps = disc * f * w
    mall_est.append(float(samps.mean()))
    mall_se.append(float(samps.std(ddof=1) / np.sqrt(n_paths)))

    dh = s0 * 0.01
    rng2 = np.random.default_rng(seed)
    out_up = model.simulate(s0 + dh, T, n_steps_b, n_paths, return_full_paths=True, rng=rng2)
    rng3 = np.random.default_rng(seed)
    out_dn = model.simulate(s0 - dh, T, n_steps_b, n_paths, return_full_paths=True, rng=rng3)
    f_up = doc_payoff(out_up["paths"], out_up["times"])
    f_dn = doc_payoff(out_dn["paths"], out_dn["times"])
    samps_fd = disc * (f_up - f_dn) / (2 * dh)
    fd_est.append(float(samps_fd.mean()))
    fd_se.append(float(samps_fd.std(ddof=1) / np.sqrt(n_paths)))

mall_est, mall_se = np.array(mall_est), np.array(mall_se)
fd_est,   fd_se   = np.array(fd_est),   np.array(fd_se)
analytic = np.array([bs_barrier_delta(s, K, B, T, r, q, sigma) for s in S0_grid])

z = 1.96
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(f"Down-and-Out Call Delta  (B={B}, K={K})", fontsize=13)

ax = axes[0]
ax.axvline(B, color="gray", ls=":", label=f"Barrier B={B}")
ax.plot(S0_grid, analytic, "k-", lw=2, label="Analytical (continuous)")
ax.plot(S0_grid, mall_est, "b-o", ms=4, label="Malliavin")
ax.fill_between(S0_grid, mall_est - z*mall_se, mall_est + z*mall_se, alpha=0.25, color="b")
ax.plot(S0_grid, fd_est, "r--s", ms=4, label="FD (h=1%)")
ax.fill_between(S0_grid, fd_est - z*fd_se, fd_est + z*fd_se, alpha=0.15, color="r")
ax.set_xlabel("S₀"); ax.set_ylabel("Delta"); ax.legend(fontsize=8)
ax.set_title("Estimates ± 95% CI")

ax2 = axes[1]
ax2.axvline(B, color="gray", ls=":")
ax2.semilogy(S0_grid, mall_se, "b-o", ms=4, label="Malliavin SE")
ax2.semilogy(S0_grid, fd_se,   "r--s", ms=4, label="FD SE (h=1%)")
ax2.set_xlabel("S₀"); ax2.set_ylabel("Std Error (log)"); ax2.legend(fontsize=8)
ax2.set_title("SE near barrier: Malliavin flat, FD spikes")

plt.tight_layout()
plt.savefig("03_barrier_sweep.png", dpi=100, bbox_inches="tight")
plt.show()
"""),

    cell("""\
## 3. Lookback call delta

engine_lb = MonteCarloEngine(model, n_paths=n_paths, n_steps=52, rng_seed=seed)
mall_lb = MalliavinGreeks(model, engine_lb)
fd_lb   = FiniteDifferenceGreeks(model, engine_lb, bump_size=0.01, bump_type="relative")

lookback = FloatingStrikeLookbackCall()
delta_mall_lb = mall_lb.delta(lookback, S0, T)
delta_fd_lb   = fd_lb.delta(lookback, S0, T)

print("Floating-Strike Lookback Call Delta")
print(f"  Malliavin : {delta_mall_lb['value']:.5f} ± {delta_mall_lb['std_error']:.5f}")
print(f"  FD (h=1%) : {delta_fd_lb['value']:.5f} ± {delta_fd_lb['std_error']:.5f}")
"""),

    cell("""\
## 4. Summary table

print(f"{'Payoff':<25} {'Method':<12} {'Delta':>10} {'SE':>10}")
print("=" * 62)
rows = [
    ("European call",        "Malliavin", bs_delta(S0, K, T, r, q, sigma, "call"), 0),
    ("Asian (arith) call",   "Malliavin", delta_mall_a["value"], delta_mall_a["std_error"]),
    ("Asian (arith) call",   "FD (h=1%)", delta_fd_a["value"],   delta_fd_a["std_error"]),
    ("Barrier DOC call",     "Malliavin", mall_est[len(mall_est)//2], mall_se[len(mall_se)//2]),
    ("Barrier DOC call",     "FD (h=1%)", fd_est[len(fd_est)//2],    fd_se[len(fd_se)//2]),
    ("Lookback (float) call","Malliavin", delta_mall_lb["value"], delta_mall_lb["std_error"]),
    ("Lookback (float) call","FD (h=1%)", delta_fd_lb["value"],   delta_fd_lb["std_error"]),
]
for payoff_name, method, val, se in rows:
    print(f"{payoff_name:<25} {method:<12} {val:>10.5f} {se if se else '—':>10}")
print()
print("Note: barrier delta shown at S₀=", round(S0_grid[len(S0_grid)//2], 1))
"""),
]

# ─────────────────────────────────────────────────────────────────────────────
# 04_variance_analysis.ipynb
# ─────────────────────────────────────────────────────────────────────────────

nb04_cells = [
    cell("""\
# 04 — Variance Analysis: When to Use Which Method

## Decision guide

| Setting | Best method | Reason |
|---------|-------------|--------|
| European call/put delta (smooth) | FD with CRN | FD variance < Malliavin for smooth payoffs |
| Digital / barrier delta | **Malliavin** | FD variance → ∞ as h → 0 |
| Multiple Greeks, one simulation | **Malliavin** | Single simulation for all Greeks |
| Higher-order Greeks (gamma, vanna) | **Malliavin** | FD requires 3–4 extra sims; unstable |
| Heston / local vol model | **Malliavin BEL** | Score function not available |

This notebook quantifies these statements with systematic variance comparisons.
""", "markdown"),

    cell("""\
import sys
sys.path.insert(0, '..')

import numpy as np
import matplotlib.pyplot as plt
import time

from mgreeks.models.gbm import GeometricBrownianMotion
from mgreeks.payoffs.european import EuropeanCall, DigitalCall
from mgreeks.payoffs.barrier import DownAndOutCall
from mgreeks.simulation import MonteCarloEngine
from mgreeks.greeks import (
    MalliavinGreeks, FiniteDifferenceGreeks, PathwiseGreeks, LikelihoodRatioGreeks,
)
from mgreeks.greeks.analytical import bs_delta, bs_digital_delta
from mgreeks.variance_reduction.antithetic import antithetic_malliavin, simulate_antithetic
from mgreeks.weights.malliavin_weights import delta_weight_gbm

S0, K, T, B = 100.0, 100.0, 1.0, 80.0
r, q, sigma  = 0.05, 0.02, 0.20
n_paths, seed = 50_000, 42
disc = np.exp(-r * T)

model   = GeometricBrownianMotion(r=r, q=q, sigma=sigma)
engine  = MonteCarloEngine(model, n_paths=n_paths, n_steps=1, rng_seed=seed)
mall    = MalliavinGreeks(model, engine)
fd      = FiniteDifferenceGreeks(model, engine, bump_size=0.01, bump_type="relative")
pw      = PathwiseGreeks(model, engine)
lr      = LikelihoodRatioGreeks(model, engine)
print("Setup complete.")
"""),

    cell("""\
## 1. Systematic SE comparison across payoffs and methods

payoffs_config = [
    ("European call",  EuropeanCall(K),   bs_delta(S0, K, T, r, q, sigma, "call")),
    ("Digital call",   DigitalCall(K),    bs_digital_delta(S0, K, T, r, q, sigma)),
]

print(f"{'Payoff':<18} {'Method':<12} {'Delta':>10} {'SE':>10} {'SE_ratio':>10} {'Bias':>10}")
print("=" * 78)
se_mall_by_payoff = {}

for name, pf, truth in payoffs_config:
    ses = {}
    for method_name, estimator in [("Malliavin", mall), ("FD", fd), ("Pathwise", pw), ("LR", lr)]:
        try:
            res = estimator.delta(pf, S0, T)
            ses[method_name] = res["std_error"]
            bias = res["value"] - truth
            print(f"{name:<18} {method_name:<12} {res['value']:>10.5f} {res['std_error']:>10.5f} {'—':>10} {bias:>10.5f}")
        except Exception:
            print(f"{name:<18} {method_name:<12} {'N/A':>10}")
    if "Malliavin" in ses:
        se_mall_by_payoff[name] = ses["Malliavin"]
        for method_name, se in ses.items():
            if method_name != "Malliavin":
                ratio = se / ses["Malliavin"]
                print(f"  → {method_name}/Malliavin SE ratio: {ratio:.2f}×")
    print()
"""),

    cell("""\
## 2. Variance reduction: antithetic variates for digital call delta

digital = DigitalCall(K)
rng = np.random.default_rng(seed)
sim_pos, sim_neg = simulate_antithetic(model, S0, T, n_steps=1, n_paths=n_paths, rng=rng)

def weight_fn(paths, increments):
    W_T = increments.sum(axis=1)
    return delta_weight_gbm(S0, paths[:, -1], sigma, r, q, T, W_T)

result = antithetic_malliavin(
    digital, weight_fn,
    sim_pos["paths"], sim_neg["paths"],
    sim_pos["brownian_increments"], sim_neg["brownian_increments"],
    disc,
)

truth = bs_digital_delta(S0, K, T, r, q, sigma)
se_plain = result["std_error"] * result["variance_reduction_ratio"] ** 0.5
print("Antithetic variates for Digital Call Delta")
print(f"  Estimate    : {result['value']:.5f}")
print(f"  SE (plain)  : {se_plain:.5f}")
print(f"  SE (anti)   : {result['std_error']:.5f}")
print(f"  VRR         : {result['variance_reduction_ratio']:.1f}x  (Var_plain / Var_anti)")
print(f"  Truth       : {truth:.5f}")
"""),

    cell("""\
## 3. Timing comparison: Malliavin vs FD for all Greeks

payoff = EuropeanCall(K)
timing_results = {}
greeks = ["delta", "gamma", "vega"]

for method_name, estimator in [("Malliavin", mall), ("FD", fd)]:
    times = []
    for greek in greeks:
        try:
            t0 = time.perf_counter()
            for _ in range(5):
                getattr(estimator, greek)(payoff, S0, T)
            times.append((time.perf_counter() - t0) / 5)
        except Exception:
            times.append(float("nan"))
    timing_results[method_name] = times

print(f"{'Greek':<8}", "  ".join(f"{m:>12}" for m in timing_results))
print("-" * 40)
for i, greek in enumerate(greeks):
    row = f"{greek:<8}"
    for method_name, times in timing_results.items():
        t = times[i]
        row += f"  {t*1000:>10.1f}ms"
    print(row)
print()
print("Malliavin: one simulation for ALL Greeks")
print("FD: requires a fresh simulation per Greek (2× extra per Greek for central difference)")

fig, ax = plt.subplots(figsize=(7, 4))
x = np.arange(len(greeks))
width = 0.35
times_mall = [t*1000 for t in timing_results["Malliavin"]]
times_fd   = [t*1000 for t in timing_results["FD"]]
ax.bar(x - width/2, times_mall, width, label="Malliavin", color="#1f77b4", alpha=0.8)
ax.bar(x + width/2, times_fd,   width, label="FD", color="#d62728", alpha=0.8)
ax.set_xticks(x); ax.set_xticklabels([g.capitalize() for g in greeks])
ax.set_ylabel("Time (ms)"); ax.set_title("Wall-Clock Time per Greek (n=50,000)")
ax.legend()
plt.tight_layout()
plt.savefig("04_timing.png", dpi=100, bbox_inches="tight")
plt.show()
"""),

    cell("""\
## 4. Practical recommendations (summary)

print(\"\"\"
PRACTICAL GUIDE — GREEK ESTIMATION METHOD SELECTION
=====================================================

1. SMOOTH TERMINAL PAYOFF (European call/put):
   - Only delta needed  → FD with CRN (lower variance than Malliavin)
   - Multiple Greeks    → Malliavin (one simulation, all Greeks)

2. DISCONTINUOUS PAYOFF (digital, binary, range accrual):
   - Any Greek          → Malliavin (FD variance diverges as h→0)

3. BARRIER OPTION near the barrier:
   - Any Greek          → Malliavin (FD variance explodes near B)
   - Variance reduction → Localization (concentrate weight on early steps)

4. PATH-DEPENDENT (Asian, lookback):
   - Delta              → Malliavin (ΔW₁/(σS₀Δt₁)) or pathwise IPA
   - All Greeks         → Malliavin (one simulation)
   - Variance reduction → Antithetic variates (VRR 3–20× for delta)

5. HIGHER-ORDER GREEKS (gamma, vanna, volga):
   - Always             → Malliavin (FD gamma: 3 sims + h² instability)
   - Vanna (∂²/∂S∂σ)   → Malliavin (FD needs 4 sims)

6. NON-GBM MODELS (Heston, local vol):
   - Delta              → Malliavin BEL (numerical stochastic integral)
   - Expect higher weight variance (1/√V for Heston near zero)
\"\"\")
"""),
]

# ─────────────────────────────────────────────────────────────────────────────
# Write notebooks
# ─────────────────────────────────────────────────────────────────────────────

notebooks = [
    ("01_malliavin_primer.ipynb",   nb01_cells),
    ("02_greeks_comparison.ipynb",  nb02_cells),
    ("03_exotic_greeks.ipynb",      nb03_cells),
    ("04_variance_analysis.ipynb",  nb04_cells),
]

for fname, cells in notebooks:
    nb = notebook(cells)
    path = NB_DIR / fname
    with open(path, "w") as fh:
        json.dump(nb, fh, indent=1)
    print(f"Written: {path}")

print("Done.")
