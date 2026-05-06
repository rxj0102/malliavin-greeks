"""
Deep dive into discontinuous payoff Greeks.

Demonstrates Malliavin's core advantage: finite variance for delta/gamma
of digital and barrier options, where FD has exploding variance as h→0.

Experiments
-----------
1. Digital call delta as a function of S_0 (sweep 80 → 120, K=100)
   - Malliavin (score-function) ± CI
   - FD for bump sizes h = 1%, 5%, 10%  (bias-variance tradeoff)
   - Analytical BS digital delta

2. Barrier option (DOC) delta near the barrier
   - S_0 swept from B+2 to 120
   - Malliavin remains stable; FD variance explodes near S_0 ≈ B

3. Convergence study: std_error vs n_paths
   - European call: all methods converge at 1/√n
   - Digital call: Malliavin at 1/√n; FD degrades or stalls

Output: experiments/results/discontinuous_{digital,barrier,convergence}.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
from experiments.exp_utils import (
    load_config, build_gbm, results_dir, save_or_show,
)

from mgreeks.greeks import MalliavinGreeks, FiniteDifferenceGreeks
from mgreeks.greeks.analytical import bs_digital_delta, bs_delta, bs_barrier_delta
from mgreeks.payoffs.european import DigitalCall, EuropeanCall
from mgreeks.payoffs.barrier import DownAndOutCall
from mgreeks.simulation import MonteCarloEngine
from mgreeks.weights.malliavin_weights import delta_weight_gbm


# ── Helpers ──────────────────────────────────────────────────────────────────

def _malliavin_delta(model, S0: float, K: float, T: float,
                     payoff, n_paths: int, n_steps: int, seed: int) -> tuple:
    """Return (estimate, se) for Malliavin delta."""
    out = model.simulate(S0, T, n_steps, n_paths,
                         return_full_paths=True,
                         rng=np.random.default_rng(seed))
    W_T = out["brownian_increments"].sum(axis=1)
    weight = delta_weight_gbm(S0, out["terminal"], model.sigma, model.r, model.q, T, W_T)
    f = payoff(out["paths"], out["times"])
    disc = float(np.exp(-model.r * T))
    samples = disc * f * weight
    return float(samples.mean()), float(samples.std(ddof=1) / np.sqrt(n_paths))


def _fd_delta(model, S0: float, K: float, T: float, payoff,
              h: float, n_paths: int, n_steps: int, seed: int) -> tuple:
    """Return (estimate, se) for FD central-difference delta."""
    rng_up = np.random.default_rng(seed)
    rng_dn = np.random.default_rng(seed)  # same seed → CRN

    dh = S0 * h
    out_up = model.simulate(S0 + dh, T, n_steps, n_paths,
                            return_full_paths=True, rng=rng_up)
    out_dn = model.simulate(S0 - dh, T, n_steps, n_paths,
                            return_full_paths=True, rng=rng_dn)
    disc = float(np.exp(-model.r * T))

    f_up = payoff(out_up["paths"], out_up["times"])
    f_dn = payoff(out_dn["paths"], out_dn["times"])
    samples = disc * (f_up - f_dn) / (2 * dh)
    return float(samples.mean()), float(samples.std(ddof=1) / np.sqrt(n_paths))


# ── Experiment 1: digital call delta sweep ───────────────────────────────────

def exp_digital_sweep(cfg: dict):
    """Plot digital call delta: Malliavin vs FD vs analytical, over S_0 grid."""
    model = build_gbm(cfg)
    K = cfg["option"]["K"]
    T = cfg["option"]["T"]
    n_paths = cfg["mc"]["n_paths"]
    n_steps = cfg["mc"]["n_steps_vanilla"]
    seed = cfg["seed"]
    bump_sizes = cfg["mc"]["bump_sizes"]

    disc = cfg["discontinuous"]
    S0_grid = np.linspace(
        disc["S0_grid_digital"]["start"],
        disc["S0_grid_digital"]["stop"],
        disc["S0_grid_digital"]["n_points"],
    )

    payoff = DigitalCall(K)

    # Analytical
    analytic = np.array([
        bs_digital_delta(s, K, T, model.r, model.q, model.sigma)
        for s in S0_grid
    ])

    # Malliavin
    mall_est, mall_se = zip(*[
        _malliavin_delta(model, s, K, T, payoff, n_paths, n_steps, seed)
        for s in S0_grid
    ])
    mall_est, mall_se = np.array(mall_est), np.array(mall_se)

    # FD for selected bump sizes
    fd_results = {}
    for h in [0.005, 0.01, 0.05]:
        est_h, se_h = zip(*[
            _fd_delta(model, s, K, T, payoff, h, n_paths, n_steps, seed)
            for s in S0_grid
        ])
        fd_results[h] = (np.array(est_h), np.array(se_h))

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=cfg["output"]["figsize"])
    fig.suptitle("Digital Call Delta: Malliavin vs FD vs Analytical", fontsize=12)
    z = 1.96

    ax = axes[0]
    ax.plot(S0_grid, analytic, "k-", linewidth=2, label="Analytical")
    ax.plot(S0_grid, mall_est, "b-o", markersize=4, label="Malliavin")
    ax.fill_between(S0_grid, mall_est - z * mall_se, mall_est + z * mall_se,
                    alpha=0.25, color="blue", label="Mall ±2 SE")
    colors = ["#d62728", "#ff7f0e", "#9467bd"]
    for (h, (est, se)), col in zip(fd_results.items(), colors):
        ax.plot(S0_grid, est, "--", color=col, label=f"FD h={h:.1%}")
    ax.set_xlabel("S₀"); ax.set_ylabel("Delta"); ax.legend(fontsize=7)
    ax.set_title("Estimates")

    ax2 = axes[1]
    ax2.semilogy(S0_grid, mall_se, "b-o", markersize=4, label="Malliavin SE")
    for (h, (est, se)), col in zip(fd_results.items(), colors):
        ax2.semilogy(S0_grid, se, "--", color=col, label=f"FD h={h:.1%} SE")
    ax2.set_xlabel("S₀"); ax2.set_ylabel("Std Error (log scale)")
    ax2.set_title("Standard Errors (smaller → better)")
    ax2.legend(fontsize=7)

    plt.tight_layout()
    save_or_show(fig, "discontinuous_digital", cfg)
    print("  Digital sweep done.")


# ── Experiment 2: barrier option delta near barrier ──────────────────────────

def exp_barrier_sweep(cfg: dict):
    """Plot DOC delta near the barrier: Malliavin stable, FD noisy."""
    model = build_gbm(cfg)
    K = cfg["option"]["K"]
    B = cfg["option"]["B_barrier"]
    T = cfg["option"]["T"]
    n_paths = cfg["mc"]["n_paths"]
    n_steps = cfg["mc"]["n_steps"]
    seed = cfg["seed"]

    disc_cfg = cfg["discontinuous"]
    S0_grid = np.linspace(
        disc_cfg["S0_grid_barrier"]["start"],
        disc_cfg["S0_grid_barrier"]["stop"],
        disc_cfg["S0_grid_barrier"]["n_points"],
    )
    S0_grid = S0_grid[S0_grid > B + 1]   # keep S0 above barrier

    payoff = DownAndOutCall(K, B)

    # Analytical (continuous monitoring)
    analytic = np.array([
        bs_barrier_delta(s, K, B, T, model.r, model.q, model.sigma)
        for s in S0_grid
    ])

    # Malliavin
    mall_est, mall_se = zip(*[
        _malliavin_delta(model, s, K, T, payoff, n_paths, n_steps, seed)
        for s in S0_grid
    ])
    mall_est, mall_se = np.array(mall_est), np.array(mall_se)

    # FD (1% bump)
    fd_est, fd_se = zip(*[
        _fd_delta(model, s, K, T, payoff, 0.01, n_paths, n_steps, seed)
        for s in S0_grid
    ])
    fd_est, fd_se = np.array(fd_est), np.array(fd_se)

    z = 1.96
    fig, axes = plt.subplots(1, 2, figsize=cfg["output"]["figsize"])
    fig.suptitle(f"Down-and-Out Call Delta (B={B}, K={K})", fontsize=12)

    ax = axes[0]
    ax.axvline(B, color="gray", linestyle=":", label=f"Barrier B={B}")
    ax.plot(S0_grid, analytic, "k-", linewidth=2, label="Analytical (continuous)")
    ax.plot(S0_grid, mall_est, "b-o", markersize=4, label="Malliavin")
    ax.fill_between(S0_grid, mall_est - z * mall_se, mall_est + z * mall_se,
                    alpha=0.25, color="blue")
    ax.plot(S0_grid, fd_est, "r--s", markersize=4, label="FD (h=1%)")
    ax.fill_between(S0_grid, fd_est - z * fd_se, fd_est + z * fd_se,
                    alpha=0.15, color="red")
    ax.set_xlabel("S₀"); ax.set_ylabel("Delta"); ax.legend(fontsize=8)
    ax.set_title("Estimates ± 95% CI")

    ax2 = axes[1]
    ax2.axvline(B, color="gray", linestyle=":")
    ax2.semilogy(S0_grid, mall_se, "b-o", markersize=4, label="Malliavin SE")
    ax2.semilogy(S0_grid, fd_se, "r--s", markersize=4, label="FD SE (h=1%)")
    ax2.set_xlabel("S₀"); ax2.set_ylabel("Std Error (log scale)")
    ax2.set_title("SE near barrier: Malliavin stays flat, FD spikes")
    ax2.legend(fontsize=8)

    plt.tight_layout()
    save_or_show(fig, "discontinuous_barrier", cfg)
    print("  Barrier sweep done.")


# ── Experiment 3: convergence study ─────────────────────────────────────────

def exp_convergence(cfg: dict):
    """
    Std error vs n_paths for:
      - European call delta (Malliavin + FD): both ∝ 1/√n
      - Digital call delta (Malliavin + FD):  Malliavin ∝ 1/√n, FD stalls/noisy
    """
    model = build_gbm(cfg)
    K = cfg["option"]["K"]
    S0 = cfg["option"]["S0"]
    T = cfg["option"]["T"]
    seed = cfg["seed"]
    n_grid = cfg["convergence"]["n_paths_grid"]
    n_steps_v = cfg["mc"]["n_steps_vanilla"]
    h = cfg["mc"]["bump_size"]

    euro_payoff = EuropeanCall(K)
    digital_payoff = DigitalCall(K)

    results = {
        "euro_mall": [], "euro_fd": [],
        "digital_mall": [], "digital_fd": [],
    }

    for n in n_grid:
        print(f"  n_paths={n:>8d} ...", end="", flush=True)

        _, se = _malliavin_delta(model, S0, K, T, euro_payoff, n, n_steps_v, seed)
        results["euro_mall"].append(se)

        _, se = _fd_delta(model, S0, K, T, euro_payoff, h, n, n_steps_v, seed)
        results["euro_fd"].append(se)

        _, se = _malliavin_delta(model, S0, K, T, digital_payoff, n, n_steps_v, seed)
        results["digital_mall"].append(se)

        _, se = _fd_delta(model, S0, K, T, digital_payoff, h, n, n_steps_v, seed)
        results["digital_fd"].append(se)

        print(" done")

    n_arr = np.array(n_grid, dtype=float)
    ref = results["euro_mall"][0] * np.sqrt(n_grid[0]) / np.sqrt(n_arr)  # 1/√n reference

    fig, axes = plt.subplots(1, 2, figsize=cfg["output"]["figsize"])
    fig.suptitle("Convergence: Std Error vs N paths", fontsize=12)

    for ax, key_m, key_f, title in [
        (axes[0], "euro_mall", "euro_fd", "European Call Delta"),
        (axes[1], "digital_mall", "digital_fd", "Digital Call Delta"),
    ]:
        ax.loglog(n_arr, results[key_m], "b-o", label="Malliavin")
        ax.loglog(n_arr, results[key_f], "r--s", label=f"FD (h={h:.1%})")
        ax.loglog(n_arr, ref, "k:", linewidth=1, label="1/√N reference")
        ax.set_xlabel("N paths"); ax.set_ylabel("Std Error")
        ax.set_title(title); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_or_show(fig, "discontinuous_convergence", cfg)
    print("  Convergence study done.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    cfg = load_config()
    print("=== Discontinuous Payoffs Experiment ===")
    print(f"n_paths={cfg['mc']['n_paths']}, seed={cfg['seed']}")

    print("\n[1/3] Digital call delta sweep ...")
    exp_digital_sweep(cfg)

    print("\n[2/3] Barrier option near-barrier sweep ...")
    exp_barrier_sweep(cfg)

    print("\n[3/3] Convergence study ...")
    exp_convergence(cfg)

    print("\nAll discontinuous experiments done.")
    print(f"Results in: {results_dir(cfg)}")


if __name__ == "__main__":
    main()
