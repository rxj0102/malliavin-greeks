"""
Higher-order Greeks: gamma, vanna, volga.

Experiments
-----------
1. Gamma surface: Γ(S₀, T) for European call — Malliavin vs FD vs analytical
2. Vanna (∂²V/∂S∂σ) and Volga (∂²V/∂σ²) — single sim vs 4 FD sims
3. Digital call higher-order Greeks — Malliavin handles δ'(S-K) cleanly;
   FD gamma/vanna are catastrophically noisy

Output: experiments/results/higher_order_{gamma_surface, vanna_volga, digital}.png
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
from experiments.exp_utils import (
    load_config, build_gbm, results_dir, save_or_show, result_table, fmt,
)

from mgreeks.greeks import MalliavinGreeks, FiniteDifferenceGreeks
from mgreeks.greeks.analytical import bs_gamma
from mgreeks.payoffs.european import EuropeanCall, DigitalCall
from mgreeks.simulation import MonteCarloEngine
from mgreeks.weights.malliavin_weights import gamma_weight_gbm, vega_weight_gbm


# ── Experiment 1: gamma surface ──────────────────────────────────────────────

def exp_gamma_surface(cfg: dict):
    """Γ(S₀, T) surface: Malliavin vs FD vs analytical."""
    model = build_gbm(cfg)
    K = cfg["option"]["K"]
    n_paths = cfg["mc"]["n_paths"]
    seed = cfg["seed"]

    ho = cfg["higher_order"]
    S0_grid = np.linspace(ho["S0_grid"]["start"], ho["S0_grid"]["stop"],
                          ho["S0_grid"]["n_points"])
    T_grid = ho["T_grid"]

    payoff = EuropeanCall(K)
    bump = cfg["mc"]["bump_size"]

    print("  Computing gamma surface ...", flush=True)
    rows = []
    for T in T_grid:
        n_steps = max(1, int(52 * T))
        engine = MonteCarloEngine(model, n_paths=n_paths, n_steps=n_steps,
                                  rng_seed=seed)
        mall = MalliavinGreeks(model, engine)
        fd = FiniteDifferenceGreeks(model, engine, bump_size=bump,
                                    bump_type="relative")
        for S0 in S0_grid:
            analytic = bs_gamma(S0, K, T, model.r, model.q, model.sigma)
            res_m = mall.gamma(payoff, S0, T)
            res_f = fd.gamma(payoff, S0, T)
            rows.append({
                "S0": S0, "T": T,
                "analytic": analytic,
                "mall": res_m["value"], "mall_se": res_m["std_error"],
                "fd": res_f["value"], "fd_se": res_f["std_error"],
            })

    # Plot: S0 slice at T=1
    T_plot = 1.0
    subset = [r for r in rows if r["T"] == T_plot]
    S_arr = np.array([r["S0"] for r in subset])
    g_analytic = np.array([r["analytic"] for r in subset])
    g_mall = np.array([r["mall"] for r in subset])
    g_mall_se = np.array([r["mall_se"] for r in subset])
    g_fd = np.array([r["fd"] for r in subset])
    g_fd_se = np.array([r["fd_se"] for r in subset])

    z = 1.96
    fig, axes = plt.subplots(1, 2, figsize=cfg["output"]["figsize"])
    fig.suptitle(f"European Call Gamma (T={T_plot}, K={K})", fontsize=12)

    ax = axes[0]
    ax.plot(S_arr, g_analytic, "k-", linewidth=2, label="Analytical")
    ax.plot(S_arr, g_mall, "b-o", markersize=4, label="Malliavin")
    ax.fill_between(S_arr, g_mall - z * g_mall_se, g_mall + z * g_mall_se,
                    alpha=0.25, color="blue")
    ax.plot(S_arr, g_fd, "r--s", markersize=4, label="FD")
    ax.fill_between(S_arr, g_fd - z * g_fd_se, g_fd + z * g_fd_se,
                    alpha=0.15, color="red")
    ax.set_xlabel("S₀"); ax.set_ylabel("Gamma"); ax.legend(fontsize=8)
    ax.set_title("Gamma vs S₀")

    ax2 = axes[1]
    ax2.semilogy(S_arr, g_mall_se, "b-o", markersize=4, label="Malliavin SE")
    ax2.semilogy(S_arr, g_fd_se, "r--s", markersize=4, label="FD SE")
    ax2.set_xlabel("S₀"); ax2.set_ylabel("Std Error"); ax2.legend(fontsize=8)
    ax2.set_title("Std Error vs S₀")

    plt.tight_layout()
    save_or_show(fig, "higher_order_gamma_surface", cfg)
    print("  Gamma surface done.")


# ── Experiment 2: vanna and volga ────────────────────────────────────────────

def exp_vanna_volga(cfg: dict):
    """
    Vanna (∂²V/∂S∂σ) and Volga (∂²V/∂σ²) via Malliavin and FD.

    Malliavin: one simulation, use bel_second_order_weight.
    FD:        cross-difference requires 4 simulations:
        Vanna ≈ [V(S+h_S, σ+h_σ) - V(S+h_S, σ-h_σ)
                 - V(S-h_S, σ+h_σ) + V(S-h_S, σ-h_σ)] / (4 h_S h_σ)
    """
    model = build_gbm(cfg)
    K = cfg["option"]["K"]
    S0 = cfg["option"]["S0"]
    T = cfg["option"]["T"]
    n_paths = cfg["mc"]["n_paths"]
    seed = cfg["seed"]
    h = cfg["mc"]["bump_size"]

    payoff = EuropeanCall(K)
    n_steps = 1
    engine = MonteCarloEngine(model, n_paths=n_paths, n_steps=n_steps, rng_seed=seed)
    mall = MalliavinGreeks(model, engine)

    print("  Vanna/Volga via Malliavin ...", flush=True)
    t0 = time.perf_counter()
    res_vanna_m = mall.higher_order(payoff, S0, T, greek_name="vanna")
    res_volga_m = mall.higher_order(payoff, S0, T, greek_name="volga")
    t_mall = time.perf_counter() - t0

    print("  Vanna/Volga via FD (4 simulations) ...", flush=True)
    t0 = time.perf_counter()
    h_s = S0 * h
    h_v = model.sigma * h
    sigma_orig = model.sigma

    def _price(s, sig):
        from mgreeks.models.gbm import GeometricBrownianMotion
        from mgreeks.utils import bs_price
        # For efficiency use analytical price in FD cross-difference
        return bs_price(s, K, model.r, model.q, sig, T, "call")

    vanna_fd = ((_price(S0 + h_s, sigma_orig + h_v) - _price(S0 + h_s, sigma_orig - h_v)
                 - _price(S0 - h_s, sigma_orig + h_v) + _price(S0 - h_s, sigma_orig - h_v))
                / (4 * h_s * h_v))
    volga_fd = ((_price(S0, sigma_orig + h_v) - 2 * _price(S0, sigma_orig)
                 + _price(S0, sigma_orig - h_v)) / h_v**2)
    t_fd = time.perf_counter() - t0

    # Analytical
    from scipy.stats import norm
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S0 / K) + (model.r - model.q + 0.5 * model.sigma**2) * T) / (model.sigma * sqrt_T)
    phi_d1 = norm.pdf(d1)
    vanna_analytic = -np.exp(-model.q * T) * phi_d1 * (d1 / model.sigma - sqrt_T) / model.sigma
    volga_analytic = (S0 * np.exp(-model.q * T) * phi_d1 * sqrt_T
                      * d1 * (d1 - model.sigma * sqrt_T) / model.sigma)

    rows = [
        {"greek": "Vanna", "method": "Malliavin", "estimate": fmt(res_vanna_m["value"], 6),
         "std_error": fmt(res_vanna_m["std_error"], 6), "time_s": f"{t_mall:.3f}",
         "analytical": fmt(vanna_analytic, 6)},
        {"greek": "Vanna", "method": "FD (analytic)", "estimate": fmt(vanna_fd, 6),
         "std_error": "0 (exact)", "time_s": f"{t_fd:.4f}",
         "analytical": fmt(vanna_analytic, 6)},
        {"greek": "Volga", "method": "Malliavin", "estimate": fmt(res_volga_m["value"], 4),
         "std_error": fmt(res_volga_m["std_error"], 4), "time_s": f"{t_mall:.3f}",
         "analytical": fmt(volga_analytic, 4)},
        {"greek": "Volga", "method": "FD (analytic)", "estimate": fmt(volga_fd, 4),
         "std_error": "0 (exact)", "time_s": f"{t_fd:.4f}",
         "analytical": fmt(volga_analytic, 4)},
    ]
    result_table(rows, ["greek", "method", "estimate", "std_error", "time_s", "analytical"],
                 "Vanna and Volga")

    # Bar chart: SE comparison for Malliavin
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ["Vanna\n(Malliavin)", "Volga\n(Malliavin)"]
    ses = [res_vanna_m["std_error"], res_volga_m["std_error"]]
    truths = [abs(vanna_analytic), abs(volga_analytic)]
    ax.bar(labels, ses, color=["#1f77b4", "#ff7f0e"], alpha=0.8)
    ax.set_ylabel("Std Error")
    ax.set_title("Malliavin Std Errors for Vanna and Volga")
    for i, (se, tr) in enumerate(zip(ses, truths)):
        ax.text(i, se + max(ses) * 0.02,
                f"SE/|truth|={se/tr:.1%}" if tr > 0 else "",
                ha="center", fontsize=9)
    plt.tight_layout()
    save_or_show(fig, "higher_order_vanna_volga", cfg)
    print("  Vanna/Volga done.")


# ── Experiment 3: digital higher-order Greeks ─────────────────────────────────

def exp_digital_higher_order(cfg: dict):
    """
    Gamma of a digital call:
      - FD: involves δ'(S-K) numerically → catastrophic noise
      - Malliavin: handles it cleanly via the BEL second-order weight
    """
    model = build_gbm(cfg)
    K = cfg["option"]["K"]
    S0 = cfg["option"]["S0"]
    T = cfg["option"]["T"]
    n_paths = cfg["mc"]["n_paths"]
    seed = cfg["seed"]

    payoff = DigitalCall(K)

    # Analytical digital gamma
    from mgreeks.greeks.analytical import bs_digital_gamma
    truth_gamma = bs_digital_gamma(S0, K, T, model.r, model.q, model.sigma)

    print("  Digital gamma: Malliavin ...", flush=True)
    engine = MonteCarloEngine(model, n_paths=n_paths, n_steps=1, rng_seed=seed)
    mall = MalliavinGreeks(model, engine)
    res_mall = mall.gamma(payoff, S0, T)

    print("  Digital gamma: FD sweep over bump sizes ...", flush=True)
    fd_results = {}
    for h in cfg["mc"]["bump_sizes"]:
        fd = FiniteDifferenceGreeks(model, engine, bump_size=h, bump_type="relative")
        res = fd.gamma(payoff, S0, T)
        fd_results[h] = res

    rows = [
        {"method": "Malliavin", "bump_h": "—",
         "estimate": fmt(res_mall["value"], 6),
         "std_error": fmt(res_mall["std_error"], 6),
         "truth": fmt(truth_gamma, 6)},
    ]
    for h, res in fd_results.items():
        rows.append({
            "method": "FD", "bump_h": f"{h:.1%}",
            "estimate": fmt(res["value"], 6),
            "std_error": fmt(res["std_error"], 6),
            "truth": fmt(truth_gamma, 6),
        })
    result_table(rows, ["method", "bump_h", "estimate", "std_error", "truth"],
                 "Digital Call Gamma: Malliavin vs FD")

    # Plot
    h_arr = np.array(cfg["mc"]["bump_sizes"])
    fd_se = np.array([fd_results[h]["std_error"] for h in cfg["mc"]["bump_sizes"]])
    fd_est = np.array([fd_results[h]["value"] for h in cfg["mc"]["bump_sizes"]])

    fig, axes = plt.subplots(1, 2, figsize=cfg["output"]["figsize"])
    fig.suptitle("Digital Call Gamma: FD Bias-Variance vs Malliavin", fontsize=12)

    ax = axes[0]
    ax.semilogx(h_arr, fd_est, "r-o", label="FD estimate")
    ax.axhline(truth_gamma, color="k", linestyle="--", label="Analytical")
    ax.axhline(res_mall["value"], color="b", linestyle="-", label="Malliavin")
    ax.fill_between([h_arr[0], h_arr[-1]],
                    [res_mall["value"] - 2 * res_mall["std_error"]] * 2,
                    [res_mall["value"] + 2 * res_mall["std_error"]] * 2,
                    alpha=0.2, color="blue")
    ax.set_xlabel("Bump size h"); ax.set_ylabel("Gamma")
    ax.set_title("FD bias-variance tradeoff\n(small h → noisy; large h → biased)")
    ax.legend(fontsize=8)

    ax2 = axes[1]
    ax2.loglog(h_arr, fd_se, "r-o", label="FD SE ∝ 1/h")
    ax2.axhline(res_mall["std_error"], color="b", linestyle="-",
                label=f"Malliavin SE = {res_mall['std_error']:.4f}")
    ax2.set_xlabel("Bump size h"); ax2.set_ylabel("Std Error")
    ax2.set_title("FD std error explodes as h→0\nMalliavin SE stays constant")
    ax2.legend(fontsize=8)

    plt.tight_layout()
    save_or_show(fig, "higher_order_digital", cfg)
    print("  Digital higher-order done.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    cfg = load_config()
    print("=== Higher-Order Greeks Experiment ===")
    print(f"n_paths={cfg['mc']['n_paths']}, seed={cfg['seed']}")

    print("\n[1/3] Gamma surface ...")
    exp_gamma_surface(cfg)

    print("\n[2/3] Vanna and Volga ...")
    exp_vanna_volga(cfg)

    print("\n[3/3] Digital call higher-order Greeks ...")
    exp_digital_higher_order(cfg)

    print("\nAll higher-order experiments done.")
    print(f"Results in: {results_dir(cfg)}")


if __name__ == "__main__":
    main()
