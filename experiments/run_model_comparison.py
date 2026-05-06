"""
Compare Malliavin Greeks across three models for a European call.

Models
------
1. GBM         — closed-form Malliavin weights (W_T / (σ S_0 T))
2. CEV (β=0.7) — weights via first-variation process (Euler simulation)
3. Heston      — BEL weight using inverse square-root of spot variance

For each model we compute delta and vega of a European call, then report:
  - MC estimate ± SE
  - Wall-clock time
  - SE ratio vs FD (efficiency of Malliavin for that model)

Key insight: as models get more complex, the Malliavin weight has higher
variance (variational equation amplifies noise), but remains valid for
discontinuous payoffs where FD breaks down.

Output
------
experiments/results/model_comparison_{estimates, se_comparison}.png
Console: comparison table
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
    load_config, build_gbm, build_heston, build_local_vol,
    results_dir, save_or_show, result_table, fmt,
)

from mgreeks.greeks import MalliavinGreeks, FiniteDifferenceGreeks
from mgreeks.greeks.analytical import bs_delta, bs_vega
from mgreeks.payoffs.european import EuropeanCall, DigitalCall
from mgreeks.simulation import MonteCarloEngine
from mgreeks.weights.malliavin_weights import delta_weight_heston, vega_weight_heston


# ── GBM delta/vega ───────────────────────────────────────────────────────────

def _gbm_greeks(model, payoff, S0, T, n_paths, n_steps, seed, bump):
    engine = MonteCarloEngine(model, n_paths=n_paths, n_steps=n_steps, rng_seed=seed)
    mall = MalliavinGreeks(model, engine)
    fd = FiniteDifferenceGreeks(model, engine, bump_size=bump, bump_type="relative")

    t0 = time.perf_counter()
    d_m = mall.delta(payoff, S0, T)
    v_m = mall.vega(payoff, S0, T)
    t_mall = time.perf_counter() - t0

    t0 = time.perf_counter()
    d_f = fd.delta(payoff, S0, T)
    v_f = fd.vega(payoff, S0, T)
    t_fd = time.perf_counter() - t0

    return {
        "delta_mall": d_m, "vega_mall": v_m, "t_mall": t_mall,
        "delta_fd": d_f, "vega_fd": v_f, "t_fd": t_fd,
    }


# ── CEV: FD for delta (Malliavin BEL requires full variational SDE integration)

def _cev_greeks(model, payoff, S0, T, n_paths, n_steps, seed, bump):
    """
    CEV Malliavin delta uses the pathwise IPA estimator (works for smooth payoffs).
    IPA: delta = E[e^{-rT} f'(S_T) × J_{0,T}]  where J_{0,T} = ∂S_T/∂S_0.
    For a European call, f'(S_T) = 1_{S_T > K}.

    Note: For DISCONTINUOUS payoffs the BEL formula is needed, which requires
    solving the variational SDE for σ_loc — not implemented here.
    """
    engine = MonteCarloEngine(model, n_paths=n_paths, n_steps=n_steps, rng_seed=seed)
    fd = FiniteDifferenceGreeks(model, engine, bump_size=bump, bump_type="relative")

    out = model.simulate(S0, T, n_steps, n_paths, return_full_paths=True,
                         rng=np.random.default_rng(seed))
    disc = float(np.exp(-model.r * T))

    t0 = time.perf_counter()
    # Pathwise IPA for CEV: delta = E[1_{S_T>K} × J_{0,T}] × disc
    # J_{0,T} computed via the variational equation (approximated by ratio chain)
    J_full = model.first_variation(out["paths"], out["brownian_increments"],
                                   s_index=0)
    J_T = J_full[:, -1]   # ∂S_T/∂S_0 (ratio-chain approximation)

    from mgreeks.payoffs.european import EuropeanCall
    K_val = float(out["terminal"].mean() * 0)    # zero placeholder
    indicator = (out["terminal"] > K_val).astype(float)
    # For European call with payoff f(S_T) = max(S_T-K, 0):
    # f'(S_T) = 1_{S_T > K}
    K_strike = payoff.K if hasattr(payoff, "K") else 100.0
    f_deriv = (out["terminal"] > K_strike).astype(float)
    samples_d = disc * f_deriv * J_T
    t_mall = time.perf_counter() - t0

    d_m = {"value": float(samples_d.mean()),
           "std_error": float(samples_d.std(ddof=1) / np.sqrt(n_paths))}

    t0 = time.perf_counter()
    d_f = fd.delta(payoff, S0, T)
    try:
        v_f = fd.vega(payoff, S0, T)
    except AttributeError:
        v_f = None
    t_fd = time.perf_counter() - t0

    return {
        "delta_mall": d_m, "vega_mall": None, "t_mall": t_mall,
        "delta_fd": d_f, "vega_fd": v_f, "t_fd": t_fd,
    }


# ── Heston delta/vega via explicit BEL weights ────────────────────────────────

def _heston_greeks(model, payoff, S0, T, n_paths, n_steps, seed, bump):
    from mgreeks.models.gbm import GeometricBrownianMotion

    engine = MonteCarloEngine(model, n_paths=n_paths, n_steps=n_steps, rng_seed=seed)
    fd = FiniteDifferenceGreeks(model, engine, bump_size=bump, bump_type="relative")

    out = model.simulate(S0, T, n_steps, n_paths, return_full_paths=True,
                         scheme="euler", rng=np.random.default_rng(seed))
    disc = float(np.exp(-model.r * T))
    f = payoff(out["paths"], out["times"])

    t0 = time.perf_counter()
    w_d = delta_weight_heston(
        out["paths"], out["v_paths"],
        out["brownian_increments"], out["v_brownian_increments"],
        S0, model.V0, T, out["times"], model,
    )
    w_v = vega_weight_heston(
        out["paths"], out["v_paths"],
        out["brownian_increments"], out["v_brownian_increments"],
        S0, model.V0, T, out["times"], model,
    )
    samples_d = disc * f * w_d
    samples_v = disc * f * w_v
    t_mall = time.perf_counter() - t0

    d_m = {"value": float(samples_d.mean()),
           "std_error": float(samples_d.std(ddof=1) / np.sqrt(n_paths))}
    v_m = {"value": float(samples_v.mean()),
           "std_error": float(samples_v.std(ddof=1) / np.sqrt(n_paths))}

    t0 = time.perf_counter()
    d_f = fd.delta(payoff, S0, T)
    try:
        v_f = fd.vega(payoff, S0, T)
    except AttributeError:
        v_f = None
    t_fd = time.perf_counter() - t0

    return {
        "delta_mall": d_m, "vega_mall": v_m, "t_mall": t_mall,
        "delta_fd": d_f, "vega_fd": v_f, "t_fd": t_fd,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    cfg = load_config()
    S0 = cfg["option"]["S0"]
    K = cfg["option"]["K"]
    T = cfg["option"]["T"]
    n_paths = cfg["mc"]["n_paths"]
    n_steps = cfg["mc"]["n_steps"]
    seed = cfg["seed"]
    bump = cfg["mc"]["bump_size"]

    gbm = build_gbm(cfg)
    heston = build_heston(cfg)
    cev = build_local_vol(cfg)

    euro_payoff = EuropeanCall(K)
    digital_payoff = DigitalCall(K)

    analytic_delta = bs_delta(S0, K, T, gbm.r, gbm.q, gbm.sigma, "call")
    analytic_vega = bs_vega(S0, K, T, gbm.r, gbm.q, gbm.sigma)

    print("=== Model Comparison Experiment ===")
    print(f"n_paths={n_paths}, n_steps={n_steps}, seed={seed}\n")

    results = {}

    print("[1/3] GBM ...", flush=True)
    results["GBM"] = _gbm_greeks(gbm, euro_payoff, S0, T, n_paths, n_steps, seed, bump)

    print("[2/3] CEV (β=0.7) ...", flush=True)
    results["CEV"] = _cev_greeks(cev, euro_payoff, S0, T, n_paths, n_steps, seed, bump)

    print("[3/3] Heston ...", flush=True)
    results["Heston"] = _heston_greeks(
        heston, euro_payoff, S0, T, n_paths, n_steps, seed, bump
    )

    # ── Table ────────────────────────────────────────────────────────────────
    rows = []
    for model_name, res in results.items():
        for greek_key, analytic in [("delta", analytic_delta), ("vega", analytic_vega)]:
            mall_res = res.get(f"{greek_key}_mall")
            fd_res = res.get(f"{greek_key}_fd")
            t_m = res["t_mall"]
            t_f = res["t_fd"]

            if mall_res:
                se_ratio = (fd_res["std_error"] / mall_res["std_error"]
                            if fd_res and mall_res["std_error"] > 0 else float("nan"))
                rows.append({
                    "model": model_name, "greek": greek_key, "method": "Malliavin",
                    "estimate": fmt(mall_res["value"], 5),
                    "std_error": fmt(mall_res["std_error"], 6),
                    "SE_ratio_vs_FD": f"{se_ratio:.2f}",
                    "time_s": fmt(t_m, 3),
                    "analytic": fmt(analytic, 5) if analytic else "—",
                })
            if fd_res:
                rows.append({
                    "model": model_name, "greek": greek_key, "method": "FD",
                    "estimate": fmt(fd_res["value"], 5),
                    "std_error": fmt(fd_res["std_error"], 6),
                    "SE_ratio_vs_FD": "1.00",
                    "time_s": fmt(t_f, 3),
                    "analytic": fmt(analytic, 5) if analytic else "—",
                })

    result_table(rows,
                 ["model", "greek", "method", "estimate", "std_error",
                  "SE_ratio_vs_FD", "time_s", "analytic"],
                 "Model Comparison: Malliavin vs FD (European Call)")

    # ── Digital call delta comparison (shows Malliavin advantage) ─────────────
    print("\n── Digital Call Delta (FD is unreliable near K) ──")
    digital_rows = []
    gbm_d = _gbm_greeks(gbm, digital_payoff, S0, T, n_paths // 5, 1, seed, bump)
    from mgreeks.greeks.analytical import bs_digital_delta
    digital_truth = bs_digital_delta(S0, K, T, gbm.r, gbm.q, gbm.sigma)
    for method_key, label in [("delta_mall", "Malliavin"), ("delta_fd", "FD")]:
        r = gbm_d[method_key]
        digital_rows.append({
            "method": label,
            "estimate": fmt(r["value"], 5),
            "std_error": fmt(r["std_error"], 6),
            "truth": fmt(digital_truth, 5),
            "bias": fmt(r["value"] - digital_truth, 5),
        })
    result_table(digital_rows, ["method", "estimate", "std_error", "truth", "bias"],
                 "Digital Call Delta (GBM)")

    # ── Plots ─────────────────────────────────────────────────────────────────
    _plot_model_comparison(results, cfg)


def _plot_model_comparison(results: dict, cfg: dict):
    model_names = list(results.keys())
    greeks = ["delta", "vega"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Model Comparison: Malliavin vs FD — European Call", fontsize=13)

    for row_idx, greek in enumerate(greeks):
        estimates_m = [results[m][f"{greek}_mall"]["value"]
                       if results[m][f"{greek}_mall"] else np.nan
                       for m in model_names]
        ses_m = [results[m][f"{greek}_mall"]["std_error"]
                 if results[m][f"{greek}_mall"] else np.nan
                 for m in model_names]
        estimates_f = [results[m][f"{greek}_fd"]["value"]
                       if results[m][f"{greek}_fd"] else np.nan
                       for m in model_names]
        ses_f = [results[m][f"{greek}_fd"]["std_error"]
                 if results[m][f"{greek}_fd"] else np.nan
                 for m in model_names]

        x = np.arange(len(model_names))
        width = 0.35

        ax = axes[row_idx][0]
        ax.bar(x - width / 2, estimates_m, width, label="Malliavin",
               color="#1f77b4", alpha=0.8, yerr=ses_m, capsize=4)
        ax.bar(x + width / 2, estimates_f, width, label="FD",
               color="#d62728", alpha=0.8, yerr=ses_f, capsize=4)
        ax.set_xticks(x); ax.set_xticklabels(model_names)
        ax.set_ylabel(greek.capitalize()); ax.legend(fontsize=8)
        ax.set_title(f"{greek.capitalize()} Estimates ± SE")

        ax2 = axes[row_idx][1]
        ax2.bar(x - width / 2, ses_m, width, label="Malliavin SE",
                color="#1f77b4", alpha=0.8)
        ax2.bar(x + width / 2, ses_f, width, label="FD SE",
                color="#d62728", alpha=0.8)
        ax2.set_xticks(x); ax2.set_xticklabels(model_names)
        ax2.set_ylabel("Std Error"); ax2.legend(fontsize=8)
        ax2.set_title(f"{greek.capitalize()} Std Errors (smaller → better)")

    plt.tight_layout()
    save_or_show(fig, "model_comparison", cfg)
    print("  Model comparison plot done.")


if __name__ == "__main__":
    main()
