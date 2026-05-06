"""
Systematic variance comparison across all four Greek estimation methods.

For each (option_type, greek) combination:
  1. Run Malliavin, FD, Pathwise (where applicable), LR methods
  2. Record: estimate, std_error, wall-clock time
  3. Where analytical result exists, record absolute bias

Outputs:
  - experiments/results/variance_comparison.png  — Var(FD)/Var(Mall) bar chart
  - experiments/results/efficiency_table.png     — (Var × Time) ratio heatmap
  - Console: full numeric table
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
    load_config, build_gbm, build_payoff, results_dir, save_or_show,
    timed, result_table, fmt,
)

from mgreeks.greeks import (
    MalliavinGreeks, FiniteDifferenceGreeks, PathwiseGreeks, LikelihoodRatioGreeks,
)
from mgreeks.greeks.analytical import (
    bs_delta, bs_gamma, bs_vega, bs_digital_delta, bs_digital_gamma,
    bs_barrier_delta,
)
from mgreeks.simulation import MonteCarloEngine
from mgreeks.weights.malliavin_weights import delta_weight_path_dependent


# ── Analytical truth table ───────────────────────────────────────────────────

def _analytical_delta(option_type: str, cfg: dict):
    S0 = cfg["option"]["S0"]
    K = cfg["option"]["K"]
    T = cfg["option"]["T"]
    r = cfg["model"]["r"]
    q = cfg["model"]["q"]
    sigma = cfg["model"]["sigma"]
    B = cfg["option"]["B_barrier"]

    if option_type == "european_call":
        return bs_delta(S0, K, T, r, q, sigma, "call")
    if option_type == "european_put":
        return bs_delta(S0, K, T, r, q, sigma, "put")
    if option_type == "digital_call":
        return bs_digital_delta(S0, K, T, r, q, sigma)
    if option_type == "down_and_out_call":
        return bs_barrier_delta(S0, K, B, T, r, q, sigma)
    return None   # Asian, lookback: no simple closed form for delta


def _analytical_gamma(option_type: str, cfg: dict):
    S0 = cfg["option"]["S0"]
    K = cfg["option"]["K"]
    T = cfg["option"]["T"]
    r = cfg["model"]["r"]
    q = cfg["model"]["q"]
    sigma = cfg["model"]["sigma"]
    if option_type in ("european_call", "european_put"):
        return bs_gamma(S0, K, T, r, q, sigma)
    if option_type == "digital_call":
        return bs_digital_gamma(S0, K, T, r, q, sigma)
    return None


def _analytical_vega(option_type: str, cfg: dict):
    from mgreeks.greeks.analytical import bs_vega
    S0 = cfg["option"]["S0"]
    K = cfg["option"]["K"]
    T = cfg["option"]["T"]
    r = cfg["model"]["r"]
    q = cfg["model"]["q"]
    sigma = cfg["model"]["sigma"]
    if option_type in ("european_call", "european_put"):
        return bs_vega(S0, K, T, r, q, sigma)
    return None


_TRUTH_FNS = {"delta": _analytical_delta, "gamma": _analytical_gamma,
              "vega": _analytical_vega}


# ── Per-method Greek computation ─────────────────────────────────────────────

def _run_greek(method_name: str, greek_name: str, payoff, estimators: dict,
               S0: float, T: float, model, sim_out: dict):
    """Run one (method, greek) combination. Returns result dict or None."""
    mall: MalliavinGreeks = estimators["mall"]
    fd: FiniteDifferenceGreeks = estimators["fd"]
    pw: PathwiseGreeks = estimators["pw"]
    lr: LikelihoodRatioGreeks = estimators["lr"]

    disc = float(np.exp(-model.r * T))
    dW = sim_out["brownian_increments"]

    try:
        if method_name == "malliavin":
            if greek_name == "delta":
                # Use path-dependent weight for path-dep payoffs for unbiasedness
                pd = delta_weight_path_dependent(
                    sim_out["paths"], dW, model, S0, T)
                f = payoff(sim_out["paths"], sim_out["times"])
                samples = f * pd
                vals = disc * samples
                mean = float(vals.mean())
                se = float(vals.std(ddof=1) / np.sqrt(len(vals)))
                return {"value": mean, "std_error": se}
            elif greek_name == "gamma":
                return mall.gamma(payoff, S0, T, sim_result=sim_out)
            elif greek_name == "vega":
                return mall.vega(payoff, S0, T, sim_result=sim_out)

        elif method_name == "fd":
            if greek_name == "delta":
                return fd.delta(payoff, S0, T)
            elif greek_name == "gamma":
                return fd.gamma(payoff, S0, T)
            elif greek_name == "vega":
                return fd.vega(payoff, S0, T)

        elif method_name == "pathwise":
            if greek_name == "delta":
                return pw.delta(payoff, S0, T, sim_result=sim_out)
            elif greek_name == "vega":
                return pw.vega(payoff, S0, T, sim_result=sim_out)
            else:
                return None   # pathwise gamma not supported for all payoffs

        elif method_name == "lr":
            if greek_name == "delta":
                return lr.delta(payoff, S0, T, sim_result=sim_out)
            elif greek_name == "gamma":
                return lr.gamma(payoff, S0, T, sim_result=sim_out)
            elif greek_name == "vega":
                return lr.vega(payoff, S0, T, sim_result=sim_out)

    except (NotImplementedError, Exception):
        return None

    return None


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    cfg = load_config()
    S0 = cfg["option"]["S0"]
    T = cfg["option"]["T"]
    n_paths = cfg["mc"]["n_paths"]
    n_steps = cfg["mc"]["n_steps"]
    seed = cfg["seed"]

    model = build_gbm(cfg)
    engine = MonteCarloEngine(model, n_paths=n_paths, n_steps=n_steps, rng_seed=seed)

    estimators = {
        "mall": MalliavinGreeks(model, engine),
        "fd":   FiniteDifferenceGreeks(model, engine,
                                       bump_size=cfg["mc"]["bump_size"],
                                       bump_type="relative"),
        "pw":   PathwiseGreeks(model, engine),
        "lr":   LikelihoodRatioGreeks(model, engine),
    }

    option_types = cfg["variance_comparison"]["option_types"]
    greeks = cfg["variance_comparison"]["greeks"]
    methods = ["malliavin", "fd", "pathwise", "lr"]

    rows = []
    # {(option_type, greek): {method: se}}
    se_matrix: dict = {}
    time_matrix: dict = {}

    for opt_type in option_types:
        payoff = build_payoff(opt_type, cfg)
        print(f"\n── {opt_type} ──")

        # One shared simulation for Malliavin/LR/pathwise
        sim_out = model.simulate(S0, T, n_steps, n_paths,
                                 return_full_paths=True,
                                 rng=np.random.default_rng(seed))

        for greek in greeks:
            truth_fn = _TRUTH_FNS.get(greek)
            truth = truth_fn(opt_type, cfg) if truth_fn else None
            se_matrix[(opt_type, greek)] = {}
            time_matrix[(opt_type, greek)] = {}

            for method in methods:
                t0 = __import__("time").perf_counter()
                res = _run_greek(method, greek, payoff, estimators, S0, T, model, sim_out)
                elapsed = __import__("time").perf_counter() - t0

                if res is None:
                    row = {"option": opt_type, "greek": greek, "method": method,
                           "estimate": "N/A", "std_error": "N/A",
                           "bias": "N/A", "time_s": f"{elapsed:.3f}"}
                    rows.append(row)
                    continue

                est = res["value"]
                se = res["std_error"]
                bias = f"{est - truth:.5f}" if truth is not None else "—"

                se_matrix[(opt_type, greek)][method] = se
                time_matrix[(opt_type, greek)][method] = elapsed

                row = {
                    "option": opt_type[:18],
                    "greek": greek,
                    "method": method,
                    "estimate": fmt(est, 5),
                    "std_error": fmt(se, 6),
                    "bias": bias,
                    "time_s": f"{elapsed:.3f}",
                }
                if truth is not None:
                    row["truth"] = fmt(truth, 5)
                rows.append(row)
                print(f"  {greek:6s}  {method:10s}  est={est:.5f}  se={se:.6f}  t={elapsed:.3f}s  bias={bias}")

    result_table(rows,
                 ["option", "greek", "method", "estimate", "std_error", "bias", "time_s"],
                 "Variance Comparison — GBM")

    # ── Plot: variance ratio Var(FD)/Var(Mall) ────────────────────────────────
    _plot_variance_ratios(se_matrix, time_matrix, option_types, greeks, cfg)


def _plot_variance_ratios(se_matrix, time_matrix, option_types, greeks, cfg):
    import matplotlib.pyplot as plt

    keys = [(o, g) for o in option_types for g in greeks]
    labels = [f"{o[:10]}\n{g}" for (o, g) in keys]

    var_ratio_fd  = []
    var_ratio_lr  = []
    eff_ratio     = []

    for key in keys:
        semap = se_matrix.get(key, {})
        tmap  = time_matrix.get(key, {})

        se_m = semap.get("malliavin")
        se_f = semap.get("fd")
        se_l = semap.get("lr")
        tm   = tmap.get("malliavin")
        tf   = tmap.get("fd")

        var_ratio_fd.append((se_f / se_m) ** 2 if se_m and se_f else float("nan"))
        var_ratio_lr.append((se_l / se_m) ** 2 if se_m and se_l else float("nan"))

        # Efficiency: (Var × Time) ratio = (se_f² × tf) / (se_m² × tm)
        if se_m and se_f and tm and tf and tm > 0:
            eff_ratio.append((se_f**2 * tf) / (se_m**2 * tm))
        else:
            eff_ratio.append(float("nan"))

    x = np.arange(len(keys))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=cfg["output"]["figsize"])
    fig.suptitle("Malliavin vs Finite-Difference Greeks — GBM", fontsize=13)

    ax = axes[0]
    bars_fd = ax.bar(x - width / 2, var_ratio_fd, width, label="Var(FD)/Var(Mall)",
                     color="#d62728", alpha=0.8)
    bars_lr = ax.bar(x + width / 2, var_ratio_lr, width, label="Var(LR)/Var(Mall)",
                     color="#1f77b4", alpha=0.8)
    ax.axhline(1.0, color="k", linestyle="--", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Variance ratio (> 1 → Malliavin wins)")
    ax.set_title("Variance Ratio")
    ax.legend(fontsize=8)
    ax.set_ylim(bottom=0)

    ax2 = axes[1]
    ax2.bar(x, eff_ratio, color="#2ca02c", alpha=0.8)
    ax2.axhline(1.0, color="k", linestyle="--", linewidth=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=7)
    ax2.set_ylabel("(Var × Time) FD / (Var × Time) Mall")
    ax2.set_title("Efficiency Ratio (Var × Time)")
    ax2.set_ylim(bottom=0)

    plt.tight_layout()
    save_or_show(fig, "variance_comparison", cfg)


if __name__ == "__main__":
    main()
