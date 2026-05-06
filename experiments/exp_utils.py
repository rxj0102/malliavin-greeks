"""
Shared utilities for all experiment scripts.

Provides:
  load_config()        — read config.yaml and merge CLI overrides
  build_gbm()          — GeometricBrownianMotion from config
  build_heston()       — HestonModel from config
  build_local_vol()    — LocalVolModel (CEV) from config
  build_payoff()       — payoff object by name
  result_table()       — pretty-print a comparison table
  save_or_show()       — save figure to results/ or display interactively
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml


# ── Config loading ────────────────────────────────────────────────────────────

_CONFIG_PATH = Path(__file__).parent / "config.yaml"


def load_config(extra_args: list[str] = None) -> dict:
    """Load config.yaml and allow CLI overrides for numeric keys."""
    with open(_CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--n_paths", type=int)
    parser.add_argument("--n_steps", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--show_plots", action="store_true", default=None)
    parser.add_argument("--no_plots", action="store_true")
    args, _ = parser.parse_known_args(extra_args)

    if args.n_paths is not None:
        cfg["mc"]["n_paths"] = args.n_paths
    if args.n_steps is not None:
        cfg["mc"]["n_steps"] = args.n_steps
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.show_plots:
        cfg["output"]["show_plots"] = True
    if args.no_plots:
        cfg["output"]["save_plots"] = False
        cfg["output"]["show_plots"] = False

    return cfg


# ── Model builders ────────────────────────────────────────────────────────────

def build_gbm(cfg: dict):
    from mgreeks.models.gbm import GeometricBrownianMotion
    m = cfg["model"]
    return GeometricBrownianMotion(r=m["r"], q=m["q"], sigma=m["sigma"])


def build_heston(cfg: dict):
    from mgreeks.models.heston import HestonModel
    h = cfg["heston"]
    m = cfg["model"]
    return HestonModel(
        r=m["r"], q=m["q"],
        kappa=h["kappa"], theta=h["theta"],
        xi=h["xi"], rho=h["rho"], V0=h["V0"],
    )


def build_local_vol(cfg: dict):
    from mgreeks.models.local_vol import LocalVolModel
    lv = cfg["local_vol_cev"]
    m = cfg["model"]
    return LocalVolModel(
        r=m["r"], q=m["q"],
        model_type="cev",
        sigma0=lv["sigma0"],
        beta=lv["beta"],
        S_ref=lv["S_ref"],
    )


# ── Payoff builders ───────────────────────────────────────────────────────────

def build_payoff(name: str, cfg: dict):
    K = cfg["option"]["K"]
    B = cfg["option"]["B_barrier"]

    if name == "european_call":
        from mgreeks.payoffs.european import EuropeanCall
        return EuropeanCall(K)
    if name == "european_put":
        from mgreeks.payoffs.european import EuropeanPut
        return EuropeanPut(K)
    if name == "digital_call":
        from mgreeks.payoffs.european import DigitalCall
        return DigitalCall(K)
    if name == "arithmetic_asian_call":
        from mgreeks.payoffs.asian import ArithmeticAsianCall
        return ArithmeticAsianCall(K)
    if name == "down_and_out_call":
        from mgreeks.payoffs.barrier import DownAndOutCall
        return DownAndOutCall(K, B)
    if name == "lookback_call":
        from mgreeks.payoffs.lookback import FloatingStrikeLookbackCall
        return FloatingStrikeLookbackCall()
    raise ValueError(f"Unknown payoff name: {name!r}")


# ── Output helpers ────────────────────────────────────────────────────────────

def results_dir(cfg: dict) -> Path:
    d = Path(cfg["output"]["dir"])
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_or_show(fig, name: str, cfg: dict):
    import matplotlib.pyplot as plt
    out = cfg["output"]
    if out.get("save_plots", True):
        p = results_dir(cfg) / f"{name}.png"
        fig.savefig(p, dpi=out.get("dpi", 120), bbox_inches="tight")
        print(f"  saved → {p}")
    if out.get("show_plots", False):
        plt.show()
    plt.close(fig)


# ── Timing wrapper ────────────────────────────────────────────────────────────

def timed(fn, *args, **kwargs):
    """Run fn(*args, **kwargs) and return (result, elapsed_seconds)."""
    t0 = time.perf_counter()
    res = fn(*args, **kwargs)
    return res, time.perf_counter() - t0


# ── Pretty table printer ──────────────────────────────────────────────────────

def result_table(rows: list[dict], cols: list[str], title: str = "") -> None:
    if title:
        print(f"\n{'='*70}")
        print(f"  {title}")
        print(f"{'='*70}")
    widths = {c: max(len(c), max(len(str(r.get(c, ""))) for r in rows))
              for c in cols}
    header = "  ".join(f"{c:<{widths[c]}}" for c in cols)
    print(header)
    print("-" * len(header))
    for row in rows:
        print("  ".join(f"{str(row.get(c,'')):<{widths[c]}}" for c in cols))
    print()


def fmt(x: Any, decimals: int = 5) -> str:
    if isinstance(x, float):
        return f"{x:.{decimals}f}"
    return str(x)
