"""
Monte Carlo simulation engine.

MonteCarloEngine is the primary user-facing entry point.  It wraps a
StochasticModel and dispatches to Greek estimators based on the requested
method.

Supported Greeks and methods
----------------------------
Greek       : 'delta', 'gamma', 'vega', 'rho', 'theta'
method      : 'malliavin'  — score-function / LR weight
              'pathwise'   — IPA (smooth payoffs only)
              'finite_diff'— central finite differences
              'lr'         — alias for 'malliavin'
"""

from __future__ import annotations

from typing import Optional, Union
import numpy as np

from mgreeks.models.base import StochasticModel
from mgreeks.payoffs.european import _Payoff


# ---------------------------------------------------------------------------
# Confidence-interval helper
# ---------------------------------------------------------------------------

def _ci(samples: np.ndarray, disc: float, alpha: float = 0.95) -> dict:
    """Return price, std_error, ci_lower, ci_upper from discounted samples."""
    n = len(samples)
    vals = disc * samples
    mean = float(vals.mean())
    stderr = float(vals.std(ddof=1) / np.sqrt(n))
    z = 1.959964  # 95% two-sided
    return {
        "price": mean,
        "std_error": stderr,
        "ci_lower": mean - z * stderr,
        "ci_upper": mean + z * stderr,
        "n_paths": n,
    }


# ---------------------------------------------------------------------------
# MonteCarloEngine
# ---------------------------------------------------------------------------

class MonteCarloEngine:
    """
    Prices options and computes Greeks via Monte Carlo.

    Parameters
    ----------
    model    : StochasticModel — the underlying asset dynamics
    n_paths  : number of Monte Carlo paths
    n_steps  : time-steps per path (more needed for path-dependent payoffs)
    rng_seed : seed for the random number generator (None → non-reproducible)
    """

    def __init__(
        self,
        model: StochasticModel,
        n_paths: int = 100_000,
        n_steps: int = 252,
        rng_seed: Optional[int] = None,
    ):
        self.model = model
        self.n_paths = int(n_paths)
        self.n_steps = int(n_steps)
        self.rng_seed = rng_seed

    def _rng(self) -> np.random.Generator:
        return np.random.default_rng(self.rng_seed)

    # ------------------------------------------------------------------
    # Price
    # ------------------------------------------------------------------

    def price(
        self,
        payoff: _Payoff,
        S0: float,
        T: float,
        **model_kwargs,
    ) -> dict:
        """
        Estimate the option price  V = e^{-rT} E[f(S)].

        Parameters
        ----------
        payoff : callable payoff object
        S0     : initial spot price
        T      : time to maturity (years)
        **model_kwargs : passed through to model.simulate()

        Returns
        -------
        dict with keys: price, std_error, ci_lower, ci_upper, n_paths
        """
        r = getattr(self.model, "r", 0.0)
        disc = np.exp(-r * T)

        sim = self.model.simulate(
            S0, T, self.n_steps, self.n_paths,
            return_full_paths=True, rng=self._rng(), **model_kwargs
        )
        paths = sim["paths"]
        times = sim["times"]

        payoffs = payoff(paths, times)
        return _ci(payoffs, disc)

    # ------------------------------------------------------------------
    # Single Greek
    # ------------------------------------------------------------------

    def greek(
        self,
        payoff: _Payoff,
        S0: float,
        T: float,
        greek_name: str,
        method: str = "malliavin",
        bump: float = 0.01,
        **model_kwargs,
    ) -> dict:
        """
        Estimate a single Greek.

        Parameters
        ----------
        greek_name : 'delta', 'gamma', 'vega', 'rho', 'theta'
        method     : 'malliavin' / 'lr', 'pathwise', 'finite_diff'
        bump       : relative bump size for finite differences

        Returns
        -------
        dict with keys: greek, std_error, ci_lower, ci_upper, n_paths, method
        """
        method = method.lower()
        if method == "lr":
            method = "malliavin"

        if method == "malliavin":
            result = self._malliavin_greek(payoff, S0, T, greek_name, **model_kwargs)
        elif method == "pathwise":
            result = self._pathwise_greek(payoff, S0, T, greek_name, **model_kwargs)
        elif method == "finite_diff":
            result = self._fd_greek(payoff, S0, T, greek_name, bump, **model_kwargs)
        else:
            raise ValueError(f"Unknown method {method!r}. Use 'malliavin', 'pathwise', or 'finite_diff'.")

        result["method"] = method
        return result

    # ------------------------------------------------------------------
    # All Greeks
    # ------------------------------------------------------------------

    def all_greeks(
        self,
        payoff: _Payoff,
        S0: float,
        T: float,
        method: str = "malliavin",
        bump: float = 0.01,
    ) -> dict:
        """
        Estimate delta, gamma, vega, rho, theta in a single call.

        Returns dict mapping Greek name → result dict.
        """
        names = ["delta", "gamma", "vega", "rho", "theta"]
        return {
            name: self.greek(payoff, S0, T, name, method=method, bump=bump)
            for name in names
        }

    # ------------------------------------------------------------------
    # Malliavin / score-function estimators
    # ------------------------------------------------------------------

    def _malliavin_greek(
        self,
        payoff: _Payoff,
        S0: float,
        T: float,
        greek_name: str,
        **model_kwargs,
    ) -> dict:
        from mgreeks.models.gbm import GeometricBrownianMotion

        r = getattr(self.model, "r", 0.0)
        disc = np.exp(-r * T)

        sim = self.model.simulate(
            S0, T, self.n_steps, self.n_paths,
            return_full_paths=True, rng=self._rng(), **model_kwargs
        )
        paths = sim["paths"]
        times = sim["times"]
        dW = sim["brownian_increments"]

        f = payoff(paths, times)

        if isinstance(self.model, GeometricBrownianMotion):
            weight = self._gbm_weight(greek_name, S0, T, paths, dW)
        else:
            raise NotImplementedError(
                f"Malliavin weights not yet implemented for {type(self.model).__name__}. "
                "Use method='finite_diff'."
            )

        samples = f * weight
        result = _ci(samples, disc)
        result["greek"] = result.pop("price")
        result["std_error"] = result.pop("std_error")
        return result

    def _gbm_weight(
        self,
        greek_name: str,
        S0: float,
        T: float,
        paths: np.ndarray,
        dW: np.ndarray,
    ) -> np.ndarray:
        """Malliavin / score-function weights for GBM."""
        sigma = self.model.sigma
        W_T = dW.sum(axis=1)          # terminal Brownian motion

        if greek_name == "delta":
            return W_T / (sigma * S0 * T)

        elif greek_name == "gamma":
            return (W_T * (W_T - sigma * T) - T) / (sigma**2 * T**2 * S0**2)

        elif greek_name == "vega":
            return (W_T**2 - T) / (sigma * T) - W_T

        elif greek_name == "rho":
            # score_r = W_T/σ (no T factor); discount adds −T
            return W_T / sigma - T

        elif greek_name == "theta":
            # Uses score w.r.t. T; discount factor correction applied separately
            r = self.model.r
            q = self.model.q
            mu = r - q - 0.5 * sigma**2
            score_T = mu + sigma * W_T / T
            return score_T

        else:
            raise ValueError(f"Unknown greek_name {greek_name!r}")

    # ------------------------------------------------------------------
    # Pathwise (IPA) estimators
    # ------------------------------------------------------------------

    def _pathwise_greek(
        self,
        payoff: _Payoff,
        S0: float,
        T: float,
        greek_name: str,
        **model_kwargs,
    ) -> dict:
        if greek_name not in ("delta",):
            raise ValueError(
                f"Pathwise method supports 'delta' only; got {greek_name!r}. "
                "Use method='malliavin' or 'finite_diff' for other Greeks."
            )

        r = getattr(self.model, "r", 0.0)
        disc = np.exp(-r * T)

        sim = self.model.simulate(
            S0, T, self.n_steps, self.n_paths,
            return_full_paths=True, rng=self._rng(), **model_kwargs
        )
        paths = sim["paths"]
        times = sim["times"]

        # IPA delta: E[f'(S_T) · S_T / S_0]
        deriv = payoff.payoff_deriv(paths, times)         # ∂f/∂S_T
        ST = paths[:, -1]
        samples = deriv * ST / S0

        result = _ci(samples, disc)
        result["greek"] = result.pop("price")
        return result

    # ------------------------------------------------------------------
    # Finite-difference estimators
    # ------------------------------------------------------------------

    def _fd_greek(
        self,
        payoff: _Payoff,
        S0: float,
        T: float,
        greek_name: str,
        bump: float,
        **model_kwargs,
    ) -> dict:
        r = getattr(self.model, "r", 0.0)
        disc = np.exp(-r * T)
        rng = self._rng()

        def _price_at(S0_: float, **extra) -> np.ndarray:
            sim = self.model.simulate(
                S0_, T, self.n_steps, self.n_paths,
                return_full_paths=True, rng=rng, **model_kwargs
            )
            return payoff(sim["paths"], sim["times"])

        if greek_name == "delta":
            h = bump * S0
            rng = self._rng()
            sim_base = self.model.simulate(S0, T, self.n_steps, self.n_paths,
                                           return_full_paths=True, rng=rng, **model_kwargs)
            f_base = payoff(sim_base["paths"], sim_base["times"])
            rng_up = np.random.default_rng(self.rng_seed)
            sim_up = self.model.simulate(S0 + h, T, self.n_steps, self.n_paths,
                                         return_full_paths=True, rng=rng_up, **model_kwargs)
            f_up = payoff(sim_up["paths"], sim_up["times"])
            rng_dn = np.random.default_rng(self.rng_seed)
            sim_dn = self.model.simulate(S0 - h, T, self.n_steps, self.n_paths,
                                         return_full_paths=True, rng=rng_dn, **model_kwargs)
            f_dn = payoff(sim_dn["paths"], sim_dn["times"])
            samples = (f_up - f_dn) / (2 * h)

        elif greek_name == "gamma":
            h = bump * S0
            rng_c = np.random.default_rng(self.rng_seed)
            sim_c = self.model.simulate(S0, T, self.n_steps, self.n_paths,
                                        return_full_paths=True, rng=rng_c, **model_kwargs)
            f_c = payoff(sim_c["paths"], sim_c["times"])
            rng_u = np.random.default_rng(self.rng_seed)
            sim_u = self.model.simulate(S0 + h, T, self.n_steps, self.n_paths,
                                        return_full_paths=True, rng=rng_u, **model_kwargs)
            f_u = payoff(sim_u["paths"], sim_u["times"])
            rng_d = np.random.default_rng(self.rng_seed)
            sim_d = self.model.simulate(S0 - h, T, self.n_steps, self.n_paths,
                                        return_full_paths=True, rng=rng_d, **model_kwargs)
            f_d = payoff(sim_d["paths"], sim_d["times"])
            samples = (f_u - 2 * f_c + f_d) / (h**2)

        elif greek_name == "vega":
            sigma0 = self.model.sigma
            h = bump * sigma0
            self.model.sigma = sigma0 + h
            rng_u = np.random.default_rng(self.rng_seed)
            sim_u = self.model.simulate(S0, T, self.n_steps, self.n_paths,
                                        return_full_paths=True, rng=rng_u, **model_kwargs)
            f_u = payoff(sim_u["paths"], sim_u["times"])
            self.model.sigma = sigma0 - h
            rng_d = np.random.default_rng(self.rng_seed)
            sim_d = self.model.simulate(S0, T, self.n_steps, self.n_paths,
                                        return_full_paths=True, rng=rng_d, **model_kwargs)
            f_d = payoff(sim_d["paths"], sim_d["times"])
            self.model.sigma = sigma0     # restore
            samples = (f_u - f_d) / (2 * h)

        elif greek_name == "rho":
            r0 = self.model.r
            h = max(bump * r0, 1e-4)
            self.model.r = r0 + h
            rng_u = np.random.default_rng(self.rng_seed)
            sim_u = self.model.simulate(S0, T, self.n_steps, self.n_paths,
                                        return_full_paths=True, rng=rng_u, **model_kwargs)
            f_u = payoff(sim_u["paths"], sim_u["times"])
            price_u = np.exp(-(r0 + h) * T) * f_u.mean()
            self.model.r = r0 - h
            rng_d = np.random.default_rng(self.rng_seed)
            sim_d = self.model.simulate(S0, T, self.n_steps, self.n_paths,
                                        return_full_paths=True, rng=rng_d, **model_kwargs)
            f_d = payoff(sim_d["paths"], sim_d["times"])
            price_d = np.exp(-(r0 - h) * T) * f_d.mean()
            self.model.r = r0     # restore
            rho = (price_u - price_d) / (2 * h)
            # Return scalar result wrapped consistently
            return {
                "greek": float(rho),
                "std_error": float("nan"),
                "ci_lower": float("nan"),
                "ci_upper": float("nan"),
                "n_paths": self.n_paths,
            }

        elif greek_name == "theta":
            h = max(bump * T, 1e-4)
            rng_u = np.random.default_rng(self.rng_seed)
            sim_u = self.model.simulate(S0, T + h, self.n_steps, self.n_paths,
                                        return_full_paths=True, rng=rng_u, **model_kwargs)
            f_u = payoff(sim_u["paths"], sim_u["times"])
            price_u = np.exp(-self.model.r * (T + h)) * f_u.mean()
            rng_d = np.random.default_rng(self.rng_seed)
            sim_d = self.model.simulate(S0, T - h, self.n_steps, self.n_paths,
                                        return_full_paths=True, rng=rng_d, **model_kwargs)
            f_d = payoff(sim_d["paths"], sim_d["times"])
            price_d = np.exp(-self.model.r * (T - h)) * f_d.mean()
            theta = -(price_u - price_d) / (2 * h)   # sign: Theta = -dV/dT
            return {
                "greek": float(theta),
                "std_error": float("nan"),
                "ci_lower": float("nan"),
                "ci_upper": float("nan"),
                "n_paths": self.n_paths,
            }

        else:
            raise ValueError(f"Unknown greek_name {greek_name!r}")

        result = _ci(samples, disc)
        result["greek"] = result.pop("price")
        return result
