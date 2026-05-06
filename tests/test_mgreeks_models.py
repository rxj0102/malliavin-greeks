"""
Tests for mgreeks model implementations.

Checks:
  GBM     : exact simulation shapes, martingale, log-variance, Malliavin derivative
  Heston  : shapes, martingale (approx), paths finite, QE vs Euler
  LocalVol: shapes, martingale (approx)
  MultiAsset: shapes, correlation structure, Malliavin derivative
"""

import numpy as np
import pytest

from mgreeks.models.gbm import GeometricBrownianMotion
from mgreeks.models.heston import HestonModel
from mgreeks.models.local_vol import LocalVolModel
from mgreeks.models.multidimensional import MultiAssetGBM


# ---------------------------------------------------------------------------
# GBM
# ---------------------------------------------------------------------------

class TestGBM:
    S0, T, n_steps, n_paths = 100.0, 1.0, 52, 50_000
    model = GeometricBrownianMotion(r=0.05, q=0.02, sigma=0.20)

    def sim(self, seed=42):
        return self.model.simulate(
            self.S0, self.T, self.n_steps, self.n_paths,
            return_full_paths=True, rng=np.random.default_rng(seed)
        )

    def test_shapes(self):
        out = self.sim()
        assert out["paths"].shape == (self.n_paths, self.n_steps + 1)
        assert out["brownian_increments"].shape == (self.n_paths, self.n_steps)
        assert out["terminal"].shape == (self.n_paths,)
        assert len(out["times"]) == self.n_steps + 1

    def test_initial_condition(self):
        out = self.sim()
        np.testing.assert_allclose(out["paths"][:, 0], self.S0)

    def test_martingale(self):
        """E[S_T] = S0 * exp((r-q)*T) under risk-neutral measure."""
        expected = self.S0 * np.exp((self.model.r - self.model.q) * self.T)
        out = self.sim()
        mean_ST = out["terminal"].mean()
        # 4-sigma tolerance
        std_err = out["terminal"].std() / np.sqrt(self.n_paths)
        assert abs(mean_ST - expected) < 4 * std_err, (
            f"Martingale failed: {mean_ST:.4f} vs {expected:.4f}"
        )

    def test_log_variance(self):
        """Var[log S_T] = σ² T."""
        expected_var = self.model.sigma**2 * self.T
        out = self.sim()
        log_ST = np.log(out["terminal"])
        observed_var = log_ST.var()
        np.testing.assert_allclose(observed_var, expected_var, rtol=0.03)

    def test_malliavin_derivative(self):
        """D_s S_t = σ S_t for any s ≤ t."""
        out = self.sim()
        paths = out["paths"]
        dW = out["brownian_increments"]
        D = self.model.malliavin_derivative(paths, dW, s_index=0, t_index=self.n_steps)
        expected = self.model.sigma * paths[:, self.n_steps]
        np.testing.assert_allclose(D, expected)

    def test_malliavin_derivative_order(self):
        """s_index > t_index raises ValueError."""
        out = self.sim()
        with pytest.raises(ValueError):
            self.model.malliavin_derivative(out["paths"], out["brownian_increments"],
                                            s_index=5, t_index=3)

    def test_first_variation(self):
        """J_t = S_t / S_0 starting from t=0."""
        out = self.sim()
        J = self.model.first_variation(out["paths"], s_index=0)
        expected = out["paths"] / self.S0
        np.testing.assert_allclose(J, expected, rtol=1e-10)

    def test_log_density_gradient_delta(self):
        """Score w.r.t. S0: W_T / (σ T S0)."""
        out = self.sim()
        ST = out["terminal"]
        score = self.model.log_density_gradient(self.S0, ST, self.T, "S0")
        W_T = out["brownian_increments"].sum(axis=1)
        expected = W_T / (self.model.sigma * self.T * self.S0)
        np.testing.assert_allclose(score, expected, rtol=1e-10)

    def test_log_density_gradient_vega(self):
        """Score w.r.t. sigma: (W_T²-T)/(σT) - W_T."""
        out = self.sim()
        ST = out["terminal"]
        score = self.model.log_density_gradient(self.S0, ST, self.T, "sigma")
        W_T = out["brownian_increments"].sum(axis=1)
        expected = (W_T**2 - self.T) / (self.model.sigma * self.T) - W_T
        np.testing.assert_allclose(score, expected, rtol=1e-10)

    def test_log_density_gradient_rho(self):
        """Score w.r.t. r: W_T / σ."""
        out = self.sim()
        ST = out["terminal"]
        score = self.model.log_density_gradient(self.S0, ST, self.T, "r")
        W_T = out["brownian_increments"].sum(axis=1)
        expected = W_T / self.model.sigma
        np.testing.assert_allclose(score, expected, rtol=1e-10)

    def test_reproducibility(self):
        """Same seed → same paths."""
        out1 = self.sim(seed=0)
        out2 = self.sim(seed=0)
        np.testing.assert_array_equal(out1["paths"], out2["paths"])

    def test_different_seeds(self):
        """Different seeds → different paths."""
        out1 = self.sim(seed=1)
        out2 = self.sim(seed=2)
        assert not np.array_equal(out1["paths"], out2["paths"])

    def test_W_T_from_increments(self):
        """W_T = sum of dW increments."""
        out = self.sim()
        W_T = self.model.W_T_from_increments(out["brownian_increments"])
        expected = out["brownian_increments"].sum(axis=1)
        np.testing.assert_array_equal(W_T, expected)

    def test_terminal_no_full_path(self):
        """return_full_paths=False returns None for paths."""
        out = self.model.simulate(
            self.S0, self.T, self.n_steps, 1000,
            return_full_paths=False, rng=np.random.default_rng(0)
        )
        assert out["paths"] is None
        assert out["terminal"].shape == (1000,)


# ---------------------------------------------------------------------------
# Heston
# ---------------------------------------------------------------------------

class TestHeston:
    S0, T, n_steps, n_paths = 100.0, 1.0, 100, 20_000
    model = HestonModel(
        r=0.05, q=0.0, kappa=2.0, theta=0.04,
        xi=0.3, rho=-0.7, V0=0.04
    )

    def sim(self, scheme="qe", seed=42):
        return self.model.simulate(
            self.S0, self.T, self.n_steps, self.n_paths,
            return_full_paths=True, scheme=scheme,
            rng=np.random.default_rng(seed)
        )

    def test_shapes_qe(self):
        out = self.sim("qe")
        assert out["paths"].shape == (self.n_paths, self.n_steps + 1)
        assert out["v_paths"].shape == (self.n_paths, self.n_steps + 1)

    def test_shapes_euler(self):
        out = self.sim("euler")
        assert out["paths"].shape == (self.n_paths, self.n_steps + 1)

    def test_initial_condition(self):
        out = self.sim()
        np.testing.assert_allclose(out["paths"][:, 0], self.S0)
        np.testing.assert_allclose(out["v_paths"][:, 0], self.model.V0)

    def test_paths_finite(self):
        out = self.sim()
        assert np.all(np.isfinite(out["paths"]))
        assert np.all(np.isfinite(out["v_paths"]))

    def test_spot_positive(self):
        """S_t > 0 for all paths and times."""
        out = self.sim()
        assert np.all(out["paths"] > 0)

    def test_variance_nonneg_qe(self):
        """QE scheme guarantees V_t ≥ 0."""
        out = self.sim("qe")
        assert np.all(out["v_paths"] >= 0), "QE should keep variance non-negative"

    def test_martingale_approx(self):
        """E[S_T] ≈ S0 * exp(r*T) within 2%."""
        out = self.sim()
        expected = self.S0 * np.exp(self.model.r * self.T)
        mean_ST = out["terminal"].mean()
        np.testing.assert_allclose(mean_ST, expected, rtol=0.02)

    def test_feller(self):
        assert self.model.feller_satisfied()

    def test_invalid_scheme(self):
        with pytest.raises(ValueError):
            self.model.simulate(self.S0, self.T, 10, 100,
                                scheme="invalid", rng=np.random.default_rng(0))


# ---------------------------------------------------------------------------
# LocalVol
# ---------------------------------------------------------------------------

class TestLocalVol:
    S0, T, n_steps, n_paths = 100.0, 1.0, 100, 20_000
    model_cev = LocalVolModel(r=0.05, q=0.0, model_type="cev",
                              sigma0=0.20, beta=0.5, S_ref=100.0)
    model_quad = LocalVolModel(r=0.05, q=0.0, model_type="quadratic",
                               a=0.20, b=0.001, c=0.0, S_ref=100.0)

    def sim(self, model, seed=42):
        return model.simulate(
            self.S0, self.T, self.n_steps, self.n_paths,
            return_full_paths=True, rng=np.random.default_rng(seed)
        )

    def test_cev_shapes(self):
        out = self.sim(self.model_cev)
        assert out["paths"].shape == (self.n_paths, self.n_steps + 1)

    def test_quad_shapes(self):
        out = self.sim(self.model_quad)
        assert out["paths"].shape == (self.n_paths, self.n_steps + 1)

    def test_paths_positive(self):
        out = self.sim(self.model_cev)
        assert np.all(out["paths"] > 0)

    def test_paths_finite(self):
        out = self.sim(self.model_quad)
        assert np.all(np.isfinite(out["paths"]))

    def test_martingale_approx(self):
        """E[S_T] ≈ S0*exp(r*T) within 3% for CEV."""
        out = self.sim(self.model_cev)
        expected = self.S0 * np.exp(self.model_cev.r * self.T)
        np.testing.assert_allclose(out["terminal"].mean(), expected, rtol=0.03)


# ---------------------------------------------------------------------------
# MultiAssetGBM
# ---------------------------------------------------------------------------

class TestMultiAssetGBM:
    S0 = np.array([100.0, 80.0, 120.0])
    corr = np.array([[1.0, 0.5, -0.3],
                     [0.5, 1.0,  0.2],
                     [-0.3, 0.2, 1.0]])
    sigma = np.array([0.20, 0.25, 0.15])
    T, n_steps, n_paths = 1.0, 52, 30_000
    model = MultiAssetGBM(r=0.05, q=0.0, sigma=sigma, correlation=corr, n_assets=3)

    def sim(self, seed=42):
        return self.model.simulate(
            self.S0, self.T, self.n_steps, self.n_paths,
            return_full_paths=True, rng=np.random.default_rng(seed)
        )

    def test_shapes(self):
        out = self.sim()
        # paths: (n_paths, n_assets, n_steps+1)
        assert out["paths"].shape == (self.n_paths, 3, self.n_steps + 1)

    def test_initial_condition(self):
        out = self.sim()
        for i, s0 in enumerate(self.S0):
            np.testing.assert_allclose(out["paths"][:, i, 0], s0)

    def test_paths_positive(self):
        out = self.sim()
        assert np.all(out["paths"] > 0)

    def test_marginal_martingale(self):
        """Each asset satisfies E[S^i_T] = S0^i * exp(r*T) within 2%."""
        out = self.sim()
        for i, s0 in enumerate(self.S0):
            expected = s0 * np.exp(self.model.r * self.T)
            mean_ST = out["paths"][:, i, -1].mean()
            np.testing.assert_allclose(mean_ST, expected, rtol=0.02)

    def test_correlation_structure(self):
        """
        Check that cross-asset log-return correlations match the input ρ_ij.
        With 30k paths and T=1 this converges to within ~rtol=0.05.
        """
        out = self.sim()
        log_ST = np.log(out["paths"][:, :, -1] / self.S0)   # (n_paths, 3)
        empir_corr = np.corrcoef(log_ST.T)
        np.testing.assert_allclose(empir_corr, self.corr, atol=0.06)

    def test_malliavin_derivative_ij(self):
        """D^j_s S^i_t = σ_i * L_{ij} * S^i_t."""
        out = self.sim()
        L = self.model.chol
        for asset in range(3):
            for brownian in range(3):
                D = self.model.malliavin_derivative_ij(
                    out["paths"], s_index=0, t_index=self.n_steps,
                    asset=asset, brownian=brownian
                )
                expected = self.sigma[asset] * L[asset, brownian] * out["paths"][:, asset, self.n_steps]
                np.testing.assert_allclose(D, expected, rtol=1e-10)
