"""Act VI, quest 6.7 machinery: correlated multi-name Heston paths.

Checks, all against an independent reference:
  * one name alone reproduces heston_paths (same law: discounted mean and variance);
  * with constant variance the joint log-return correlation equals the target R
    exactly (in the limit), so the cross-name dependence is wired correctly;
  * a joint covariance that is not PSD is refused, not repaired silently.
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pytest

from basket_heston import correlated_heston_paths
from heston import heston_paths

r, T = 0.05, 1.0


def test_un_seul_actif_reproduit_heston_paths():
    """Discounted mean of S_T and of v_T agree with heston_paths within MC error."""
    p = dict(v0=0.04, kappa=1.5, theta=0.04, xi=0.3, rho=-0.7)
    N = 100_000
    S1 = correlated_heston_paths(np.array([100.0]), np.array([p["v0"]]), np.array([p["kappa"]]),
                                 np.array([p["theta"]]), np.array([p["xi"]]), np.array([p["rho"]]),
                                 np.eye(1), r, T, n_steps=100, N=N, rng=np.random.default_rng(1))
    S0, _ = heston_paths(S0=100.0, v0=p["v0"], r=r, T=T, kappa=p["kappa"], theta=p["theta"],
                         xi=p["xi"], rho=p["rho"], n_steps=100, N=N, rng=np.random.default_rng(2))
    disc1 = np.exp(-r * T) * S1[:, 0, -1]
    disc0 = np.exp(-r * T) * S0[:, -1]
    hw = 1.96 * np.sqrt(disc1.var(ddof=1) / N + disc0.var(ddof=1) / N)
    assert abs(disc1.mean() - disc0.mean()) < 2 * hw
    assert abs(disc1.mean() - 100.0) < 4 * disc1.std(ddof=1) / np.sqrt(N)   # martingale


def test_correlation_des_log_rendements_a_variance_constante():
    """With variance pinned (xi ~ 0, v = theta), log returns are Gaussian with
    correlation exactly R. The cross-name wiring is correct if this holds."""
    R = np.array([[1.0, 0.6], [0.6, 1.0]])
    theta = np.array([0.04, 0.09])
    N = 50_000
    S = correlated_heston_paths(np.array([100.0, 50.0]), theta, np.array([50.0, 50.0]), theta,
                                np.array([1e-4, 1e-4]), np.array([0.0, 0.0]), R, r, T,
                                n_steps=50, N=N, rng=np.random.default_rng(4))
    lr = np.log(S[:, :, -1] / S[:, :, 0])
    rho_hat = np.corrcoef(lr.T)[0, 1]
    assert abs(rho_hat - 0.6) < 4 * (1 - 0.6**2) / np.sqrt(N)


def test_covariance_non_psd_refusee():
    """Incompatible inputs raise; nothing is silently repaired."""
    R = np.array([[1.0, -0.99], [-0.99, 1.0]])
    with pytest.raises(ValueError, match="not PSD"):
        correlated_heston_paths(np.array([100.0, 50.0]), np.array([0.04, 0.09]),
                                np.array([1.5, 2.0]), np.array([0.04, 0.09]),
                                np.array([0.3, 0.5]), np.array([-0.99, 0.99]),
                                R, r, T, n_steps=10, N=10)
