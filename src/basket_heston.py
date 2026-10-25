"""Quest 6.7 support: correlated multi-name Heston paths.

Each name follows its own calibrated Heston (6.2) with effective spot
S_eff = F e^{-rT}. The dependence across names is the historical correlation
R (6.4) on the SPOT Brownians. Each name keeps its own spot-variance
correlation rho_i. The joint Gaussian vector is ordered (W_S_1..n, W_v_1..n)
with covariance

    [[ R      , diag(rho) ],
     [ diag(rho), I        ]]

It must be positive semi-definite; the function checks this and refuses
otherwise, rather than silently repairing a model input.

Output has the repo's multi-asset shape, (N, n_assets, n_steps+1), so
multi_asset.brc_price and worst_of_performance apply unchanged.
"""

import numpy as np


def joint_covariance(R: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """2n x 2n covariance of (W_S, W_v) for spot correlation R and spot-var rho_i."""
    n = len(rho)
    D = np.diag(rho)
    return np.block([[R, D], [D, np.eye(n)]])


def correlated_heston_paths(S_eff: np.ndarray,
                            v0: np.ndarray,
                            kappa: np.ndarray,
                            theta: np.ndarray,
                            xi: np.ndarray,
                            rho: np.ndarray,
                            R: np.ndarray,
                            r: float,
                            T: float,
                            n_steps: int,
                            N: int = 100_000,
                            rng: np.random.Generator | None = None,
                            ) -> np.ndarray:
    """Joint spot paths of n Heston names, Euler with full truncation per name.

    Params
    ------
    S_eff : (n,) effective spots F_i e^{-rT}.
    v0, kappa, theta, xi, rho : (n,) per-name Heston parameters.
    R : (n, n) spot correlation between names, unit diagonal, PSD.
    Returns
    -------
    S : (N, n, n_steps+1) spot paths, S[:, i, 0] == S_eff[i].

    Raises
    ------
    ValueError : if the joint covariance is not positive semi-definite.
    """
    n = len(S_eff)
    cov = joint_covariance(R, rho)
    eig = np.linalg.eigvalsh(cov)
    if eig.min() < -1e-10:
        raise ValueError(f"joint covariance not PSD (min eigenvalue {eig.min():.2e}); "
                         "the cross-name correlation and the spot-variance correlations "
                         "are jointly incompatible")
    L = np.linalg.cholesky(cov + 1e-14 * np.eye(2 * n))

    if rng is None:
        rng = np.random.default_rng()
    dt = T / n_steps
    Z = rng.standard_normal((N, n_steps, 2 * n)) @ L.T
    WS, Wv = Z[:, :, :n], Z[:, :, n:]

    S = np.zeros((N, n, n_steps + 1))
    S[:, :, 0] = S_eff
    v_brut = np.broadcast_to(v0, (N, n)).astype(float).copy()
    v_prev = v_brut.copy()
    for t in range(n_steps):
        v_plus = np.maximum(v_brut, 0.0)
        v_brut = v_prev + kappa*(theta - v_plus)*dt + xi*np.sqrt(v_plus*dt)*Wv[:, t, :]
        incr = (r - 0.5*v_plus)*dt + np.sqrt(v_plus*dt)*WS[:, t, :]
        S[:, :, t+1] = S[:, :, t]*np.exp(incr)
        v_prev = np.maximum(v_brut, 0.0)
    return S
