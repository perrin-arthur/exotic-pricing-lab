"""Surface SVI (Gatheral & Jacquier 2014): one arbitrage-aware surface across expiries.

    w(k, theta) = theta/2 * (1 + rho*phi(theta)*k + sqrt((phi(theta)*k + rho)^2 + 1 - rho^2))
    phi(theta)  = eta * theta^(-gamma)

k = ln(K/F) is forward log-moneyness, theta the ATM total variance of the expiry.
Global parameters (rho, eta, gamma) are shared by all expiries; each expiry has its
own theta. Unlike independent slices, one surface can be checked for:

  * calendar arbitrage: w(k, T_i) <= w(k, T_j) for T_i < T_j, all k;
  * butterfly arbitrage: the Breeden-Litzenberger density g(k) >= 0, all k.

Both are enforced as hinge penalties on a fixed k-grid during the fit, and
verified again on a finer grid afterwards. A surface that fails the final check
is reported as such, not accepted.
"""

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from bs import call_bs

SSVI_GLOBAL = ("rho", "eta", "gamma")
K_GRID = np.linspace(-0.6, 0.6, 61)
K_CHECK = np.linspace(-0.6, 0.6, 1201)


def ssvi_total_variance(k: np.ndarray, theta: float, rho: float, eta: float, gamma: float) -> np.ndarray:
    phi = eta * theta ** (-gamma)
    return 0.5 * theta * (1 + rho * phi * k + np.sqrt((phi * k + rho) ** 2 + 1 - rho ** 2))


def butterfly_density(k: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Gatheral's g(k): non-negative iff the slice admits no butterfly arbitrage."""
    dk = k[1] - k[0]
    wp = np.gradient(w, dk)
    wpp = np.gradient(wp, dk)
    return (1 - k * wp / (2 * w)) ** 2 - wp ** 2 / 4 * (1 / w + 0.25) + wpp / 2


def _slices(quotes: pd.DataFrame):
    out = []
    for expiry, sub in quotes.groupby("expiry", sort=False):
        out.append((expiry, sub))
    out.sort(key=lambda x: x[1]["T"].iloc[0])
    return out


def _unpack(x: np.ndarray, n: int):
    return x[:3], x[3:3 + n]


def _model_calls(sub: pd.DataFrame, theta: float, glob, r: float) -> np.ndarray:
    rho, eta, gamma = glob
    F, T = sub["F"].iloc[0], sub["T"].iloc[0]
    k = np.log(sub["K"].to_numpy() / F)
    w = np.maximum(ssvi_total_variance(k, theta, rho, eta, gamma), 1e-12)
    iv = np.sqrt(w / T)
    S_eff = np.full(len(k), F * np.exp(-r * T))
    return call_bs(S_eff, sub["K"].to_numpy(), iv, r, T)


def _penalties(theta: np.ndarray, slices, glob, scale: float) -> np.ndarray:
    rho, eta, gamma = glob
    pen = []
    ws = []
    for (_, sub), th in zip(slices, theta):
        w = ssvi_total_variance(K_GRID, th, rho, eta, gamma)
        ws.append(w)
        g = butterfly_density(K_GRID, w)
        pen.append(scale * np.maximum(0.0, -g))
    for a, b in zip(ws[:-1], ws[1:]):
        pen.append(scale * np.maximum(0.0, a - b))
    return np.concatenate(pen)


def calibrate_ssvi(quotes: pd.DataFrame, r: float, penalty_weight: float = 50.0) -> dict:
    """Joint fit of one SSVI surface to all expiries of one name, half-spread residuals."""
    slices = _slices(quotes)
    n = len(slices)
    T0 = np.array([sub["T"].iloc[0] for _, sub in slices])
    theta0 = np.maximum.accumulate(0.04 * T0)   # neutral seed: 20% vol, increasing in T
    x0 = np.concatenate([[-0.3, 0.6, 0.4], theta0])
    lo = np.concatenate([[-0.999, 1e-3, 0.01], np.full(n, 1e-5)])
    hi = np.concatenate([[0.999, 5.0, 0.99], np.full(n, 5.0)])

    def fun(x):
        glob, theta = _unpack(x, n)
        res = []
        for (_, sub), th in zip(slices, theta):
            mod = _model_calls(sub, th, glob, r)
            res.append((mod - sub["C_mid"].to_numpy()) / sub["half"].to_numpy())
        res = np.concatenate(res)
        return np.concatenate([res, _penalties(theta, slices, glob, penalty_weight)])

    best = least_squares(fun, x0=np.clip(x0, lo + 1e-9, hi - 1e-9), bounds=(lo, hi),
                         x_scale="jac", max_nfev=400)
    glob, theta = _unpack(best.x, n)
    res_quotes = np.concatenate([(_model_calls(sub, th, glob, r) - sub["C_mid"].to_numpy())
                                 / sub["half"].to_numpy() for (_, sub), th in zip(slices, theta)])
    return dict(
        params=dict(zip(SSVI_GLOBAL, map(float, glob))),
        theta={e: float(th) for (e, _), th in zip(slices, theta)},
        T={e: float(sub["T"].iloc[0]) for e, sub in slices},
        residuals=res_quotes,
        frac_inside=float(np.mean(np.abs(res_quotes) <= 1.0)),
        median_abs_res=float(np.median(np.abs(res_quotes))),
    )


def check_no_arbitrage(params: dict, theta_by_T: list[tuple[float, float]]) -> dict:
    """Verify calendar and butterfly conditions on a fine grid.

    theta_by_T : list of (T, theta) for the calibrated expiries, any order.
    Returns the number of violating grid points for each condition, and the
    worst values.
    """
    rho, eta, gamma = params["rho"], params["eta"], params["gamma"]
    items = sorted(theta_by_T)
    W = np.array([ssvi_total_variance(K_CHECK, th, rho, eta, gamma) for _, th in items])
    cal = W[1:] - W[:-1]
    bf = np.array([butterfly_density(K_CHECK, w)[20:-20] for w in W])
    return dict(
        calendar_violations=int(np.sum(cal < -1e-12)),
        calendar_worst=float(cal.min()) if cal.size else 0.0,
        butterfly_violations=int(np.sum(bf < -1e-9)),
        butterfly_worst=float(bf.min()),
    )


def ssvi_term_calls(params: dict, theta_by_T: list[tuple[float, float]], quotes: pd.DataFrame, r: float) -> np.ndarray:
    """Price quotes on any expiry from the surface. theta interpolated linearly in T;
    beyond the calibrated range theta scales linearly in T from the nearest expiry (flat vol)."""
    items = sorted(theta_by_T)
    Ts = np.array([t for t, _ in items])
    ths = np.array([th for _, th in items])
    out = np.empty(len(quotes))
    for expiry, sub in quotes.groupby("expiry", sort=False):
        F, T = sub["F"].iloc[0], sub["T"].iloc[0]
        if T <= Ts[0]:
            theta = ths[0] * T / Ts[0]
        elif T >= Ts[-1]:
            theta = ths[-1] * T / Ts[-1]
        else:
            theta = float(np.interp(T, Ts, ths))
        glob = (params["rho"], params["eta"], params["gamma"])
        out[quotes.index.get_indexer(sub.index)] = _model_calls(sub, theta, glob, r)
    return out
