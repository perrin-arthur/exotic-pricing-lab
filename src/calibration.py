"""Act VI calibration: quotes in call-equivalent form, Heston and SVI fits.

Everything is measured in the same unit, the HALF bid-ask spread of each quote.
A residual of 1 means the model price sits exactly on the edge of the observed
market. Comparing models in that unit makes Heston, SVI and flat BS directly
comparable without choosing a vol-space or price-space convention.

Forward handling. Commodity ETFs carry a cost of carry (lease rate, roll) that
is not a dividend yield. Rather than estimating q, each expiry uses the forward
read from put-call parity on the cleaned quotes, F_T = median(K + e^{rT}(C-P)).
A Heston model with spot S_eff = F_T * e^{-rT} has forward F_T exactly, so the
carry is absorbed per expiry and no q enters the pricing.

Quotes used for fitting are call-equivalent and out-of-the-money: calls for
K >= F, and puts converted to calls by parity (C = P + (F-K)e^{-rT}) for K < F.
"""

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from bs import call_bs
from heston import heston_call, heston_cf


def forward_by_expiry(clean: pd.DataFrame, r: float, min_pairs: int = 3) -> pd.DataFrame:
    """Forward per expiry from put-call parity, with the pair count it rests on.

    Returns a frame indexed by expiry with columns F, T, n_pairs. Expiries with
    fewer than min_pairs call/put pairs are dropped: a median of two points is
    not a forward.
    """
    calls = clean[clean["type"] == "call"].set_index(["strike", "expiry"])
    puts = clean[clean["type"] == "put"].set_index(["strike", "expiry"])
    common = calls.index.intersection(puts.index)

    rows = []
    for expiry in sorted({e for _, e in common}):
        idx = [(k, e) for k, e in common if e == expiry]
        K = np.array([k for k, _ in idx])
        T = calls.loc[idx[0], "T"]
        C = calls.loc[idx, "mid"].to_numpy()
        P = puts.loc[idx, "mid"].to_numpy()
        if len(idx) < min_pairs:
            continue
        F = np.median(K + np.exp(r * T) * (C - P))
        rows.append(dict(expiry=expiry, F=float(F), T=float(T), n_pairs=len(idx)))
    return pd.DataFrame(rows).set_index("expiry")


def call_equivalent_quotes(clean: pd.DataFrame, r: float) -> pd.DataFrame:
    """Out-of-the-money quotes, all expressed as call prices, with their half spread.

    Columns: expiry, T, F, K, C_mid, half. C_mid is a call price; half is the
    half bid-ask of the quote actually used (the put's, for converted strikes).
    """
    fwd = forward_by_expiry(clean, r)
    out = []
    for expiry, row in fwd.iterrows():
        F, T = row["F"], row["T"]
        sub = clean[clean["expiry"] == expiry]
        for K in sorted(sub["strike"].unique()):
            if K >= F:
                q = sub[(sub["strike"] == K) & (sub["type"] == "call")]
                if len(q) == 0:
                    continue
                q = q.iloc[0]
                C_mid, half = q["mid"], (q["ask"] - q["bid"]) / 2
            else:
                q = sub[(sub["strike"] == K) & (sub["type"] == "put")]
                if len(q) == 0:
                    continue
                q = q.iloc[0]
                C_mid = q["mid"] + (F - K) * np.exp(-r * T)
                half = (q["ask"] - q["bid"]) / 2
            out.append(dict(expiry=expiry, T=T, F=F, K=float(K), C_mid=float(C_mid), half=float(half)))
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Heston
# ---------------------------------------------------------------------------

HESTON_NAMES = ("v0", "kappa", "theta", "xi", "rho")
HESTON_BOUNDS = (
    [1e-4, 0.05, 1e-4, 0.01, -0.99],   # lower
    [1.0, 20.0, 1.0, 3.0, 0.99],       # upper
)


def heston_model_calls(quotes: pd.DataFrame, params: dict, r: float) -> np.ndarray:
    """Heston call price for each quote, under the per-expiry forward (see module doc)."""
    out = np.empty(len(quotes))
    for i, (F, T, K) in enumerate(zip(quotes["F"], quotes["T"], quotes["K"])):
        S_eff = F * np.exp(-r * T)
        out[i] = heston_call(S_eff, K, params["v0"], r, T,
                             params["kappa"], params["theta"], params["xi"], params["rho"])
    return out


def heston_residuals(x: np.ndarray, quotes: pd.DataFrame, r: float) -> np.ndarray:
    params = dict(zip(HESTON_NAMES, x))
    return (heston_model_calls_fast(quotes, params, r) - quotes["C_mid"].to_numpy()) / quotes["half"].to_numpy()


def calibrate_heston(quotes: pd.DataFrame, r: float, starts: list[list[float]]) -> dict:
    """Least-squares fit of the five Heston parameters to all expiries of one name.

    Objective: sum of squared residuals in half-spread units, joint over expiries.
    Several starting points are tried and the best sum of squares is kept:
    Heston is non-convex and one start is not evidence of a global fit.

    Returns
    -------
    dict with params (name -> float), residuals (array, half-spread units),
    sse, max_abs_res, frac_inside (share of quotes with |residual| <= 1).
    """
    best = None
    for x0 in starts:
        res = least_squares(heston_residuals, x0=np.asarray(x0, dtype=float),
                            args=(quotes, r), bounds=HESTON_BOUNDS,
                            x_scale="jac", max_nfev=200)
        if best is None or res.cost < best.cost:
            best = res
    resid = best.fun
    return dict(
        params=dict(zip(HESTON_NAMES, map(float, best.x))),
        residuals=resid,
        sse=float(np.sum(resid**2)),
        max_abs_res=float(np.max(np.abs(resid))),
        frac_inside=float(np.mean(np.abs(resid) <= 1.0)),
    )


# ---------------------------------------------------------------------------
# Fast Heston pricing for calibration
# ---------------------------------------------------------------------------

from numpy.polynomial.legendre import leggauss

_PANEL, _M, _U_MAX = 10.0, 12, 1000.0   # composite Gauss-Legendre on [0, 1000]
_X, _W = leggauss(_M)
_EDGES = np.arange(0.0, _U_MAX + 1e-12, _PANEL)
_A, _B = _EDGES[:-1, None], _EDGES[1:, None]
_U = (0.5 * (_B - _A) * _X[None, :] + 0.5 * (_A + _B)).ravel()
_WU = (0.5 * (_B - _A) * _W[None, :]).ravel()


def heston_calls_grid(S_eff: float, Ks: np.ndarray, T: float, params: dict, r: float) -> np.ndarray:
    """Heston call prices for several strikes of ONE expiry, on a shared quadrature.

    Same Gil-Pelaez decomposition as heston.heston_call (P1, P2). The
    characteristic function is evaluated once per expiry on 1 200 composite
    Gauss-Legendre nodes, and the strikes share it. Validated against
    heston_call to 2.3e-8 in tests/test_calibration.py.
    """
    u = _U
    cf_a = heston_cf(u, S_eff, params["v0"], r, T, params["kappa"], params["theta"],
                     params["xi"], params["rho"])
    cf_b = heston_cf(u - 1j, S_eff, params["v0"], r, T, params["kappa"], params["theta"],
                     params["xi"], params["rho"])
    Ks = np.asarray(Ks, dtype=float)
    phase = np.exp(-1j * u[None, :] * np.log(Ks)[:, None])
    P2 = 0.5 + np.sum(_WU * np.real(phase * cf_a[None, :] / (1j * u[None, :])), axis=1) / np.pi
    P1 = 0.5 + np.sum(_WU * np.real(phase * cf_b[None, :] / (1j * u[None, :] * S_eff * np.exp(r * T))),
                      axis=1) / np.pi
    return S_eff * P1 - Ks * np.exp(-r * T) * P2


def heston_model_calls_fast(quotes: pd.DataFrame, params: dict, r: float) -> np.ndarray:
    """heston_model_calls, grouped by expiry through heston_calls_grid."""
    out = np.empty(len(quotes))
    for expiry, sub in quotes.groupby("expiry", sort=False):
        F, T = sub["F"].iloc[0], sub["T"].iloc[0]
        S_eff = F * np.exp(-r * T)
        out[quotes.index.get_indexer(sub.index)] = heston_calls_grid(
            S_eff, sub["K"].to_numpy(), T, params, r)
    return out


# ---------------------------------------------------------------------------
# SVI, one expiry at a time (quest 6.3): the descriptive second baseline
# ---------------------------------------------------------------------------

SVI_NAMES = ("a", "b", "rho", "m", "sigma")
SVI_BOUNDS = (
    np.array([-0.5, 1e-6, -0.999, -1.0, 1e-3]),   # lower
    np.array([1.0, 5.0, 0.999, 1.0, 2.0]),        # upper
)


def svi_total_variance(k: np.ndarray, a: float, b: float, rho: float, m: float, sigma: float) -> np.ndarray:
    """Raw SVI total variance w(k) = a + b*(rho*(k-m) + sqrt((k-m)^2 + sigma^2)).

    k is log-moneyness ln(K/F) against the expiry's forward.
    """
    d = k - m
    return a + b * (rho * d + np.sqrt(d * d + sigma * sigma))


def svi_model_calls(quotes: pd.DataFrame, params: dict, r: float) -> np.ndarray:
    """Black-Scholes call prices with the SVI implied vol of each quote.

    Quotes of one expiry only: the forward and T are read from the first row.
    The model is priced on the same effective spot as Heston, S_eff = F e^{-rT}.
    """
    F, T = quotes["F"].iloc[0], quotes["T"].iloc[0]
    S_eff = F * np.exp(-r * T)
    k = np.log(quotes["K"].to_numpy() / F)
    w = np.maximum(svi_total_variance(k, *[params[n] for n in SVI_NAMES]), 1e-12)
    iv = np.sqrt(w / T)
    return np.array([call_bs(S_eff, K, v, r, T) for K, v in zip(quotes["K"].to_numpy(), iv)])


def svi_residuals(x: np.ndarray, quotes: pd.DataFrame, r: float) -> np.ndarray:
    params = dict(zip(SVI_NAMES, x))
    return (svi_model_calls(quotes, params, r) - quotes["C_mid"].to_numpy()) / quotes["half"].to_numpy()


def calibrate_svi_slice(quotes: pd.DataFrame, r: float) -> dict:
    """Fit raw SVI to one expiry, on the same half-spread residuals as Heston.

    Returns the same keys as calibrate_heston, plus min_total_variance: the
    smallest w(k) on a grid, a check that the slice does not go negative.
    """
    T = quotes["T"].iloc[0]
    atm_w = 0.04 * T   # neutral seed: 20% vol, independent of the data
    starts = [
        [atm_w, 0.1, -0.3, 0.0, 0.1],
        [atm_w, 0.3, 0.3, 0.1, 0.3],
        [atm_w, 0.05, -0.6, -0.05, 0.05],
    ]
    best = None
    for x0 in starts:
        x0 = np.clip(np.asarray(x0, dtype=float), SVI_BOUNDS[0] + 1e-9, SVI_BOUNDS[1] - 1e-9)
        res = least_squares(svi_residuals, x0=x0, args=(quotes, r), bounds=SVI_BOUNDS,
                            x_scale="jac", max_nfev=300)
        if best is None or res.cost < best.cost:
            best = res
    grid = np.linspace(-1.0, 1.0, 401)
    w_grid = svi_total_variance(grid, *best.x)
    resid = best.fun
    return dict(
        params=dict(zip(SVI_NAMES, map(float, best.x))),
        residuals=resid,
        sse=float(np.sum(resid**2)),
        max_abs_res=float(np.max(np.abs(resid))),
        frac_inside=float(np.mean(np.abs(resid) <= 1.0)),
        min_total_variance=float(w_grid.min()),
    )


# ---------------------------------------------------------------------------
# Out-of-sample comparison (quest 6.5)
# ---------------------------------------------------------------------------

def flat_bs_calls(quotes: pd.DataFrame, sigma: float, r: float) -> np.ndarray:
    """Black-Scholes call prices at one flat vol, on the per-expiry effective spot."""
    return np.array([call_bs(F * np.exp(-r * T), K, sigma, r, T)
                     for F, T, K in zip(quotes["F"], quotes["T"], quotes["K"])])


def atm_vol(quotes: pd.DataFrame, r: float) -> float:
    """Implied vol of the quote closest to the forward, by bisection on call_bs.

    The baseline a desk would use: one number, read at the money on the last
    training expiry, then applied flat.
    """
    row = quotes.iloc[np.argmin(np.abs(np.log(quotes["K"].to_numpy() / quotes["F"].to_numpy())))]
    F, T, K, target = row["F"], row["T"], row["K"], row["C_mid"]
    S_eff = F * np.exp(-r * T)
    lo, hi = 1e-4, 5.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if call_bs(S_eff, K, mid, r, T) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def svi_term_calls(slices: dict, quotes: pd.DataFrame, r: float) -> np.ndarray:
    """SVI prices on expiries that were NOT calibrated, by linear interpolation in T.

    slices : {expiry: (T, params)} for the calibrated expiries.
    Total variance w(k, T) is interpolated linearly in T at fixed forward
    log-moneyness k. Beyond the last calibrated expiry, w scales linearly in T
    from that slice (flat vol). Linear-in-T total variance keeps calendar
    arbitrage out of the interpolation.
    """
    items = sorted(slices.values(), key=lambda x: x[0])
    Ts = np.array([t for t, _ in items])
    out = np.empty(len(quotes))
    for expiry, sub in quotes.groupby("expiry", sort=False):
        F, T = sub["F"].iloc[0], sub["T"].iloc[0]
        S_eff = F * np.exp(-r * T)
        k = np.log(sub["K"].to_numpy() / F)
        if T <= Ts[0]:
            w = svi_total_variance(k, *[items[0][1][n] for n in SVI_NAMES]) * (T / Ts[0])
        elif T >= Ts[-1]:
            w = svi_total_variance(k, *[items[-1][1][n] for n in SVI_NAMES]) * (T / Ts[-1])
        else:
            j = np.searchsorted(Ts, T) - 1
            (Ta, pa), (Tb, pb) = items[j], items[j + 1]
            wa = svi_total_variance(k, *[pa[n] for n in SVI_NAMES])
            wb = svi_total_variance(k, *[pb[n] for n in SVI_NAMES])
            w = wa + (T - Ta) / (Tb - Ta) * (wb - wa)
        iv = np.sqrt(np.maximum(w, 1e-12) / T)
        out[quotes.index.get_indexer(sub.index)] = [
            call_bs(S_eff, K, v, r, T) for K, v in zip(sub["K"].to_numpy(), iv)]
    return out


def holdout_summary(model_calls: np.ndarray, quotes: pd.DataFrame) -> dict:
    """Share inside the band and median |residual| (half-spreads) on held-out quotes."""
    res = (model_calls - quotes["C_mid"].to_numpy()) / quotes["half"].to_numpy()
    return dict(frac_inside=float(np.mean(np.abs(res) <= 1.0)),
                median_abs_res=float(np.median(np.abs(res))),
                n=int(len(res)))


# ---------------------------------------------------------------------------
# Bates = Heston + Merton log-normal jumps (short-end wings)
# ---------------------------------------------------------------------------

BATES_NAMES = ("v0", "kappa", "theta", "xi", "rho", "lam", "mu_j", "delta")
BATES_BOUNDS = (
    np.array([1e-4, 0.05, 1e-4, 0.01, -0.99, 0.0, -0.5, 0.01]),
    np.array([1.0, 20.0, 1.0, 3.0, 0.99, 5.0, 0.5, 1.0]),
)


def _cf_fn(S_eff: float, T: float, params: dict, r: float):
    """Characteristic function of ln S_T (Heston, optionally plus Merton jumps)."""
    from heston import heston_cf

    def phi(u):
        base = heston_cf(u, S_eff, params["v0"], r, T, params["kappa"], params["theta"],
                         params["xi"], params["rho"])
        if "lam" not in params:
            return base
        lam, mu, d = params["lam"], params["mu_j"], params["delta"]
        kbar = np.exp(mu + 0.5 * d * d) - 1.0
        jump = lam * T * (np.exp(1j * u * mu - 0.5 * d * d * u * u) - 1.0 - 1j * u * kbar)
        return base * np.exp(jump)
    return phi


def calls_grid_cf(S_eff: float, Ks: np.ndarray, T: float, params: dict, r: float) -> np.ndarray:
    """Call prices for several strikes of one expiry, for any model in _cf_fn."""
    phi = _cf_fn(S_eff, T, params, r)
    u = _U
    cf_a = phi(u)
    cf_b = phi(u - 1j)
    Ks = np.asarray(Ks, dtype=float)
    phase = np.exp(-1j * u[None, :] * np.log(Ks)[:, None])
    P2 = 0.5 + np.sum(_WU * np.real(phase * cf_a[None, :] / (1j * u[None, :])), axis=1) / np.pi
    P1 = 0.5 + np.sum(_WU * np.real(phase * cf_b[None, :] / (1j * u[None, :] * S_eff * np.exp(r * T))),
                      axis=1) / np.pi
    return S_eff * P1 - Ks * np.exp(-r * T) * P2


def model_calls_cf(quotes: pd.DataFrame, params: dict, r: float) -> np.ndarray:
    out = np.empty(len(quotes))
    for expiry, sub in quotes.groupby("expiry", sort=False):
        F, T = sub["F"].iloc[0], sub["T"].iloc[0]
        out[quotes.index.get_indexer(sub.index)] = calls_grid_cf(
            F * np.exp(-r * T), sub["K"].to_numpy(), T, params, r)
    return out


def _residuals_named(x, names, quotes, r):
    params = dict(zip(names, x))
    return (model_calls_cf(quotes, params, r) - quotes["C_mid"].to_numpy()) / quotes["half"].to_numpy()


def multistart_fit(quotes: pd.DataFrame, r: float, names, lo, hi, n_starts: int = 12, seed: int = 0) -> dict:
    """Least-squares fit from n_starts Sobol-spread starting points. Keeps the best
    local minimum found. This is a best-found result, not a proof of the global one."""
    from scipy.stats import qmc
    sampler = qmc.Sobol(d=len(names), scramble=True, seed=seed)
    U = sampler.random(n_starts)
    starts = lo + U * (hi - lo)
    best, sse_all = None, []
    for x0 in starts:
        res = least_squares(_residuals_named, x0=np.clip(x0, lo + 1e-9, hi - 1e-9),
                            args=(names, quotes, r), bounds=(lo, hi), x_scale="jac", max_nfev=300)
        sse_all.append(float(2 * res.cost))
        if best is None or res.cost < best.cost:
            best = res
    resid = best.fun
    return dict(
        params=dict(zip(names, map(float, best.x))),
        residuals=resid,
        frac_inside=float(np.mean(np.abs(resid) <= 1.0)),
        median_abs_res=float(np.median(np.abs(resid))),
        best_sse=float(2 * best.cost),
        sse_spread=[float(s) for s in sorted(sse_all)],
    )
