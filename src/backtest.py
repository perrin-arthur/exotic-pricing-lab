"""Quest 6.6: temporal backtest, parameter stability and J+1 predictive power.

Hypothesis, fixed before any data is read (Arthur's, see PROGRESS):
  H1. Heston, fitted per day, has day-to-day parameters more STABLE than SVI's
      (median over parameters of the coefficient of variation, Heston vs SVI).
  H2. Heston fitted on day J predicts the day J+1 smile better than SVI fitted
      on day J (RMSE in half-spread units on day J+1 quotes, parameters frozen).
  Both must hold to confirm the hypothesis. A flat-vol baseline (yesterday's ATM
  vol, carried flat) is reported alongside.

Frozen-parameter conventions:
  Heston: parameters frozen, priced at the day J+1 forward and maturity.
  SVI: the day-J implied vol curve sqrt(w_J(k)/T_J) is frozen in forward
  log-moneyness, and day J+1 quotes are priced at that vol with T_{J+1}.
  Expiries present on both days only. The baseline freezes the day-J ATM vol.

The series is built by scripts/collect_market.py run daily. Until MIN_DAYS
snapshot days exist, run_backtest refuses to run.
"""

import json
import pathlib

import numpy as np
import pandas as pd

from bs import call_bs
from calibration import (atm_vol, call_equivalent_quotes, calibrate_heston, calibrate_svi_slice,
                         heston_model_calls_fast, svi_total_variance)
from market_data import clean_option_chain

MIN_DAYS = 5
DATA = pathlib.Path(__file__).resolve().parent.parent / "data"
HESTON_STARTS = [[0.04, 1.5, 0.04, 0.5, -0.5], [0.01, 5.0, 0.01, 1.0, -0.8], [0.09, 0.8, 0.09, 0.3, -0.3]]
NAMES = ["GLD", "SLV", "USO"]
SVI_NAMES = ("a", "b", "rho", "m", "sigma")


def snapshot_days(data_dir: pathlib.Path = DATA) -> list[str]:
    return sorted(p.stem.removeprefix("chains_") for p in data_dir.glob("chains_*.csv"))


def coefficient_of_variation(values) -> float:
    values = np.asarray(values, dtype=float)
    return float(values.std(ddof=1) / abs(values.mean()))


def _quotes_for_day(data_dir, day, name):
    raw = pd.read_csv(data_dir / f"chains_{day}.csv", dtype={"expiry": str})
    rate = json.loads((data_dir / f"rate_{day}.json").read_text())
    clean = clean_option_chain(raw[raw.ticker == name], rate["spots"][name], r=rate["r"])
    return call_equivalent_quotes(clean, rate["r"]), rate["r"]


def fit_day(data_dir: pathlib.Path, day: str) -> dict:
    """Heston joint and per-expiry SVI for every name on one day."""
    out = {}
    for name in NAMES:
        q, r = _quotes_for_day(data_dir, day, name)
        h = calibrate_heston(q, r, HESTON_STARTS)
        svi = {}
        for expiry, sub in q.groupby("expiry"):
            s = calibrate_svi_slice(sub, r)
            svi[expiry] = dict(params=s["params"], T=float(sub["T"].iloc[0]), F=float(sub["F"].iloc[0]))
        out[name] = dict(r=r, heston=h["params"], svi=svi, atm_vol={
            e: atm_vol(q[q.expiry == e], r) for e in q["expiry"].unique()})
    return out


def j1_errors(data_dir: pathlib.Path, day_j: str, day_j1: str, fit_j: dict) -> dict:
    """RMSE (half-spread units) on day J+1 quotes, parameters frozen from day J."""
    out = {}
    for name in NAMES:
        q1, r = _quotes_for_day(data_dir, day_j1, name)
        f = fit_j[name]
        common = [e for e in q1["expiry"].unique() if e in f["svi"]]
        q1 = q1[q1["expiry"].isin(common)]
        if q1.empty:
            out[name] = None
            continue
        pred_h = heston_model_calls_fast(q1, f["heston"], r)
        pred_s = np.empty(len(q1))
        pred_b = np.empty(len(q1))
        for expiry, sub in q1.groupby("expiry", sort=False):
            idx = q1.index.get_indexer(sub.index)
            sj = f["svi"][expiry]
            k = np.log(sub["K"].to_numpy() / sub["F"].iloc[0])
            w_j = np.maximum(_svi_w(k, sj), 1e-12)
            # frozen day-J implied vol curve, evaluated at the day J+1 log-moneyness
            iv = np.sqrt(w_j / sj["T"])
            S_eff = sub["F"].iloc[0] * np.exp(-r * sub["T"].iloc[0])
            pred_s[idx] = call_bs(np.full(len(k), S_eff), sub["K"].to_numpy(), iv, r, sub["T"].iloc[0])
            pred_b[idx] = call_bs(np.full(len(k), S_eff), sub["K"].to_numpy(),
                                  f["atm_vol"][expiry], r, sub["T"].iloc[0])
        y = q1["C_mid"].to_numpy()
        half = q1["half"].to_numpy()
        rmse = lambda p: float(np.sqrt(np.mean(((p - y) / half) ** 2)))
        out[name] = dict(n=int(len(q1)), heston=rmse(pred_h), svi=rmse(pred_s), flat_atm=rmse(pred_b))
    return out


def _svi_w(k, sj):
    p = sj["params"]
    return svi_total_variance(k, p["a"], p["b"], p["rho"], p["m"], p["sigma"])


def run_backtest(data_dir: pathlib.Path = DATA, min_days: int = MIN_DAYS) -> dict:
    days = snapshot_days(data_dir)
    if len(days) < min_days:
        raise RuntimeError(f"6.6 needs >= {min_days} daily snapshots, have {len(days)}")
    fits = {d: fit_day(data_dir, d) for d in days}

    stability = {}
    for name in NAMES:
        heston_cv = {p: coefficient_of_variation([fits[d][name]["heston"][p] for d in days])
                     for p in fits[days[0]][name]["heston"]}
        svi_cv_by_param = {p: [] for p in SVI_NAMES}
        expiries = set.intersection(*(set(fits[d][name]["svi"]) for d in days))
        for e in expiries:
            for p in SVI_NAMES:
                svi_cv_by_param[p].append(coefficient_of_variation([fits[d][name]["svi"][e]["params"][p] for d in days]))
        svi_cv = {p: float(np.median(v)) for p, v in svi_cv_by_param.items()}
        stability[name] = dict(heston_cv=heston_cv, svi_cv_median=svi_cv,
                               heston_median_cv=float(np.median(list(heston_cv.values()))),
                               svi_median_cv=float(np.median(list(svi_cv.values()))))

    j1 = {}
    for dj, dj1 in zip(days[:-1], days[1:]):
        j1[f"{dj}->{dj1}"] = j1_errors(data_dir, dj, dj1, fits[dj])
    return dict(days=days, stability=stability, j1=j1)
