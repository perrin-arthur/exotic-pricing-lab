"""BOSS 6: worst-of BRC on market-calibrated Heston marginals, historical correlation.

Run:       .venv/bin/python scripts/boss6_brc_calibrated.py
Validated: pytest tests/test_boss6.py

Inputs (committed in data/):
  chains_D.csv, rate_D.json            market snapshot (6.1)
  heston_calibration_D.json            per-name joint Heston (6.2)
  history.csv                          2y closes for the correlation (6.4)

Method. The BRC matures at the longest expiry common to all three names. Each
name evolves under its calibrated Heston with effective spot F_T e^{-rT} (carry
absorbed by the per-expiry forward). Dependence across names is the repaired
historical correlation. Benchmark: flat Black-Scholes per name at its own
at-the-money vol on that expiry, with the SAME correlation, so the only thing
that changes between the two prices is the marginal model.

Outputs:
  figures/boss6_brc_calibrated.png
  figures/boss6_results.json
"""

import json
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from basket_heston import correlated_heston_paths
from calibration import atm_vol, call_equivalent_quotes, forward_by_expiry
from correlation import historical_correlation
from market_data import clean_option_chain
from multi_asset import brc_price, correlated_gbm_paths, repair_correlation_clip, worst_of_performance

NAMES = ["GLD", "SLV", "USO"]
C_COUPON, H_BARRIER = 0.10, 0.70
N_PATHS, N_STEPS = 40_000, 52
LAMBDAS = [0.0, 0.25, 0.5, 0.75, 1.0]
DATA = ROOT / "data"
FIG = ROOT / "figures"


def latest(pattern):
    return sorted(DATA.glob(pattern))[-1]


def main():
    rate = json.loads(latest("rate_*.json").read_text())
    r = rate["r"]
    raw = pd.read_csv(latest("chains_*.csv"), dtype={"expiry": str})
    heston = json.loads(latest("heston_calibration_*.json").read_text())
    prices = pd.read_csv(DATA / "history.csv", index_col=0, parse_dates=True)[NAMES]

    clean = {t: clean_option_chain(raw[raw.ticker == t], rate["spots"][t], r=r) for t in NAMES}
    fwd = {t: forward_by_expiry(clean[t], r) for t in NAMES}
    common = sorted(set.intersection(*(set(fwd[t].index) for t in NAMES)))
    T_expiry = common[-1]
    T = float(fwd[NAMES[0]].loc[T_expiry, "T"])

    S_eff = np.array([fwd[t].loc[T_expiry, "F"] * np.exp(-r * T) for t in NAMES])
    p = [heston[t]["params"] for t in NAMES]
    v0, kap, th, xi, rho = (np.array([d[k] for d in p]) for k in ["v0", "kappa", "theta", "xi", "rho"])

    flat_vol = np.array([atm_vol(call_equivalent_quotes(clean[t], r).query("expiry == @T_expiry"), r)
                         for t in NAMES])

    raw_R, R = historical_correlation(prices)
    I = np.eye(len(NAMES))

    def run(Rm, seed):
        S_h = correlated_heston_paths(S_eff, v0, kap, th, xi, rho, Rm, r, T,
                                      N_STEPS, N_PATHS, np.random.default_rng(seed))
        S_b = correlated_gbm_paths(S_eff, flat_vol, Rm, r, T, N_STEPS, N_PATHS,
                                   np.random.default_rng(seed + 1))
        out = {}
        for name, paths in (("heston", S_h), ("bs_flat", S_b)):
            price, hw = brc_price(paths, C_COUPON, H_BARRIER, r, T)
            perf = worst_of_performance(paths)[:, -1]
            out[name] = dict(price=float(price), half_width=float(hw),
                             prob_knock_in=float(np.mean(worst_of_performance(paths).min(axis=1) < H_BARRIER)),
                             perf_T_median=float(np.median(perf)))
            if name == "heston" and Rm is R:
                out["_perf_heston"] = perf
        return out

    base = run(R, 100)
    perf_heston = base.pop("_perf_heston")
    sweep = []
    for lam in LAMBDAS:
        Rl = repair_correlation_clip((1 - lam) * I + lam * R)
        res = run(Rl, 200 + int(100 * lam))
        sweep.append(dict(lam=lam, heston=res["heston"], bs_flat=res["bs_flat"]))

    results = dict(
        params=dict(C=C_COUPON, H=H_BARRIER, r=r, T=T, T_expiry=T_expiry, n_paths=N_PATHS,
                    n_steps=N_STEPS, names=NAMES),
        marginals={t: dict(S_eff=float(S_eff[i]), flat_vol=float(flat_vol[i]),
                           heston=p[i]) for i, t in enumerate(NAMES)},
        correlation=dict(raw=raw_R.tolist(), repaired=R.tolist()),
        base=base, sweep=sweep,
    )
    FIG.mkdir(exist_ok=True)
    (FIG / "boss6_results.json").write_text(json.dumps(results, indent=2))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4))
    lam = [s["lam"] for s in sweep]
    for key, col, lab in (("heston", "#1f77b4", "Heston calibré"), ("bs_flat", "#7f7f7f", "BS plat")):
        y = [s[key]["price"] for s in sweep]
        e = [s[key]["half_width"] for s in sweep]
        a1.errorbar(lam, y, yerr=e, fmt="o-", color=col, capsize=4, lw=2, label=lab)
    a1.set_xlabel("intensité de corrélation λ (0 = indépendant, 1 = historique)")
    a1.set_ylabel(f"prix du BRC (coupon {C_COUPON:.0%}, barrière {H_BARRIER:.0%})")
    a1.set_title("Prix worst-of : marginales calibrées vs BS plat")
    a1.legend()
    a2.hist(perf_heston, bins=60, color="#1f77b4", alpha=0.8)
    a2.axvline(H_BARRIER, color="#d62728", ls="--", label="barrière")
    a2.set_xlabel("performance du pire actif à maturité")
    a2.set_ylabel("chemins")
    a2.set_title("Distribution de la performance (Heston, corrélation historique)")
    a2.legend()
    plt.tight_layout()
    fig.savefig(FIG / "boss6_brc_calibrated.png", dpi=150)
    print(json.dumps(dict(T_expiry=T_expiry, T=T, base=base), indent=2))


if __name__ == "__main__":
    main()
