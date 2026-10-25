"""Fit one arbitrage-checked SSVI surface per name, and its holdout on the longest expiry.

Run: .venv/bin/python scripts/fit_ssvi.py
Writes data/ssvi_surface_D.json. Reads the same committed snapshot as the other
Act VI scripts.
"""

import json
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from calibration import call_equivalent_quotes
from market_data import clean_option_chain
from ssvi import calibrate_ssvi, check_no_arbitrage, ssvi_term_calls

DATA = ROOT / "data"


def latest(pattern):
    return sorted(DATA.glob(pattern))[-1]


def main():
    rate = json.loads(latest("rate_*.json").read_text())
    r = rate["r"]
    raw = pd.read_csv(latest("chains_*.csv"), dtype={"expiry": str})
    out = {}
    for t in ["GLD", "SLV", "USO"]:
        q = call_equivalent_quotes(clean_option_chain(raw[raw.ticker == t], rate["spots"][t], r=r), r)
        fit = calibrate_ssvi(q, r)
        theta_T = [(fit["T"][e], th) for e, th in fit["theta"].items()]
        arb = check_no_arbitrage(fit["params"], theta_T)

        last = max(q["expiry"].unique(), key=lambda e: q.loc[q.expiry == e, "T"].iloc[0])
        train, test = q[q.expiry != last], q[q.expiry == last]
        fit_h = calibrate_ssvi(train, r)
        th_h = [(fit_h["T"][e], th) for e, th in fit_h["theta"].items()]
        arb_h = check_no_arbitrage(fit_h["params"], th_h)
        pred = ssvi_term_calls(fit_h["params"], th_h, test, r)
        res_h = (pred - test["C_mid"].to_numpy()) / test["half"].to_numpy()

        out[t] = dict(
            full=dict(params=fit["params"], theta=fit["theta"], T=fit["T"],
                      frac_inside=fit["frac_inside"], median_abs_res=fit["median_abs_res"],
                      arbitrage=arb),
            holdout=dict(expiry=last, params=fit_h["params"], arbitrage=arb_h,
                         frac_inside=float((abs(res_h) <= 1).mean()),
                         median_abs_res=float(np.median(np.abs(res_h)))),
        )
        print(t, "full frac", round(fit["frac_inside"], 3), "med", round(fit["median_abs_res"], 2),
              "arb", arb, "| holdout frac", round(out[t]["holdout"]["frac_inside"], 3),
              "med", round(out[t]["holdout"]["median_abs_res"], 2), flush=True)
    (DATA / "ssvi_surface_2026-10-25.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
