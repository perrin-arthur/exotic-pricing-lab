"""Heston and Bates on the term smile: multi-start, per name, best local minimum found.

Run: .venv/bin/python scripts/fit_heston_bates.py
Writes data/heston_bates_D.json.
"""

import json
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from calibration import (BATES_BOUNDS, BATES_NAMES, call_equivalent_quotes, multistart_fit)
from market_data import clean_option_chain

DATA = ROOT / "data"
HESTON_NAMES = BATES_NAMES[:5]
HESTON_BOUNDS = (BATES_BOUNDS[0][:5], BATES_BOUNDS[1][:5])


def latest(pattern):
    return sorted(DATA.glob(pattern))[-1]


def main():
    rate = json.loads(latest("rate_*.json").read_text())
    r = rate["r"]
    raw = pd.read_csv(latest("chains_*.csv"), dtype={"expiry": str})
    out = {}
    for t in ["GLD", "SLV", "USO"]:
        q = call_equivalent_quotes(clean_option_chain(raw[raw.ticker == t], rate["spots"][t], r=r), r)
        h = multistart_fit(q, r, HESTON_NAMES, *HESTON_BOUNDS, n_starts=12)
        b = multistart_fit(q, r, BATES_NAMES, *BATES_BOUNDS, n_starts=12)
        out[t] = dict(
            n_quotes=int(len(q)),
            heston=dict(params=h["params"], frac_inside=h["frac_inside"],
                        median_abs_res=h["median_abs_res"], best_sse=h["best_sse"],
                        sse_spread=h["sse_spread"]),
            bates=dict(params=b["params"], frac_inside=b["frac_inside"],
                       median_abs_res=b["median_abs_res"], best_sse=b["best_sse"],
                       sse_spread=b["sse_spread"]),
        )
        print(t, "heston", round(h["frac_inside"], 3), round(h["median_abs_res"], 2),
              "| bates", round(b["frac_inside"], 3), round(b["median_abs_res"], 2), flush=True)
    (DATA / "heston_bates_2026-10-25.json").write_text(json.dumps(out, indent=2))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
