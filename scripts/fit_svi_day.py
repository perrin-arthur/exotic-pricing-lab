"""Per-expiry SVI calibration for one snapshot date, per name.

Run: .venv/bin/python scripts/fit_svi_day.py YYYY-MM-DD
Writes data/svi_calibration_YYYY-MM-DD.json. A name that cannot be calibrated
(no expiry with at least three call/put pairs after cleaning) is recorded with
its status and reason, not skipped silently.
"""

import json
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from calibration import call_equivalent_quotes, calibrate_svi_slice
from market_data import clean_option_chain

DATA = ROOT / "data"
NAMES = ["GLD", "SLV", "USO"]


def main(day: str):
    raw = pd.read_csv(DATA / f"chains_{day}.csv", dtype={"expiry": str})
    rate = json.loads((DATA / f"rate_{day}.json").read_text())
    r = rate["r"]
    out = {}
    for t in NAMES:
        clean = clean_option_chain(raw[raw.ticker == t], rate["spots"][t], r=r)
        q = call_equivalent_quotes(clean, r) if len(clean) else pd.DataFrame()
        if q.empty:
            out[t] = dict(status="not calibrable", reason="no expiry with >= 3 call/put pairs after cleaning",
                          n_clean=int(len(clean)))
            print(t, out[t]["status"], flush=True)
            continue
        slices, tot, n = {}, 0.0, 0
        for e, sl in q.groupby("expiry"):
            s = calibrate_svi_slice(sl, r)
            tot += s["frac_inside"] * len(sl)
            n += len(sl)
            slices[e] = dict(n=len(sl), frac_inside=s["frac_inside"],
                             median_abs_res=float(np.median(np.abs(s["residuals"]))),
                             min_total_variance=s["min_total_variance"], params=s["params"],
                             T=float(sl["T"].iloc[0]))
        out[t] = dict(status="calibrated", frac_inside=tot / n, n_quotes=n, n_clean=int(len(clean)),
                      slices=slices)
        print(t, "frac_inside", round(tot / n, 3), flush=True)
    (DATA / f"svi_calibration_{day}.json").write_text(json.dumps(dict(snapshot=day, names=out), indent=2))
    print("DONE", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
