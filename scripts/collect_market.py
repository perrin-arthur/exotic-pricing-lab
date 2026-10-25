"""Snapshot the market inputs of Act VI: option chains, history, short rate.

Run:  .venv/bin/python scripts/collect_market.py
Writes, for today's date D:
  data/chains_D.csv        raw option chains, GLD / SLV / USO, term structure
                           20 to 400 days out, at most 12 expiries per name
  data/history.csv         2y daily adjusted close, one column per ticker
  data/rate_D.json         short rate (^IRX, 13-week T-bill, continuous), spot per ticker

Run once per day to build the time series quest 6.6 needs. Yahoo gives only the
CURRENT chain, so the history of chains is only as long as the snapshots kept.
"""

import json
import pathlib
import sys
from datetime import date

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
from market_data import fetch_option_chain, fetch_underlying_history

TICKERS = ["GLD", "SLV", "USO"]
MAX_EXPIRIES = 12
MIN_DAYS = 20      # skip the weeklies: a BRC lives about a year, its smile is the term one
MAX_DAYS = 400
ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def short_rate() -> float:
    """^IRX quotes the 13-week T-bill as a percent discount rate; convert to a
    continuously compounded rate for a 0.25-year bill."""
    last = float(yf.Ticker("^IRX").history(period="5d")["Close"].dropna().iloc[-1])
    return float(np.log(1 + last / 100 * 0.25) / 0.25)


def main() -> None:
    today = date.today().isoformat()
    DATA.mkdir(exist_ok=True)

    hist = pd.concat({t: fetch_underlying_history(t, period="2y") for t in TICKERS}, axis=1)
    hist.to_csv(DATA / "history.csv")

    spots = {t: float(hist[t].dropna().iloc[-1]) for t in TICKERS}
    r = short_rate()

    chains = []
    for t in TICKERS:
        c = fetch_option_chain(t, spots[t], max_expiries=MAX_EXPIRIES,
                               min_days=MIN_DAYS, max_days=MAX_DAYS)
        c.insert(0, "ticker", t)
        chains.append(c)
    pd.concat(chains, ignore_index=True).to_csv(DATA / f"chains_{today}.csv", index=False)

    with open(DATA / f"rate_{today}.json", "w") as f:
        json.dump({"date": today, "r": r, "spots": spots}, f, indent=2)
    print(f"{today}: r = {r:.5f}, spots = {spots}")


if __name__ == "__main__":
    main()
