"""Real market data ingestion: option chains and underlying history (yfinance).

Act VI's foundation: everything downstream (Heston calibration 6.2, SVI 6.3,
historical correlation 6.4) is only as trustworthy as the cleaning step here.
yfinance returns raw, unfiltered option chains -- crossed quotes, stale
zero-volume strikes, and bid-ask spreads wide enough to make a "price" nearly
meaningless. This module's job is to turn that into something worth
calibrating on, and to prove it with a model-free check: forward call-put
parity, which holds regardless of which smile model gets fit next.

Names used in Act VI: GLD, SLV, USO (commodity ETFs -- yfinance exposes no
option chain on raw commodity futures, only on the listed ETFs that
replicate them).
"""

from datetime import date

import numpy as np
import pandas as pd
import yfinance as yf


def fetch_underlying_history(ticker: str, period: str = "2y") -> pd.Series:
    """Daily adjusted close history for one ticker.

    Feeds quest 6.4 (historical correlation reconstruction). Not exercised
    beyond a shape sanity check by this quest's own tests.

    Params
    ------
    ticker : yfinance symbol, e.g. "GLD".
    period : yfinance lookback window string, e.g. "2y", "1y", "6mo".

    Returns
    -------
    pd.Series of adjusted close prices, indexed by date, ticker name as
    `.name`.
    """
    hist = yf.Ticker(ticker).history(period=period, auto_adjust=True)
    close = hist["Close"].dropna().astype(float)
    close.name = ticker
    return close


def fetch_option_chain(ticker: str, S0: float, max_expiries: int = 4) -> pd.DataFrame:
    """Raw option chain (calls and puts, several expiries) for one ticker.

    Params
    ------
    ticker       : yfinance symbol, e.g. "GLD".
    S0           : current spot, used only to help the caller later restrict
                   strikes near the money -- not applied as a filter here.
    max_expiries : how many of the nearest listed expiries to pull.

    Returns
    -------
    pd.DataFrame, one row per (strike, expiry, type), columns at least:
        strike  : float
        expiry  : str, 'YYYY-MM-DD' as given by yfinance
        T       : float, years to expiry, ACT/365 from TODAY (not a fixed
                  date -- this must be computed at call time)
        type    : 'call' or 'put'
        bid          : float
        ask          : float
        mid          : float, (bid + ask) / 2
        volume       : float, traded volume of the day (NaN if not reported)
        openInterest : float, open interest (NaN if not reported)
    """
    tk = yf.Ticker(ticker)
    today = date.today()
    frames = []
    for expiry in list(tk.options)[:max_expiries]:
        T = (date.fromisoformat(expiry) - today).days / 365.0
        chain = tk.option_chain(expiry)
        for typ, side in (("call", chain.calls), ("put", chain.puts)):
            f = pd.DataFrame({
                "strike": side["strike"].astype(float),
                "expiry": expiry,
                "T": T,
                "type": typ,
                "bid": side["bid"].astype(float),
                "ask": side["ask"].astype(float),
                "volume": side["volume"].astype(float),
                "openInterest": side["openInterest"].astype(float),
            })
            f["mid"] = (f["bid"] + f["ask"]) / 2
            frames.append(f)
    return pd.concat(frames, ignore_index=True)


def clean_option_chain(chain: pd.DataFrame, S0: float, r: float, q: float = 0.0,
                       max_rel_spread: float = 0.5) -> pd.DataFrame:
    """Filter a raw option chain down to quotes worth calibrating on.

    A row survives only if all four hold:
        1. bid and ask are present, strictly positive, and not crossed (bid <= ask);
        2. volume > 0 OR openInterest > 0 (a quote nobody has traded is stale);
        3. the relative spread (ask - bid) / mid <= max_rel_spread, which removes
           deep ITM/OTM strikes where the spread dominates the price;
        4. T > 0.

    S0, r and q are accepted for the signature but deliberately NOT used to
    filter. A parity-based filter here would make the forward-parity test in
    tests/test_market_data.py circular: it would only check the filter against
    itself. Parity is the check, not a rule.

    Params
    ------
    chain : output of fetch_option_chain, with volume and openInterest columns.
    S0    : current spot of the underlying (unused by the filter, see above).
    r     : risk-free rate (annualised, continuously compounded), unused likewise.
    q     : net cost-of-carry for THIS underlying (unused likewise). Do not
            assume it is the same across GLD, SLV, USO, and do not read it off
            a yfinance "dividend yield" field, which does not mean the same
            thing for a commodity ETF as it does for a stock.
    max_rel_spread : largest accepted (ask - bid) / mid.

    Returns
    -------
    pd.DataFrame, same columns as `chain`, surviving rows only, original index.

    Raises
    ------
    KeyError : if a column needed by the rules above is missing.
    """
    required = {"strike", "expiry", "T", "type", "bid", "ask", "volume", "openInterest"}
    missing = required - set(chain.columns)
    if missing:
        raise KeyError(f"clean_option_chain: colonnes manquantes {sorted(missing)}")

    bid, ask = chain["bid"], chain["ask"]
    mid = (bid + ask) / 2
    priced = (bid > 0) & (ask > 0) & (bid <= ask)   # NaN compares False: rejected
    traded = (chain["volume"].fillna(0) > 0) | (chain["openInterest"].fillna(0) > 0)
    tight = (ask - bid) / mid <= max_rel_spread
    alive = chain["T"] > 0

    return chain.loc[priced & traded & tight & alive].copy()
