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
    raise NotImplementedError("QUÊTE 6.1")


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
        bid     : float
        ask     : float
        mid     : float, (bid + ask) / 2
    """
    raise NotImplementedError("QUÊTE 6.1")


def clean_option_chain(chain: pd.DataFrame, S0: float, r: float, q: float = 0.0) -> pd.DataFrame:
    """Filter a raw option chain down to quotes worth calibrating on.

    Must reject (at least): crossed quotes (bid > ask), zero or missing
    bid/ask, zero volume AND zero open interest together (a stale quote
    nobody has traded), and strikes deep enough ITM/OTM that the spread
    dominates the mid-price.

    Params
    ------
    chain : output of fetch_option_chain.
    S0    : current spot of the underlying.
    r     : risk-free rate (annualised, continuously compounded).
    q     : net cost-of-carry for THIS underlying -- do not assume it is the
            same across GLD, SLV, USO, and do not read it off a yfinance
            "dividend yield" field, which does not mean the same thing for a
            commodity ETF as it does for a stock.

    Returns
    -------
    pd.DataFrame, same columns as `chain`, surviving rows only.
    """
    raise NotImplementedError("QUÊTE 6.1")
