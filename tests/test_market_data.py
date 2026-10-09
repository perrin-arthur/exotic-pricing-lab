"""Act VI, quest 6.1: market data ingestion and cleaning (GLD, SLV, USO).

Same mechanic as every act so far: everything is @pytest.mark.xfail(strict=True)
until src/market_data.py is implemented -- XPASS is Arthur's signal, nobody
removes the marker for him.

The validation that matters here is model-free: forward call-put parity on
the CLEANED mid-quotes. It holds whatever smile model gets fit next (6.2,
6.3), so a chain that fails it was not cleaned properly -- the bug is in
clean_option_chain, not in a pricing model that does not exist yet.

Run: .venv/bin/python -m pytest tests/test_market_data.py -v
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pytest

from market_data import clean_option_chain, fetch_option_chain, fetch_underlying_history

TICKER = "GLD"
R = 0.045  # approximate short rate at fetch time; not the point of this test


@pytest.mark.xfail(strict=True, reason="QUÊTE 6.1")
def test_fetch_option_chain_colonnes_et_types():
    """Sanity shape/columns on a real fetch -- one ticker, one call."""
    hist = fetch_underlying_history(TICKER, period="5d")
    S0 = float(hist.iloc[-1])

    chain = fetch_option_chain(TICKER, S0, max_expiries=2)

    expected_cols = {"strike", "expiry", "T", "type", "bid", "ask", "mid"}
    assert expected_cols.issubset(chain.columns)
    assert len(chain) > 0
    assert set(chain["type"].unique()) <= {"call", "put"}
    assert (chain["T"] > 0).all()
    assert np.allclose(chain["mid"], (chain["bid"] + chain["ask"]) / 2)


@pytest.mark.xfail(strict=True, reason="QUÊTE 6.1")
def test_clean_option_chain_respecte_parite_forward():
    """The independent reference: forward parity on surviving mid-quotes.

    C - P = S0*exp(-q*T) - K*exp(-r*T), for options sharing the same strike
    and expiry. Checked PER EXPIRY (q may differ across names, and across
    expiries if the ETF's roll cost is not flat in time -- see src pitfalls).
    The residual must fall inside a band set by the OBSERVED bid-ask spread,
    not an arbitrary absolute tolerance: the spot used here is never exactly
    simultaneous with the option quotes.
    """
    hist = fetch_underlying_history(TICKER, period="5d")
    S0 = float(hist.iloc[-1])

    raw = fetch_option_chain(TICKER, S0, max_expiries=3)
    clean = clean_option_chain(raw, S0, r=R, q=0.0)

    calls = clean[clean["type"] == "call"].set_index(["strike", "expiry"])
    puts = clean[clean["type"] == "put"].set_index(["strike", "expiry"])
    common = calls.index.intersection(puts.index)
    assert len(common) > 5, "pas assez de paires call/put survivantes pour tester la parité"

    C = calls.loc[common, "mid"]
    P = puts.loc[common, "mid"]
    K = np.array([k for k, _ in common])
    T = calls.loc[common, "T"].to_numpy()
    spread = (calls.loc[common, "ask"] - calls.loc[common, "bid"]).to_numpy() + \
             (puts.loc[common, "ask"] - puts.loc[common, "bid"]).to_numpy()

    forward_rhs = S0 - K * np.exp(-R * T)
    residual = (C.to_numpy() - P.to_numpy()) - forward_rhs

    assert np.all(np.abs(residual) < spread), (
        "résidu de parité hors de la bande de bid-ask -- le nettoyage a "
        "laissé passer du bruit de microstructure"
    )
