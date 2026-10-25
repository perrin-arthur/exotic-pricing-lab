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
import pandas as pd
import pytest

from bs import call_bs, put_bs
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


# ---------------------------------------------------------------------------
# Offline checks for clean_option_chain (no network needed).
#
# A Black-Scholes chain with a known q: mids ARE the closed-form prices, so
# forward parity is an exact identity on every survivor. These tests fix the
# filter before any real data exists. They do not replace the real parity test
# above, which stays xfail until fetch_option_chain can reach yfinance.
# ---------------------------------------------------------------------------

S0_SYN, R_SYN, Q_SYN, SIGMA_SYN = 100.0, 0.045, 0.02, 0.25
EXPIRIES_SYN = (0.25, 0.5)
STRIKES_SYN = np.arange(70.0, 131.0, 5.0)
MAX_REL = 0.5


def _chaine_synthetique() -> pd.DataFrame:
    """Black-Scholes chain. Spread grows with price, so the relative spread
    is large for cheap deep-OTM quotes and small near the money."""
    rows = []
    for T in EXPIRIES_SYN:
        for K in STRIKES_SYN:
            prices = (("call", call_bs(S0_SYN, K, SIGMA_SYN, R_SYN, T, Q_SYN)),
                      ("put", put_bs(S0_SYN, K, SIGMA_SYN, R_SYN, T, Q_SYN)))
            for typ, price in prices:
                half = 0.01 + 0.02 * price
                rows.append(dict(strike=K, expiry=f"T={T}", T=T, type=typ,
                                 bid=price - half, ask=price + half, mid=price,
                                 volume=100.0, openInterest=500.0))
    return pd.DataFrame(rows)


def _locate(chain, K, T, typ):
    mask = (chain["strike"] == K) & np.isclose(chain["T"], T) & (chain["type"] == typ)
    return chain.index[mask][0]


def _injecte_mauvais_quotes(chain):
    """Plant one defect per rule. Returns (dirty chain, indices that must go)."""
    bad = chain.copy()

    i = _locate(bad, 100.0, 0.25, "call")         # crossed: bid > ask
    bad.loc[i, "bid"] = bad.loc[i, "ask"] + 0.10
    crossed = i

    i = _locate(bad, 95.0, 0.25, "put")           # zero bid
    bad.loc[i, "bid"] = 0.0
    zero_bid = i

    i = _locate(bad, 95.0, 0.5, "put")            # missing ask
    bad.loc[i, "ask"] = np.nan
    missing_ask = i

    i = _locate(bad, 105.0, 0.5, "call")          # stale: no volume, no OI
    bad.loc[i, ["volume", "openInterest"]] = 0.0
    stale = i

    i = _locate(bad, 70.0, 0.5, "put")            # deep OTM, spread dominates
    bad.loc[i, ["bid", "ask"]] = [0.001, 0.30]
    wide = i

    rejected = {crossed, zero_bid, missing_ask, stale, wide}

    i = _locate(bad, 110.0, 0.25, "call")         # OI > 0 but volume 0: must SURVIVE
    bad.loc[i, "volume"] = 0.0
    kept_by_oi = i

    return bad, rejected, kept_by_oi


def test_clean_option_chain_rejette_chaque_defaut_et_garde_le_reste():
    """Each planted defect is rejected by its own rule, and nothing else changes.

    The rejection set is restated independently: the injected rows, plus every
    base row whose relative spread exceeds the threshold. The OI-only row must
    survive, which pins the OR in the volume/open-interest rule.
    """
    dirty, rejected, kept_by_oi = _injecte_mauvais_quotes(_chaine_synthetique())

    clean = clean_option_chain(dirty, S0_SYN, r=R_SYN, q=Q_SYN, max_rel_spread=MAX_REL)

    rel = (dirty["ask"] - dirty["bid"]) / ((dirty["bid"] + dirty["ask"]) / 2)
    expected = set(dirty.index[~dirty.index.isin(rejected) & (rel <= MAX_REL)])

    assert set(clean.index) == expected
    assert kept_by_oi in clean.index
    assert rejected.isdisjoint(clean.index)


def test_clean_option_chain_parite_synthetique_exacte():
    """Forward parity on survivors, exact to 1e-10: prices are closed-form here.

    This is the model-free check the real test will run, applied where the
    answer is known. If it fails, the filter kept a pair that does not belong.
    """
    chain = _chaine_synthetique()
    clean = clean_option_chain(chain, S0_SYN, r=R_SYN, q=Q_SYN, max_rel_spread=MAX_REL)

    calls = clean[clean["type"] == "call"].set_index(["strike", "expiry"])
    puts = clean[clean["type"] == "put"].set_index(["strike", "expiry"])
    common = calls.index.intersection(puts.index)
    assert len(common) > 5

    C = calls.loc[common, "mid"].to_numpy()
    P = puts.loc[common, "mid"].to_numpy()
    K = np.array([k for k, _ in common])
    T = calls.loc[common, "T"].to_numpy()

    residual = (C - P) - (S0_SYN * np.exp(-Q_SYN * T) - K * np.exp(-R_SYN * T))
    assert np.max(np.abs(residual)) < 1e-10


def test_clean_option_chain_ne_depend_ni_de_s0_ni_de_r_ni_de_q():
    """Anti-circularity guard: the filter must not read the parity inputs.

    Same chain, absurd S0, r and q: the surviving index must be unchanged.
    Otherwise the parity test would be checking a filter built on parity.
    """
    dirty, _, _ = _injecte_mauvais_quotes(_chaine_synthetique())

    a = clean_option_chain(dirty, S0_SYN, r=R_SYN, q=Q_SYN)
    b = clean_option_chain(dirty, 1e3, r=0.9, q=-0.5)

    assert a.index.equals(b.index)


def test_clean_option_chain_refuse_une_colonne_manquante():
    """Without volume and openInterest the stale-quote rule cannot run: fail loud."""
    chain = _chaine_synthetique().drop(columns=["openInterest"])

    with pytest.raises(KeyError, match="openInterest"):
        clean_option_chain(chain, S0_SYN, r=R_SYN, q=Q_SYN)
