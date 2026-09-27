"""Closed-form Black-Scholes, Monte-Carlo pricers, paths, implied vol, barriers.

Each test validates against an independent reference: a closed-form price, a
model-free identity, or a pathwise identity. Tolerances follow the nature of the
comparison -- a confidence interval for Monte-Carlo, machine precision for
algebra.

Run: .venv/bin/python -m pytest tests/test_pricers.py -v
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pytest

import barriers
import implied_vol
from bs import call_bs, put_bs, vega, digital_call_bs, digital_call_replication
from mc_engine import gbm_paths, pricer_mc_call, delta_mc

PARAMS = dict(S0=100, K=100, sigma=0.2, r=0.05, T=1)
# gbm_paths takes no strike, hence a separate dict.
PATH_PARAMS = dict(S0=100, sigma=0.2, r=0.05, T=1)

def test_bs_reference_value():
    assert abs(call_bs(**PARAMS) - 10.4506) < 1e-3

def test_call_bs_dividende():
    """C(S0, K, sigma, r, T, q) == C(S0*exp(-qT), K, sigma, r, T, 0).

    A call on an underlying paying a continuous dividend IS a dividend-free call
    on the discounted spot: ln(S0/K) + (r-q)T = ln(S0*exp(-qT)/K) + rT.
    Unlike call-put parity, which holds by construction since put_bs is DEFINED
    by it, this identity breaks if the -q is missing from d1.
    """
    q, T = 0.03, 1.0
    lhs = call_bs(100, 100, 0.2, 0.05, T, q)
    rhs = call_bs(100*np.exp(-q*T), 100, 0.2, 0.05, T, 0.0)
    assert abs(lhs - rhs) < 1e-14

def test_put_call_parity_bs():
    lhs = call_bs(**PARAMS) - put_bs(**PARAMS)
    rhs = 100 - 100*np.exp(-0.05)
    assert abs(lhs - rhs) < 1e-10        # analytical: machine precision

def test_mc_within_ci():
    # Explicit rng: determinism is a decision of the test, not a property of the
    # pricer, whose default is fresh entropy.
    price, ci = pricer_mc_call(**PARAMS, N=100_000, rng=np.random.default_rng(42))
    assert abs(price - call_bs(**PARAMS)) < ci * 1.5   # margin on the CI

def test_delta_crn():
    d = delta_mc(**PARAMS, h=0.1, N=100_000, seed=42)
    assert abs(d - 0.6368) < 0.01



def test_gbm_paths_shape_et_depart():
    paths = gbm_paths(**PATH_PARAMS, n_steps=50, N=10_000,
                      rng=np.random.default_rng(42))
    assert paths.shape == (10_000, 51)          # n_steps+1 columns
    # paths[:, 0] is an ARRAY of 10 000 values, so np.all is required, otherwise
    # "truth value of an array is ambiguous". The equality is EXACT: it is
    # guaranteed by the leading zero column of the cumulative sum, not by a
    # favourable rounding.
    assert np.all(paths[:, 0] == PATH_PARAMS["S0"])

def test_gbm_paths_call_europeen():
    """The payoff on paths[:, -1] must recover the closed-form BS price."""
    N = 50_000
    paths = gbm_paths(**PATH_PARAMS, n_steps=50, N=N,
                      rng=np.random.default_rng(42))
    disc = np.exp(-0.05) * np.maximum(paths[:, -1] - 100, 0.0)
    price = disc.mean()
    ci = 1.96 * disc.std(ddof=1) / np.sqrt(N)
    assert abs(price - call_bs(**PARAMS)) < ci * 1.5

def test_gbm_paths_loi_independante_de_n_steps():
    N = 50_000
    prices, cis = [], []
    for n_steps in (1, 100):
        paths = gbm_paths(**PATH_PARAMS, n_steps=n_steps, N=N,
                          rng=np.random.default_rng(42))
        disc = np.exp(-0.05) * np.maximum(paths[:, -1] - 100, 0.0)
        prices.append(disc.mean())
        cis.append(1.96 * disc.std(ddof=1) / np.sqrt(N))
    # independent draws -> the CIs add in quadrature
    ci_comb = np.hypot(*cis)
    assert abs(prices[0] - prices[1]) < ci_comb * 1.5


def test_implied_vol_call():
    eps = np.finfo(float).eps
    failures = []
    for K in [70, 100, 130]:
        for T in [0.1, 1.0, 5.0]:
            p = call_bs(100, K, 0.2, 0.05, T, q=0.0)
            v = vega(100, K, 0.2, 0.05, T, 0.0)
            sigma = implied_vol.implied_vol_call(100, K, 0.05, T, p, q=0.0,
                                                 tol=1e-10, max_iter=100)
            err = abs(sigma - 0.2)
            tol_sigma = max(100 * eps * p / v, 100 * eps)
            print(f"K={K:3d} T={T:4.1f}  p={p:11.6g}  vega={v:9.2e}  "
                  f"err={err:.2e}  tol={tol_sigma:.2e}")
            if err >= tol_sigma:
                failures.append((K, T, err, tol_sigma))
    assert not failures, failures


def test_di_do_van():
    """DI + DO = vanilla put, on the SAME paths.

    Pathwise identity: 1{min<H} + 1{min>=H} = 1 on every path, so the
    Monte-Carlo noise is rigorously identical on both sides and cancels in the
    subtraction. Tolerance 1e-12 (floating-point rounding), NOT a confidence
    interval. True for any H: this is not a property of the barrier level.
    """
    # Explicit rng: without it the LEVEL assertion below (2.94 sigma) would fail
    # about one run in 300 with no code having changed.
    paths = gbm_paths(**PATH_PARAMS, n_steps=50, N=100_000,
                      rng=np.random.default_rng(42))
    put = put_bs(**PARAMS)
    put_mc, ci_van = barriers.van_put(paths, K=100, r=0.05, T=1.0)

    for H in (80, 90, 120):
        di, _ = barriers.di_put(paths, K=100, H=H, r=0.05, T=1.0)
        do, _ = barriers.do_put(paths, K=100, H=H, r=0.05, T=1.0)
        print(f"H={H:3d}  DI={di:7.4f}  DO={do:7.4f}  "
              f"DI/vanille={di/put_mc:6.1%}  ecart parite={abs(di+do-put_mc):.2e}")
        assert abs(di + do - put_mc) < 1e-12

    # Level: Monte-Carlo against analytics -> the tolerance is the CI. Outside
    # the loop, since it does not depend on H.
    assert abs(put_mc - put) < ci_van * 1.5


def test_digital_call_bs_vs_mc():
    """digital_call_bs contre une estimation Monte-Carlo indépendante.

    Référence : moyenne actualisée de l'indicatrice 1{S_T > K} sur des
    trajectoires GBM à un pas -- rien à voir avec la formule fermée testée.
    """
    S0, K, sigma, r, T = 100.0, 100.0, 0.2, 0.05, 1.0
    N = 200_000
    rng = np.random.default_rng(7)

    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * rng.standard_normal(N))
    payoff = np.exp(-r * T) * (ST > K).astype(float)
    price_mc = payoff.mean()
    ci = 1.96 * payoff.std(ddof=1) / np.sqrt(N)

    price = digital_call_bs(S0, K, sigma, r, T)
    print(f"digital_call_bs={price:.5f}  MC={price_mc:.5f}+/-{ci:.5f}")

    assert abs(price - price_mc) < ci * 1.5


def test_digital_call_replication_converge_en_h2():
    """La réplication par call spread converge vers la formule fermée en O(h^2).

    Testé comme une PENTE (ratio de convergence quand h est divisé par 2),
    pas comme un seuil sur une seule valeur de h -- une formule fausse rate le
    taux de convergence, pas seulement le niveau.
    """
    S0, K, sigma, r, T = 100.0, 100.0, 0.2, 0.05, 1.0
    ref = digital_call_bs(S0, K, sigma, r, T)

    hs = [0.4, 0.2, 0.1, 0.05]
    errors = [abs(digital_call_replication(S0, h, K, sigma, r, T) - ref) for h in hs]

    print("h:", hs)
    print("erreurs:", errors)

    ratios = [errors[i] / errors[i + 1] for i in range(len(errors) - 1)]
    print("ratios (attendu ~4, division par 2 de h -> erreur /4):", ratios)

    assert all(3.0 < ratio < 5.0 for ratio in ratios)
