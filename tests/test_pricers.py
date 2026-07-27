import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

import numpy as np

import implied_vol
from bs import call_bs, put_bs, vega
from mc_engine import pricer_mc_call, delta_mc

PARAMS = dict(S0=100, K=100, sigma=0.2, r=0.05, T=1)

def test_bs_reference_value():
    assert abs(call_bs(**PARAMS) - 10.4506) < 1e-3

def test_put_call_parity_bs():
    lhs = call_bs(**PARAMS) - put_bs(**PARAMS)
    rhs = 100 - 100*np.exp(-0.05)
    assert abs(lhs - rhs) < 1e-10        # analytique : précision machine

def test_mc_within_ci():
    price, ci = pricer_mc_call(**PARAMS, N=100_000)
    assert abs(price - call_bs(**PARAMS)) < ci * 1.5   # marge sur l'IC

def test_delta_crn():
    d = delta_mc(**PARAMS, h=0.1, N=100_000, seed=42)
    assert abs(d - 0.6368) < 0.01

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