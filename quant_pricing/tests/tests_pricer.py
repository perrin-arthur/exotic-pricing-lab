import sys, pathlib
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
from bs import call_bs, put_bs
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