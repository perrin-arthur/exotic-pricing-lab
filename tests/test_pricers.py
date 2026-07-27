import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

import numpy as np

import barriers
import implied_vol
from bs import call_bs, put_bs, vega
from mc_engine import gbm_paths, pricer_mc_call, delta_mc

PARAMS = dict(S0=100, K=100, sigma=0.2, r=0.05, T=1)
# gbm_paths ne prend pas de strike : dict separe, sinon TypeError sur K.
PATH_PARAMS = dict(S0=100, sigma=0.2, r=0.05, T=1)

def test_bs_reference_value():
    assert abs(call_bs(**PARAMS) - 10.4506) < 1e-3

def test_call_bs_dividende():
    """C(S0, K, sigma, r, T, q) == C(S0*exp(-qT), K, sigma, r, T, 0).

    Un call sur sous-jacent a dividende continu EST un call sans dividende
    sur le spot escompte : ln(S0/K) + (r-q)T = ln(S0*exp(-qT)/K) + rT.
    Contrairement a la parite put-call (vraie par construction puisque put_bs
    est DEFINIE par la parite), cette identite casse si le -q manque dans d1.
    """
    q, T = 0.03, 1.0
    lhs = call_bs(100, 100, 0.2, 0.05, T, q)
    rhs = call_bs(100*np.exp(-q*T), 100, 0.2, 0.05, T, 0.0)
    assert abs(lhs - rhs) < 1e-14

def test_put_call_parity_bs():
    lhs = call_bs(**PARAMS) - put_bs(**PARAMS)
    rhs = 100 - 100*np.exp(-0.05)
    assert abs(lhs - rhs) < 1e-10        # analytique : précision machine

def test_mc_within_ci():
    # rng explicite : le determinisme est une decision du TEST, pas une
    # propriete du pricer (dont le defaut est une entropie fraiche).
    price, ci = pricer_mc_call(**PARAMS, N=100_000, rng=np.random.default_rng(42))
    assert abs(price - call_bs(**PARAMS)) < ci * 1.5   # marge sur l'IC

def test_delta_crn():
    d = delta_mc(**PARAMS, h=0.1, N=100_000, seed=42)
    assert abs(d - 0.6368) < 0.01



def test_gbm_paths_shape_et_depart():
    paths = gbm_paths(**PATH_PARAMS, n_steps=50, N=10_000,
                      rng=np.random.default_rng(42))
    assert paths.shape == (10_000, 51)          # n_steps+1 colonnes
    # paths[:, 0] est un ARRAY de 10 000 valeurs -> np.all obligatoire, sinon
    # "truth value of an array is ambiguous". Egalite EXACTE : c'est la colonne
    # de zeros du cumsum qui la garantit, pas un arrondi favorable.
    assert np.all(paths[:, 0] == PATH_PARAMS["S0"])

def test_gbm_paths_call_europeen():
    """Le payoff sur paths[:, -1] doit retrouver le prix BS fermé."""
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
    # tirages independants -> les IC s'ajoutent en quadrature
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
    """DI + DO = put vanille, sur les MEMES trajectoires.

    Identite pathwise : 1{min<H} + 1{min>=H} = 1 sur chaque chemin, donc le
    bruit MC est rigoureusement identique des deux cotes et s'annule dans la
    soustraction. Tolerance 1e-12 (arrondi flottant), PAS un IC.
    Vraie pour tout H : ce n'est pas une propriete du niveau de barriere.
    """
    # rng explicite : sans lui, l'assert de NIVEAU ci-dessous (2.94 sigma)
    # echouerait ~1 run sur 300 sans qu'aucun code n'ait change.
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

    # niveau : MC contre analytique -> tolerance = l'IC. Hors de la boucle,
    # elle ne depend pas de H.
    assert abs(put_mc - put) < ci_van * 1.5
