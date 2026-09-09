"""Variance reduction: control variates, antithetic variates, pilot coefficient.

Every test validates against an independent reference: a closed-form price, an
exactly known moment, or an algebraic identity. Tolerances are chosen
accordingly -- a confidence interval where the comparison is statistical, 1e-12
where it is algebraic.

Run: .venv/bin/python -m pytest tests/test_variance_reduction.py -v
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

import json

import numpy as np

import barriers
import mc_engine
from bs import put_bs
from mc_engine import (control_variate, control_variate_antithetic,
                       gbm_paths, gbm_paths_antithetic, pilot_c)

PARAMS = dict(S0=100, K=100, sigma=0.2, r=0.05, T=1)
PATH_PARAMS = dict(S0=100, sigma=0.2, r=0.05, T=1)
ROOT = pathlib.Path(__file__).resolve().parent.parent


def _echantillon_correle(n, rng):
    """Synthetic (Y, X, EX, EY) with no finance in it and exactly known moments.

    X = 1 + Z1                      -> E[X] = 1
    Y = 3 + 2*Z1 + 0.5*Z2           -> E[Y] = 3, Cov(Y,X) = 2, Var(X) = 1
    hence c* = 2 and rho* = 2/sqrt(4.25) = 0.97014...
    """
    Z1 = rng.standard_normal(n)
    Z2 = rng.standard_normal(n)
    return 3.0 + 2.0*Z1 + 0.5*Z2, 1.0 + Z1, 1.0, 3.0


def _half_width(v):
    """95% CI half-width of an i.i.d. sample -- the library convention."""
    return 1.96 * v.std(ddof=1) / np.sqrt(len(v))


# ---------------------------------------------------------------------------
# control_variate
# ---------------------------------------------------------------------------


def test_cv_unbiased():
    """The estimate falls within its own CI around the true E[Y] = 3, and its
    half-width is strictly smaller than that of the raw Monte-Carlo."""
    n = 20_000
    Y, X, EX, EY = _echantillon_correle(n, np.random.default_rng(42))
    est, hw, c_hat, rho_hat = control_variate(Y, X, EX)

    assert abs(est - EY) < hw
    assert hw < _half_width(Y)
    assert abs(c_hat - 2.0) < 0.05                 # c* = 2
    assert abs(rho_hat - 2/np.sqrt(4.25)) < 0.02   # rho* = 0.9701
    assert -1.0 <= rho_hat <= 1.0


def test_cv_c_zero_reproduit_le_mc_brut():
    """c = 0 must reproduce the raw Monte-Carlo on Y exactly.

    Base case: if this breaks, the fault is not in the estimation of c but in
    the plumbing -- the mean, the half-width, or the order of the tuple.
    """
    n = 5_000
    Y, X, EX, _ = _echantillon_correle(n, np.random.default_rng(0))
    est, hw, c_hat, _ = control_variate(Y, X, EX, c=0.0)

    assert abs(est - Y.mean()) < 1e-12
    assert abs(hw - _half_width(Y)) < 1e-12
    assert abs(c_hat) < 1e-15          # an imposed c is returned as given


def test_cv_utilise_bien_EX():
    """Guards the parenthesisation: Y - c*(X - EX), not Y - c*X - EX.

    EX = 1000 makes the mistake impossible to miss: the faulty version shifts
    the estimate by ~3000. The second assertion additionally catches the sign
    error Y + c*(X - EX), which doubles the variance instead of reducing it.
    """
    n = 20_000
    rng = np.random.default_rng(1)
    Z1 = rng.standard_normal(n)
    Z2 = rng.standard_normal(n)
    X = 1000.0 + Z1
    Y = 3.0 + 2.0*Z1 + 0.5*Z2

    est, hw, _, _ = control_variate(Y, X, EX=1000.0)

    assert abs(est - 3.0) < 0.05
    assert hw < _half_width(Y)


def test_cv_half_width_sur_le_residu():
    """The half-width must be computed on the residual Z = Y - c(X - EX), not Y.

    Y and X are nearly collinear here, so the residual is almost constant and
    the half-width must collapse. Computed on Y it would not move at all, and
    the variance gain -- the entire point of the technique -- would be
    invisible.
    """
    n = 10_000
    rng = np.random.default_rng(2)
    Z1 = rng.standard_normal(n)
    Z2 = rng.standard_normal(n)
    X = Z1
    Y = 5.0 + Z1 + 1e-6*Z2

    est, hw, _, _ = control_variate(Y, X, EX=0.0)

    assert hw < 0.01 * _half_width(Y)
    assert abs(est - 5.0) < 1e-4


def test_cv_rho_invariant_par_echelle():
    """Guards operator precedence: rho = cov/(sd_Y*sd_X), not cov/sd_Y*sd_X.

    Python reads the faulty form as (cov/sd_Y)*sd_X. Nothing shows while the
    scales are of order 1; multiplying X by 1000 sends rho outside [-1, 1].

    Incidentally, the control is invariant under a rescaling of the control
    itself, c absorbing the factor, so the estimate and the half-width must not
    move either.
    """
    n = 10_000
    Y, X, EX, _ = _echantillon_correle(n, np.random.default_rng(3))

    est1, hw1, c1, rho1 = control_variate(Y, X, EX)
    est2, hw2, c2, rho2 = control_variate(Y, 1000.0*X, 1000.0*EX)

    assert abs(rho2 - rho1) < 1e-10
    assert -1.0 <= rho2 <= 1.0
    assert abs(1000.0*c2 - c1) < 1e-8
    assert abs(est2 - est1) < 1e-9
    assert abs(hw2 - hw1) < 1e-9


def test_cv_retourne_des_floats():
    """Four Python floats, not np.float64 nor 0-d arrays.

    The barrier sweep scripts serialise these values to JSON, and json.dump
    rejects np.float64.
    """
    Y, X, EX, _ = _echantillon_correle(500, np.random.default_rng(4))
    out = control_variate(Y, X, EX)

    assert isinstance(out, tuple) and len(out) == 4
    for v in out:
        assert type(v) is float
    json.dumps(list(out))          # must not raise TypeError


def test_cv_pas_de_N_fantome(monkeypatch):
    """The sample size must be len(Y), never a module-level N.

    Two checks:
      1. the half-width scales as 1/sqrt(n) -- an N captured from the module
         would freeze it and the ratio would collapse;
      2. planting mc_engine.N = 999_999 must change nothing.
    The call goes through a local function, with no ambient variable that could
    be captured by accident.
    """
    def run(n, seed):
        Y, X, EX, _ = _echantillon_correle(n, np.random.default_rng(seed))
        return control_variate(Y, X, EX)

    hw_petit = run(137, 1)[1]
    hw_grand = run(137*4, 1)[1]
    assert 1.7 < hw_petit/hw_grand < 2.3       # factor 2 expected

    avant = run(137, 1)
    monkeypatch.setattr(mc_engine, "N", 999_999, raising=False)
    apres = run(137, 1)
    assert avant == apres


# ---------------------------------------------------------------------------
# Degenerate case: the control equals the quantity of interest
# ---------------------------------------------------------------------------

def test_cv_degenere_Y_egal_X():
    """Y = X = vanilla put, EX = the closed-form Black-Scholes price.

    The control then knows the sampling error exactly, subtracts all of it, and
    the estimator returns the closed-form price with a numerically null
    half-width. The Monte-Carlo has disappeared.

    Tolerance 1e-12 rather than a CI: this is an algebraic identity, no
    statistical argument is involved.
    """
    N = 20_000
    paths = gbm_paths(**PATH_PARAMS, n_steps=50, N=N,
                      rng=np.random.default_rng(7))
    disc = np.exp(-PARAMS["r"]*PARAMS["T"]) * np.maximum(
        PARAMS["K"] - paths[:, -1], 0.0)
    EX = put_bs(**PARAMS)

    est, hw, c_hat, rho_hat = control_variate(disc, disc, EX)

    assert abs(est - EX) < 1e-12
    assert hw < 1e-12
    assert abs(c_hat - 1.0) < 1e-10
    assert abs(rho_hat - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# Control variate applied to the down-and-in put
# ---------------------------------------------------------------------------

def test_di_put_payoffs_valeurs_a_la_main():
    """Toy paths, results worked out by hand. Two failure modes targeted.

    np.max vs np.maximum: np.max(K - ST) returns a single scalar, the maximum
    over the whole array, so all three paths would get the same payoff.
    Axis of the minimum: paths.min() without axis=1 returns the global minimum,
    so path 1, which never touches, would count as knocked in.

    No path touches H exactly, so the strict-versus-loose convention is not
    exercised here; it only has to match di_put.
    """
    K, H, r, T = 100.0, 85.0, 0.10, 2.0
    paths = np.array([
        [100.,  95.,  80.,  90.],   # min=80  < 85 : knocked in  -> put = 10
        [100.,  98.,  96.,  94.],   # min=94 >= 85 : intact      -> put = 6 but Y=0
        [100.,  90.,  84., 130.],   # min=84  < 85 : knocked in  -> put = 0
    ])
    disc = np.exp(-r*T)

    Y, X = barriers.di_put_payoffs(paths, K=K, H=H, r=r, T=T)

    assert Y.shape == (3,) and X.shape == (3,)
    assert np.allclose(X, disc*np.array([10.0, 6.0, 0.0]), atol=1e-12)
    assert np.allclose(Y, disc*np.array([10.0, 0.0, 0.0]), atol=1e-12)


def test_di_put_payoffs_actualisation_coherente():
    """Y, X and EX must live in the same units.

    EX = put_bs is a discounted price. If X came back undiscounted, X.mean() -
    EX would no longer measure a sampling error but a unit mismatch, and the
    control would shift the price instead of stabilising it. Locked down by
    requiring that the two sample means reproduce the plain pricers exactly.
    """
    N = 20_000
    paths = gbm_paths(**PATH_PARAMS, n_steps=50, N=N,
                      rng=np.random.default_rng(42))
    Y, X = barriers.di_put_payoffs(paths, K=100, H=90, r=0.05, T=1.0)

    assert Y.shape == X.shape == (N,)
    assert abs(Y.mean() - barriers.di_put(paths, K=100, H=90, r=0.05, T=1.0)[0]) < 1e-12
    assert abs(X.mean() - barriers.van_put(paths, K=100, r=0.05, T=1.0)[0]) < 1e-12
    # the control is the vanilla put: its mean must sit near the closed form
    assert abs(X.mean() - put_bs(**PARAMS)) < _half_width(X) * 1.5


def test_di_put_cv_accord_avec_mc_brut():
    """Same price as the raw Monte-Carlo, within the CI, on the SAME paths.

    The gap between the two estimators is exactly c*(X_bar - EX): it is of the
    order of the CONTROL's confidence interval, no more. A larger gap means the
    control is not centred -- wrong EX, wrong sigma, wrong discounting.
    """
    paths = gbm_paths(**PATH_PARAMS, n_steps=50, N=50_000,
                      rng=np.random.default_rng(11))
    mc, ci_mc = barriers.di_put(paths, K=100, H=90, r=0.05, T=1.0)
    _, ci_van = barriers.van_put(paths, K=100, r=0.05, T=1.0)
    est, hw, c_hat, rho_hat = barriers.di_put_cv(paths, K=100, H=90,
                                                 r=0.05, T=1.0, sigma=0.2)

    assert abs(est - mc) < ci_van
    assert 0.0 < rho_hat < 1.0
    assert 0.0 < c_hat < 2.0
    assert hw < ci_mc


def test_di_put_cv_reduit_la_variance():
    """The gain must exist, and must never turn into a loss.

    H at 95% of spot: the DI put is nearly the vanilla put, rho is very high, a
    factor 2 on the half-width is required.
    H at 60% of spot: the DI put no longer resembles the vanilla, rho collapses
    -- but an optimal c cannot degrade the estimator, since at worst c -> 0. The
    ratio must stay below 1.
    """
    paths = gbm_paths(**PATH_PARAMS, n_steps=50, N=50_000,
                      rng=np.random.default_rng(12))
    ratios = {}
    for H in (95.0, 60.0):
        _, ci_mc = barriers.di_put(paths, K=100, H=H, r=0.05, T=1.0)
        _, hw, _, rho = barriers.di_put_cv(paths, K=100, H=H, r=0.05, T=1.0,
                                           sigma=0.2)
        ratios[H] = hw / ci_mc
        print(f"H={H:5.1f}  rho={rho:6.3f}  ratio IC={ratios[H]:6.3f}")

    assert ratios[95.0] < 0.5
    assert ratios[60.0] <= 1.0


# ---------------------------------------------------------------------------
# Antithetic variates, alone and combined with the control
# ---------------------------------------------------------------------------


def test_gbm_paths_antithetic_partage_Z():
    """Exact identity: log(up) + log(down) does not depend on the draw.

    The two branches differ only by the sign of the sigma term, so their sum in
    log space is the deterministic drift path, doubled. Two separate calls to
    standard_normal -- that is, two independent draws and no antithetic
    structure at all -- break this identity immediately.
    """
    n_steps, N = 10, 1_000
    up, down = gbm_paths_antithetic(**PATH_PARAMS, n_steps=n_steps, N=N,
                                    rng=np.random.default_rng(3))

    assert up.shape == down.shape == (N, n_steps + 1)
    assert np.all(up[:, 0] == PATH_PARAMS["S0"])
    assert np.all(down[:, 0] == PATH_PARAMS["S0"])
    assert not np.allclose(up, down)          # otherwise Z = 0 everywhere

    t = np.linspace(0.0, PATH_PARAMS["T"], n_steps + 1)
    attendu = 2.0*(np.log(PATH_PARAMS["S0"])
                   + (PATH_PARAMS["r"] - 0.5*PATH_PARAMS["sigma"]**2)*t)
    assert np.max(np.abs(np.log(up) + np.log(down) - attendu)) < 1e-10



def test_cv_antithetic_N_est_le_nombre_de_paires():
    """Dividing by 2N instead of N would make the half-width wrong by sqrt(2).

    The 2N draws are not 2N independent observations. Pairs are formed FIRST,
    giving an i.i.d. sample of N points, and the control is applied AFTERWARDS.
    This test states the exact equality with that definition: if the four
    outputs do not match to 1e-12, the order of operations is wrong.
    """
    n = 8_000
    rng = np.random.default_rng(5)
    Z = rng.standard_normal(n)
    X_up, X_down = np.exp(Z), np.exp(-Z)
    EX = float(np.exp(0.5))                      # E[e^Z], exact
    Y_up = np.maximum(X_up - 1.0, 0.0)
    Y_down = np.maximum(X_down - 1.0, 0.0)

    attendu = control_variate(0.5*(Y_up + Y_down), 0.5*(X_up + X_down), EX)
    obtenu = control_variate_antithetic(Y_up, Y_down, X_up, X_down, EX)

    for a, b in zip(attendu, obtenu):
        assert abs(a - b) < 1e-12


def test_cv_antithetic_domine_chaque_technique_seule():
    """At an EQUAL path budget (2N), the combination beats each technique alone.

    Comparing half-widths at different budgets is meaningless, so all four
    estimators get 2N paths. The vanilla put plays the role of Y, its payoff
    being monotone in Z, which is where antithetic variates are at their best;
    the control is the discounted forward, whose expectation is known exactly:
    E[e^{-rT} S_T] = S0.
    """
    N = 25_000                                   # -> 2N = 50 000 paths
    S0, K, r, T = 100.0, 100.0, 0.05, 1.0
    disc = np.exp(-r*T)

    up, down = gbm_paths_antithetic(**PATH_PARAMS, n_steps=25, N=N,
                                    rng=np.random.default_rng(21))
    plat = gbm_paths(**PATH_PARAMS, n_steps=25, N=2*N,
                     rng=np.random.default_rng(22))

    put = lambda p: disc*np.maximum(K - p[:, -1], 0.0)
    fwd = lambda p: disc*p[:, -1]

    hw_brut = _half_width(put(plat))
    hw_av = _half_width(0.5*(put(up) + put(down)))
    _, hw_cv, _, _ = control_variate(put(plat), fwd(plat), S0)
    _, hw_both, _, _ = control_variate_antithetic(put(up), put(down),
                                                  fwd(up), fwd(down), S0)
    print(f"brut={hw_brut:.5f}  AV={hw_av:.5f}  CV={hw_cv:.5f}  AV+CV={hw_both:.5f}")

    assert hw_both < hw_av
    assert hw_both < hw_cv
    assert hw_both < hw_brut


# ---------------------------------------------------------------------------
# Control coefficient frozen on an independent pilot run
# ---------------------------------------------------------------------------


def test_pilot_c_est_un_scalaire():
    """pilot_c knows neither EX nor discounting: it is a ratio of moments.

    Two invariances prove it:
      * translating Y by a constant leaves c unchanged, since the moments are
        centred;
      * multiplying X by a divides c by a.
    """
    Y, X, _, _ = _echantillon_correle(5_000, np.random.default_rng(6))
    c = pilot_c(Y, X)

    assert type(c) is float
    assert abs(pilot_c(Y + 7.0, X) - c) < 1e-10
    assert abs(pilot_c(Y, 3.0*X) - c/3.0) < 1e-10


def test_pilot_c_proche_du_c_plein():
    """A 10 000-path pilot is enough: c does not need to be precise.

    An error on c does not bias the price, it only forfeits part of the variance
    gain -- the variance is quadratic around the optimum, hence flat there.
    """
    pilote = gbm_paths(**PATH_PARAMS, n_steps=50, N=10_000,
                       rng=np.random.default_rng(31))
    plein = gbm_paths(**PATH_PARAMS, n_steps=50, N=100_000,
                      rng=np.random.default_rng(32))

    Yp, Xp = barriers.di_put_payoffs(pilote, K=100, H=90, r=0.05, T=1.0)
    c_pilote = pilot_c(Yp, Xp)
    _, _, c_plein, _ = barriers.di_put_cv(plein, K=100, H=90, r=0.05, T=1.0,
                                          sigma=0.2)
    print(f"c_pilote={c_pilote:.4f}  c_plein={c_plein:.4f}")

    assert abs(c_pilote - c_plein) / abs(c_plein) < 0.05


def test_pilot_c_elimine_le_biais():
    """c estimated on an INDEPENDENT sample gives an exactly centred estimator.

    Sanity check: eight independent repetitions with the pilot c frozen,
    averaged, must land on a high-precision reference. This does not prove the
    absence of the O(1/N) bias -- it is far too small for that; the proof is
    theoretical, a frozen c being deterministic, so E[c(X_bar - EX)] = 0. It
    does catch any reuse of the pilot paths.
    """
    ref_paths = gbm_paths(**PATH_PARAMS, n_steps=20, N=200_000,
                          rng=np.random.default_rng(99))
    ref, ci_ref = barriers.di_put(ref_paths, K=100, H=90, r=0.05, T=1.0)

    pilote = gbm_paths(**PATH_PARAMS, n_steps=20, N=10_000,
                       rng=np.random.default_rng(100))
    Yp, Xp = barriers.di_put_payoffs(pilote, K=100, H=90, r=0.05, T=1.0)
    c = pilot_c(Yp, Xp)

    estimations, hws = [], []
    for k in range(8):
        paths = gbm_paths(**PATH_PARAMS, n_steps=20, N=50_000,
                          rng=np.random.default_rng(200 + k))
        est, hw, _, _ = barriers.di_put_cv(paths, K=100, H=90, r=0.05, T=1.0,
                                           sigma=0.2, c=c)
        estimations.append(est)
        hws.append(hw)

    moyenne = float(np.mean(estimations))
    hw_moyenne = float(np.mean(hws)) / np.sqrt(len(estimations))
    print(f"ref={ref:.5f}+/-{ci_ref:.5f}  moyenne CV={moyenne:.5f}+/-{hw_moyenne:.5f}")

    assert abs(moyenne - ref) < np.hypot(ci_ref, hw_moyenne) * 1.5


# ---------------------------------------------------------------------------
# Barrier sweep artefacts
# ---------------------------------------------------------------------------


def test_boss2_artefacts():
    """Checks the artefacts produced by scripts/boss2_barrier_sweep.py.

    Run: .venv/bin/python scripts/boss2_barrier_sweep.py
    It must produce figures/boss2_barrier_sweep.png and
    figures/boss2_results.json.

    What the JSON has to show: as the barrier rises towards the spot, the DI put
    resembles the vanilla put more and more, rho increases, and the ratio of
    half-widths collapses.
    """
    fig = ROOT / "figures" / "boss2_barrier_sweep.png"
    res = ROOT / "figures" / "boss2_results.json"
    assert fig.exists(), "figure manquante"
    assert res.exists(), "resultats manquants"

    data = json.loads(res.read_text())
    sweep = data["sweep"]
    assert len(sweep) >= 6

    h = [p["H_pct"] for p in sweep]
    rho = [p["rho"] for p in sweep]
    ratio = [p["ratio_half_width"] for p in sweep]

    assert h == sorted(h)
    assert abs(h[0] - 0.60) < 1e-9 and abs(h[-1] - 0.95) < 1e-9
    assert all(0.0 < x <= 1.0 for x in ratio)          # never degrades
    assert all(-1.0 <= x <= 1.0 for x in rho)
    # rho increasing in H, with a tolerance for Monte-Carlo noise
    assert all(rho[i+1] > rho[i] - 0.02 for i in range(len(rho) - 1))
    assert rho[-1] > rho[0]
    assert ratio[-1] < 0.5                             # large gain near the spot
