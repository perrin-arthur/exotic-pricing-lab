"""Single-asset Heston: paths, characteristic function, prices, smile.

Everything tested here is independent of the implementation: martingale
identities, exact CIR moments, the degenerate limit towards Black-Scholes, and
call-put parity. No test compares Heston against Heston.

Run: .venv/bin/python -m pytest tests/test_heston.py -v
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

import json

import numpy as np

from bs import call_bs
from heston import (heston_call, heston_cf, heston_paths, heston_put,
                    heston_smile)

# Reference parameter set. Feller holds: 2*kappa*theta = 0.12 > 0.09 = xi^2, so
# the variance does not stick at zero -- these tests measure the healthy regime,
# not the pathological one.
HP = dict(S0=100.0, v0=0.04, r=0.05, T=1.0,
          kappa=1.5, theta=0.04, xi=0.3, rho=-0.7)
ROOT = pathlib.Path(__file__).resolve().parent.parent


def _half_width(v):
    """95% CI half-width -- the library convention."""
    return 1.96 * v.std(ddof=1) / np.sqrt(len(v))


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def test_heston_paths_shape_et_depart():
    """Base contract: shapes, exact starting points, non-negative variance.

    The v >= 0 check is not cosmetic: a single negative value under a square
    root produces a nan that contaminates the whole path, and a nan in a payoff
    gives a nan price, which is loud. The real danger is the nan that vanishes
    inside a np.maximum(..., 0) and returns a silently wrong price.
    """
    n_steps, N = 50, 5_000
    S, v = heston_paths(**HP, n_steps=n_steps, N=N,
                        rng=np.random.default_rng(1))

    assert S.shape == v.shape == (N, n_steps + 1)
    assert np.all(S[:, 0] == HP["S0"])
    assert np.all(v[:, 0] == HP["v0"])
    assert np.all(v >= 0.0)
    assert np.all(np.isfinite(S)) and np.all(np.isfinite(v))
    assert np.all(S > 0.0)          # log scheme: exact positivity



def test_heston_paths_degenere_black_scholes():
    """xi = 0 with v0 = theta: Heston is no longer Heston, it is Black-Scholes.

    Without vol-of-vol and started at its long-run mean, the variance stops
    moving: v_t = theta for all t, exactly, since the drift kappa*(theta-theta)
    and the diffusion both vanish. The spot is then a GBM of volatility
    sqrt(theta), and the Monte-Carlo price must land on the closed form.

    If this fails, there is no point looking further: the skeleton of the scheme
    is wrong.
    """
    params = dict(HP, xi=0.0, v0=HP["theta"])
    N = 100_000
    S, v = heston_paths(**params, n_steps=100, N=N,
                        rng=np.random.default_rng(2))

    assert np.max(np.abs(v - HP["theta"])) < 1e-12      # variance frozen

    K = 100.0
    disc = np.exp(-HP["r"]*HP["T"]) * np.maximum(S[:, -1] - K, 0.0)
    ref = call_bs(HP["S0"], K, np.sqrt(HP["theta"]), HP["r"], HP["T"])
    assert abs(disc.mean() - ref) < _half_width(disc) * 1.5



def test_heston_paths_martingale():
    """E[e^{-rT} S_T] = S0, a model-free identity.

    It assumes nothing about the variance dynamics, only that the spot drifts at
    r under the risk-neutral measure. It catches a missing drift, a missing
    -0.5*v_t, a dt that is not T/n_steps, and a correlation applied on the wrong
    side -- none of which show up on a plot of the paths.
    """
    N = 200_000
    S, _ = heston_paths(**HP, n_steps=250, N=N, rng=np.random.default_rng(3))
    disc = np.exp(-HP["r"]*HP["T"]) * S[:, -1]

    assert abs(disc.mean() - HP["S0"]) < _half_width(disc) * 1.5



def test_heston_paths_moyenne_de_la_variance():
    """E[v_T] = theta + (v0 - theta)*exp(-kappa*T), the exact CIR mean.

    The starting point is deliberately away from the long-run mean (v0 = 0.09
    for theta = 0.04): starting at v0 = theta would make the mean constant and
    the test would pass even with a wrong kappa. Here it pins kappa, theta and
    the time step at once.

    Tolerance is 3% relative rather than a CI: the Euler scheme with truncation
    carries a discretisation bias -- truncation can only push the variance up --
    which does not shrink as N grows. This is a tolerance on the SCHEME, not on
    the statistics.
    """
    params = dict(HP, v0=0.09)
    N = 100_000
    _, v = heston_paths(**params, n_steps=250, N=N,
                        rng=np.random.default_rng(4))

    attendu = HP["theta"] + (0.09 - HP["theta"])*np.exp(-HP["kappa"]*HP["T"])
    mesure = v[:, -1].mean()
    print(f"E[v_T] attendu={attendu:.6f}  mesure={mesure:.6f}")

    assert abs(mesure - attendu) / attendu < 0.03



def test_heston_paths_correlation_du_premier_pas():
    """rho must be readable in the data, not only in the signature.

    At the first step v equals v0 > 0 everywhere, so no truncation has occurred
    yet, both increments are exactly gaussian, and the two normalised draws can
    be recovered. Their empirical correlation is rho.

    This pins three things at once: the Cholesky factorisation with rho in the
    right place, the ordering of the two gaussians, and the fact that the spot
    step reads v_t rather than v_{t+dt}.
    """
    n_steps, N = 20, 200_000
    S, v = heston_paths(**HP, n_steps=n_steps, N=N,
                        rng=np.random.default_rng(5))
    dt = HP["T"]/n_steps

    Z_v = (v[:, 1] - HP["v0"] - HP["kappa"]*(HP["theta"] - HP["v0"])*dt) \
        / (HP["xi"]*np.sqrt(HP["v0"]*dt))
    Z_S = (np.log(S[:, 1]/HP["S0"]) - (HP["r"] - 0.5*HP["v0"])*dt) \
        / np.sqrt(HP["v0"]*dt)

    rho_hat = float(np.corrcoef(Z_S, Z_v)[0, 1])
    print(f"rho impose={HP['rho']}  rho mesure={rho_hat:.4f}")

    assert abs(Z_v.std(ddof=1) - 1.0) < 0.02      # correctly normalised
    assert abs(Z_S.std(ddof=1) - 1.0) < 0.02
    assert abs(rho_hat - HP["rho"]) < 0.01


# ---------------------------------------------------------------------------
# Characteristic function and semi-analytical prices
# ---------------------------------------------------------------------------


def test_heston_cf_proprietes_de_fonction_caracteristique():
    """Three identities satisfied by ANY characteristic function.

    They depend neither on Heston nor on the parameters: they are properties of
    E[exp(i*u*X)] for real X. They cost three lines and catch half of the
    transcription errors -- an inverted sign almost always breaks |phi| <= 1.
    """
    phi0 = heston_cf(0.0, **HP)
    assert abs(phi0 - 1.0) < 1e-12

    u = np.linspace(-50.0, 50.0, 201)
    phi = np.asarray(heston_cf(u, **HP), dtype=complex)
    assert np.all(np.abs(phi) <= 1.0 + 1e-10)

    phi_plus = heston_cf(3.7, **HP)
    phi_moins = heston_cf(-3.7, **HP)
    assert abs(phi_moins - np.conj(phi_plus)) < 1e-10



def test_heston_cf_limite_gaussienne():
    """xi -> 0 with v0 = theta: log S_T is gaussian and its cf is known.

    m = log S0 + (r - theta/2)*T, variance = theta*T, hence
    phi(u) = exp(i*u*m - u^2*theta*T/2).

    Why xi = 1e-3 rather than 0: the formula has xi^2 in a denominator. At
    xi = 0 that is a division by zero, and at xi = 1e-7 it is a difference of
    two nearly equal numbers multiplied by 1/xi^2 -- catastrophic cancellation
    returns noise. Measured, the error decreases down to xi ~ 1e-5 and then
    RISES again.

    What is tested is not a tolerance but a RATE of convergence: the departure
    from the gaussian is O(xi), the skew term rho*xi*u^3 being first order
    because it is the correlation that breaks the symmetry. Dividing xi by ten
    must divide the error by ten. A scale factor is far more discriminating than
    a threshold: a wrong formula misses the slope, not just the level.
    """
    u = np.array([0.5, 1.0, 2.0, 5.0])
    m = np.log(HP["S0"]) + (HP["r"] - 0.5*HP["theta"])*HP["T"]
    attendu = np.exp(1j*u*m - 0.5*u**2*HP["theta"]*HP["T"])

    def ecart(xi):
        obtenu = np.asarray(heston_cf(u, **dict(HP, xi=xi, v0=HP["theta"])),
                            dtype=complex)
        return float(np.max(np.abs(obtenu - attendu)))

    e3, e4 = ecart(1e-3), ecart(1e-4)
    print(f"ecart(xi=1e-3)={e3:.3e}  ecart(xi=1e-4)={e4:.3e}  ratio={e3/e4:.2f}")

    assert e3 < 1e-3
    assert 8.0 < e3/e4 < 12.0          # linear convergence in xi



def test_heston_call_limite_black_scholes():
    """xi -> 0 with v0 = theta: the Heston price returns the BS price.

    Tolerance 1e-3 rather than 1e-10: the limit is O(xi), the skew term
    rho*xi*u^3 being first order. This tests a convergence, not an identity, and
    xi = 1e-4 keeps the model error under the tolerance without dropping into
    the cancellation zone below xi ~ 1e-5. A genuine formula error -- branch
    cut, sign, truncated integration range -- shows up at 0.1 or more, never at
    1e-3.
    """
    params = dict(HP, xi=1e-4, v0=HP["theta"])
    for K in (80.0, 100.0, 120.0):
        obtenu = heston_call(K=K, **params)
        ref = call_bs(HP["S0"], K, np.sqrt(HP["theta"]), HP["r"], HP["T"])
        print(f"K={K:6.1f}  heston={obtenu:.6f}  bs={ref:.6f}")
        assert abs(obtenu - ref) < 1e-3


def test_heston_call_bornes_et_monotonie():
    """Model-free bounds and decrease in K.

    max(S0 - K*exp(-rT), 0) <= C <= S0: violating these offers an arbitrage. A
    violated lower bound is the typical symptom of a truncated integration range
    under-estimating P1.
    """
    strikes = np.array([70.0, 85.0, 100.0, 115.0, 130.0])
    prix = np.array([heston_call(K=float(K), **HP) for K in strikes])

    assert all(type(heston_call(K=float(K), **HP)) is float for K in (100.0,))
    borne_inf = np.maximum(HP["S0"] - strikes*np.exp(-HP["r"]*HP["T"]), 0.0)
    assert np.all(prix >= borne_inf - 1e-10)
    assert np.all(prix <= HP["S0"] + 1e-10)
    assert np.all(np.diff(prix) < 0.0)



def test_heston_call_accord_avec_le_monte_carlo():
    """The two routes must meet: semi-analytical price within the MC interval.

    This is the cross-validation of the module: two computations sharing no line
    of code -- a quadrature against an Euler scheme -- returning the same number.
    A systematic gap of the same sign across the three strikes signals a
    discretisation bias rather than noise.
    """
    N = 200_000
    S, _ = heston_paths(**HP, n_steps=250, N=N, rng=np.random.default_rng(6))
    disc_factor = np.exp(-HP["r"]*HP["T"])

    for K in (90.0, 100.0, 110.0):
        payoff = disc_factor*np.maximum(S[:, -1] - K, 0.0)
        mc, hw = payoff.mean(), _half_width(payoff)
        exact = heston_call(K=K, **HP)
        print(f"K={K:6.1f}  MC={mc:.4f}+/-{hw:.4f}  exact={exact:.4f}")
        assert abs(exact - mc) < hw * 2.0


def test_heston_put_parite_call_put():
    """C - P = S0 - K*exp(-rT), to 1e-10.

    Parity is model-free: it assumes no dynamics, only no-arbitrage. So no
    statistical argument and no generous tolerance -- this is algebra, like the
    DI + DO = vanilla identity in test_pricers.py.
    """
    for K in (80.0, 100.0, 125.0):
        c = heston_call(K=K, **HP)
        p = heston_put(K=K, **HP)
        attendu = HP["S0"] - K*np.exp(-HP["r"]*HP["T"])
        assert abs((c - p) - attendu) < 1e-10


# ---------------------------------------------------------------------------
# Implied volatility smile
# ---------------------------------------------------------------------------


def test_smile_plat_quand_xi_tend_vers_zero():
    """No vol-of-vol, no smile: the surface is flat at sqrt(theta).

    The same degenerate limit seen from implied volatilities. It also locks down
    the wiring into implied_vol_call: inverting a put with a call solver blows
    this up immediately.
    """
    params = dict(HP, xi=1e-3, v0=HP["theta"])
    strikes = np.linspace(80.0, 120.0, 9)
    iv = heston_smile(strikes=strikes, **params)

    assert iv.shape == strikes.shape
    assert np.max(np.abs(iv - np.sqrt(HP["theta"]))) < 1e-3



def test_smile_skew_negatif_quand_rho_negatif():
    """rho < 0 -> OTM puts quote richer -> implied volatility decreasing in K.

    This is the market fact Black-Scholes cannot reproduce, and the economic
    reason structured products exist: someone wants to buy downside protection,
    someone has to sell it. Leverage (rho < 0) is the mechanism producing it.
    """
    strikes = np.linspace(80.0, 120.0, 9)
    iv = heston_smile(strikes=strikes, **HP)          # rho = -0.7

    print("skew :", np.round(iv, 4))
    assert np.all(np.diff(iv) < 0.0)                  # strictly decreasing
    assert iv[0] - iv[-1] > 0.01                      # economically readable slope



def test_smile_symetrique_et_convexe_quand_rho_nul():
    """rho = 0: no slope left, but still curvature.

    The dissociation is the point: rho makes the SLOPE, xi makes the CURVATURE.
    At rho = 0 the smile is symmetric in log-moneyness and the money is its
    minimum -- a genuine smile, not a skew.
    """
    params = dict(HP, rho=0.0)
    F = HP["S0"]*np.exp(HP["r"]*HP["T"])              # forward, not the spot
    k = np.array([-0.2, -0.1, 0.0, 0.1, 0.2])
    strikes = F*np.exp(k)
    iv = heston_smile(strikes=strikes, **params)

    assert abs(iv[0] - iv[-1]) < 5e-3                 # symmetry
    assert abs(iv[1] - iv[3]) < 5e-3
    assert iv[2] < iv[1] and iv[2] < iv[3]            # convexity



def test_smile_courbure_croissante_en_xi():
    """Doubling the vol-of-vol deepens the smile.

    Curvature is measured by a second difference in log-moneyness, at rho = 0 to
    isolate the effect of xi from that of the slope.
    """
    F = HP["S0"]*np.exp(HP["r"]*HP["T"])
    strikes = F*np.exp(np.array([-0.15, 0.0, 0.15]))

    def courbure(xi):
        iv = heston_smile(strikes=strikes, **dict(HP, rho=0.0, xi=xi))
        return iv[0] - 2*iv[1] + iv[2]

    c_petit, c_grand = courbure(0.2), courbure(0.5)
    print(f"courbure xi=0.2 : {c_petit:.5f}   xi=0.5 : {c_grand:.5f}")

    assert c_petit > 0.0
    assert c_grand > c_petit * 1.5


# ---------------------------------------------------------------------------
# Heston barrier sweep artefacts
# ---------------------------------------------------------------------------

def test_boss3_artefacts():
    """Checks the artefacts produced by scripts/boss3_heston_barrier.py.

    Run: .venv/bin/python scripts/boss3_heston_barrier.py

    The setup is the one of the Black-Scholes sweep -- DI put, vanilla put as
    control -- transposed under Heston. The only thing that changes is where EX
    comes from: heston_put instead of put_bs. The variance reduction machinery
    is reused unmodified, which is the point: it is a statistical technique,
    independent of the model.
    """
    fig = ROOT / "figures" / "boss3_heston.png"
    res = ROOT / "figures" / "boss3_results.json"
    assert fig.exists(), "figure manquante"
    assert res.exists(), "resultats manquants"

    data = json.loads(res.read_text())

    smile = data["smile"]
    assert len(smile) >= 7
    iv = [p["iv"] for p in smile]
    assert all(0.05 < x < 1.0 for x in iv)
    assert iv[0] > iv[-1]                    # negative skew, rho < 0

    sweep = data["sweep"]
    assert len(sweep) >= 6
    h = [p["H_pct"] for p in sweep]
    rho = [p["rho"] for p in sweep]
    ratio = [p["ratio_half_width"] for p in sweep]

    assert h == sorted(h)
    assert abs(h[0] - 0.60) < 1e-9 and abs(h[-1] - 0.95) < 1e-9
    assert all(0.0 < x <= 1.0 for x in ratio)
    assert all(rho[i+1] > rho[i] - 0.02 for i in range(len(rho) - 1))
    assert ratio[-1] < 0.5
