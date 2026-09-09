"""ACTE III — Heston mono-actif. Tests ecrits AVANT le code.

Meme mecanique que l'Acte II : tout est en @pytest.mark.xfail(strict=True).
Tant que la fonction leve NotImplementedError -> XFAIL. Des que ton code est
juste -> XPASS, la suite passe au rouge, et c'est TON signal pour retirer le
marqueur.

Lance : .venv/bin/python -m pytest tests/test_heston.py -v

Ce qui est teste ici est choisi pour etre INDEPENDANT de ton implementation :
identites de martingale, moments exacts du CIR, limite degeneree vers
Black-Scholes, parite call-put. Aucun test ne compare Heston a Heston.
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

import json

import numpy as np
import pytest

from bs import call_bs
from heston import (heston_call, heston_cf, heston_paths, heston_put,
                    heston_smile)

# Jeu de parametres de reference de l'acte. Feller : 2*kappa*theta = 0.12 > 0.09
# = xi^2, donc la variance ne colle pas a zero — les tests ne mesurent pas le
# comportement pathologique, ils mesurent le cas sain.
HP = dict(S0=100.0, v0=0.04, r=0.05, T=1.0,
          kappa=1.5, theta=0.04, xi=0.3, rho=-0.7)
ROOT = pathlib.Path(__file__).resolve().parent.parent


def _half_width(v):
    """Demi-largeur IC 95% — la convention du repo."""
    return 1.96 * v.std(ddof=1) / np.sqrt(len(v))


# ---------------------------------------------------------------------------
# QUETE 3.1 — trajectoires                                            [40 XP]
# ---------------------------------------------------------------------------


def test_heston_paths_shape_et_depart():
    """Contrat de base : formes, points de depart exacts, variance positive.

    Le v >= 0 n'est pas cosmetique : une seule valeur negative sous une racine
    produit un nan qui contamine toute la trajectoire, et un nan dans un
    payoff donne un prix nan — bruyant. Le vrai danger est le nan qui
    disparait dans un np.maximum(..., 0) et rend un prix silencieusement faux.
    """
    n_steps, N = 50, 5_000
    S, v = heston_paths(**HP, n_steps=n_steps, N=N,
                        rng=np.random.default_rng(1))

    assert S.shape == v.shape == (N, n_steps + 1)
    assert np.all(S[:, 0] == HP["S0"])
    assert np.all(v[:, 0] == HP["v0"])
    assert np.all(v >= 0.0)
    assert np.all(np.isfinite(S)) and np.all(np.isfinite(v))
    assert np.all(S > 0.0)          # log-schema : positivite exacte



def test_heston_paths_degenere_black_scholes():
    """xi = 0 et v0 = theta : Heston N'EST PLUS Heston, c'est Black-Scholes.

    Le boss de tutoriel de l'acte, meme role que la quete 2.2. Sans vol-of-vol
    et demarre a sa moyenne, la variance ne bouge plus : v_t = theta pour tout
    t, exactement (le drift kappa*(theta - theta) est nul, la diffusion aussi).
    Le spot est alors un GBM de vol sqrt(theta), et le prix MC doit retomber
    sur la formule fermee de l'Acte I.

    Si ce test echoue, inutile de regarder le reste : le squelette du schema
    est faux.
    """
    params = dict(HP, xi=0.0, v0=HP["theta"])
    N = 100_000
    S, v = heston_paths(**params, n_steps=100, N=N,
                        rng=np.random.default_rng(2))

    assert np.max(np.abs(v - HP["theta"])) < 1e-12      # variance figee

    K = 100.0
    disc = np.exp(-HP["r"]*HP["T"]) * np.maximum(S[:, -1] - K, 0.0)
    ref = call_bs(HP["S0"], K, np.sqrt(HP["theta"]), HP["r"], HP["T"])
    assert abs(disc.mean() - ref) < _half_width(disc) * 1.5



def test_heston_paths_martingale():
    """E[e^{-rT} S_T] = S0. Identite MODEL-FREE : elle ne suppose rien sur la
    dynamique de la variance, seulement que le drift du spot est r sous la
    probabilite risque-neutre.

    C'est le test le plus rentable de l'acte : il attrape un drift oublie, un
    -0.5*v_t manquant, un dt qui n'est pas T/n_steps, une correlation appliquee
    du mauvais cote. Aucune de ces erreurs ne se voit sur un graphe de
    trajectoires.
    """
    N = 200_000
    S, _ = heston_paths(**HP, n_steps=250, N=N, rng=np.random.default_rng(3))
    disc = np.exp(-HP["r"]*HP["T"]) * S[:, -1]

    assert abs(disc.mean() - HP["S0"]) < _half_width(disc) * 1.5



def test_heston_paths_moyenne_de_la_variance():
    """E[v_T] = theta + (v0 - theta)*exp(-kappa*T), moyenne exacte du CIR.

    On demarre DELIBEREMENT hors de la moyenne de long terme (v0 = 0.09 pour
    theta = 0.04) : si tu partais a v0 = theta, la moyenne serait constante et
    le test passerait meme avec un kappa faux. Ici le test pince a la fois
    kappa, theta et le pas de temps.

    Tolerance 3% relative et non un IC : le schema d'Euler avec troncature
    introduit un biais de discretisation (la troncature ne peut que remonter la
    variance), qui ne disparait pas quand N grandit. C'est une tolerance de
    SCHEMA, pas de statistique — et c'est exactement la nuance a savoir
    expliquer.
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
    """rho doit se lire dans les donnees, pas seulement dans la signature.

    Au premier pas, v vaut v0 > 0 partout : aucune troncature n'a encore eu
    lieu, donc les deux increments sont EXACTEMENT gaussiens et on peut
    remonter aux deux tirages normalises. Leur correlation empirique est rho.

    Ce test pince trois choses d'un coup : le Cholesky (rho au bon endroit),
    l'ordre des deux gaussiennes (inverser Z1 et Z2 change rho en rho aussi,
    mais melanger W_S et W_v casse le signe du skew), et le fait que le pas du
    spot lit v_t et non v_{t+dt}.
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

    assert abs(Z_v.std(ddof=1) - 1.0) < 0.02      # bien normalise
    assert abs(Z_S.std(ddof=1) - 1.0) < 0.02
    assert abs(rho_hat - HP["rho"]) < 0.01


# ---------------------------------------------------------------------------
# QUETE 3.2 — semi-analytique                                         [50 XP]
# ---------------------------------------------------------------------------


def test_heston_cf_proprietes_de_fonction_caracteristique():
    """Trois identites que TOUTE fonction caracteristique verifie.

    Elles ne dependent ni de Heston ni de tes parametres : ce sont les
    proprietes de E[exp(i*u*X)] pour X reelle. Elles coutent trois lignes et
    attrapent la moitie des fautes de recopie de la formule — un signe inverse
    quelque part fait presque toujours sauter |phi| <= 1.
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
    """xi -> 0 avec v0 = theta : log S_T est gaussienne, on connait sa cf.

    m = log S0 + (r - theta/2)*T, variance = theta*T, donc
    phi(u) = exp(i*u*m - u^2*theta*T/2).

    Pourquoi xi = 1e-3 et pas 0 : la formule fait apparaitre xi^2 au
    denominateur. A xi = 0 c'est une division par zero, et a xi = 1e-7 c'est
    une soustraction de deux nombres presque egaux multipliee par 1/xi^2 —
    l'annulation catastrophique te rend du bruit (mesure : l'erreur descend
    jusqu'a xi ~ 1e-5 puis REMONTE). Savoir a quelle distance de la
    singularite on peut encore tester est un reflexe de calcul numerique.

    Et on ne teste pas une tolerance seule, on teste une VITESSE de
    convergence : l'ecart a la gaussienne est en O(xi) — terme de skew en
    rho*xi*u^3, d'ordre UN parce que c'est la correlation qui brise la
    symetrie. Diviser xi par 10 doit donc diviser l'ecart par 10. Un facteur
    d'echelle est bien plus discriminant qu'un seuil : une formule fausse rate
    la pente, pas seulement le niveau.
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
    assert 8.0 < e3/e4 < 12.0          # convergence lineaire en xi



def test_heston_call_limite_black_scholes():
    """xi -> 0, v0 = theta : le prix Heston redonne le prix BS.

    Tolerance 1e-3 et non 1e-10 : la limite est en O(xi) — le terme de skew
    rho*xi*u^3 est d'ordre UN, cf. test_heston_cf_limite_gaussienne. On ne
    teste donc pas une identite mais une convergence, et on prend xi = 1e-4
    pour que l'ecart de modele passe sous la tolerance sans descendre dans la
    zone d'annulation catastrophique (en dessous de xi ~ 1e-5). Ne cherche pas
    a resserrer — une VRAIE erreur de formule (coupure de branche, signe,
    borne d'integration trop courte) se voit a 0.1 ou plus, jamais a 1e-3.
    """
    params = dict(HP, xi=1e-4, v0=HP["theta"])
    for K in (80.0, 100.0, 120.0):
        obtenu = heston_call(K=K, **params)
        ref = call_bs(HP["S0"], K, np.sqrt(HP["theta"]), HP["r"], HP["T"])
        print(f"K={K:6.1f}  heston={obtenu:.6f}  bs={ref:.6f}")
        assert abs(obtenu - ref) < 1e-3


def test_heston_call_bornes_et_monotonie():
    """Bornes model-free et decroissance en K.

    max(S0 - K*exp(-rT), 0) <= C <= S0 : violer ces bornes, c'est offrir un
    arbitrage. Une borne inferieure violee est le symptome typique d'une borne
    d'integration trop courte (l'integrale tronquee sous-estime P1).
    """
    strikes = np.array([70.0, 85.0, 100.0, 115.0, 130.0])
    prix = np.array([heston_call(K=float(K), **HP) for K in strikes])

    assert all(type(heston_call(K=float(K), **HP)) is float for K in (100.0,))
    borne_inf = np.maximum(HP["S0"] - strikes*np.exp(-HP["r"]*HP["T"]), 0.0)
    assert np.all(prix >= borne_inf - 1e-10)
    assert np.all(prix <= HP["S0"] + 1e-10)
    assert np.all(np.diff(prix) < 0.0)



def test_heston_call_accord_avec_le_monte_carlo():
    """Les deux routes doivent se rejoindre : semi-analytique dans l'IC du MC.

    C'est LA validation croisee de l'acte : deux calculs qui ne partagent
    aucune ligne de code (une quadrature contre un schema d'Euler) et qui
    doivent donner le meme nombre. Un ecart systematique du meme signe sur les
    trois strikes signale un biais de discretisation, pas du bruit.
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
    """C - P = S0 - K*exp(-rT), a 1e-10.

    La parite est MODEL-FREE : elle ne depend d'aucune hypothese de dynamique,
    seulement de l'absence d'arbitrage. Donc pas d'argument statistique, pas de
    tolerance genereuse — c'est de l'algebre, comme la parite DI+DO de la
    quete 1.7.
    """
    for K in (80.0, 100.0, 125.0):
        c = heston_call(K=K, **HP)
        p = heston_put(K=K, **HP)
        attendu = HP["S0"] - K*np.exp(-HP["r"]*HP["T"])
        assert abs((c - p) - attendu) < 1e-10


# ---------------------------------------------------------------------------
# QUETE 3.3 — le smile                                                [40 XP]
# ---------------------------------------------------------------------------


def test_smile_plat_quand_xi_tend_vers_zero():
    """Sans vol-of-vol, pas de smile : la surface est plate a sqrt(theta).

    C'est le meme boss degenere, vu depuis les vols implicites. Il verrouille
    au passage le branchement sur implied_vol_call : si tu inverses un put avec
    un solveur de call, ce test explose immediatement.
    """
    params = dict(HP, xi=1e-3, v0=HP["theta"])
    strikes = np.linspace(80.0, 120.0, 9)
    iv = heston_smile(strikes=strikes, **params)

    assert iv.shape == strikes.shape
    assert np.max(np.abs(iv - np.sqrt(HP["theta"]))) < 1e-3



def test_smile_skew_negatif_quand_rho_negatif():
    """rho < 0 -> les puts OTM cotent plus cher -> vol implicite decroissante.

    C'est LE fait de marche que Black-Scholes ne peut pas reproduire, et la
    raison economique pour laquelle les produits structures existent : quelqu'un
    veut acheter de la protection a la baisse, quelqu'un doit la vendre.
    L'effet de levier (rho < 0) est le mecanisme du modele qui le produit.
    """
    strikes = np.linspace(80.0, 120.0, 9)
    iv = heston_smile(strikes=strikes, **HP)          # rho = -0.7

    print("skew :", np.round(iv, 4))
    assert np.all(np.diff(iv) < 0.0)                  # strictement decroissante
    assert iv[0] - iv[-1] > 0.01                      # pente economiquement lisible



def test_smile_symetrique_et_convexe_quand_rho_nul():
    """rho = 0 : plus de pente, mais toujours de la courbure.

    La dissociation est le point a retenir : rho fait la PENTE, xi fait la
    COURBURE. A rho = 0 le smile est symetrique en log-moneyness et la monnaie
    est son minimum — un vrai "smile", pas un "skew".
    """
    params = dict(HP, rho=0.0)
    F = HP["S0"]*np.exp(HP["r"]*HP["T"])              # forward, pas le spot
    k = np.array([-0.2, -0.1, 0.0, 0.1, 0.2])
    strikes = F*np.exp(k)
    iv = heston_smile(strikes=strikes, **params)

    assert abs(iv[0] - iv[-1]) < 5e-3                 # symetrie
    assert abs(iv[1] - iv[3]) < 5e-3
    assert iv[2] < iv[1] and iv[2] < iv[3]            # convexite



def test_smile_courbure_croissante_en_xi():
    """Doubler la vol-of-vol creuse le smile.

    On mesure la courbure par une difference seconde en log-moneyness, a rho
    nul pour isoler l'effet de xi de celui de la pente.
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
# BOSS 3 — DI put sous Heston, avec variable de controle                [80 XP]
# ---------------------------------------------------------------------------

def test_boss3_artefacts():
    """Le boss se valide sur ses artefacts, comme le BOSS 2.

    Lance : .venv/bin/python scripts/boss3_heston_barrier.py

    L'idee : reprendre EXACTEMENT le dispositif du BOSS 2 (DI put, variable de
    controle = put vanille) mais sous Heston. La seule chose qui change est
    l'origine de EX : ce n'est plus put_bs, c'est heston_put — ton prix
    semi-analytique de la quete 3.2. Tout l'Acte II se rebranche sans une ligne
    de modification, et c'est precisement ce qu'on veut montrer : la reduction
    de variance est independante du modele.
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
    assert iv[0] > iv[-1]                    # skew negatif, rho < 0

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
